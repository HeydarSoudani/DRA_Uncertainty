"""Belief model training, one phase per call (``src/belief_model_training``).

    1  data          build a training dataset: --dataset (infoseek | webshaper_frames |
                     trqa_validation), --limit (the next N questions; 0 = rewrite the
                     files only), --source (webshaper_frames: webshaper, frames)
    2  trajectories  dra_inference on --dataset / --subset (sft | rl | val) for each
                     agent (--policy-model, default shared.policy_models): generate,
                     then the teacher's beliefs on sft and val; then the statistics
    3  sft           SFT samples from the teacher's replies, LoRA SFT, merged into
                     {BELIEF_TRAINING_ROOT}/sft/final (torchrun for several GPUs)
    4  rl            GRPO over belief chains from the SFT model; --beliefs gold (run
                     rl) or own (run rl_own_init)
    5  evaluate      dra_inference --stage belief --belief-model on --dataset /
                     --subset for each agent (a trained model or a baseline)

Settings: --config (YAML; sections shared, data, trajectories, sft, rl) and
--set section.key=value (repeatable).  The trained models are served by
dra_inference as qwen3.5_4b_sft, qwen3.5_4b_rl, qwen3.5_4b_rl_own_init.
"""

import argparse
import logging
from pathlib import Path

import yaml
from dotenv import load_dotenv

load_dotenv()

from belief_model_training.config import RL_BELIEFS, load_config

_CONFIG_DEFAULT = str(Path(__file__).resolve().parents[1] / "configs" / "belief_training.yaml")

#: The flags each phase takes (besides --phase, --config, --set).
PHASE_FLAGS = {
    1: {"dataset", "limit", "source"},
    2: {"dataset", "subset", "policy_model", "limit"},
    3: set(),
    4: {"beliefs"},
    5: {"dataset", "subset", "policy_model", "belief_model", "limit"},
}
REQUIRED = {1: {"dataset"}, 2: {"dataset", "subset"}, 5: {"dataset", "belief_model"}}


def _non_negative_int(value: str) -> int:
    n = int(value)
    if n < 0:
        raise argparse.ArgumentTypeError(f"must be >= 0, got {n}")
    return n


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Belief model training, one phase per call.",
                                     formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__,
                                     allow_abbrev=False)
    parser.add_argument("--phase", type=int, required=True, choices=sorted(PHASE_FLAGS))
    parser.add_argument("--config", type=str, default=_CONFIG_DEFAULT, help="YAML settings.")
    parser.add_argument("--set", action="append", default=[], metavar="SECTION.KEY=VALUE",
                        help="Override a config key (the value is read as YAML).")
    parser.add_argument("--dataset", type=str, default=None, help="Phases 1, 2, 5: the dataset.")
    parser.add_argument("--subset", type=str, default=None,
                        help="Phase 2: the split (sft, rl, val); phase 5: the dataset's subset (dra_inference --subset).")
    parser.add_argument("--policy-model", type=str, default=None,
                        help="Phases 2, 5: one agent (default: every agent of shared.policy_models).")
    parser.add_argument("--limit", type=_non_negative_int, default=None,
                        help="Phase 1: the next N questions; phases 2, 5: the queries each dra_inference stage runs.")
    parser.add_argument("--source", type=str, default=None, help="Phase 1, webshaper_frames: sources, comma-separated.")
    parser.add_argument("--belief-model", type=str, default=None, help="Phase 5: a dra_inference --belief-model.")
    parser.add_argument("--beliefs", type=str, default=None, choices=list(RL_BELIEFS),
                        help="Phase 4: starting beliefs (rl.beliefs): gold or own.")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = _parse_args()
    given = {k for k in ("dataset", "subset", "policy_model", "limit", "source", "belief_model", "beliefs")
             if getattr(args, k) is not None}
    extra = given - PHASE_FLAGS[args.phase]
    if extra:
        raise SystemExit(f"--phase {args.phase} takes no {', '.join('--' + k.replace('_', '-') for k in sorted(extra))}")
    missing = REQUIRED.get(args.phase, set()) - given
    if missing:
        raise SystemExit(f"--phase {args.phase} needs {', '.join('--' + k.replace('_', '-') for k in sorted(missing))}")

    overrides = {}
    for item in args.set:
        key, sep, value = item.partition("=")
        if not sep:
            raise SystemExit(f"--set {item}: expected SECTION.KEY=VALUE")
        overrides[key.strip()] = yaml.safe_load(value)
    if args.beliefs is not None:
        overrides["rl.beliefs"] = args.beliefs
    cfg = load_config(args.config, overrides)
    agents = [args.policy_model] if args.policy_model else list(cfg.shared.policy_models)

    if args.phase == 1:
        from belief_model_training.data.build import build_dataset
        sources = [s.strip() for s in args.source.split(",") if s.strip()] if args.source else None
        build_dataset(cfg, args.dataset, args.limit, sources)
    elif args.phase == 2:
        from belief_model_training.trajectories.run import run_trajectories
        run_trajectories(cfg, args.dataset, args.subset, agents, args.limit)
    elif args.phase == 3:
        from belief_model_training.sft.train import run_sft
        run_sft(cfg)
    elif args.phase == 4:
        from belief_model_training.rl.loop import run_rl
        run_rl(cfg)
    else:
        from belief_model_training.evaluate.run import run_evaluation
        run_evaluation(cfg, args.belief_model, args.dataset, args.subset, agents, args.limit)


if __name__ == "__main__":
    main()

# python experiments/training/train_belief_model.py --phase 1 --dataset infoseek --limit 500
# python experiments/training/train_belief_model.py --phase 1 --dataset trqa_validation
# python experiments/training/train_belief_model.py --phase 2 --dataset infoseek --subset sft --limit 50
# python experiments/training/train_belief_model.py --phase 2 --dataset infoseek --subset rl
# torchrun --nproc_per_node 4 experiments/training/train_belief_model.py --phase 3
# python experiments/training/train_belief_model.py --phase 4 --beliefs gold
# python experiments/training/train_belief_model.py --phase 5 --belief-model qwen3.5_4b_rl --dataset infoseek --subset val
# python experiments/training/train_belief_model.py --phase 5 --belief-model qwen3.5_4b_rl --dataset browsecomp_plus
