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
--set section.key=value (repeatable); --retriever, --workers, --belief-workers
and --beliefs override shared.retriever, trajectories.workers,
trajectories.belief_workers and rl.beliefs.  The trained models are served by
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
    2: {"dataset", "subset", "policy_model", "retriever", "workers", "belief_workers", "limit"},
    3: {"retriever"},
    4: {"retriever", "beliefs"},
    5: {"dataset", "subset", "policy_model", "retriever", "belief_workers", "belief_model", "limit"},
}
REQUIRED = {1: {"dataset"}, 2: {"dataset", "subset"}, 5: {"dataset", "belief_model"}}
#: The flags that override a config key.
CONFIG_FLAGS = {"retriever": "shared.retriever", "workers": "trajectories.workers",
                "belief_workers": "trajectories.belief_workers", "beliefs": "rl.beliefs"}


def _non_negative_int(value: str) -> int:
    n = int(value)
    if n < 0:
        raise argparse.ArgumentTypeError(f"must be >= 0, got {n}")
    return n


def _positive_int(value: str) -> int:
    n = int(value)
    if n < 1:
        raise argparse.ArgumentTypeError(f"must be >= 1, got {n}")
    return n


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Belief model training, one phase per call.",
                                     formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__,
                                     allow_abbrev=False)
    # ── Run ─────────────────────────────────────────────────────────────────
    parser.add_argument("--phase", type=int, required=True, choices=sorted(PHASE_FLAGS), help="Phase to run: 1 data, 2 trajectories, 3 sft, 4 rl, 5 evaluate (see below).")
    parser.add_argument("--config", type=str, default=_CONFIG_DEFAULT, help="YAML settings (sections shared, data, trajectories, sft, rl).")
    parser.add_argument("--set", action="append", default=[], metavar="SECTION.KEY=VALUE", help="Override a config key (the value is read as YAML); repeatable.")
    # ── Shared (each phase takes the ones listed in PHASE_FLAGS) ────────────
    parser.add_argument("--dataset", type=str, default=None, help="Phases 1, 2, 5: the dataset. Phase 1: infoseek | webshaper_frames | trqa_validation; phase 2: a built training dataset; phase 5: a training dataset or a dra_inference --dataset.")
    parser.add_argument("--subset", type=str, default=None, help="Phases 2, 5: phase 2: the split (sft, rl, val); phase 5: the dataset's subset (dra_inference --subset).")
    parser.add_argument("--policy-model", type=str, default=None, help="Phases 2, 5: one agent (default: every agent of shared.policy_models).")
    parser.add_argument("--retriever", type=str, default=None, help="Phases 2 to 5: the retriever of the runs (overrides shared.retriever); phases 3, 4 read the runs made with it.")
    parser.add_argument("--workers", type=_positive_int, default=None, help="Phase 2: query worker processes of the generate stage (dra_inference --workers; overrides trajectories.workers).")
    parser.add_argument("--belief-workers", type=_positive_int, default=None, help="Phases 2, 5: threads of the belief stage (dra_inference --belief-workers; overrides trajectories.belief_workers).")
    parser.add_argument("--limit", type=_non_negative_int, default=None, help="Phases 1, 2, 5: phase 1: the next N questions (0 = rewrite the files only); phases 2, 5: the queries each dra_inference stage runs.")
    # ── Phase 1: data ───────────────────────────────────────────────────────
    parser.add_argument("--source", type=str, default=None, help="webshaper_frames only: sources, comma-separated (webshaper, frames).")
    # ── Phase 4: rl ─────────────────────────────────────────────────────────
    parser.add_argument("--beliefs", type=str, default=None, choices=list(RL_BELIEFS), help="Starting beliefs (overrides rl.beliefs): gold (run rl) or own (run rl_own_init).")
    # ── Phase 5: evaluate ───────────────────────────────────────────────────
    parser.add_argument("--belief-model", type=str, default=None, help="A dra_inference --belief-model (a trained model or a baseline).")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = _parse_args()
    given = {k for k in set().union(*PHASE_FLAGS.values()) if getattr(args, k) is not None}
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
    for flag, key in CONFIG_FLAGS.items():
        if getattr(args, flag) is not None:
            overrides[key] = getattr(args, flag)
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
