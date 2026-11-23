"""Belief model training, one stage per call (``src/belief_model_training``).

    data_prepare    the questions of a training dataset worth its gold
              annotation, no LLM call: --dataset (infoseek_hard | frames |
              webshaper | trqa_val); source as loaded in
              data/training_datasets_raw/{dataset}, the questions that pass
              the checks in its prepared.jsonl
    data_relevance  the gold annotation, qrels and sft / rl / val split of
              the prepared questions (LLM calls): --dataset, --limit (the
              next N prepared questions; 0 = rewrite the files only)
    generate  dra_inference --stage generate on --dataset / --subset (sft | rl |
              val) for each agent (--policy-model, default shared.policy_models);
              then the statistics
    teacher   dra_inference --stage belief --belief-model {--teacher, default
              shared.teacher} on --dataset / --subset (sft | val) for each
              agent, over the generated trajectories
    filter    what is trained on (CPU): the SFT samples from the teacher's
              replies (--teacher) and the RL episodes, under
              {BELIEF_TRAINING_ROOT}/filter (rules: the filter section)
    sft       LoRA SFT of --student (default shared.student) on the filtered
              samples, merged into {BELIEF_TRAINING_ROOT}/{student}/sft/final
              (torchrun for several GPUs)
    rl        GRPO over belief chains of the filtered episodes, from
              --init-model (default the student's sft/final); --beliefs gold
              (run rl) or own (run rl_own_init)
    evaluate  dra_inference --stage belief --belief-model on --dataset /
              --subset for each agent (a trained model or a baseline)

Settings: --config (YAML; sections shared, data (stages data_prepare and
data_relevance), generate, belief_stage,
filter, sft, rl); to change one, edit the YAML or pass another file.  The
flags --teacher, --student, --init-model and --beliefs overwrite their YAML
keys.  The trained models are served by dra_inference as {student}_sft,
{student}_rl, {student}_rl_own_init (e.g. qwen3.5_4b_rl).
"""

import argparse
import logging
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from belief_model_training.config import RL_BELIEFS, load_config

_CONFIG_DEFAULT = str(Path(__file__).resolve().parents[1] / "configs" / "belief_training.yaml")

STAGES = ("data_prepare", "data_relevance", "generate", "teacher", "filter", "sft", "rl", "evaluate")

#: The flags each stage takes (besides --stage, --config).
STAGE_FLAGS = {
    "data_prepare":   {"dataset"},
    "data_relevance": {"dataset", "limit"},
    "generate": {"dataset", "subset", "policy_model", "limit"},
    "teacher":  {"dataset", "subset", "policy_model", "limit", "teacher"},
    "filter":   {"teacher"},
    "sft":      {"student"},
    "rl":       {"student", "init_model", "beliefs"},
    "evaluate": {"dataset", "subset", "policy_model", "limit", "belief_model"},
}
REQUIRED = {
    "data_prepare":   {"dataset"},
    "data_relevance": {"dataset"},
    "generate": {"dataset", "subset"},
    "teacher":  {"dataset", "subset"},
    "evaluate": {"dataset", "belief_model"},
}
#: The flags that overwrite a key of the YAML.
CONFIG_KEYS = {
    "teacher":    "shared.teacher",
    "student":    "shared.student",
    "init_model": "rl.init_model",
    "beliefs":    "rl.beliefs",
}


def _non_negative_int(value: str) -> int:
    n = int(value)
    if n < 0:
        raise argparse.ArgumentTypeError(f"must be >= 0, got {n}")
    return n


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Belief model training, one stage per call.",
                                     formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__,
                                     allow_abbrev=False)
    # ── Run ─────────────────────────────────────────────────────────────────
    parser.add_argument("--config", type=str, default=_CONFIG_DEFAULT, help="YAML settings (default: experiments/configs/belief_training.yaml).")
    parser.add_argument("--stage", type=str, required=True, choices=STAGES, help="Stage to run (see the stages below).")
    # ── Stages data_prepare, data_relevance, generate, teacher, evaluate ────
    parser.add_argument("--dataset", type=str, default=None, help="data_prepare, data_relevance, generate, teacher (required): infoseek_hard | frames | webshaper | trqa_val; evaluate (required): also any dra_inference --dataset.")
    parser.add_argument("--subset", type=str, default=None, help="generate (required): sft | rl | val; teacher (required): sft | val; evaluate: any dra_inference --subset.")
    parser.add_argument("--policy-model", type=str, default=None, help="generate, teacher, evaluate: one agent (default: every agent of shared.policy_models).")
    parser.add_argument("--limit", type=_non_negative_int, default=None, help="data_relevance: process the next N prepared questions (0 = rewrite the files only); generate, teacher, evaluate: run N queries.")
    # ── Stages teacher, filter ──────────────────────────────────────────────
    parser.add_argument("--teacher", type=str, default=None, help="teacher, filter: the teacher --belief-model (default: shared.teacher).")
    # ── Stages sft, rl ──────────────────────────────────────────────────────
    parser.add_argument("--student", type=str, default=None, help="sft, rl: the student --belief-model (default: shared.student); its trained models are served as {student}_sft, {student}_rl, {student}_rl_own_init.")
    parser.add_argument("--init-model", type=str, default=None, help="rl: the checkpoint RL starts from (default: the student's sft/final).")
    parser.add_argument("--beliefs", type=str, default=None, choices=RL_BELIEFS, help="rl: starting beliefs, gold (run rl) or own, the student's init call (run rl_own_init) (default: rl.beliefs).")
    # ── Stage evaluate ──────────────────────────────────────────────────────
    parser.add_argument("--belief-model", type=str, default=None, help="evaluate (required): a trained or baseline --belief-model of dra_inference.")
    return parser.parse_args()


def _flag(name: str) -> str:
    return "--" + name.replace("_", "-")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = _parse_args()
    stage = args.stage
    given = {k for k in set().union(*STAGE_FLAGS.values()) if getattr(args, k) is not None}
    extra = given - STAGE_FLAGS[stage]
    if extra:
        raise SystemExit(f"--stage {stage} takes no {', '.join(_flag(k) for k in sorted(extra))}")
    missing = REQUIRED.get(stage, set()) - given
    if missing:
        raise SystemExit(f"--stage {stage} needs {', '.join(_flag(k) for k in sorted(missing))}")

    overrides = {CONFIG_KEYS[k]: getattr(args, k) for k in given & set(CONFIG_KEYS)}
    try:
        cfg = load_config(args.config, overrides)
    except ValueError as e:
        raise SystemExit(f"--stage {stage}: {e}")
    agents = [args.policy_model] if args.policy_model else list(cfg.shared.policy_models)

    if stage == "data_prepare":
        from belief_model_training.data.build import prepare_dataset
        prepare_dataset(cfg, args.dataset)
    elif stage == "data_relevance":
        from belief_model_training.data.build import relevance_dataset
        relevance_dataset(cfg, args.dataset, args.limit)
    elif stage == "generate":
        from belief_model_training.trajectories.run import run_trajectories
        run_trajectories(cfg, args.dataset, args.subset, agents, args.limit)
    elif stage == "teacher":
        from belief_model_training.teacher.run import run_teacher
        run_teacher(cfg, args.dataset, args.subset, agents, args.limit)
    elif stage == "filter":
        from belief_model_training.filtering.run import run_filter
        run_filter(cfg)
    elif stage == "sft":
        from belief_model_training.sft.train import run_sft
        run_sft(cfg)
    elif stage == "rl":
        from belief_model_training.rl.loop import run_rl
        run_rl(cfg)
    else:
        from belief_model_training.evaluate.run import run_evaluation
        run_evaluation(cfg, args.belief_model, args.dataset, args.subset, agents, args.limit)


if __name__ == "__main__":
    main()

# python experiments/training/train_belief_model.py --stage data_prepare --dataset infoseek_hard
# python experiments/training/train_belief_model.py --stage data_prepare --dataset trqa_val
# python experiments/training/train_belief_model.py --stage data_prepare --dataset frames
# python experiments/training/train_belief_model.py --stage data_prepare --dataset webshaper
# python experiments/training/train_belief_model.py --stage data_relevance --dataset infoseek_hard --limit 50
# python experiments/training/train_belief_model.py --stage data_relevance --dataset trqa_val
# python experiments/training/train_belief_model.py --stage data_relevance --dataset frames --limit 30
# python experiments/training/train_belief_model.py --stage data_relevance --dataset webshaper --limit 30
# python experiments/training/train_belief_model.py --stage generate --dataset infoseek_hard --subset sft --limit 50
# python experiments/training/train_belief_model.py --stage generate --dataset infoseek_hard --subset rl
# python experiments/training/train_belief_model.py --stage teacher --dataset infoseek_hard --subset sft
# python experiments/training/train_belief_model.py --stage teacher --dataset infoseek_hard --subset val
# python experiments/training/train_belief_model.py --stage filter
# torchrun --nproc_per_node 4 experiments/training/train_belief_model.py --stage sft
# python experiments/training/train_belief_model.py --stage rl
# python experiments/training/train_belief_model.py --stage rl --beliefs own
# python experiments/training/train_belief_model.py --stage evaluate --belief-model qwen3.5_4b_rl --dataset infoseek_hard --subset val
# python experiments/training/train_belief_model.py --stage evaluate --belief-model qwen3.5_4b_rl --dataset browsecomp_plus
