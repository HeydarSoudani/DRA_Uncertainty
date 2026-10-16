"""Train the belief model (the belief updater call) with the agent frozen.

Agent-BRACE (arXiv 2605.11436) adapted to retrieval: the belief model
b_t = f(query, b_{t-1}, o_t) is trained, here with the agent frozen.  One
phase per call (``--phase N``); each phase reads the previous one's output
from the run directory ``{output_root}/{run_name}/``:

    0  training queries                          phase0_queries/        TODO
    1  frozen agent + teacher belief model in     phase1_trajectories/
       monitor mode -> episodes and SFT samples
    2  SFT warm-up of the student (LoRA, merged)  phase2_sft/
    3  RL of the student (Agent-BRACE belief PPO) phase3_rl/            rewards TODO
    4  frozen agent with off / teacher / student  phase4_eval/
       belief model, existing evaluators

Settings: ``--config`` (YAML, one section per phase plus ``shared``), then
the flags below, then ``--set section.key=value`` (any key, e.g.
``--set rl.learning_rate=5e-6``).  The agent and retriever settings of the
inference runs (phases 1 and 4) come from ``shared.inference_config``
(experiments/configs/dra_inference.yaml) and are never repeated here.
The merged settings of each phase are saved as ``phase{N}_config.yaml``.

Examples:
    python experiments/belief_model_training.py --phase 1 --run-name trqa_glm --agent glm
    torchrun --nproc_per_node 4 experiments/belief_model_training.py --phase 2 --run-name trqa_glm
    python experiments/belief_model_training.py --phase 3 --run-name trqa_glm
    python experiments/belief_model_training.py --phase 4 --run-name trqa_glm --eval-splits trqa:wiki1:test
"""

import argparse
import logging
import os
import sys
from pathlib import Path

# This script has the name of the src/belief_model_training package: drop its
# own directory from the import path so that the package is imported.
_HERE = Path(__file__).resolve().parent
sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != _HERE]

import yaml
from dotenv import load_dotenv

load_dotenv()

from belief_model_training.config import RunPaths, load_config

_CONFIG_DEFAULT = str(_HERE / "configs" / "belief_model_training.yaml")

# CLI flag (dest) -> config key.
_FLAGS = {
    # shared
    "run_name": "shared.run_name",
    "output_root": "shared.output_root",
    "seed": "shared.seed",
    "retriever": "shared.retriever",
    "num_gpus": "shared.num_gpus",
    # phase 1
    "agent": "trajectories.agent",
    "teacher_model": "trajectories.teacher_model",
    "criteria_model": "trajectories.criteria_model",
    # phase 2
    "student_model": "sft.student_model",
    # phase 3
    "student_init": "rl.student_init",
    "judge_model": "rl.judge_model",
    "rl_env": "rl.env",
    # phase 4
    "belief_model": "eval.belief_model",
    "eval_agent": "eval.agent",
    "eval_splits": "eval.splits",
    "eval_variants": "eval.variants",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the belief model with the agent frozen (Agent-BRACE adapted to retrieval).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        allow_abbrev=False,
    )
    # ── Phase and settings files ────────────────────────────────────────────
    parser.add_argument("--phase", type=int, required=True, choices=[0, 1, 2, 3, 4], help="0 training queries (TODO), 1 teacher trajectories, 2 SFT, 3 RL, 4 evaluation.")
    parser.add_argument("--config", type=str, default=_CONFIG_DEFAULT, help="YAML settings, one section per phase plus 'shared'.")
    parser.add_argument("--set", action="append", default=[], metavar="SECTION.KEY=VALUE", help="Override any config key; repeatable; lists comma-separated.")

    # ── Shared ──────────────────────────────────────────────────────────────
    parser.add_argument("--run-name", type=str, default="default", help="Run directory name under the output root.")
    parser.add_argument("--output-root", type=str, default=None, help="Root of the training runs (None = OUTPUT_ROOT/belief_model_training).")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--retriever", type=str, default="qwen3_emb_4b", help="Retriever of the inference runs (phases 1 and 4).")
    parser.add_argument("--num-gpus", type=int, default=4, help="dra_inference workers (phases 1 and 4).")

    # ── Phase 1: teacher trajectories ───────────────────────────────────────
    parser.add_argument("--agent", type=str, default="glm", help="Frozen agent (--agentic-model of dra_inference), e.g. glm, oss_20b.")
    parser.add_argument("--teacher-model", type=str, default="openrouter/qwen/qwen3.6-27b", help="Teacher belief updater LLM.")
    parser.add_argument("--criteria-model", type=str, default="openrouter/qwen/qwen3.6-27b", help="Criteria extraction LLM (frozen; phases 1 and 4).")

    # ── Phase 2: SFT ────────────────────────────────────────────────────────
    parser.add_argument("--student-model", type=str, default="Qwen/Qwen3-4B-Instruct-2507", help="Base checkpoint of the student belief model.")

    # ── Phase 3: RL ─────────────────────────────────────────────────────────
    parser.add_argument("--student-init", type=str, default=None, help="Start checkpoint (None = the merged phase 2 model).")
    parser.add_argument("--judge-model", type=str, default=None, help="LLM judge of the rewards (rewards TODO).")
    parser.add_argument("--rl-env", type=str, default="replay", choices=["replay", "online"], help="replay: phase 1 trajectories; online: TODO.")

    # ── Phase 4: evaluation ─────────────────────────────────────────────────
    parser.add_argument("--belief-model", type=str, default=None, help="Student checkpoint (None = the final phase 3 model).")
    parser.add_argument("--eval-agent", type=str, default=None, help="Agent (None = the phase 1 agent).")
    parser.add_argument("--eval-splits", type=str, default=None, help="Comma-separated dataset[:subset[:dataset_year]] splits.")
    parser.add_argument("--eval-variants", type=str, default="off,teacher,student", help="Comma-separated subset of off,teacher,student.")

    args = parser.parse_args()
    # The defaults above document the standard values; only the flags given
    # on the command line override the --config file.
    for action in parser._actions:
        action.default = argparse.SUPPRESS
    args.given = set(vars(parser.parse_args()))
    return args


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = _parse_args()

    overrides = {key: getattr(args, dest) for dest, key in _FLAGS.items() if dest in args.given}
    for item in args.set:
        key, sep, value = item.partition("=")
        if not sep:
            raise SystemExit(f"--set expects SECTION.KEY=VALUE, got {item!r}")
        overrides[key.strip()] = value.strip()
    cfg = load_config(args.config, overrides)
    paths = RunPaths(cfg)

    if int(os.environ.get("RANK", "0")) == 0:
        paths.root.mkdir(parents=True, exist_ok=True)
        (paths.root / f"phase{args.phase}_config.yaml").write_text(yaml.safe_dump(cfg.to_dict(), sort_keys=False))
    print(f"[belief_model_training] phase {args.phase}, run {paths.root}")

    if args.phase == 0:
        from belief_model_training.queries import build_training_queries
        build_training_queries(cfg, paths)
    elif args.phase == 1:
        from belief_model_training.trajectories import run_trajectories
        run_trajectories(cfg, paths)
    elif args.phase == 2:
        from belief_model_training.sft import run_sft
        run_sft(cfg, paths)
    elif args.phase == 3:
        from belief_model_training.rl import run_rl
        run_rl(cfg, paths)
    else:
        from belief_model_training.evaluate import run_eval
        run_eval(cfg, paths)


if __name__ == "__main__":
    main()
