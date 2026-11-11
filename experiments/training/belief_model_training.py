"""Train the belief model with the agent frozen.

Agent-BRACE (arXiv 2605.11436) adapted to retrieval: the belief model is
trained, here with the agent frozen.  Two calls are trained: the belief
update, b_t = f(query, b_{t-1}, agent turns, new passages) -> the
observations of the turns and the passages and the candidate updates of
the criteria they touch, on the dataset's gold criteria (the criteria init is
not trained); and, on a short-answer query, the final scorer, s_T =
f(query, b_{T-1}, answer turn).  The updates are rewarded over whole
trajectories, per criterion: the assignment of the sources to it and the
calibration of its status at every step, plus well-formed calls; the
scorer by s_T against the correctness of the agent's answer (on a
report, the certainty of the final beliefs).
Training uses only the queries with the full gold annotation (each query's
``split`` in the dataset's queries file).  One phase per call (``--phase N``):

    1  dra_inference with the frozen agent on     run_outputs/ (dra_inference layout)
       one split of the training dataset (--split):
       sft and val in monitor mode with the
       teacher belief model, rl in off mode
    2  teacher scores, SFT warm-up of the student phase2_sft/
       (LoRA, merged) on the teacher's replies
       (sft, val runs)
    3  RL of the student (GRPO on its whole       phase3_rl/
       belief chains, Agent-BRACE PPO update; rl
       run, validation on the val run)

Phases 2 and 3 read the phase 1 runs of ``--policy-model`` and write under
``{output_root}/{training_data}/{policy_model}/``.

The trained belief model is evaluated with experiments/inference/dra_inference.py
(see experiments/training/README.md, "Evaluation").

Settings: ``--config`` (YAML, one section per phase plus ``shared``), then
the flags below.  The agent, retriever and query scorer settings of the
inference run (phase 1) come from ``shared.inference_config``
(experiments/configs/dra_inference.yaml) and are never repeated here.
The merged settings of each phase are saved as ``phase{N}_config.yaml``.

The training data is a training dataset (``--training-data``), built and
split into sft / rl / val by experiments/training/build_training_dataset.py
under ``data/training_datasets/{dataset}`` and run by dra_inference as
dataset ``{dataset}``, subset = split: infoseek (the wiki-25-512 corpus and
its indices under ``DATA_ROOT/infoseek``), trqa_validation (TRQA's
wiki_partial corpus and indices) or webshaper_frames (val only).

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
from indexing_corpus_dataset.layout import BUILT_DATASETS, TRAINING_SPLITS
from utils.cli_setup import _sm_bool

_CONFIG_DEFAULT = str(_HERE.parent / "configs" / "belief_model_training.yaml")

# CLI flag (dest) -> config key.
_FLAGS = {
    # shared
    "training_data": "shared.training_data",
    "policy_model": "shared.policy_model",
    "output_root": "shared.output_root",
    "seed": "shared.seed",
    "retriever": "shared.retriever",
    "num_gpus": "shared.num_gpus",
    "gpu_ids": "shared.gpu_ids",
    # phase 1
    "split": "trajectories.split",
    "teacher_belief_model": "trajectories.teacher_belief_model",
    "limit": "trajectories.limit",
    "quiet": "trajectories.quiet",
    # phase 2
    "student_belief_model": "sft.student_belief_model",
    # phase 3
    "rl_init_belief_model": "rl.init_belief_model",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the belief model with the agent frozen (Agent-BRACE adapted to retrieval).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        allow_abbrev=False,
    )
    # ── Phase and settings files ────────────────────────────────────────────
    parser.add_argument("--phase", type=int, required=True, choices=[1, 2, 3], help="1 trajectories (dra_inference on a dataset split), 2 SFT, 3 RL.")
    parser.add_argument("--config", type=str, default=_CONFIG_DEFAULT, help="YAML settings, one section per phase plus 'shared'.")

    # ── Shared ──────────────────────────────────────────────────────────────
    parser.add_argument("--training-data", type=str, default="infoseek", choices=sorted(BUILT_DATASETS), help="Training dataset (dataset of dra_inference); webshaper_frames has a val split only.")
    parser.add_argument("--policy-model", type=str, default="oss_20b", help="Frozen agent (--policy-model of dra_inference), e.g. glm, oss_20b, oss_120b: phase 1 runs it, phases 2-3 read its runs and write under {output root}/{training data}/{policy model}.")
    parser.add_argument("--retriever", type=str, default="qwen3_emb_4b", help="Retriever of the inference runs (phase 1).")
    parser.add_argument("--output-root", type=str, default=None, help="Root of the phase 2-3 runs (None = OUTPUT_ROOT/belief_model_training).")
    parser.add_argument("--num-gpus", type=int, default=4, help="dra_inference workers (phase 1).")
    parser.add_argument("--gpu-ids", type=str, default=None, help="GPUs of the dra_inference workers (phase 1), comma-separated physical ids, e.g. 0,1,2,3 (None = gpu_ids of the inference config).")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")

    # ── Phase 1: trajectories ───────────────────────────────────────────────
    parser.add_argument("--split", type=str, default="sft", choices=list(TRAINING_SPLITS), help="Dataset split to run: sft and val in monitor mode with the teacher, rl in off mode.")
    parser.add_argument("--teacher-belief-model", type=str, default="openrouter/qwen/qwen3.6-27b", help="Teacher belief model (criteria init and belief updates; phase 2 checks the runs used it).")
    parser.add_argument("--limit", type=int, default=None, help="The next N queries of the split not run yet (None = all).")
    parser.add_argument("--quiet", type=_sm_bool, nargs="?", const=True, default=False, help="Print minimal logs (no per-step agent output), as dra_inference --quiet.")

    # ── Phase 2: SFT ────────────────────────────────────────────────────────
    parser.add_argument("--student-belief-model", type=str, default="Qwen/Qwen3.5-4B", help="Base checkpoint of the student belief model (non-thinking).")

    # ── Phase 3: RL ─────────────────────────────────────────────────────────
    parser.add_argument("--rl-init-belief-model", type=str, default=None, help="Start checkpoint of the student belief model (None = the merged phase 2 model).")

    args = parser.parse_args()
    # The defaults above document the standard values; only the flags given
    # on the command line override the --config file.  --policy-model is the
    # exception: its default always applies (the agent is chosen here).
    for action in parser._actions:
        if action.dest != "policy_model":
            action.default = argparse.SUPPRESS
    args.given = set(vars(parser.parse_args()))
    return args


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = _parse_args()

    overrides = {key: getattr(args, dest) for dest, key in _FLAGS.items() if dest in args.given}
    cfg = load_config(args.config, overrides)
    paths = RunPaths(cfg)

    if args.phase == 1:
        # Phase 1 writes the inference runs only, in dra_inference's layout.
        from belief_model_training.trajectories import run_trajectories
        run_trajectories(cfg)
        return

    if int(os.environ.get("RANK", "0")) == 0:
        paths.root.mkdir(parents=True, exist_ok=True)
        (paths.root / f"phase{args.phase}_config.yaml").write_text(yaml.safe_dump(cfg.to_dict(), sort_keys=False))
    print(f"[belief_model_training] phase {args.phase}, run {paths.root}")

    if args.phase == 2:
        from belief_model_training.sft import run_sft
        run_sft(cfg, paths)
    else:
        from belief_model_training.rl import run_rl
        run_rl(cfg, paths)


if __name__ == "__main__":
    main()


# Examples:
# python experiments/training/belief_model_training.py --phase 1 --split sft --limit 1 --training-data trqa_validation
# python experiments/training/belief_model_training.py --phase 1 --split sft --limit 1
# python experiments/training/belief_model_training.py --phase 1 --split rl --limit 1
# python experiments/training/belief_model_training.py --phase 1 --split val --limit 1
# python experiments/training/belief_model_training.py --phase 2
# python experiments/training/belief_model_training.py --phase 3
