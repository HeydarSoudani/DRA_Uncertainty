"""Phase 1: trajectories.

``experiments/inference/dra_inference.py`` runs the frozen agent on one split
(``trajectories.split``) of the training pool ``shared.training_data``
(``experiments/training/build_training_pool.py``): every query of the split
with qrels, or the next ``limit`` of them.  The mode follows the split
(``runs.SPLIT_MODE``):

    sft, val  monitor: the teacher belief model writes the criteria (init,
              read from the criteria bank when banked) and the belief
              updates, the agent never sees them; the query scorer is
              frozen (``query_scorer_model`` of the inference config,
              default the teacher)
    rl        off: no belief model (phase 3 replays the steps with the student)

Inference resume applies: queries already run are skipped, so a call after
the pool has grown, or with a larger ``limit``, runs only the new queries.
The runs are written in dra_inference's layout under its output root, and
nothing else: phases 2 and 3 read them (``runs.load_episodes``).
"""

import subprocess
import sys
from typing import List

from .config import REPO_ROOT, TrainingConfig
from .runs import SPLIT_MODE


def inference_command(cfg: TrainingConfig) -> List[str]:
    """The ``experiments/inference/dra_inference.py`` command of phase 1."""
    tc = cfg.trajectories
    split = cfg.pool_split(tc.split)
    mode = SPLIT_MODE[tc.split]
    cmd = [
        sys.executable, str(REPO_ROOT / "experiments" / "inference" / "dra_inference.py"),
        "--config", cfg.shared.inference_config,
        "--dataset", split.dataset,
        "--subset", split.subset,
        "--retriever", cfg.shared.retriever,
        "--policy-model", cfg.shared.policy_model,
        "--belief-mode", mode,
        "--num-gpus", str(cfg.shared.num_gpus),
        "--quiet",
    ]
    if cfg.shared.gpu_ids:
        cmd += ["--gpu-ids", cfg.shared.gpu_ids]
    if mode == "monitor":
        cmd += [
            "--llm-criteria", tc.teacher_belief_model,
            "--belief-judge-model", tc.teacher_belief_model,
            "--criteria-bank", "true",
        ]
    if tc.limit is not None:
        cmd += ["--limit", str(tc.limit)]
    return cmd


def run_trajectories(cfg: TrainingConfig) -> None:
    tc = cfg.trajectories
    split = cfg.pool_split(tc.split)
    mode = SPLIT_MODE[tc.split]
    teacher = f", teacher {tc.teacher_belief_model}" if mode == "monitor" else ""
    print(f"[phase 1] {split.name}; {mode} mode; "
          f"policy model {cfg.shared.policy_model}{teacher}")
    cmd = inference_command(cfg)
    print(f"[phase 1] {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)
