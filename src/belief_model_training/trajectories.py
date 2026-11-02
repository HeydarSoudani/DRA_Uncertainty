"""Phase 1: trajectories.

``experiments/inference/dra_inference.py`` runs the frozen agent on one split
(``trajectories.split``) of the training pool ``shared.training_data``
(``experiments/training/build_training_pool.py``): the queries with that
``split`` in the pool's queries file (``--subset``: dra_inference reads the
split as a view, ``layout.is_pool_view``): those with the full gold
annotation and criterion texts, and in ``val`` also the out-of-domain ones
(no full annotation: deep_qa's WebShaper and FRAMES), or the next ``limit``
of them.  The mode follows the split (``runs.SPLIT_MODE``):

    sft, val  monitor: the belief model starts from the query's gold
              criteria (``--criteria-source gold``), or for an
              out-of-domain query from the teacher's own
              (``--llm-criteria``, through the criteria bank), and the
              teacher writes the belief updates
              (statuses and answer candidates), the agent never sees them;
              the query scorer is frozen (``query_scorer_model`` of the
              inference config, default the teacher).  Training uses only
              the in-domain queries; the out-of-domain ones are for
              evaluation
    rl        off: no belief model (phase 3 runs the student's chains)

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
        "--quiet", str(tc.quiet).lower(),
    ]
    if cfg.shared.gpu_ids:
        cmd += ["--gpu-ids", cfg.shared.gpu_ids]
    if mode == "monitor":
        cmd += [
            "--llm-criteria", tc.teacher_belief_model,
            "--belief-judge-model", tc.teacher_belief_model,
            "--criteria-source", "gold",
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
