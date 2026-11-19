"""dra_inference calls of phases 2 and 5: the inference pipeline itself, run
as a child process with the training's settings as its flags (nothing of
the pipeline is repeated here)."""

import subprocess
import sys
from typing import List, Optional

from .config import REPO_ROOT, TrainingConfig

DRA_INFERENCE = REPO_ROOT / "experiments" / "inference" / "dra_inference.py"


def run_dra_inference(cfg: TrainingConfig, *, stage: str, dataset: str, subset: Optional[str], policy_model: str,
                      belief_model: Optional[str] = None, limit: Optional[int] = None,
                      extra: Optional[List[str]] = None) -> None:
    """``dra_inference.py --stage {stage}`` on *subset* of *dataset* with
    *policy_model*'s run (and *belief_model* for a belief stage); stops when
    it fails."""
    tr = cfg.trajectories
    cmd = [sys.executable, str(DRA_INFERENCE), "--stage", stage, "--config", str(cfg.inference_config_path),
           "--dataset", dataset, "--policy-model", policy_model, "--retriever", cfg.shared.retriever,
           "--workers", str(tr.workers)]
    if subset is not None:
        cmd += ["--subset", subset]
    if belief_model is not None:
        cmd += ["--belief-model", belief_model]
        if tr.belief_workers is not None:
            cmd += ["--belief-workers", str(tr.belief_workers)]
    if limit is not None:
        cmd += ["--limit", str(limit)]
    if tr.quiet:
        cmd += ["--quiet"]
    cmd += list(extra or [])
    print(f"[train_belief_model] {' '.join(cmd[1:])}", flush=True)
    done = subprocess.run(cmd, cwd=str(REPO_ROOT))
    if done.returncode != 0:
        raise SystemExit(f"dra_inference --stage {stage} failed (exit {done.returncode})")
