"""dra_inference calls of the stages generate, teacher and evaluate: the inference pipeline itself, run
as a child process with the training's settings as its flags (nothing of
the pipeline is repeated here)."""

import subprocess
import sys
from typing import Optional

from .config import REPO_ROOT, TrainingConfig

DRA_INFERENCE = REPO_ROOT / "experiments" / "inference" / "dra_inference.py"


def run_dra_inference(cfg: TrainingConfig, *, stage: str, dataset: str, subset: Optional[str], policy_model: str,
                      belief_model: Optional[str] = None, limit: Optional[int] = None) -> None:
    """``dra_inference.py --stage {stage}`` (generate or belief) on *subset*
    of *dataset* with *policy_model*'s run (and *belief_model* for the
    belief stage); the generate stage takes the ``generate`` settings,
    the belief stage the ``belief_stage`` ones.  Stops when it fails."""
    cmd = [sys.executable, str(DRA_INFERENCE), "--stage", stage, "--config", str(cfg.inference_config_path),
           "--dataset", dataset, "--policy-model", policy_model, "--retriever", cfg.shared.retriever]
    if subset is not None:
        cmd += ["--subset", subset]
    if stage == "generate":
        cmd += ["--workers", str(cfg.generate.workers)]
        quiet = cfg.generate.quiet
    elif stage == "belief":
        if belief_model is None:
            raise ValueError("the belief stage needs a belief model")
        cmd += ["--belief-model", belief_model]
        if cfg.belief_stage.belief_workers is not None:
            cmd += ["--belief-workers", str(cfg.belief_stage.belief_workers)]
        quiet = cfg.belief_stage.quiet
    else:
        raise ValueError(f"stage {stage!r}: expected generate or belief")
    if limit is not None:
        cmd += ["--limit", str(limit)]
    if quiet:
        cmd += ["--quiet"]
    print(f"[train_belief_model] {' '.join(cmd[1:])}", flush=True)
    done = subprocess.run(cmd, cwd=str(REPO_ROOT))
    if done.returncode != 0:
        raise SystemExit(f"dra_inference --stage {stage} failed (exit {done.returncode})")
