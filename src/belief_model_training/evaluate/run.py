"""Phase 5: evaluate a belief model on saved runs.

``dra_inference --stage belief --belief-model {name}`` on the generated run
of each agent: the belief model over the saved trajectories, then its
evaluation, added to the run's ``eval/metrics.md`` next to the other belief
models (one ``belief_model`` row each: s_T AUROC and Brier against y, the
belief measures).  *name* is any ``--belief-model`` of dra_inference: a
trained one (``qwen3.5_4b_sft``, ``qwen3.5_4b_rl``, ``qwen3.5_4b_rl_own_init``,
served from its checkpoint on local vLLM) or a prompted one (the
baselines).  The runs must be generated (phase 2 for the training
datasets' val split; the test sets' own runs otherwise).
"""

from typing import List, Optional

from utils.config import BELIEF_MODEL_TO_LLM, TRAINED_BELIEF_RUNS, trained_belief_checkpoint

from ..config import TrainingConfig
from ..inference import run_dra_inference


def run_evaluation(cfg: TrainingConfig, belief_model: str, dataset: str, subset: Optional[str],
                   policy_models: List[str], limit: Optional[int]) -> None:
    if belief_model not in BELIEF_MODEL_TO_LLM:
        raise SystemExit(f"--belief-model {belief_model}: expected one of {sorted(BELIEF_MODEL_TO_LLM)}")
    if belief_model in TRAINED_BELIEF_RUNS:
        checkpoint = trained_belief_checkpoint(TRAINED_BELIEF_RUNS[belief_model])
        if not (checkpoint / "config.json").exists():
            raise SystemExit(f"--belief-model {belief_model}: no checkpoint at {checkpoint}; train it first "
                             f"(phase 3 for sft, phase 4 for rl)")
    for policy in policy_models:
        print(f"\n[phase 5] {belief_model} on {dataset}" + (f"/{subset}" if subset else "") + f", agent {policy}")
        run_dra_inference(cfg, stage="belief", dataset=dataset, subset=subset, policy_model=policy,
                          belief_model=belief_model, limit=limit)
