"""Stage generate: the frozen agents' trajectories on a training dataset split.

Per agent (``shared.policy_models``, or the one given),
``dra_inference --stage generate`` on the split: the agent's trajectories
and the judged correctness of its answers (``eval/generation/``), the y
of training.  The stage resumes (the queries already done are skipped);
``limit`` caps the queries it runs.  Then the statistics of the runs
(``trajectories.stats``).  The teacher's beliefs on these trajectories
are stage teacher (``teacher``).
"""

from typing import List, Optional

from ..shared.config import TrainingConfig
from ..shared.inference import run_dra_inference
from .stats import run_stats


def run_trajectories(cfg: TrainingConfig, dataset: str, split: str, policy_models: List[str],
                     limit: Optional[int]) -> None:
    from indexing_corpus_dataset.shared.layout import BUILT_DATASETS, TRAINING_SPLITS

    if dataset not in BUILT_DATASETS:
        raise SystemExit(f"--dataset {dataset}: stage generate runs on a training dataset ({sorted(BUILT_DATASETS)})")
    if split not in TRAINING_SPLITS:
        raise SystemExit(f"--subset {split}: expected one of {list(TRAINING_SPLITS)}")
    for policy in policy_models:
        print(f"\n[stage generate] {dataset}/{split}/{policy}: generate")
        run_dra_inference(cfg, stage="generate", dataset=dataset, subset=split, policy_model=policy, limit=limit)
    run_stats(cfg, [dataset], split, policy_models)
