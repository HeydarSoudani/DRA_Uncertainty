"""Phase 2: the frozen agents' trajectories on a training dataset split, and
the teacher's beliefs on them.

Per agent (``shared.policy_models``, or the one given):

1. ``dra_inference --stage generate`` on the split: the agent's trajectories
   and the judged correctness of its answers (``eval/generation/``), the y
   of training;
2. on the ``sft`` and ``val`` splits of a training dataset,
   ``dra_inference --stage belief --belief-model {shared.teacher}``: the
   teacher's beliefs, the SFT samples (phase 3).  The ``rl`` split needs
   the trajectories only; a dataset that is not trained on (webshaper_frames,
   out of domain) gets no teacher run.

Both stages resume (the queries already done are skipped); ``limit`` caps
the queries each stage runs.  Then the statistics of the runs
(``trajectories.stats``).
"""

from typing import List, Optional

from ..config import TrainingConfig
from ..inference import run_dra_inference
from .stats import run_stats

#: Splits whose trajectories get the teacher's beliefs.
TEACHER_SPLITS = ("sft", "val")


def run_trajectories(cfg: TrainingConfig, dataset: str, split: str, policy_models: List[str],
                     limit: Optional[int]) -> None:
    from indexing_corpus_dataset.layout import BUILT_DATASETS, TRAINING_SPLITS

    if dataset not in BUILT_DATASETS:
        raise SystemExit(f"--dataset {dataset}: phase 2 runs on a training dataset ({sorted(BUILT_DATASETS)})")
    if split not in TRAINING_SPLITS:
        raise SystemExit(f"--split {split}: expected one of {list(TRAINING_SPLITS)}")
    teacher = split in TEACHER_SPLITS and dataset in cfg.shared.training_datasets
    for policy in policy_models:
        print(f"\n[phase 2] {dataset}/{split}/{policy}: generate" + (f", then teacher {cfg.shared.teacher}"
                                                                      if teacher else ""))
        run_dra_inference(cfg, stage="generate", dataset=dataset, subset=split, policy_model=policy, limit=limit)
        if teacher:
            run_dra_inference(cfg, stage="belief", dataset=dataset, subset=split, policy_model=policy,
                              belief_model=cfg.shared.teacher, limit=limit)
    run_stats(cfg, [dataset], split, policy_models)
