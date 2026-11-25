"""Stage teacher: the teacher's beliefs over the generated trajectories.

Per agent (``shared.policy_models``, or the one given),
``dra_inference --stage belief --belief-model {teacher}`` (``shared.teacher``
or --teacher) on the ``sft`` or ``val`` split: the teacher belief model over
the saved trajectories (``belief/{teacher}/`` of the run), the SFT samples
of stage filter,
and its evaluation (s_T against y) in the run's ``eval/metrics.md``.  The
``rl`` split needs no teacher.  The stage resumes (the queries already done
are skipped) and runs only over finished trajectories; ``limit`` caps the
queries it runs.
"""

from typing import List, Optional

from ..shared.config import TrainingConfig
from ..shared.inference import run_dra_inference
from ..shared.runs import find_run

#: The splits whose trajectories get the teacher's beliefs.
TEACHER_SPLITS = ("sft", "val")


def run_teacher(cfg: TrainingConfig, dataset: str, split: str, policy_models: List[str],
                limit: Optional[int]) -> None:
    from indexing_corpus_dataset.shared.dataset_loaders import load_queries
    from indexing_corpus_dataset.shared.layout import BUILT_DATASETS, DATASET_SPECS, dataset_root
    from utils.run_files import scan_queries
    from evaluation.shared.files import METRICS_FILE

    if dataset not in BUILT_DATASETS:
        raise SystemExit(f"--dataset {dataset}: stage teacher runs on a training dataset ({sorted(BUILT_DATASETS)})")
    if split not in TEACHER_SPLITS:
        raise SystemExit(f"--subset {split}: the teacher runs on {list(TEACHER_SPLITS)} only "
                         "(the rl split needs the trajectories only)")
    teacher = cfg.shared.teacher
    queries = load_queries(dataset_root(dataset), split, DATASET_SPECS[dataset].query_key, dataset=dataset)
    for policy in policy_models:
        run_dir = find_run(cfg.inference_output, dataset, split, cfg.shared.retriever, policy)
        if run_dir is None:
            print(f"\n[stage teacher] {dataset}/{split}/{policy}: no run; run --stage generate on this split first")
            continue
        finished = len(set(scan_queries(run_dir)[0]) & set(queries))
        print(f"\n[stage teacher] {dataset}/{split}/{policy}: teacher {teacher} over {finished} finished "
              f"trajectories ({len(queries) - finished} of {len(queries)} queries without one)")
        run_dra_inference(cfg, stage="belief", dataset=dataset, subset=split, policy_model=policy,
                          belief_model=teacher, limit=limit)
        print(f"[stage teacher] {dataset}/{split}/{policy}: teacher s_T against y in {run_dir / METRICS_FILE}")
