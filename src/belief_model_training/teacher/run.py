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

The stage first banks the gold beliefs of every query of the dataset,
every split (``bank_teacher_beliefs``): on a dataset whose beliefs come
from the question (infoseek_hard, webshaper: multi-clue, as
BrowseComp-Plus) the teacher's init beliefs; on trqa_val the gold beliefs
made from its rows, with the teacher's texts (as TRQA's at inference).
They are the gold beliefs of training (``runs.read_gold_rows``), the rl
split's included, which gets no teacher run.  The belief stage then reads
them from the bank.
"""

from typing import List, Optional

from ..config import TrainingConfig
from ..inference import run_dra_inference
from ..runs import find_run

#: The splits whose trajectories get the teacher's beliefs.
TEACHER_SPLITS = ("sft", "val")


def bank_teacher_beliefs(dataset: str, teacher: str) -> None:
    """Bank the gold beliefs of every query of *dataset* (all splits) as
    the belief stage of the *teacher* makes them: its init beliefs when
    they come from the question, else (trqa_val) those made from the
    dataset's rows, with its texts (the queries whose row has gold
    beliefs; FRAMES' have none).  The ones banked already are kept
    (``prepare_beliefs``)."""
    from belief_model_component import build_belief_source, prepare_beliefs
    from indexing_corpus_dataset.dataset_loaders import load_queries
    from indexing_corpus_dataset.layout import ALL_SPLIT, DATASET_SPECS, QUESTION_GOLD, dataset_root, gold_beliefs_path
    from utils.config import BELIEF_MODEL_TO_LLM
    from utils.io_utils import iter_jsonl

    spec = DATASET_SPECS[dataset]
    from_question = spec.gold_beliefs == QUESTION_GOLD
    queries = load_queries(dataset_root(dataset), ALL_SPLIT, spec.query_key, dataset=dataset)
    if not from_question:
        with_beliefs = {str(r["id"]) for r in iter_jsonl(gold_beliefs_path(dataset)) if r["members"]}
        queries = {qid: q for qid, q in queries.items() if qid in with_beliefs}
        if not queries:
            return
    what = "init beliefs" if from_question else "gold belief texts"
    print(f"\n[stage teacher] {dataset}: the teacher's {what} of all {len(queries)} queries (every split)")
    source = build_belief_source(dataset, ALL_SPLIT, belief_model=BELIEF_MODEL_TO_LLM[teacher],
                                 max_beliefs=spec.max_beliefs, fallback=False)
    result = prepare_beliefs(source, queries)
    if result["failed"]:
        print(f"[stage teacher] {dataset}: {len(result['failed'])} queries without beliefs (asked again by the "
              "next run); training leaves them out")


def run_teacher(cfg: TrainingConfig, dataset: str, split: str, policy_models: List[str],
                limit: Optional[int]) -> None:
    from indexing_corpus_dataset.dataset_loaders import load_queries
    from indexing_corpus_dataset.layout import BUILT_DATASETS, DATASET_SPECS, dataset_root
    from utils.run_files import scan_queries
    from evaluation.metrics_file import METRICS_FILE

    if dataset not in BUILT_DATASETS:
        raise SystemExit(f"--dataset {dataset}: stage teacher runs on a training dataset ({sorted(BUILT_DATASETS)})")
    if split not in TEACHER_SPLITS:
        raise SystemExit(f"--subset {split}: the teacher runs on {list(TEACHER_SPLITS)} only "
                         "(the rl split needs the trajectories only)")
    teacher = cfg.shared.teacher
    bank_teacher_beliefs(dataset, teacher)
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
