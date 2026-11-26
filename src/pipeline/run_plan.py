"""The stages of a run and its plan: what a run works on, settled before
any model is loaded."""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

from indexing_corpus_dataset.shared.dataset_loaders import load_qrels, load_split, resolve_split_id
from indexing_corpus_dataset.shared.layout import DATASET_SPECS, build_dataset_dir_name
from utils.run_files import scan_queries

from .run_config import (
    build_run_config,
    build_run_name_for_pipeline,
    check_run_config,
    check_run_dataset,
    update_run_config_evaluation,
)

#: The stages of a run (``--stage``; see ``experiments/inference/dra_inference.py``).
STAGES = ("generate", "belief", "eval", "all")
GENERATES = ("generate", "all")
BELIEVES = ("belief", "all")


@dataclass
class RunPlan:
    """What a run works on, settled before any model is loaded (see
    :func:`prepare_run`)."""
    dataset: str
    split: str                      # file-level split id, e.g. "wiki2_test"
    queries: Dict[str, str]         # the queries still to generate (resume and --limit applied)
    all_questions: Dict[str, str]   # every query of the split, for accuracy evaluation
    qrels: Dict[str, Dict[str, int]]         # at min_relevance_score
    answers: Dict[str, str]
    run_dir: str
    limit: Optional[int]            # --limit: the cap of each stage's queries


def prepare_run(data_path: str, output_path: str, policy_model: str, subset: Optional[str] = None,
                dataset_year: Optional[str] = None, query_key: Optional[str] = None,
                limit: Optional[int] = None, **kwargs) -> RunPlan:
    """Load the split, name the run directory and find the queries left to
    generate (the generate stage).

    Done before any model, retriever or vLLM server is set up, so a run with
    nothing left to generate sets up none.

    Args:
        data_path:     Path to dataset directory.
        output_path:   Root path for saving results.
        policy_model:  Agent to use (internal name); one of ALL_AGENTS.
        subset:        Dataset subset identifier, e.g. "news" or "technical"
                       for neuclir, "wiki2" for trqa.
        dataset_year:  Dataset year (neuclir "2024", ragtime "2025"); trqa
                       carries its eval split (test|validation) here.
        query_key:     Key in each JSONL query record to use as the query text.
                       None = the dataset's default (layout.DATASET_SPECS),
                       e.g. "request" for neuclir.  Records without the key
                       are not run.
        limit:         Cap the number of queries (for quick tests), applied
                       to the queries left to generate.
        **kwargs:      The pipeline kwargs (``assemble_pipeline_kwargs``).

    ``limit`` and ``query_ids_file`` choose which queries the generate and
    belief stages run; the evaluation always scores every finished query of
    the run directory (``pipeline.runner.run_pipeline``).
    """
    from deep_research_agents.agents import ALL_AGENTS

    if policy_model not in ALL_AGENTS:
        raise ValueError(
            f"Unknown policy_model '{policy_model}'. Choose from: {', '.join(ALL_AGENTS)}"
        )
    kwargs = dict(kwargs)

    # ==================== Resolve file-level dataset identifier ====================
    # Build the split string used to locate queries/qrels files on disk
    # (e.g. "2024_news", "2025", "wiki2_test"); see resolve_split_id.
    dataset = kwargs.pop("dataset")
    qrels_data_path = kwargs.pop("qrels_data_path", None) or data_path
    file_data_set = resolve_split_id(dataset, dataset_year, subset)

    if query_key is None:
        query_key = DATASET_SPECS[dataset].query_key

    # ==================== Load Dataset ====================
    # The qrels are read once; the evaluation and the belief model use the
    # grades at min_relevance_score.  Only queries with a relevant doc are run.
    queries, all_qrels, answers = load_split(data_path, file_data_set, query_key=query_key, dataset=dataset)
    if qrels_data_path != data_path:
        all_qrels = load_qrels(qrels_data_path, file_data_set, dataset=dataset)
    min_rel_score = kwargs.get("min_relevance_score")
    qrels = {}
    for qid, docs in all_qrels.items():
        kept = {did: rel for did, rel in docs.items() if min_rel_score is None or rel >= min_rel_score}
        if kept:
            qrels[qid] = kept
    queries = {qid: q for qid, q in queries.items() if qid in qrels}
    if answers:
        print(f"Loaded {len(answers)} ground-truth answers (accuracy evaluation available)")

    # Only the queries listed in the file (one id per line).
    query_ids_file = kwargs.pop("query_ids_file", None)
    if query_ids_file:
        wanted = {line.strip() for line in Path(query_ids_file).read_text().splitlines() if line.strip()}
        queries = {qid: q for qid, q in queries.items() if qid in wanted}
        print(f"Kept {len(queries)} queries listed in {query_ids_file} ({len(wanted)} ids)")

    # Keep the full set of questions for accuracy evaluation (before resume filtering)
    all_questions = dict(queries)

    stage = kwargs["stage"]
    if stage in BELIEVES:
        from belief_model_component import check_belief_inputs
        check_belief_inputs(dataset, kwargs["belief_source"], kwargs["belief_model"], kwargs["max_beliefs"])

    # ==================== Resume: skip already-generated queries ====================
    # Named and recorded by the CLI --policy-model name (oss_20b, not oss).
    llm_model = kwargs.pop("llm_model")
    policy_model_cli = kwargs.pop("policy_model_cli")
    run_name = build_run_name_for_pipeline(policy_model=policy_model_cli, llm_model=llm_model, **kwargs)
    dataset_dir = build_dataset_dir_name(dataset, file_data_set, query_key, kwargs["retriever_name"])
    run_dir = str(Path(output_path) / dataset_dir / run_name)

    print(f"\n{'=' * 80}")
    print(f"[OUTPUT] Loading/saving results from: {run_dir}")
    print(f"{'=' * 80}")

    run_config = build_run_config(
        policy_model_cli, llm_model, dataset=dataset, split=file_data_set, query_key=query_key, **kwargs,
    )
    if stage in GENERATES:
        # Every trajectory of the run directory is generated with the same
        # settings: a resume with other ones stops here, before any query runs.
        check_run_config(run_dir, run_config)
    else:
        # The belief and eval stages read the saved run with this call's
        # qrels and corpus (the dataset section, which must be the run's);
        # only the evaluation settings (e.g. the judge) may change.
        check_run_dataset(run_dir, run_config)
        update_run_config_evaluation(run_dir, run_config)

    if stage not in GENERATES:
        queries = {}
    else:
        generated = scan_queries(run_dir)[0]
        if generated:
            print(f"Found {len(generated)} already generated queries — skipping them")
            original_count = len(queries)
            queries = {qid: q for qid, q in queries.items() if qid not in generated}
            print(f"Remaining: {len(queries)} queries (skipped {original_count - len(queries)})")
        if limit is not None and limit > 0:
            queries = dict(list(queries.items())[:limit])
            print(f"Limited to {len(queries)} queries for testing")
        if not queries:
            print("All queries already generated")

    return RunPlan(
        dataset=dataset, split=file_data_set, queries=queries,
        all_questions=all_questions, qrels=qrels, answers=answers,
        run_dir=run_dir, limit=limit,
    )
