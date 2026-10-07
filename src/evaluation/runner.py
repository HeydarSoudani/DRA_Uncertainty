"""Evaluation of an inference run, used by ``experiments/dra_inference.py``.

    save_query_outputs      Write one query's files under the run directory.
    build_evaluators        Instantiate the evaluators of a dataset.
    load_run_results        Read the run's queries back from their saved files.
    evaluate_and_save       Run every evaluator and write ``summary.json``
                            (grouped: retrieval / generation /
                            trajectory), plus the terminal log.

A run with an output directory is always evaluated from its saved files, at
the end of the run as with ``--eval-only``, so both write the same
``summary.json``.  The LLM-judged evaluators keep their verdicts in the run
directory (``accuracy.jsonl``, ``report_eval/``) and judge only what
changed.
The surfaced-doc fusion metrics come from
:func:`evaluation.retrieval.fusion.run_fusion_eval`, which the caller runs
first and passes to :func:`evaluate_and_save`.
"""

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Union

from tqdm import tqdm

from indexing_corpus_dataset.dataset_loaders import load_nuggets
from indexing_corpus_dataset.layout import DATASET_SPECS
from utils.io_utils import load_result_from_saved_files

from .answer import AccuracyEvaluator, ArgueReportEvaluator, NumericMatchEvaluator
from .common import RULE, write_json
from .generation import GenerationEvaluator
from .retrieval import CitedDocEvaluator, SeenDocEvaluator, SurfacedDocEvaluator
from .retrieval.metrics import DEFAULT_K_VALUES
from .trajectory import TrajectoryEvaluator, save_trajectory
from .uncertainty import save_uncertainty

logger = logging.getLogger(__name__)

#: Per-query output directories under the run directory.
QUERY_OUTPUT_DIRS = (
    "retrieval/surfaced", "retrieval/seen", "retrieval/cited",
    "generation", "trajectory", "uncertainty",
)


def save_query_outputs(run_dir: Union[str, Path], query_id: str, question: str, result: Dict[str, Any]) -> None:
    """Write every per-query file of *result* under *run_dir*.

    ``retrieval/{surfaced,seen,cited}/{qid}.trec``, ``generation/{qid}.md``,
    ``trajectory/{qid}.jsonl`` and ``uncertainty/{qid}.jsonl`` (the last only
    when the uncertainty estimator ran).
    """
    run_dir = Path(run_dir)
    SurfacedDocEvaluator.save_item(query_id, result, run_dir / "retrieval" / "surfaced")
    SeenDocEvaluator.save_item(query_id, result, run_dir / "retrieval" / "seen")
    CitedDocEvaluator.save_item(query_id, result, run_dir / "retrieval" / "cited")
    GenerationEvaluator.save_item(query_id, result, run_dir / "generation")
    save_trajectory(query_id, question, result, run_dir / "trajectory")
    save_uncertainty(query_id, question, result, run_dir / "uncertainty")


@dataclass
class Evaluators:
    """The evaluators of one run.

    ``accuracy`` is None without ground-truth answers or for a dataset
    without ``answer_eval``; ``report`` is None for a dataset without
    ``report_eval`` (``layout.DATASET_SPECS``).
    """
    seen: SeenDocEvaluator
    cited: CitedDocEvaluator
    trajectory: TrajectoryEvaluator
    generation: GenerationEvaluator
    accuracy: Optional[Union[AccuracyEvaluator, NumericMatchEvaluator]] = None
    report: Optional[ArgueReportEvaluator] = None


def build_evaluators(
    qrels: Dict,
    kwargs: Dict[str, Any],
    answers: Optional[Dict[str, str]] = None,
    questions: Optional[Dict[str, str]] = None,
    dataset: Optional[str] = None,
    graded_qrels: Optional[Dict] = None,
    data_path: Optional[Union[str, Path]] = None,
    split: Optional[str] = None,
) -> Evaluators:
    """Instantiate the evaluators of a run.

    Args:
        qrels:        Qrels dict loaded from the dataset.
        kwargs:       Pipeline configuration (read: ``k_values``,
                      ``interleaving_window``, ``rrf_k``, ``judge_model``,
                      ``corpus_path``).
        answers:      ``query_id -> ground-truth answer``.  The dataset's
                      ``answer_eval`` (``layout.DATASET_SPECS``) picks the
                      accuracy evaluator: ``"numeric_match"`` (TRQA)
                      :class:`NumericMatchEvaluator`, ``"llm_judge"``
                      (BrowseComp-Plus) :class:`AccuracyEvaluator`.
        questions:    ``query_id -> question text``, for the judges.
        dataset:      Dataset name.
        graded_qrels: ``{query_id: {doc_id: gain}}`` (official gains) for
                      GradedRecall@N and as the NDCG gains.
        data_path:    Dataset directory and *split*: the nuggets of a dataset
                      whose ``report_eval`` is ``"argue"``
                      (:class:`ArgueReportEvaluator`).
    """
    retrieval_kwargs = dict(
        qrels=qrels,
        k_values=kwargs.get("k_values") or list(DEFAULT_K_VALUES),
        interleaving_window=kwargs.get("interleaving_window", 3),
        rrf_k=kwargs.get("rrf_k", 60),
        graded_qrels=graded_qrels,
    )
    evaluators = Evaluators(
        seen=SeenDocEvaluator(**retrieval_kwargs),
        cited=CitedDocEvaluator(**retrieval_kwargs),
        trajectory=TrajectoryEvaluator(),
        generation=GenerationEvaluator(),
    )

    spec = DATASET_SPECS[dataset] if dataset else None
    answer_eval = spec.answer_eval if spec else "llm_judge"
    judge_kwargs = {"judge_model": kwargs["judge_model"]} if kwargs.get("judge_model") else {}
    if answers and answer_eval == "numeric_match":
        evaluators.accuracy = NumericMatchEvaluator(answers=answers)
    elif answers and answer_eval == "llm_judge":
        evaluators.accuracy = AccuracyEvaluator(answers=answers, questions=questions, **judge_kwargs)
    if spec is not None and spec.report_eval == "argue":
        evaluators.report = ArgueReportEvaluator(
            nuggets=load_nuggets(data_path, split), questions=questions or {},
            corpus_path=kwargs["corpus_path"], max_chars=spec.report_chars, **judge_kwargs,
        )
    return evaluators


def load_run_results(run_dir: Union[str, Path], query_ids: Iterable[str]) -> Dict[str, Dict[str, Any]]:
    """``{query_id: result}`` read back from the run's saved files
    (``utils.io_utils.load_result_from_saved_files``); a query without files
    is left out."""
    run_dir = Path(run_dir)
    query_ids = sorted(query_ids)
    if not query_ids:
        return {}
    start = time.time()
    with ThreadPoolExecutor(max_workers=min(8, len(query_ids))) as pool:
        loaded = list(tqdm(pool.map(lambda qid: load_result_from_saved_files(run_dir, qid), query_ids),
                           total=len(query_ids), desc="Loading results", unit="query"))
    results = {qid: r for qid, r in zip(query_ids, loaded) if r}
    print(f"Loaded {len(results)} queries from {run_dir} in {time.time() - start:.1f}s")
    return results


def _fmt(value: Optional[float], digits: int = 4) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def evaluate_and_save(
    results: Dict[str, Any],
    evaluators: Evaluators,
    run_dir: Optional[Union[str, Path]],
    fusion_metrics: Optional[Dict[str, Any]] = None,
) -> None:
    """Run every evaluator, print the results and write ``summary.json``.

    One grouped ``summary`` dict is the source of ``summary.json``; the
    terminal log prints its headline numbers in the same order::

        {
          "num_queries": N,
          "retrieval":  {"seen": {...}, "cited": {...}, "fusion": {...}},
          "generation": {"correctness": {...}   # datasets with answers
                         | "nuggets": {...},     # report datasets (Auto-ARGUE)
                         "stats": {...}},
          "trajectory": {...},
        }

    The log shows the seen docs only; cited and fusion are in the file.

    Args:
        results:        Unified results dict keyed by query_id.
        evaluators:     From :func:`build_evaluators`.
        run_dir:        Run directory; None skips every file.
        fusion_metrics: Per-method surfaced-doc metrics from
                        :func:`evaluation.retrieval.fusion.run_fusion_eval`,
                        nested under ``retrieval.fusion``.
    """
    ev = evaluators
    timings: Dict[str, float] = {}

    def timed(name: str, fn, *args):
        start = time.time()
        out = fn(*args)
        timings[name] = time.time() - start
        return out

    generation_metrics = timed("generation", ev.generation.evaluate, results)
    trajectory_metrics = timed("trajectory", ev.trajectory.evaluate, results)
    cited_metrics = timed("cited", ev.cited.evaluate, results)
    seen_metrics = timed("seen", ev.seen.evaluate, results)
    accuracy_metrics = timed("accuracy", ev.accuracy.evaluate, results, run_dir) if ev.accuracy else {}
    report_metrics = timed("report", ev.report.evaluate, results, run_dir) if ev.report else {}

    # Every evaluator must cover the same queries.
    num_queries = generation_metrics.get("num_queries", 0)
    counts = {
        "trajectory": trajectory_metrics.get("num_queries", 0) if trajectory_metrics else num_queries,
        "cited_doc_retrieval": cited_metrics.get("num_queries", 0) if cited_metrics else num_queries,
        "seen_doc_retrieval": seen_metrics.get("num_queries", 0) if seen_metrics else num_queries,
        "accuracy": accuracy_metrics.get("num_evaluated", 0) if accuracy_metrics else num_queries,
        "report": (report_metrics["num_evaluated"] + report_metrics["num_without_nuggets"]
                   + report_metrics["num_judge_failures"]) if report_metrics else num_queries,
    }
    mismatches = {k: v for k, v in counts.items() if v != num_queries}
    if mismatches:
        logger.warning("Evaluator query-count mismatch! Expected %d (from generation). "
                       "Mismatches: %s", num_queries, mismatches)

    # ── Grouped summary ──────────────────────────────────────────────────
    summary: Dict[str, Any] = {"num_queries": num_queries}

    retrieval = {k: v for k, v in (("seen", seen_metrics), ("cited", cited_metrics),
                                   ("fusion", fusion_metrics)) if v}
    if retrieval:
        summary["retrieval"] = retrieval

    generation: Dict[str, Any] = {}
    if accuracy_metrics:
        generation["correctness"] = {k: accuracy_metrics[k] for k in (
            "accuracy", "num_correct", "num_evaluated", "num_judge_errors", "exact_match", "soft_exact_match",
        ) if k in accuracy_metrics}
    if report_metrics:
        generation["nuggets"] = ev.report.summary(report_metrics)
    generation["stats"] = generation_metrics
    summary["generation"] = generation

    if trajectory_metrics:
        summary["trajectory"] = trajectory_metrics

    # ── Terminal log, in summary order ───────────────────────────────────
    print(f"\n{RULE}\nEVALUATION SUMMARY  ({num_queries} queries)\n{RULE}")

    at_n = seen_metrics.get("Metrics@N", {})
    print("Retrieval (seen docs)")
    if at_n:
        line = (f"  Recall@N {_fmt(at_n.get('Recall@N'))} | Precision@N {_fmt(at_n.get('Precision@N'))}"
                f" | F1@N {_fmt(at_n.get('F1@N'))}")
        if "GradedRecall@N" in at_n:
            line += f" | GradedRecall@N {_fmt(at_n['GradedRecall@N'])}"
        print(line + f" | avg N {at_n.get('avg_N', 0):.1f}")
    else:
        print("  n/a")

    print("Generation")
    if accuracy_metrics:
        print(f"  Correctness: accuracy {_fmt(accuracy_metrics.get('accuracy'))}"
              f" ({accuracy_metrics.get('num_correct', 0)}/{accuracy_metrics.get('num_evaluated', 0)},"
              f" judge errors {accuracy_metrics.get('num_judge_errors', 0)})")
    if report_metrics:
        print("\n".join(ev.report.summary_lines(report_metrics)))
    print(f"  Length {generation_metrics.get('avg_generation_length', 0):.0f} chars"
          f" / {generation_metrics.get('avg_generation_words', 0):.0f} words"
          f" | citations {generation_metrics.get('avg_citations', 0):.1f}")

    if trajectory_metrics:
        t = trajectory_metrics

        def mean(key: str, block: Optional[Dict] = None) -> float:
            return ((block or t).get(key) or {}).get("mean", 0.0)

        print("Trajectory")
        print(f"  steps {mean('steps'):.1f} | searches {mean('search_steps'):.1f}"
              f" | docs seen/search {mean('docs_seen_per_search'):.1f}"
              f" | no search {t.get('queries_no_search', 0)}"
              f" | forced answer {t.get('queries_with_force_answer', 0)}"
              f" | max iter {t.get('queries_max_iter_reached', 0)}")
        tokens = t.get("tokens") or {}
        if tokens.get("num_queries_with_tokens"):
            print(f"  tokens/query {mean('total_tokens', tokens):.0f}"
                  f" (in {mean('input_tokens', tokens):.0f}, out {mean('output_tokens', tokens):.0f})"
                  f" | LLM calls/query {mean('llm_calls', tokens):.1f}")
    print(RULE)
    print("  Evaluation time: " + ", ".join(f"{k} {v:.1f}s" for k, v in timings.items()))

    # ── Files ────────────────────────────────────────────────────────────
    if not run_dir:
        return
    run_dir = Path(run_dir)
    if accuracy_metrics:
        ev.accuracy.save_results(accuracy_metrics, run_dir / "accuracy.jsonl")
    write_json(run_dir / "summary.json", summary)
    print(f"  ✓ Saved summary: {run_dir}/summary.json")
