"""Evaluation of an inference run, used by ``experiments/inference/dra_inference.py``
(and so by belief model training's phase 1, a dra_inference run).

    save_query_outputs      Write one query's files under the run directory.
    build_evaluators        Instantiate the evaluators of a dataset.
    load_run_results        Read the run's queries back from their saved files.
    evaluate_and_save       Run every evaluator, print the log block and write
                            ``metrics.md`` and ``per_query.csv``
                            (``evaluation.report``).

The groups, in the order of the files and the log:

    retrieval    Recall@N, Precision@N of the seen docs (N = docs seen)
    generation   accuracy (QA) or Auto-ARGUE nugget coverage and F1 (reports)
    uncertainty  s_T, its formula ablations and the agent's verbal confidence
                 against correctness (QA, belief model on); a report's s_T
                 against its nugget coverage (reports, belief model on)
    belief       the belief update on gold documents (datasets with gold
                 per criterion, belief model on)
    trajectory   searches, agent and belief model tokens, forced answers

A run with an output directory is always evaluated from its saved files, at
the end of the run as with ``--eval-only``, so both write the same files.
The LLM-judged evaluators keep their verdicts in the run directory
(``accuracy.jsonl``, ``report_eval/``, ``belief_eval/judge_cache.jsonl``)
and judge only what changed.
"""

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union

from tqdm import tqdm

from indexing_corpus_dataset.dataset_loaders import load_nuggets
from indexing_corpus_dataset.layout import DATASET_SPECS
from utils.io_utils import load_result_from_saved_files

from .answer import AccuracyEvaluator, ArgueReportEvaluator, NumericMatchEvaluator
from .belief import BeliefUpdateEvaluator, load_belief_files, read_run_config, save_belief
from .belief.evaluator import BELIEF_EVAL_DIR
from .common import RULE
from .metrics import mean
from .generation import GenerationEvaluator
from .gold.criteria import GOLD_DATASETS
from .report import METRICS_FILE, PER_QUERY_FILE, Row, log_lines, write_metrics, write_per_query
from .retrieval import CitedDocEvaluator, SeenDocEvaluator, SurfacedDocEvaluator
from .trajectory import TrajectoryEvaluator, save_trajectory
from .uncertainty import ReportUncertaintyEvaluator, UncertaintyEvaluator

logger = logging.getLogger(__name__)

#: Per-query output directories under the run directory (``belief/`` is
#: created by its writer, only when the belief model is on).
QUERY_OUTPUT_DIRS = (
    "retrieval/surfaced", "retrieval/seen", "retrieval/cited",
    "generation", "trajectory",
)


def save_query_outputs(run_dir: Union[str, Path], query_id: str, question: str, result: Dict[str, Any]) -> None:
    """Write every per-query file of *result* under *run_dir*.

    ``retrieval/{surfaced,seen,cited}/{qid}.trec``, ``generation/{qid}.md``,
    ``belief/{qid}.jsonl`` (only when the belief model ran), then
    ``trajectory/{qid}.jsonl``: its meta line marks the query as finished
    (``utils.io_utils.get_processed_queries``), so it is written last.
    """
    run_dir = Path(run_dir)
    SurfacedDocEvaluator.save_item(query_id, result, run_dir / "retrieval" / "surfaced")
    SeenDocEvaluator.save_item(query_id, result, run_dir / "retrieval" / "seen")
    CitedDocEvaluator.save_item(query_id, result, run_dir / "retrieval" / "cited")
    GenerationEvaluator.save_item(query_id, result, run_dir / "generation")
    save_belief(query_id, question, result, run_dir / "belief")
    save_trajectory(query_id, question, result, run_dir / "trajectory")


@dataclass
class Evaluators:
    """The evaluators of one run.

    ``accuracy`` is None without ground-truth answers or for a dataset
    without ``answer_eval``; ``report`` is None for a dataset without
    ``report_eval`` (``layout.DATASET_SPECS``); ``uncertainty`` is None for
    a report dataset, ``report_uncertainty`` for any other; ``belief`` is None for a dataset without
    per-criterion gold (``evaluation.gold.criteria``).  The last two score
    a run only when its belief files exist.
    """
    seen: SeenDocEvaluator
    trajectory: TrajectoryEvaluator
    accuracy: Optional[Union[AccuracyEvaluator, NumericMatchEvaluator]] = None
    report: Optional[ArgueReportEvaluator] = None
    uncertainty: Optional[UncertaintyEvaluator] = None
    report_uncertainty: Optional[ReportUncertaintyEvaluator] = None
    belief: Optional[BeliefUpdateEvaluator] = None
    dataset: Optional[str] = None


def build_evaluators(
    qrels: Dict,
    kwargs: Dict[str, Any],
    answers: Optional[Dict[str, str]] = None,
    questions: Optional[Dict[str, str]] = None,
    dataset: Optional[str] = None,
    data_path: Optional[Union[str, Path]] = None,
    split: Optional[str] = None,
) -> Evaluators:
    """Instantiate the evaluators of a run.

    Args:
        qrels:        Qrels dict loaded from the dataset (at the run's
                      ``min_relevance_score``).
        kwargs:       Pipeline configuration (read: ``judge_model``,
                      ``corpus_path``).
        answers:      ``query_id -> ground-truth answer``.  The dataset's
                      ``answer_eval`` (``layout.DATASET_SPECS``) picks the
                      accuracy evaluator: ``"numeric_match"`` (TRQA)
                      :class:`NumericMatchEvaluator`, ``"llm_judge"``
                      (BrowseComp-Plus, infoseek, webshaper_frames) :class:`AccuracyEvaluator`.
        questions:    ``query_id -> question text``, for the judges.
        dataset:      Dataset name.
        data_path:    Dataset directory and *split*: the nuggets of a dataset
                      whose ``report_eval`` is ``"argue"``
                      (:class:`ArgueReportEvaluator`); *split* also locates
                      TRQA's gold criteria (the belief update evaluation).
    """
    evaluators = Evaluators(seen=SeenDocEvaluator(qrels), trajectory=TrajectoryEvaluator(), dataset=dataset)

    spec = DATASET_SPECS[dataset] if dataset else None
    answer_eval = spec.answer_eval if spec else "llm_judge"
    judge_model = kwargs.get("judge_model") or None
    judge_kwargs = {"judge_model": judge_model} if judge_model else {}
    if answers and answer_eval == "numeric_match":
        evaluators.accuracy = NumericMatchEvaluator(answers=answers)
    elif answers and answer_eval == "llm_judge":
        evaluators.accuracy = AccuracyEvaluator(answers=answers, questions=questions, **judge_kwargs)
    if spec is not None and spec.report_eval == "argue":
        evaluators.report = ArgueReportEvaluator(
            nuggets=load_nuggets(data_path, split), questions=questions or {},
            corpus_path=kwargs["corpus_path"], **judge_kwargs,
        )
    if spec is not None and spec.task == "qa" and evaluators.accuracy is not None:
        evaluators.uncertainty = UncertaintyEvaluator()
    if spec is not None and spec.task == "report" and evaluators.report is not None:
        evaluators.report_uncertainty = ReportUncertaintyEvaluator()
    if dataset in GOLD_DATASETS:
        evaluators.belief = BeliefUpdateEvaluator(dataset, split, kwargs.get("corpus_path"), judge_model)
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


def _header(run_dir: Optional[Path], dataset: Optional[str], num_queries: int,
            num_unfinished: Optional[int] = None) -> Dict[str, Any]:
    """The run's identity for the header of ``metrics.md`` and the log
    (``run_config.json``); ``queries`` counts the finished (evaluated)
    ones, ``unfinished`` those of the split not finished yet."""
    config = read_run_config(run_dir) if run_dir is not None else {}
    data, agent, belief = config.get("dataset") or {}, config.get("agent") or {}, config.get("belief_model") or {}
    mode = belief.get("mode") or "?"
    source = belief.get("criteria_source_effective")
    return {
        "run": str(run_dir) if run_dir else "-",
        "dataset": f"{data.get('dataset') or dataset} / {data.get('split') or '-'}",
        "agent": agent.get("policy_model") or "-",
        "belief": f"{mode}{f' ({source} criteria)' if source and mode != 'off' else ''}",
        "queries": num_queries,
        **({"unfinished": num_unfinished} if num_unfinished is not None else {}),
        "evaluated": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }


def _mean(values) -> float:
    """The mean of the values that are not None, NaN when there is none."""
    return mean([v for v in values if v is not None])


def evaluate_and_save(
    results: Dict[str, Any],
    evaluators: Evaluators,
    run_dir: Optional[Union[str, Path]],
    num_unfinished: Optional[int] = None,
) -> None:
    """Run every evaluator, print the log block and write ``metrics.md``,
    ``per_query.csv`` (``evaluation.report``) and the evaluators' own files
    (``accuracy.jsonl``, ``report_eval/``, ``belief_eval/``).

    Args:
        results:    Unified results dict keyed by query_id.
        evaluators: From :func:`build_evaluators`.
        run_dir:    Run directory; None skips every file (and the
                    uncertainty and belief evaluations, which read the
                    belief files).
        num_unfinished: Queries of the split not finished yet (the
                    header's ``unfinished``); None: not recorded.
    """
    ev = evaluators
    run_dir = Path(run_dir) if run_dir else None
    timings: Dict[str, float] = {}

    def timed(name: str, fn, *args):
        start = time.time()
        out = fn(*args)
        timings[name] = time.time() - start
        return out

    rows: List[Row] = []
    per_query: Dict[str, Dict[str, Any]] = {qid: {} for qid in results}

    def add(group: str, metric: str, value: Any, ci_low=None, ci_high=None, n=None) -> None:
        rows.append(Row(group, metric, value, ci_low, ci_high, n))

    def merge(values: Dict[str, Dict[str, Any]]) -> None:
        for qid, rec in values.items():
            per_query.setdefault(qid, {}).update(rec)

    # ── Retrieval: the seen docs ─────────────────────────────────────────
    seen = timed("retrieval", ev.seen.evaluate, results)
    merge(seen)
    add("retrieval", "Recall@N", _mean(r["recall_at_n"] for r in seen.values()), n=len(seen))
    add("retrieval", "Precision@N", _mean(r["precision_at_n"] for r in seen.values()), n=len(seen))
    add("retrieval", "N (docs seen)", _mean(r["n_seen"] for r in seen.values()), n=len(seen))

    # ── Generation: correctness or nuggets ───────────────────────────────
    accuracy = timed("accuracy", ev.accuracy.evaluate, results, run_dir) if ev.accuracy else {}
    if accuracy and run_dir is not None:
        # Saved at once: the verdicts are paid for, and a later evaluator may fail.
        ev.accuracy.save_results(accuracy, run_dir / "accuracy.jsonl")
    if accuracy:
        merge({r["query_id"]: {"correct": None if r.get("judge_error") else bool(r["correct"]),
                               "verbal_confidence": r.get("verbal_confidence")} for r in accuracy["per_query"]})
        add("generation", "accuracy", accuracy["accuracy"], n=accuracy["num_evaluated"])
        add("generation", "correct", accuracy["num_correct"])
        add("generation", "evaluated", accuracy["num_evaluated"])
        add("generation", "judge errors", accuracy.get("num_judge_errors", 0))
    report = timed("report", ev.report.evaluate, results, run_dir) if ev.report else {}
    if report:
        weighted = ev.dataset == "neuclir"
        keys = ["nugget_coverage", "f1"] + (["nugget_coverage_weighted", "f1_weighted"] if weighted else [])
        merge({r["query_id"]: {k: r.get(k) for k in keys} for r in report["per_query"]})
        names = {"nugget_coverage": "nugget coverage", "f1": "nugget F1",
                 "nugget_coverage_weighted": "nugget coverage (weighted)", "f1_weighted": "nugget F1 (weighted)"}
        for key in keys:
            add("generation", names[key], report["metrics"].get(key), n=report["num_evaluated"])
        add("generation", "evaluated", report["num_evaluated"])
        add("generation", "without nuggets", report["num_without_nuggets"])
        add("generation", "empty reports", report["num_empty_reports"])
        add("generation", "reports without citations", report["num_uncited_reports"])
        add("generation", "reports with unmapped citations", report["num_unmapped_reports"])
        add("generation", "judge failures", report["num_judge_failures"])

    # ── Uncertainty and belief update (belief files) ─────────────────────
    has_belief = run_dir is not None and (run_dir / "belief").is_dir()
    # Each belief file is read once, for every evaluation below.
    belief_files = timed("belief files", load_belief_files, run_dir, list(results)) if has_belief else {}
    if belief_files and ev.uncertainty is not None and accuracy:
        finals = {q: f[2] for q, f in belief_files.items()}
        unc = timed("uncertainty", ev.uncertainty.evaluate, finals, accuracy["per_query"])
        if unc:
            merge({q: {k: v for k, v in r.items() if k != "y"} for q, r in unc["per_query"].items()})
            for metric, value, lo, hi, n in unc["rows"]:
                add("uncertainty", metric, value, lo, hi, n)
    if belief_files and ev.report_uncertainty is not None and report:
        finals = {q: f[2] for q, f in belief_files.items()}
        unc = timed("uncertainty", ev.report_uncertainty.evaluate, finals, report["per_query"])
        if unc:
            merge(unc["per_query"])
            for metric, value, lo, hi, n in unc["rows"]:
                add("uncertainty", metric, value, lo, hi, n)
    bel: Dict[str, Any] = {}
    if belief_files and ev.belief is not None:
        bel = timed("belief", ev.belief.evaluate, run_dir, belief_files, list(results))
        if bel:
            merge(bel["per_query"])
            for metric, value, lo, hi, n in bel["rows"]:
                add("belief", metric, value, lo, hi, n)

    # ── Trajectory ───────────────────────────────────────────────────────
    traj = timed("trajectory", ev.trajectory.evaluate, results)
    belief_tokens = {q: (f[0].get("tokens") or {}).get("total_tokens") for q, f in belief_files.items()}
    for qid, rec in traj.items():
        if qid in belief_tokens:
            rec["belief_tokens"] = belief_tokens[qid]
    merge(traj)
    add("trajectory", "searches", _mean(r["searches"] for r in traj.values()), n=len(traj))
    add("trajectory", "agent tokens", _mean(r["agent_tokens"] for r in traj.values()),
        n=sum(r["agent_tokens"] is not None for r in traj.values()))
    if any(v is not None for v in belief_tokens.values()):
        add("trajectory", "belief tokens", _mean(belief_tokens.values()),
            n=sum(v is not None for v in belief_tokens.values()))
    add("trajectory", "forced answers", sum(r["forced_answer"] for r in traj.values()), n=len(traj))
    add("trajectory", "max iterations reached", sum(r["max_iter"] for r in traj.values()), n=len(traj))

    # ── Log and files ────────────────────────────────────────────────────
    header = _header(run_dir, ev.dataset, len(results), num_unfinished)
    title = f"EVALUATION  {header['dataset']} / {header['agent']} / {header['belief']}  (n={len(results)})"
    print(f"\n{RULE}")
    print("\n".join(log_lines(title, rows)))
    print(RULE)
    print("  Evaluation time: " + ", ".join(f"{k} {v:.1f}s" for k, v in timings.items()))
    if run_dir is None:
        return
    write_metrics(run_dir / METRICS_FILE, header, rows)
    write_per_query(run_dir / PER_QUERY_FILE, per_query)
    # The grouped JSON summary of earlier evaluations is replaced by these files.
    (run_dir / "summary.json").unlink(missing_ok=True)
    print(f"  Saved {run_dir / METRICS_FILE} and {PER_QUERY_FILE}"
          f"{f' and {BELIEF_EVAL_DIR}/events.csv' if bel else ''}")
