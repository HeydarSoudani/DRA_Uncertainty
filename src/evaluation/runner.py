"""Evaluation of an inference run, used by ``experiments/inference/dra_inference.py``.

    build_evaluators        Instantiate the evaluators of a dataset.
    load_run_results        Read the run's queries back from their saved files.
    evaluate_and_save       Run every evaluator, print the log block and write
                            ``eval/metrics.md`` and ``eval/per_query.csv``
                            (``evaluation.metrics_file``).

The groups, in the order of the files and the log:

    retrieval    Recall@N, Precision@N of the seen docs (N = docs seen)
    generation   accuracy (QA) or Auto-ARGUE nugget coverage and F1 (reports)
    uncertainty  s_T, its formula ablations and the agent's verbal confidence
                 against correctness (QA, belief files); a report's s_T
                 against its nugget coverage (reports, belief files)
    belief       the finished queries with a belief file of the current
                 schema (belief files); the belief's change at gold and
                 control events, and per
                 criterion its recall, answer reached and status calibration
                 (datasets with gold per criterion, belief files); the
                 share of update calls with issues, and every issue code
                 that occurs (belief files, ``belief.issues``)
    trajectory   searches, agent and belief model tokens, forced answers

A run with an output directory is always evaluated from its saved files,
whatever its stage (``--stage``), so every stage writes the same files.
The LLM-judged evaluators keep their verdicts in the run directory
under ``eval/`` (``eval/generation/``: ``accuracy.jsonl`` or the Auto-ARGUE
cache; ``eval/belief/judge_cache.jsonl``) and judge only what changed.
"""

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union

from indexing_corpus_dataset.dataset_loaders import load_nuggets
from indexing_corpus_dataset.layout import DATASET_SPECS
from utils.run_files import BELIEF_DIR, load_belief_files, load_result_from_saved_files
from utils.run_files.run_config import belief_model_of, read_run_config
from utils.progress import step_iter

from .belief import GOLD_DATASETS, BeliefUpdateEvaluator, issue_summary
from .generation import AccuracyEvaluator, ArgueReportEvaluator, NumericMatchEvaluator
from .metrics_file import (EVAL_DIR, METRICS_FILE, PER_QUERY_FILE, RULE, GroupResult, Row, Section, log_lines,
                           write_metrics, write_per_query)
from .retrieval import SeenDocEvaluator
from .trajectory import TrajectoryEvaluator
from .uncertainty import ReportUncertaintyEvaluator, UncertaintyEvaluator

logger = logging.getLogger(__name__)

@dataclass
class Evaluators:
    """The evaluators of one run.

    ``accuracy`` is None without ground-truth answers or for a dataset
    without ``answer_eval``; ``report`` is None for a dataset without
    ``report_eval`` (``layout.DATASET_SPECS``); ``uncertainty`` is None for
    a report dataset, ``report_uncertainty`` for any other; ``belief`` is None for a dataset without
    per-criterion gold (``evaluation.belief.gold_rows``).  The last two score
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
    (``utils.run_files.load_result_from_saved_files``); a query without files
    is left out."""
    run_dir = Path(run_dir)
    query_ids = sorted(query_ids)
    if not query_ids:
        return {}
    start = time.time()
    with ThreadPoolExecutor(max_workers=min(8, len(query_ids))) as pool:
        loaded = list(step_iter(pool.map(lambda qid: load_result_from_saved_files(run_dir, qid), query_ids),
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
    return {
        "run": str(run_dir) if run_dir else "-",
        "dataset": f"{data.get('dataset') or dataset} / {data.get('split') or '-'}",
        "agent": agent.get("policy_model") or "-",
        "belief": (f"{belief_model_of(belief) or '-'} ({belief.get('criteria_source_effective')} criteria)"
                   if belief else "-"),
        "queries": num_queries,
        **({"unfinished": num_unfinished} if num_unfinished is not None else {}),
        "evaluated": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }


def evaluate_and_save(
    results: Dict[str, Any],
    evaluators: Evaluators,
    run_dir: Optional[Union[str, Path]],
    num_unfinished: Optional[int] = None,
) -> None:
    """Run every evaluator, print the log block and write ``metrics.md``,
    ``per_query.csv`` (``evaluation.metrics_file``) and the evaluators' own files
    (``eval/generation/``, ``eval/belief/``), all under ``eval/``.

    Each evaluator returns a :class:`GroupResult`; the groups are collected
    in the order of the files (the per-query columns keep it).

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
    rows: List[Row] = []
    sections: List[Section] = []
    per_query: Dict[str, Dict[str, Any]] = {qid: {} for qid in results}

    def timed(name: str, fn, *args):
        start = time.time()
        out = fn(*args)
        timings[name] = time.time() - start
        return out

    def collect(name: str, fn, *args) -> GroupResult:
        res = timed(name, fn, *args)
        for qid, rec in res.per_query.items():
            per_query.setdefault(qid, {}).update(rec)
        rows.extend(res.rows)
        sections.extend(res.sections)
        return res

    collect("retrieval", ev.seen.evaluate, results)
    accuracy = collect("accuracy", ev.accuracy.evaluate, results, run_dir) if ev.accuracy else GroupResult()
    report = collect("report", ev.report.evaluate, results, run_dir) if ev.report else GroupResult()

    # Each belief file is read once, for every evaluation below.
    has_belief = run_dir is not None and (run_dir / BELIEF_DIR).is_dir()
    belief_files = timed("belief files", load_belief_files, run_dir, list(results)) if has_belief else {}
    finals = {q: f[2] for q, f in belief_files.items()}
    if belief_files and ev.uncertainty is not None and accuracy.raw:
        collect("uncertainty", ev.uncertainty.evaluate, finals, accuracy.raw["per_query"])
    if belief_files and ev.report_uncertainty is not None and report.raw:
        collect("uncertainty", ev.report_uncertainty.evaluate, finals, report.raw["per_query"])
    if has_belief:
        # The belief stage saves no query whose call failed: the rest of the
        # group is over these queries only.
        rows.append(Row("belief", "queries with a belief file", len(belief_files), n=len(results)))
    belief = (collect("belief", ev.belief.evaluate, run_dir, belief_files, list(results))
              if belief_files and ev.belief is not None else GroupResult())
    if belief_files:
        collect("belief issues", issue_summary, belief_files)
    collect("trajectory", ev.trajectory.evaluate, results, belief_files)

    # ── Log and files ────────────────────────────────────────────────────
    header = _header(run_dir, ev.dataset, len(results), num_unfinished)
    title = f"EVALUATION  {header['dataset']} / {header['agent']} / {header['belief']}  (n={len(results)})"
    print(f"\n{RULE}")
    print("\n".join(log_lines(title, rows)))
    print(RULE)
    print("  Evaluation time: " + ", ".join(f"{k} {v:.1f}s" for k, v in timings.items()))
    if run_dir is None:
        return
    (run_dir / EVAL_DIR).mkdir(parents=True, exist_ok=True)
    write_metrics(run_dir / METRICS_FILE, header, rows, sections)
    write_per_query(run_dir / PER_QUERY_FILE, per_query)
    print(f"  Saved {run_dir / EVAL_DIR}/: metrics.md, per_query.csv"
          f"{' and belief/events.csv' if belief else ''}")
