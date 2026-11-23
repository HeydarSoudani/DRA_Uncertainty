"""Evaluation of an inference run, used by ``experiments/inference/dra_inference.py``.

    build_evaluators        Instantiate the evaluators of a dataset.
    load_run_results        Read the run's queries back from their saved files.
    evaluate_and_save       Run the evaluators of a scope (the generation, belief
                            models), print the log block and write
                            ``eval/metrics.md`` and ``eval/per_query.csv``
                            (``evaluation.metrics_file``), keeping the rest.

The groups, in the order of the files and the log.  The generation's
(retrieval, generation, the verbal confidence, the agent's trajectory) are
scored once; the beliefs' (s_T, belief, the belief tokens) once per belief model of
the run (``belief/{model}/``, ``utils.run_files.belief_models``), so a run
whose beliefs several belief models made reports each of them:

    retrieval    Recall@N, Precision@N of the seen docs (N = docs seen)
    generation   accuracy (QA) or Auto-ARGUE nugget coverage and F1 (reports)
    uncertainty  the agent's verbal confidence against correctness (QA,
                 the generation); the belief rule-based score and the
                 belief LLM score against correctness (QA, belief files);
                 a report's s_T against its nugget coverage (reports,
                 belief files)
    belief       evidence uptake and false uptake of s_t against the
                 qrels (QA, belief files, ``belief.uptake``); the update
                 calls, the share failing the format gate, and every issue
                 code that occurs (belief files, ``belief.issues``)
    trajectory   searches, agent tokens, forced answers; belief model tokens

A run with an output directory is always evaluated from its saved files.
Each stage (``--stage``) scores its own scope and keeps the rest of the
files: generate the generation, belief its belief model, eval and all
everything.
The LLM-judged evaluators (the generation's) keep their verdicts in the
run directory under ``eval/generation/`` (``accuracy.jsonl`` or the
Auto-ARGUE cache) and judge only what changed; the belief models'
evaluation makes no LLM call.
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
from utils.run_files import (SCHEMA_VERSION, belief_models, belief_schemas, load_belief_files,
                             load_result_from_saved_files)
from utils.run_files.run_config import belief_config_of, belief_model_of, read_run_config
from utils.progress import step_iter

from .belief import evidence_uptake, issue_summary
from .generation import AccuracyEvaluator, ArgueReportEvaluator, NumericMatchEvaluator
from .metrics_file import (EVAL_DIR, METRICS_FILE, NO_MODEL, PER_QUERY_FILE, RULE, GroupResult, Row, Section,
                           log_lines, read_header, read_metrics, read_per_query, read_sections, write_metrics,
                           write_per_query)
from .retrieval import SeenDocEvaluator
from .trajectory import TrajectoryEvaluator, belief_tokens
from .uncertainty import ReportUncertaintyEvaluator, UncertaintyEvaluator, VerbalConfidenceEvaluator

logger = logging.getLogger(__name__)

@dataclass
class Evaluators:
    """The evaluators of one run.

    ``accuracy`` is None without ground-truth answers or for a dataset
    without ``answer_eval``; ``report`` is None for a dataset without
    ``report_eval`` (``layout.DATASET_SPECS``); ``verbal`` and ``uncertainty`` are None for
    a report dataset, ``report_uncertainty`` for any other.  ``qrels`` are
    the run's on a QA dataset (the belief evidence uptake), None on a report
    dataset.  The uncertainty and belief
    evaluators score a run only when its belief files exist.
    """
    seen: SeenDocEvaluator
    trajectory: TrajectoryEvaluator
    accuracy: Optional[Union[AccuracyEvaluator, NumericMatchEvaluator]] = None
    report: Optional[ArgueReportEvaluator] = None
    verbal: Optional[VerbalConfidenceEvaluator] = None
    uncertainty: Optional[UncertaintyEvaluator] = None
    report_uncertainty: Optional[ReportUncertaintyEvaluator] = None
    qrels: Optional[Dict] = None
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
                      ``min_relevance_score``): the seen-doc retrieval and
                      the belief evidence uptake (QA datasets only).
        kwargs:       Pipeline configuration (read: ``judge_model``,
                      ``corpus_path``).
        answers:      ``query_id -> ground-truth answer``.  The dataset's
                      ``answer_eval`` (``layout.DATASET_SPECS``) picks the
                      accuracy evaluator: ``"numeric_match"`` (TRQA)
                      :class:`NumericMatchEvaluator`, ``"llm_judge"``
                      (BrowseComp-Plus, infoseek_hard) :class:`AccuracyEvaluator`.
        questions:    ``query_id -> question text``, for the judges.
        dataset:      Dataset name.
        data_path:    Dataset directory and *split*: the nuggets of a dataset
                      whose ``report_eval`` is ``"argue"``
                      (:class:`ArgueReportEvaluator`).
    """
    spec = DATASET_SPECS[dataset] if dataset else None
    # Uptake on QA datasets only: the qrels of a report dataset are pooled, so
    # an unjudged relevant document would count as a no-evidence document.
    evaluators = Evaluators(seen=SeenDocEvaluator(qrels), trajectory=TrajectoryEvaluator(), dataset=dataset,
                            qrels=qrels if spec is not None and spec.task == "qa" else None)

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
        evaluators.verbal = VerbalConfidenceEvaluator()
        evaluators.uncertainty = UncertaintyEvaluator()
    if spec is not None and spec.task == "report" and evaluators.report is not None:
        evaluators.report_uncertainty = ReportUncertaintyEvaluator()
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


def _header(run_dir: Optional[Path], dataset: Optional[str], num_queries: int, models: List[str],
            num_unfinished: Optional[int] = None) -> Dict[str, Any]:
    """The run's identity for the header of ``metrics.md`` and the log
    (``run_config.json``); ``belief`` names each belief model of *models*
    (short name, LLM, belief source; the schemas of its files when one is
    older than ``SCHEMA_VERSION``), ``queries`` counts the finished
    (evaluated) ones, ``unfinished`` those of the split not finished yet."""
    config = read_run_config(run_dir) if run_dir is not None else {}
    data, agent = config.get("dataset") or {}, config.get("agent") or {}

    def model_line(model: str) -> str:
        section = belief_config_of(config, model) or {}
        schemas = belief_schemas(run_dir, model) if run_dir is not None else {}
        older = {s: n for s, n in schemas.items() if s != SCHEMA_VERSION}
        schema = (f", schema {', '.join(f'{s} ({n} files)' for s, n in sorted(schemas.items(), key=str))}"
                  if older else "")
        return (f"{model} ({belief_model_of(section) or '-'}, {section.get('belief_source_effective')} beliefs"
                f"{schema})")

    return {
        "run": str(run_dir) if run_dir else "-",
        "dataset": f"{data.get('dataset') or dataset} / {data.get('split') or '-'}",
        "agent": agent.get("policy_model") or "-",
        "belief": "; ".join(model_line(m) for m in models) or "-",
        "queries": num_queries,
        **({"unfinished": num_unfinished} if num_unfinished is not None else {}),
        "evaluated": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }


def evaluate_and_save(
    results: Dict[str, Any],
    evaluators: Evaluators,
    run_dir: Optional[Union[str, Path]],
    num_unfinished: Optional[int] = None,
    generation: bool = True,
    models: Optional[List[str]] = None,
) -> None:
    """Run the evaluators of the scope, print the log block and write
    ``metrics.md``, ``per_query.csv`` (``evaluation.metrics_file``) and the
    evaluators' own files (``eval/generation/``),
    all under ``eval/``.

    Each evaluator returns a :class:`GroupResult`; the groups are collected
    in the order of the files (the per-query columns keep it).  The
    beliefs' groups are collected once per belief model, their rows tagged
    with the model, their per-query columns prefixed ``{model}/`` and their
    section titles suffixed `` ({model})``.

    The scope: the generation's groups (*generation*: retrieval, generation,
    the verbal confidence, trajectory) and the belief models *models*.  The
    rows, columns and sections out of the scope are kept from the run's
    ``metrics.md`` and ``per_query.csv``, so the generate stage scores the
    generation only and the belief stage adds its own model only.  A belief
    model's scope reads the answer verdicts (y) from ``eval/generation/``
    (no judge call when the generation was evaluated).  The generation is
    scored too when its saved rows cannot be kept
    (:func:`_saved_generation_stale`: no ``metrics.md``, no generation row
    in it, or another number of queries than the run's).

    Args:
        results:    Unified results dict keyed by query_id.
        evaluators: From :func:`build_evaluators`.
        run_dir:    Run directory; None skips every file (and the
                    uncertainty and belief evaluations, which read the
                    belief files).
        num_unfinished: Queries of the split not finished yet (the
                    header's ``unfinished``); None: not recorded.
        generation: Score the generation's groups.
        models:     The belief models to score; None: every belief model
                    of the run (``utils.run_files.belief_models``).
    """
    ev = evaluators
    run_dir = Path(run_dir) if run_dir else None
    run_models = belief_models(run_dir) if run_dir is not None else []
    models = run_models if models is None else [m for m in run_models if m in models]
    if not generation:
        stale = _saved_generation_stale(run_dir, len(results))
        if stale:
            print(f"  {stale}: the generation is evaluated too.")
            generation = True
    timings: Dict[str, float] = {}
    # Rows, sections and per-query values by scope: the generation (NO_MODEL) or a belief model.
    rows: Dict[str, List[Row]] = {}
    sections: Dict[str, List[Section]] = {}
    per_query: Dict[str, Dict[str, Dict[str, Any]]] = {}

    def timed(name: str, fn, *args):
        start = time.time()
        out = fn(*args)
        timings[name] = time.time() - start
        return out

    def collect(name: str, fn, *args, model: str = NO_MODEL) -> GroupResult:
        res = timed(name if model == NO_MODEL else f"{name} [{model}]", fn, *args)
        prefix = "" if model == NO_MODEL else f"{model}/"
        scope = per_query.setdefault(model, {})
        for qid, rec in res.per_query.items():
            scope.setdefault(qid, {}).update({prefix + k: v for k, v in rec.items()})
        for r in res.rows:
            r.belief_model = model
        rows.setdefault(model, []).extend(res.rows)
        sections.setdefault(model, []).extend(
            res.sections if model == NO_MODEL
            else [(f"{title} ({model})", cols, cells) for title, cols, cells in res.sections])
        return res

    if generation:
        collect("retrieval", ev.seen.evaluate, results)
        accuracy = collect("accuracy", ev.accuracy.evaluate, results, run_dir) if ev.accuracy else GroupResult()
        report = collect("report", ev.report.evaluate, results, run_dir) if ev.report else GroupResult()
        if ev.verbal is not None and accuracy.raw:
            collect("verbal", ev.verbal.evaluate, accuracy.raw["per_query"])
        collect("trajectory", ev.trajectory.evaluate, results)
    else:
        # y only (the verdicts saved by the generation's evaluation): no rows.
        accuracy = (timed("accuracy (saved verdicts)", ev.accuracy.evaluate, results, run_dir)
                    if ev.accuracy and models else GroupResult())
        report = (timed("report (saved verdicts)", ev.report.evaluate, results, run_dir)
                  if ev.report and models else GroupResult())

    # Per belief model: each belief file is read once, for every evaluation below.
    for model in models:
        rows.setdefault(model, [])
        belief_files = timed(f"belief files [{model}]", load_belief_files, run_dir, model, list(results))
        finals = {q: f[2] for q, f in belief_files.items()}
        if belief_files and ev.uncertainty is not None and accuracy.raw:
            collect("uncertainty", ev.uncertainty.evaluate, finals, accuracy.raw["per_query"], model=model)
        if belief_files and ev.report_uncertainty is not None and report.raw:
            collect("uncertainty", ev.report_uncertainty.evaluate, finals, report.raw["per_query"], model=model)
        if belief_files and ev.qrels:
            collect("belief", evidence_uptake, belief_files, ev.qrels, model=model)
        if belief_files:
            collect("belief issues", issue_summary, belief_files, model=model)
            collect("belief tokens", belief_tokens, belief_files, model=model)

    # ── Keep what is out of the scope from the run's files ─────────────────
    scopes = [NO_MODEL] + run_models
    kept = [m for m in scopes if (m == NO_MODEL and not generation) or (m != NO_MODEL and m not in models)]
    if kept and run_dir is not None:
        _keep_saved(run_dir, kept, run_models, rows, sections, per_query)

    all_rows = [r for m in scopes for r in rows.get(m, [])]
    all_sections = [sec for m in scopes for sec in sections.get(m, [])]
    merged: Dict[str, Dict[str, Any]] = {qid: {} for qid in results}
    for m in scopes:
        for qid, rec in per_query.get(m, {}).items():
            merged.setdefault(qid, {}).update(rec)

    # ── Log and files ────────────────────────────────────────────────────
    header = _header(run_dir, ev.dataset, len(results), run_models, num_unfinished)
    scored = (["generation"] if generation else []) + models
    title = (f"EVALUATION  {header['dataset']} / {header['agent']} / belief models: {', '.join(run_models) or '-'}"
             f"  (n={len(results)}; scored now: {', '.join(scored) or '-'})")
    print(f"\n{RULE}")
    print("\n".join(log_lines(title, all_rows)))
    print(RULE)
    print("  Evaluation time: " + ", ".join(f"{k} {v:.1f}s" for k, v in timings.items()))
    if run_dir is None:
        return
    (run_dir / EVAL_DIR).mkdir(parents=True, exist_ok=True)
    write_metrics(run_dir / METRICS_FILE, header, all_rows, all_sections)
    write_per_query(run_dir / PER_QUERY_FILE, merged)
    print(f"  Saved {run_dir / EVAL_DIR}/: metrics.md, per_query.csv")


def _saved_generation_stale(run_dir: Optional[Path], num_queries: int) -> Optional[str]:
    """Why the generation's rows of the run's ``metrics.md`` cannot be kept
    (an interrupted generate stage left them behind its saved queries, or
    the file is of another layout), None when they can."""
    path = run_dir / METRICS_FILE if run_dir is not None else None
    if path is None or not path.exists():
        return "No eval/metrics.md yet"
    if not any(r.belief_model == NO_MODEL for r in read_metrics(path)):
        return "No generation row in eval/metrics.md"
    saved = read_header(path).get("queries")
    if saved != str(num_queries):
        return f"eval/metrics.md scored {saved} queries, the run has {num_queries}"
    return None


def _keep_saved(run_dir: Path, kept: List[str], run_models: List[str], rows: Dict[str, List[Row]],
                sections: Dict[str, List[Section]], per_query: Dict[str, Dict[str, Dict[str, Any]]]) -> None:
    """Fill the scopes *kept* (NO_MODEL for the generation, or belief models)
    of *rows*, *sections* and *per_query* from the run's ``metrics.md`` and
    ``per_query.csv``: a row by its ``belief_model``, a section by its
    `` ({model})`` title suffix, a column by its ``{model}/`` prefix.  The
    models are those of the run and those the saved rows name, so what a
    model without a folder left (its beliefs removed) is dropped, never
    taken for the generation's."""
    metrics_path, per_query_path = run_dir / METRICS_FILE, run_dir / PER_QUERY_FILE
    saved_rows = read_metrics(metrics_path) if metrics_path.exists() else []
    owners = list(dict.fromkeys(run_models + [r.belief_model for r in saved_rows if r.belief_model != NO_MODEL]))

    def model_of_column(column: str) -> str:
        return next((m for m in owners if column.startswith(f"{m}/")), NO_MODEL)

    def model_of_section(title: str) -> str:
        return next((m for m in owners if title.endswith(f" ({m})")), NO_MODEL)

    for r in saved_rows:
        if r.belief_model in kept:
            rows.setdefault(r.belief_model, []).append(r)
    if metrics_path.exists():
        for sec in read_sections(metrics_path):
            if model_of_section(sec[0]) in kept:
                sections.setdefault(model_of_section(sec[0]), []).append(sec)
    if per_query_path.exists():
        for qid, rec in read_per_query(per_query_path).items():
            for column, value in rec.items():
                model = model_of_column(column)
                if model in kept:
                    per_query.setdefault(model, {}).setdefault(qid, {})[column] = value
