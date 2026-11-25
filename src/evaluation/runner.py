"""Evaluation of an inference run, used by ``experiments/inference/dra_inference.py``.

    build_evaluators    the evaluators that apply to a dataset (:data:`EVALUATORS`)
    load_run_results    the run's queries, read back from their saved files
    evaluate_and_save   run the evaluators of a scope, print the log block and
                        write ``eval/metrics.md`` and ``eval/per_query.csv``,
                        keeping what is out of the scope

Scopes: the generation's evaluators run once per run; the belief-model ones
once per belief model of the run (``belief/{model}/``), their rows tagged
with the model.  Each stage (``--stage``) scores its own scope: generate the
generation, belief its belief model, eval and all everything.  The groups,
their files and how to add an evaluator: ``evaluation/README.md``.
"""

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union

from indexing_corpus_dataset.shared.layout import DATASET_SPECS
from utils.run_files import (SCHEMA_VERSION, belief_models, belief_schemas, load_belief_files,
                             load_result_from_saved_files)
from utils.run_files.run_config import belief_config_of, belief_model_of, read_run_config
from utils.progress import step_iter

from . import belief, generation, retrieval, trajectory, uncertainty
from .belief import EvidenceUptakeEvaluator, IssueEvaluator
from .generation import AccuracyEvaluator, ArgueReportEvaluator, NumericMatchEvaluator
from .retrieval import SeenDocEvaluator
from .shared.context import GENERATION, EvalContext, EvalSetup
from .shared.files import (EVAL_DIR, GENERATION_EVAL_DIR, METRICS_FILE, PER_QUERY_FILE, read_header,
                           read_metrics, read_per_query, read_sections, write_metrics, write_per_query)
from .shared.log import RULE, log_lines
from .shared.result import NO_MODEL, GroupResult, Row, Section
from .trajectory import BeliefTokenEvaluator, TrajectoryEvaluator
from .uncertainty import ReportUncertaintyEvaluator, UncertaintyEvaluator, VerbalConfidenceEvaluator

logger = logging.getLogger(__name__)

#: Every evaluator, in run order: the order of ``per_query.csv``'s columns.
#: An evaluator that reads another's raw output (``provides``) comes after it.
EVALUATORS = (
    # once per run (scope GENERATION)
    SeenDocEvaluator,             # retrieval
    AccuracyEvaluator,            # generation (QA, LLM judge)       -> ctx.accuracy
    NumericMatchEvaluator,        # generation (QA, numeric match)   -> ctx.accuracy
    ArgueReportEvaluator,         # generation (reports)             -> ctx.report
    VerbalConfidenceEvaluator,    # uncertainty (reads ctx.accuracy)
    TrajectoryEvaluator,          # trajectory
    # once per belief model (scope BELIEF_MODEL)
    UncertaintyEvaluator,         # uncertainty (QA, reads ctx.accuracy)
    ReportUncertaintyEvaluator,   # uncertainty (reports, reads ctx.report)
    EvidenceUptakeEvaluator,      # belief (QA)
    IssueEvaluator,               # belief
    BeliefTokenEvaluator,         # trajectory
)

#: The log block: the generation's lines, then each belief model's.
GENERATION_LOG_LINES = (retrieval.metrics.log_line, generation.metrics.log_line,
                        uncertainty.metrics.log_line, trajectory.metrics.log_line)
MODEL_LOG_LINES = (uncertainty.metrics.model_log_line, belief.metrics.model_log_line)


@dataclass
class Evaluators:
    """The evaluators that apply to one run (:func:`build_evaluators`), by scope."""
    setup: EvalSetup
    dataset: str
    generation: List[Any] = field(default_factory=list)
    belief_model: List[Any] = field(default_factory=list)


def build_evaluators(
    qrels: Dict,
    kwargs: Dict[str, Any],
    *,
    dataset: str,
    answers: Optional[Dict[str, str]] = None,
    questions: Optional[Dict[str, str]] = None,
    data_path: Optional[Union[str, Path]] = None,
    split: Optional[str] = None,
) -> Evaluators:
    """The evaluators of :data:`EVALUATORS` that apply to the dataset (each
    one's ``from_setup``).

    Args:
        qrels:      Qrels at the run's ``min_relevance_score`` (the seen-doc
                    retrieval, the belief evidence uptake).
        kwargs:     Pipeline configuration (read: ``judge_model``, ``corpus_path``).
        dataset:    Dataset name (``layout.DATASET_SPECS``).
        answers:    ``query_id -> ground-truth answer``; the dataset's
                    ``answer_eval`` picks the accuracy evaluator.
        questions:  ``query_id -> question text``, for the judges.
        data_path:  Dataset directory, the nuggets of a report dataset are
                    read from (with *split*).
        split:      The split's name in *data_path* (the nuggets file).
    """
    setup = EvalSetup(spec=DATASET_SPECS[dataset], qrels=qrels, answers=answers,
                      questions=questions, judge_model=kwargs.get("judge_model") or None,
                      corpus_path=kwargs.get("corpus_path"), data_path=data_path, split=split)
    out = Evaluators(setup=setup, dataset=dataset)
    for cls in EVALUATORS:
        evaluator = cls.from_setup(setup)
        if evaluator is not None:
            (out.generation if cls.scope == GENERATION else out.belief_model).append(evaluator)
    return out


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
    ``metrics.md`` and ``per_query.csv`` (``shared.files``) under ``eval/``.

    The scope: the generation's evaluators (*generation*) and the belief
    models *models*.  A belief model's rows are tagged with it, its
    per-query columns prefixed ``{model}/`` and its section titles suffixed
    `` ({model})``.  What is out of the scope is kept from the run's files,
    so the generate stage scores the generation only and the belief stage
    adds its own model only.  A belief model reads the answer verdicts from
    ``eval/generation/`` (the evaluators that ``provide`` them run again on
    their saved verdicts, with no judge call, and add no row).  The
    generation is scored too when its saved rows cannot be kept
    (:func:`_saved_generation_stale`) or when a verdict is missing from
    ``eval/generation/`` (a judge failure of the generation's evaluation,
    judged again now): the verdicts the belief models are scored against
    are then always those of the generation's rows.  A belief model without
    any belief file to read keeps its saved rows.

    Args:
        results:    ``{query_id: result}`` (:func:`load_run_results`).
        evaluators: From :func:`build_evaluators`.
        run_dir:    Run directory; None skips every file (and the belief
                    models, whose files are read from it).
        num_unfinished: Queries of the split not finished yet (the
                    header's ``unfinished``); None: not recorded.
        generation: Score the generation's evaluators.
        models:     The belief models to score; None: every belief model
                    of the run (``utils.run_files.belief_models``).
    """
    ev = evaluators
    run_dir = Path(run_dir) if run_dir else None
    run_models = belief_models(run_dir) if run_dir is not None else []
    models = run_models if models is None else [m for m in run_models if m in models]
    ctx = EvalContext(results=results, run_dir=run_dir, qrels=ev.setup.qrels or {})
    if not generation:
        stale = _saved_generation_stale(run_dir, results)
        if not stale and models:
            pending = sum(e.pending(ctx) for e in ev.generation if e.provides)
            if pending:
                stale = f"{pending} verdict(s) missing from {GENERATION_EVAL_DIR}/ (judge failures)"
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

    def collect(name: str, fn, ctx: EvalContext, model: str = NO_MODEL) -> GroupResult:
        res = timed(name if model == NO_MODEL else f"{name} [{model}]", fn, ctx)
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

    for e in ev.generation:
        if generation:
            res = collect(e.name, e.evaluate, ctx)
        elif e.provides and models:
            # y only (the verdicts saved by the generation's evaluation, none
            # pending, so no judge call): no rows.
            res = timed(f"{e.name} (saved verdicts)", e.evaluate, ctx)
        else:
            continue
        if e.provides:
            setattr(ctx, e.provides, res.raw or {})

    # Per belief model: each belief file is read once, for every evaluator below.
    without_files: List[str] = []
    for model in models:
        belief_files = timed(f"belief files [{model}]", load_belief_files, run_dir, model, list(results))
        if not belief_files:
            print(f"  {model}: no belief file to evaluate; its saved rows are kept.")
            without_files.append(model)
            continue
        rows.setdefault(model, [])
        model_ctx = replace(ctx, model=model, belief_files=belief_files)
        for e in ev.belief_model:
            collect(e.name, e.evaluate, model_ctx, model=model)
    models = [m for m in models if m not in without_files]

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
    print("\n".join(log_lines(title, all_rows, GENERATION_LOG_LINES, MODEL_LOG_LINES)))
    print(RULE)
    print("  Evaluation time: " + ", ".join(f"{k} {v:.1f}s" for k, v in timings.items()))
    if run_dir is None:
        return
    (run_dir / EVAL_DIR).mkdir(parents=True, exist_ok=True)
    write_metrics(run_dir / METRICS_FILE, header, all_rows, all_sections)
    write_per_query(run_dir / PER_QUERY_FILE, merged)
    print(f"  Saved {run_dir / EVAL_DIR}/: metrics.md, per_query.csv")


def _saved_generation_stale(run_dir: Optional[Path], results: Dict[str, Any]) -> Optional[str]:
    """Why the generation's rows of the run's ``metrics.md`` cannot be kept
    (an interrupted generate stage left them behind its saved queries, the
    run's queries changed, or the file is of another layout), None when
    they can."""
    path = run_dir / METRICS_FILE if run_dir is not None else None
    if path is None or not path.exists():
        return "No eval/metrics.md yet"
    if not any(r.belief_model == NO_MODEL for r in read_metrics(path)):
        return "No generation row in eval/metrics.md"
    saved = read_header(path).get("queries")
    if saved != str(len(results)):
        return f"eval/metrics.md scored {saved} queries, the run has {len(results)}"
    per_query_path = run_dir / PER_QUERY_FILE
    if not per_query_path.exists():
        return "No eval/per_query.csv yet"
    missing = set(results) - set(read_per_query(per_query_path))
    if missing:
        return f"{len(missing)} of the run's queries are not in eval/per_query.csv"
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
