"""Table evaluator: persist and score the grid an agent builds.

Two jobs, mirroring every other evaluator in this package:

``save_item``
    Write one query's grid snapshots to ``tables/{qid}.jsonl`` -- a ``meta``
    header followed by one line per stage, the same convention
    ``ControllerEvaluator`` and ``TrajectoryEvaluator`` use.  The table has a
    history (``init`` -> each sweep -> each barrier -> ``final``) and only the
    sequence separates a nugget that was never enumerated from one a revision
    dropped later, so the stages are kept rather than just the end state.

``evaluate``
    Score those grids.  Two halves, with different requirements:

    *Against the dataset's gold intermediate information* -- did the planner
    enumerate the right nuggets (entity recall / precision / F1, each pooled
    macro and micro), name the right
    property (column recall), and did the fillers then establish the right
    values (:mod:`~evaluation.table.values`)?  Scored at ``init``, which
    isolates what the planner alone produced, at ``final``, which shows what the
    revision barriers did to it, and at every stage between; each barrier is
    scored in its own right by :mod:`~evaluation.table.stage_diff`.

    *Against the run's own qrels* -- per-phase retrieval and per-cell grounding
    (:mod:`~evaluation.table.phases`).  These need no gold table, so they are
    produced for every grid, including on datasets that ship none.

The gate is the shape of the result (``table_snapshots`` / ``table``), so an
agent that grows a grid later gets the same persistence and the same metrics for
free, and an agent without one writes nothing.

The one thing the agent's identity settles is *vocabulary*: what it calls the
phases of its own schedule and the stages of its own grid.  Those names are what
the per-phase and per-stage blocks group on, so they cannot be inferred from the
result's shape -- a phase tag is just a string.  :mod:`.vocabulary` holds one
entry per agent and an unrecognised agent falls back to GridFill's, which is
what every agent used before that module existed.  No scoring depends on the
agent; only the grouping does.

Per-query JSONL schema (``tables/{qid}.jsonl``)::

    line 1  {"record": "meta", "qid": "1_Q35657_...", "question": "...",
             "stages": ["init", "fill_round_0", "modify_round_1", "final"],
             "num_snapshots": 4}

    line 2  {"record": "snapshot", "stage": "init",
             "coverage": {"resolved": 0, "total": 100},
             "question": "...",
             "columns": [{"col_id": "c1", "name": "...", "spec": "..."}],
             "rows": [{"row_id": "r1", "entity": "Michigan", "cells": {...}}]}

    line 3  {"record": "snapshot", "stage": "fill_round_0", ...}
    ...

    line N  {"record": "phases",
             "phases": {"init": {"searches": 6, "surfaced": [[docid, ...]],
                                 "seen": [[docid, ...]]}, "fill": {...}},
             "cells":  [{"row_id": "r1", "col_id": "c1", "status": "filled",
                         "value": 10077331, "doc_ids": [...]}, ...]}

The ``phases`` line is what ``--eval-only`` reloads: the shared trajectory keeps
neither the ``phase`` tags nor the per-step document lists, and both are too
large for its meta line.
"""

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .gold import GoldTable, load_gold_tables
from .matching import match_column, match_rows, normalize_entity
from .phases import (
    build_phase_record,
    cumulative_phases,
    score_cell_grounding,
    score_phase_retrieval,
)
from .stage_diff import aggregate_barriers, barrier_pairs, score_barrier
from .values import score_cell_values
from .vocabulary import DEFAULT_VOCAB, TableVocabulary, vocabulary_for

logger = logging.getLogger(__name__)


def _snapshots(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return a result's grid snapshots, oldest first.

    Falls back to synthesising a one-stage history from ``table_init`` /
    ``table`` so a run saved before snapshots existed -- or reloaded from a
    trajectory meta line, which carries those two keys but not the full
    sequence -- still evaluates.
    """
    snapshots = result.get("table_snapshots")
    if snapshots:
        return list(snapshots)

    rebuilt: List[Dict[str, Any]] = []
    for stage, key in (("init", "table_init"), ("final", "table")):
        table = result.get(key)
        if table and table.get("rows"):
            rebuilt.append({"stage": stage, **table})
    return rebuilt


def _stage(snapshots: Sequence[Dict[str, Any]], stage: str) -> Optional[Dict[str, Any]]:
    """Return the named stage's snapshot.

    ``final`` falls back to the last snapshot taken: a run that died partway
    -- an exception, a killed worker -- never reaches the stage literally named
    "final", and its last grid is the final one it had.
    """
    for snap in snapshots:
        if snap.get("stage") == stage:
            return snap
    if stage == "final" and snapshots:
        return snapshots[-1]
    return None


def _stage_order(stage: str, vocab: TableVocabulary = DEFAULT_VOCAB) -> tuple:
    """Sort key putting stages in the order *vocab*'s schedule ran them.

    For GridFill that is ``init``, then ``fill_round_N`` / ``modify_round_{N+1}``
    alternating, then ``final`` -- so a barrier sorts immediately after the
    sweep it followed, not after the sweep sharing its number.  Anything
    unrecognised sorts last, alphabetically, rather than crashing the report.
    """
    return vocab.order_stage(stage)


def _align_stages(
    per_query_stages: Dict[str, Dict[str, Dict[str, Any]]],
    vocab: TableVocabulary = DEFAULT_VOCAB,
) -> Dict[str, List[Dict[str, Any]]]:
    """Carry each query's last grid forward across the stages it never reached.

    Without this a per-stage series compares populations rather than stages.
    Queries stop at different rounds -- ``max_modify_rounds`` is an upper bound,
    a barrier that changes nothing ends the schedule early, a run can die
    partway -- so ``modify_round_3`` would be averaged over only the runs that
    got that far, which are not a random subset of the runs that reached
    ``init``.  The stage series would then improve simply by losing its hard
    queries.

    A query that stopped kept the grid it had, so its score at every later stage
    *is* the score of its last snapshot.  Filling that in puts every stage over
    the same denominator, which is what makes the series readable left to right.

    Args:
        per_query_stages: ``{qid: {stage: scored}}``.

    Returns:
        ``{stage: [scored, ...]}`` over the union of stages, in schedule order,
        each list holding one entry per query that had reached its first stage.
    """
    # Schedule order matters twice over: it is the order the series is read in,
    # and it is the direction the carry-forward below runs.  A stage sorted into
    # the wrong slot would inherit a *later* grid's score.
    stages = sorted(
        {stage for scores in per_query_stages.values() for stage in scores},
        key=lambda s: _stage_order(s, vocab),
    )
    aligned: Dict[str, List[Dict[str, Any]]] = {stage: [] for stage in stages}
    for scores in per_query_stages.values():
        carried: Optional[Dict[str, Any]] = None
        for stage in stages:
            carried = scores.get(stage, carried)
            # Stages before this query's first one get nothing: there is no
            # earlier grid to carry, and inventing one would be a fabrication
            # rather than a carry-forward.
            if carried is not None:
                aligned[stage].append(carried)
    return {stage: scores for stage, scores in aligned.items() if scores}


def _is_revision_stage(stage: str, vocab: TableVocabulary = DEFAULT_VOCAB) -> bool:
    """True for a revision barrier's output -- ``modify_round_*`` under GridFill.

    Always False for an agent with no revision barrier, which is what keeps the
    ``revision`` and ``barriers`` blocks absent for it rather than empty.
    """
    return vocab.is_revision_stage(stage)


def _score_revision(
    per_query_stages: Dict[str, Dict[str, Dict[str, Any]]],
    vocab: TableVocabulary = DEFAULT_VOCAB,
) -> Dict[str, Any]:
    """The query-level read on what revision did: ``init`` -> ``final``.

    Two numbers per query on the same absolute scale -- entity recall of the row
    set ``init_grid`` produced, and entity recall of the row set the query ended
    with after however many revision rounds it ran, zero included.  A query that
    never revised has the second equal to the first and stays in both
    denominators, so the two are over the same queries by construction rather
    than by a special case.  That equality is what makes the gap between them
    meaningful: the mean is linear, so a macro gap is the gap of the macros only
    when both sides are averaged over the same population.

    The gap is attributable to ``modify_grid`` alone.  The row set is written in
    exactly three places -- ``_apply_init_grid``, ``_apply_modify_grid`` and
    ``_fallback_table`` -- and a fill sweep cannot touch it (``_run_fill_cell_step``
    rejects ``<grid>`` and ``<modify_grid>`` outright, ``_apply_fill`` writes
    cells only).  So no fill contamination has to be separated out here, which
    is why this needs no barrier-anchored counterpart to isolate it.

    Distinct from :func:`~evaluation.table.stage_diff.aggregate_barriers`, whose
    denominator is a barrier count: a query that ran no barrier contributes no
    row there and is absent from it entirely.  This block is per query.

    Returns ``{}`` when the run had no revision phase at all (``stop_after``
    truncation, ``max_modify_rounds`` of zero, or an agent like TaS that has no
    revision barrier in its schedule), where every query would trivially report a
    gap of zero.
    """
    pairs = [
        (scores["init"], scores["final"], any(_is_revision_stage(s, vocab) for s in scores))
        for scores in per_query_stages.values()
        if "init" in scores and "final" in scores
    ]
    if not pairs or not any(revised for _init, _final, revised in pairs):
        return {}

    init_scores = [init for init, _final, _revised in pairs]
    final_scores = [final for _init, final, _revised in pairs]
    init_macro = float(np.mean([s["entity_recall"] for s in init_scores]))
    final_macro = float(np.mean([s["entity_recall"] for s in final_scores]))
    total_gold = sum(s["n_gold_rows"] for s in init_scores)
    init_micro = _ratio(sum(s["n_matched"] for s in init_scores), total_gold)
    final_micro = _ratio(sum(s["n_matched"] for s in final_scores), total_gold)

    # Counted per query, not inferred from the gap: a mean gap of +0.02 is
    # equally consistent with every query nudging up and with a mix of +0.3 and
    # -0.3.  Barriers do drop gold rows (``gold_rows_dropped`` counts them), so
    # the negative tail is real and is reported rather than clipped away.
    deltas = [f["entity_recall"] - i["entity_recall"] for i, f, _r in pairs]
    return {
        "num_queries": len(pairs),
        "num_queries_revised": sum(1 for _i, _f, revised in pairs if revised),
        "entity_recall_init_macro": init_macro,
        "entity_recall_final_macro": final_macro,
        "entity_recall_init_micro": init_micro,
        "entity_recall_final_micro": final_micro,
        "entity_recall_gap_macro": final_macro - init_macro,
        "entity_recall_gap_micro": final_micro - init_micro,
        "num_improved": sum(1 for d in deltas if d > 0),
        "num_unchanged": sum(1 for d in deltas if d == 0),
        "num_degraded": sum(1 for d in deltas if d < 0),
    }


def _is_fallback(snapshot: Dict[str, Any]) -> bool:
    """True when this is ``_fallback_table``'s degenerate one-cell grid.

    The planner emitted nothing usable and the runner substituted a single row
    holding the question itself.  Such a run scores ~0 entity recall correctly, but
    it failed for a different reason than a badly enumerated grid, so it is
    counted separately rather than pooled in.
    """
    rows = snapshot.get("rows") or []
    if len(rows) != 1:
        return False
    return normalize_entity(rows[0].get("entity")) == normalize_entity(snapshot.get("question"))


def _score_stage(gold: GoldTable, snapshot: Dict[str, Any]) -> Dict[str, Any]:
    """Entity and column metrics for one grid against one gold table.

    The rates here are per query and carry no ``_macro`` / ``_micro`` suffix:
    that distinction is a property of how :func:`_aggregate` pools them, and one
    query admits only one weighting.  The integer counts (``n_matched``,
    ``n_gold_rows``, ``n_rows``) are returned so the micro pooling can sum them
    rather than recover them from the rates.
    """
    rows = snapshot.get("rows") or []
    columns = snapshot.get("columns") or []

    strict = match_rows(gold.entities, rows)
    lenient = match_rows(gold.entities, rows, lenient=True)
    col_id, col_score = match_column(gold.property_label, gold.property_description, columns)

    # Structure is only half the question: a grid can enumerate every gold
    # entity and still be wrong in every cell.  Scored against the same row
    # pairing and the same matched column, so the value numbers describe the
    # cells the structural numbers just counted.
    values = score_cell_values(gold, snapshot, col_id, strict)

    n_gold, n_pred = len(gold.entities), len(rows)
    hits = len(strict)
    recall = hits / n_gold if n_gold else 0.0
    precision = hits / n_pred if n_pred else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return {
        "entity_recall": recall,
        "entity_precision": precision,
        "entity_f1": f1,
        "entity_recall_lenient": len(lenient) / n_gold if n_gold else 0.0,
        "column_recall": 1.0 if col_id else 0.0,
        "column_score": col_score,
        "matched_col_id": col_id,
        "n_matched": hits,
        "n_matched_lenient": len(lenient),
        "n_gold_rows": n_gold,
        "n_rows": n_pred,
        "n_cols": len(columns),
        "is_fallback": _is_fallback(snapshot),
        **values,
    }


# Per-query rates macro-averaged into a stage's aggregate, each emitted with a
# ``_macro`` suffix beside the ``_micro`` pooling of the same quantity.
# ``matched_col_id`` and the counts describing one query only are excluded.
_MACRO_KEYS = ("entity_recall", "entity_precision", "entity_f1", "entity_recall_lenient")

# Counts reported as a per-query mean -- how big the grid was and how big the
# gold was -- rather than pooled into a rate.
_COUNT_KEYS = ("n_gold_rows", "n_rows", "n_cols")

# Cell-value keys.  Averaged over the queries where they are defined rather
# than over all of them: a stage with nothing filled yet (``init`` always, and
# any query whose gold column was never matched) reports no value metrics at
# all, and folding those in as zeros would read as "the values were wrong"
# when the truth is "there were no values to judge".
_VALUE_MEAN_KEYS = (
    "value_precision", "value_precision_exact", "value_item_recall", "value_recall",
)


def _ratio(numerator: float, denominator: float) -> float:
    """``numerator / denominator``, or 0.0 when there is nothing to divide by."""
    return float(numerator / denominator) if denominator else 0.0


def _aggregate(per_query: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Pool a stage's per-query scores both ways.

    Macro weights every question equally; micro weights every gold entity
    equally, so the two diverge exactly when performance depends on how many
    nuggets a question has -- which for TRQA (2 to 50 entities) it does.  Both
    weightings are reported for precision and F1 as well as for recall, not for
    symmetry but because the weighting bites hardest on precision: one spurious
    row costs half the precision of a two-entity question and a fiftieth of a
    fifty-entity one, and wiki2's COUNT questions are answered from the row
    count itself.

    The micro numbers are summed from the integer counts ``_score_stage``
    returns rather than recovered from the per-query rates.
    """
    if not per_query:
        return {}

    out: Dict[str, Any] = {"num_queries": len(per_query)}
    for key in _MACRO_KEYS:
        out[f"{key}_macro"] = float(np.mean([q[key] for q in per_query]))
    for key in _COUNT_KEYS:
        out[key] = float(np.mean([q[key] for q in per_query]))
    # One gold column per query, so pooling it the other way would give the same
    # number.  Reported bare rather than with a ``_macro`` suffix that would
    # imply a counterpart that does not exist.
    out["column_recall"] = float(np.mean([q["column_recall"] for q in per_query]))

    total_gold = sum(q["n_gold_rows"] for q in per_query)
    total_rows = sum(q["n_rows"] for q in per_query)
    total_hits = sum(q["n_matched"] for q in per_query)
    recall_micro = _ratio(total_hits, total_gold)
    precision_micro = _ratio(total_hits, total_rows)
    out["entity_recall_micro"] = recall_micro
    out["entity_precision_micro"] = precision_micro
    out["entity_f1_micro"] = _ratio(
        2 * precision_micro * recall_micro, precision_micro + recall_micro,
    )
    out["entity_recall_lenient_micro"] = _ratio(
        sum(q["n_matched_lenient"] for q in per_query), total_gold,
    )
    out["n_matched"] = total_hits
    out["n_gold_total"] = total_gold
    out["n_rows_total"] = total_rows
    out["n_fallback"] = sum(1 for q in per_query if q["is_fallback"])

    with_values = [q for q in per_query if q.get("n_cells_scored")]
    if with_values:
        out["num_queries_with_values"] = len(with_values)
        for key in _VALUE_MEAN_KEYS:
            out[key] = float(np.mean([q[key] for q in with_values]))
        out["n_cells_scored"] = sum(q["n_cells_scored"] for q in with_values)
        out["n_not_comparable"] = sum(q.get("n_not_comparable", 0) for q in with_values)
        # The one number here whose denominator is every query rather than the
        # ones with values: the macro rates above answer "when the run committed
        # to a value, how good was it", and pooling the queries that established
        # nothing into those would answer a different question badly.  This one
        # is the end-to-end read -- correct cells over every gold entity the run
        # was asked about -- so a query that established nothing belongs in it,
        # contributing its entities to the denominator and no correct cells.
        total_correct = sum(q["n_correct"] for q in with_values)
        out["value_recall_micro"] = float(total_correct / total_gold) if total_gold else 0.0
    return out


class TableEvaluator:
    """Persist and score per-query grids.

    Usage::

        evaluator = TableEvaluator(gold_tables=gold)

        # Per-query save (inside the query loop)
        evaluator.save_item(query_id, query_text, result, tables_dir)

        # Aggregate evaluation (after all queries)
        metrics = evaluator.evaluate(results)
        evaluator.print_results(metrics)

    Args:
        gold_tables: ``{qid: intermediate-info record}`` from
                     ``indexing_corpus_dataset.load_intermediate_info``.  Empty
                     (the default) leaves the evaluator persistence-only, which
                     is what every dataset without gold tables gets.
        qrels:       The run's relevance judgements, used only for the
                     per-phase retrieval and cell-grounding blocks.  These are
                     the same qrels the shared retrieval evaluators score
                     against, so a phase's recall is a slice of the run's
                     recall rather than a differently-defined number.  Empty
                     leaves those blocks out.
        k_values:    Cut-offs for the per-phase retrieval metrics; pass the
                     run's own so the numbers line up with the shared ones.
        fusion_method / interleaving_window / rrf_k:
                     How a phase's per-step ranked lists are consolidated.
                     Mirrors the shared evaluators' configuration for the same
                     reason.
    """

    def __init__(
        self,
        gold_tables: Optional[Dict[str, Dict[str, Any]]] = None,
        qrels: Optional[Dict[str, Dict[str, int]]] = None,
        k_values: Optional[Sequence[int]] = None,
        fusion_method: str = "interleaving",
        interleaving_window: Optional[int] = 3,
        rrf_k: int = 60,
        agentic_model: Optional[str] = None,
    ) -> None:
        self.gold: Dict[str, GoldTable] = load_gold_tables(gold_tables or {})
        self.qrels = qrels or {}
        self.k_values = list(k_values) if k_values else [1, 3, 5, 10, 25, 100]
        self.fusion_method = fusion_method
        self.interleaving_window = interleaving_window
        self.rrf_k = rrf_k
        # The one thing this evaluator needs the agent's *identity* for: what it
        # calls its own phases and stages.  Everything else still gates on the
        # shape of the result.  An unknown agent gets the default vocabulary,
        # which is what every agent got before this existed.
        self.vocab: TableVocabulary = vocabulary_for(agentic_model)

    # ------------------------------------------------------------------
    # Per-query persistence
    # ------------------------------------------------------------------

    def save_item(
        self,
        query_id: str,
        question: str,
        result: Dict[str, Any],
        output_dir,
    ) -> None:
        """Save one query's grid snapshots as a JSONL file.

        A no-op when the result carries no grid, which is how every non-grid
        agent skips this evaluator without either side naming the other.

        Args:
            query_id:   Query identifier.
            question:   Original query text.
            result:     Unified agent result dict for this query.
            output_dir: Directory where ``{query_id}.jsonl`` will be written.
        """
        snapshots = _snapshots(result)
        if not snapshots:
            return

        output_dir_str = str(output_dir).rstrip("/")
        Path(output_dir_str).mkdir(parents=True, exist_ok=True)

        def _dump(obj: Dict[str, Any]) -> str:
            return json.dumps(obj, separators=(",", ":"), default=str)

        meta = {
            "record": "meta",
            "qid": query_id,
            "question": question,
            "stages": [s.get("stage", "") for s in snapshots],
            "num_snapshots": len(snapshots),
        }

        # The per-phase view.  Built here, from the live result, because this is
        # the only moment the full per-step document lists and the ``phase``
        # tags exist together -- the shared trajectory JSONL keeps neither, and
        # is deliberately left alone.
        phase_record = build_phase_record(result, self.vocab)

        try:
            with open(f"{output_dir_str}/{query_id}.jsonl", "w") as f:
                f.write(_dump(meta) + "\n")
                for snap in snapshots:
                    f.write(_dump({"record": "snapshot", **snap}) + "\n")
                if phase_record:
                    f.write(_dump(phase_record) + "\n")
        except OSError as e:
            logger.warning("Could not write tables/%s.jsonl: %s", query_id, e)

    # ------------------------------------------------------------------
    # Sidecar reload (--eval-only)
    # ------------------------------------------------------------------

    @staticmethod
    def read_sidecar(tables_dir, query_id: str) -> Dict[str, Any]:
        """Read one query's ``tables/{qid}.jsonl`` back.

        Needed because a reloaded result carries only what the trajectory meta
        line holds -- the final grid and the init grid, not the full snapshot
        sequence and not the phase view, both of which are too large for that
        line.  Reading GridFill's own sidecar is what keeps ``--eval-only``
        equivalent to a live run without any shared file having to carry
        GridFill-specific fields.

        Returns ``{"table_snapshots": [...], "phases": {...}, "cells": [...]}``,
        or ``{}`` when the file is missing or unreadable.
        """
        path = Path(str(tables_dir)) / f"{query_id}.jsonl"
        if not path.exists():
            return {}
        snapshots: List[Dict[str, Any]] = []
        out: Dict[str, Any] = {}
        try:
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    record = obj.pop("record", "")
                    if record == "snapshot":
                        snapshots.append(obj)
                    elif record == "phases":
                        out["phases"] = obj.get("phases") or {}
                        out["cells"] = obj.get("cells") or []
        except (OSError, json.JSONDecodeError) as e:
            logger.warning("Could not read tables/%s.jsonl: %s", query_id, e)
            return {}
        if snapshots:
            out["table_snapshots"] = snapshots
        return out

    # ------------------------------------------------------------------
    # Aggregate evaluation
    # ------------------------------------------------------------------

    def evaluate(self, results: Dict[str, Dict[str, Any]], tables_dir=None) -> Dict[str, Any]:
        """Score every result that carries a grid against its gold table.

        A result carrying no grid is skipped entirely -- that is how every
        non-grid agent passes through.  A grid with no gold table still gets the
        qrels-only blocks (per-phase retrieval, cell grounding) and is left out
        of the structural ones, rather than counted as a zero that would depress
        them.  So ``num_grids`` is normally below ``num_queries`` and
        ``num_evaluated`` below ``num_grids``; both are expected, and are why
        this evaluator is left out of the runner's query-count guard.

        Args:
            results:    Unified results dict keyed by query id.
            tables_dir: The run's ``tables/`` directory.  When given, each
                        query's sidecar is read and merged over its result,
                        which is what makes ``--eval-only`` see the full
                        snapshot sequence and the phase view; a live run
                        already has both in the result and does not need it.

        Returns:
            ``{}`` when no result carried a grid.  Otherwise ``num_queries``,
            ``num_grids``, ``num_evaluated``, plus whichever blocks are
            defined.

            Qrels only: ``retrieval_by_phase`` (each phase's own searches),
            ``retrieval_cumulative`` (running prefixes of the schedule, whose
            ``after_modify`` row is every search the run made) and
            ``cell_grounding``.

            Where there is gold: an ``init`` and a ``final`` block, ``revision``
            (what the revision phase did, per query -- see
            :func:`_score_revision`), ``by_stage``
            (every stage the run passed through, each over the same queries --
            see :func:`_align_stages`), ``by_stage_reached`` (the unaligned
            view, present only when some query stopped early), ``barriers``,
            and ``by_datatype`` / ``by_aggregation`` breakdowns of the init
            stage.
        """
        if not results:
            return {}

        per_stage: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        per_query_stages: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)
        by_datatype: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        by_aggregation: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        barriers: List[Dict[str, Any]] = []
        phase_docs: Dict[str, Dict[str, Any]] = {}
        phase_cells: Dict[str, List[Dict[str, Any]]] = {}
        num_grids = 0
        num_evaluated = 0

        for query_id, result in results.items():
            if tables_dir is not None:
                sidecar = self.read_sidecar(tables_dir, query_id)
                if sidecar:
                    result = {**result, **sidecar}
            snapshots = _snapshots(result)
            if not snapshots:
                continue

            num_grids += 1

            # A live run carries the raw trajectory and derives the phase view
            # here; a reload carries it already built, merged in from the
            # sidecar above.  Both paths end in the same two structures, so the
            # numbers cannot drift between a live run and --eval-only.  This
            # half needs only qrels, so it runs for every grid -- including the
            # queries, and the datasets, that have no gold table.
            if "phases" not in result:
                derived = build_phase_record(result, self.vocab) or {}
                result = {**result, **derived}
            if result.get("phases"):
                phase_docs[query_id] = result["phases"]
            if result.get("cells"):
                phase_cells[query_id] = result["cells"]

            gold = self.gold.get(query_id)
            if gold is None:
                continue

            num_evaluated += 1

            # Every stage the run actually passed through, plus the two
            # headline ones.  ``final`` may alias the last snapshot (see
            # ``_stage``), so it is scored by name rather than by position.
            seen_stages = {str(snap.get("stage") or "") for snap in snapshots}
            for stage in sorted(seen_stages | set(self.vocab.scored_stages)):
                snapshot = _stage(snapshots, stage)
                if snapshot is None:
                    continue
                scored = _score_stage(gold, snapshot)
                per_stage[stage].append(scored)
                per_query_stages[query_id][stage] = scored
                # The breakdowns are over the planner's own output: what it can
                # enumerate depends on the property's datatype and on what the
                # aggregation demands of the row set.
                if stage == "init":
                    by_datatype[gold.datatype or "unknown"].append(scored)
                    by_aggregation[gold.aggregation or "unknown"].append(scored)

            barriers.extend(
                score_barrier(gold, before, after)
                for _stage_name, before, after in barrier_pairs(snapshots, self.vocab)
            )

        if not num_grids:
            return {}

        metrics: Dict[str, Any] = {
            "num_queries": len(results),
            "num_grids": num_grids,
            "num_evaluated": num_evaluated,
        }
        for stage in self.vocab.scored_stages:
            aggregated = _aggregate(per_stage.get(stage, []))
            if aggregated:
                metrics[stage] = aggregated

        # What the revision phase did, per query rather than per barrier: the
        # headline stages above are the two numbers it compares, so it sits
        # with them rather than with the barrier diagnostic.
        revision = _score_revision(per_query_stages, self.vocab)
        if revision:
            metrics["revision"] = revision

        # Every stage in schedule order -- init, each sweep, each barrier,
        # final.  This is what shows *where* a nugget was lost, which the two
        # headline stages can only bracket.  Aligned by ``_align_stages`` so
        # every stage is averaged over the same queries.
        by_stage = {
            stage: _aggregate(scores)
            for stage, scores in _align_stages(per_query_stages, self.vocab).items()
        }
        if by_stage:
            metrics["by_stage"] = by_stage

        # The unaligned view: each stage over only the queries that actually
        # reached it.  Kept because it is the one that says how many did, and
        # emitted only when that differs from the aligned view -- when every
        # query ran the full schedule the two blocks are identical and the
        # second is noise.
        reached = {
            stage: _aggregate(scores)
            for stage, scores in sorted(per_stage.items(), key=lambda kv: _stage_order(kv[0], self.vocab))
            if scores
        }
        if any(m["num_queries"] != by_stage.get(stage, {}).get("num_queries")
               for stage, m in reached.items()):
            metrics["by_stage_reached"] = reached

        barrier_metrics = aggregate_barriers(barriers)
        if barrier_metrics:
            metrics["barriers"] = barrier_metrics

        if self.qrels:
            phase_retrieval = {
                level: score_phase_retrieval(
                    phase_docs, self.qrels, self.k_values, level=level,
                    fusion_method=self.fusion_method, rrf_k=self.rrf_k,
                    interleaving_window=self.interleaving_window,
                    phase_order=self.vocab.retrieval_phases,
                )
                for level in ("surfaced", "seen")
            }
            phase_retrieval = {k: v for k, v in phase_retrieval.items() if v}
            if phase_retrieval:
                metrics["retrieval_by_phase"] = phase_retrieval

            # The same documents pooled as running prefixes rather than
            # disjoint slices.  ``after_modify`` is every search the run made,
            # at both levels: the agent's recall is not phase-scoped, because
            # the evidence store is not -- a fill pass may cite what ``init``
            # retrieved.  So this is the block to read for "what did the run
            # see", and ``retrieval_by_phase`` the one for "which phase earned
            # it".
            cumulative = {
                level: score_phase_retrieval(
                    cumulative_phases(phase_docs, self.vocab), self.qrels, self.k_values,
                    level=level, fusion_method=self.fusion_method, rrf_k=self.rrf_k,
                    interleaving_window=self.interleaving_window,
                    phase_order=self.vocab.cumulative_names,
                )
                for level in ("surfaced", "seen")
            }
            cumulative = {k: v for k, v in cumulative.items() if v}
            if cumulative:
                metrics["retrieval_cumulative"] = cumulative
            # Only for an agent that actually cites.  Where none is
            # recorded every filled cell is trivially uncited, and reporting
            # ``grounded_rate: 0.0`` would assert that the agent cited badly
            # rather than that it never cited at all.
            if self.vocab.records_evidence:
                grounding = score_cell_grounding(phase_cells, self.qrels)
                if grounding:
                    metrics["cell_grounding"] = grounding

        for key, breakdown in (("by_datatype", by_datatype), ("by_aggregation", by_aggregation)):
            if breakdown:
                metrics[key] = {k: _aggregate(v) for k, v in sorted(breakdown.items())}
        return metrics

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    #: Column header for the per-stage entity block, kept next to the row
    #: formatter so the two cannot drift apart.
    _STAGE_HEADER = (f"  {'stage':<16} {'entity R':^13}  {'entity P':^13}  "
                     f"{'col R':>6}  {'rows':^11} {'n':>3}   (macro / micro)")

    @staticmethod
    def _stage_line(name: str, m: Dict[str, Any]) -> str:
        """One stage's entity metrics, macro and micro side by side.

        Both weightings are printed for recall and precision rather than a
        chosen one: which of the two is flattering depends on whether the run
        does better on the small questions or the large ones, and printing only
        one invites reading it as the number.
        """
        line = (
            f"  {name:<16} "
            f"{m['entity_recall_macro']:.3f} / {m['entity_recall_micro']:.3f}  "
            f"{m['entity_precision_macro']:.3f} / {m['entity_precision_micro']:.3f}  "
            f"{m['column_recall']:6.3f}  "
            f"{m['n_rows']:5.1f}/{m['n_gold_rows']:<5.1f} {m['num_queries']:>3}"
        )
        if m.get("n_cells_scored"):
            line += (f"  |  val P {m['value_precision']:.3f}  "
                     f"R {m.get('value_recall_micro', 0.0):.3f}  "
                     f"({m['n_cells_scored']} cells)")
        return line

    def print_results(self, metrics: Dict[str, Any]) -> None:
        """Print the entity-recall view of the table metrics.

        Deliberately narrower than what :meth:`evaluate` computes: the console
        block is the stage series and what revision did to it, and nothing else.
        The barrier diagnostic, cell grounding, the per-phase and cumulative
        retrieval blocks and the datatype / aggregation breakdowns are still
        computed and still written to ``summary.json`` by :meth:`save_results`
        -- they are read there, not here, so a run's console output stays short
        enough to take in at a glance.
        """
        if not metrics:
            return

        print("\n" + "=" * 80)
        print("TABLE (grid vs. gold intermediate info)")
        print("=" * 80)
        print(f"Grids found: {metrics.get('num_grids', 0)}/{metrics['num_queries']}"
              f"  |  scored against gold: {metrics['num_evaluated']}")
        if not metrics["num_evaluated"]:
            # No gold table for this dataset: the qrels-only blocks below are
            # still real, and saying so beats printing an empty structural
            # section that reads as a failure.
            print("  (no gold intermediate info for these queries: "
                  "entity/column/value metrics are unavailable)")

        headline = [(s, metrics[s]) for s in self.vocab.scored_stages if metrics.get(s)]
        if headline:
            print(f"\n{self._STAGE_HEADER}")
            for stage, m in headline:
                label = self.vocab.stage_labels.get(stage, stage)
                print(self._stage_line(label, m))
                if m.get("n_fallback"):
                    print(f"         ({m['n_fallback']} fallback grid(s): "
                          f"init_grid emitted nothing usable)")

        # The whole schedule, not just its endpoints: this is what shows *where*
        # a nugget was lost, which init and final can only bracket.
        by_stage = metrics.get("by_stage") or {}
        if len(by_stage) > 2:
            print("\n  Per stage (the schedule in order, every stage over the "
                  "same queries):")
            print(self._STAGE_HEADER)
            for name, m in by_stage.items():
                print(self._stage_line(name, m))
            if metrics.get("by_stage_reached"):
                # Only present when it differs, i.e. when some queries stopped
                # early.  Saying so is the point: the aligned series above
                # carries their last grid forward, and this is how many runs
                # each stage actually happened in.
                counts = "  ".join(
                    f"{name}:{m['num_queries']}"
                    for name, m in metrics["by_stage_reached"].items()
                )
                print(f"    (queries that actually reached each stage -- "
                      f"later stages are carried forward above)\n      {counts}")

        revision = metrics.get("revision")
        if revision:
            # The query-level read on the revision phase, printed next to the
            # stages it compares.  modify_grid is the only thing that rewrites
            # the row set, so this gap is entirely its doing.
            gap_macro, gap_micro = (revision["entity_recall_gap_macro"],
                                    revision["entity_recall_gap_micro"])
            print("\n  Revision effect (init -> final row set, per query):")
            print(f"    entity recall  init "
                  f"{revision['entity_recall_init_macro']:.3f} / "
                  f"{revision['entity_recall_init_micro']:.3f}"
                  f"  ->  final {revision['entity_recall_final_macro']:.3f} / "
                  f"{revision['entity_recall_final_micro']:.3f}   (macro / micro)")
            print(f"    gap            {gap_macro:+.3f} / {gap_micro:+.3f}")
            print(f"    queries        {revision['num_queries']}"
                  f"  ({revision['num_queries_revised']} ran a revision round,"
                  f" {revision['num_improved']} improved,"
                  f" {revision['num_unchanged']} unchanged,"
                  f" {revision['num_degraded']} degraded)")

        print()

    def save_results(self, metrics: Dict[str, Any], output_path) -> None:
        """Write the metrics dict to *output_path* as JSON."""
        if not metrics:
            return
        path = Path(str(output_path))
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(metrics, f, indent=2, default=str)
