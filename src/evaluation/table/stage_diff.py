"""What each revision barrier did to the grid, scored against the gold rows.

``TableEvaluator`` scores the grid at ``init`` and at ``final``, which answers
"was the nugget never enumerated, or dropped later?" but not *which* barrier
dropped it, or whether a barrier that added rows added the right ones.

The barriers leave no explicit record of what they changed --
``_apply_modify_grid`` logs counts, not names.  It does not need to: rows carry
monotonically increasing ``row_id``s, so diffing consecutive snapshots by id
recovers the exact add and drop sets after the fact.  That keeps the agent free
of evaluation-only bookkeeping and means these metrics work on runs already on
disk.

Scored per barrier:

entity_recall_before / entity_recall_after
    Plain entity recall on either side of the barrier -- matched gold entities
    over all gold entities -- in the same units as every other stage, so the
    barrier's effect can be read as a delta rather than in a scale of its own.
recovery_rate
    Of the gold entities the grid was still missing *before* the barrier, the
    fraction it recovered.  This is the metric the barrier exists to move, and
    it is deliberately *not* called a recall: its denominator is what was
    missing, not what was gold, so it reads 1.0 for a barrier that recovered
    the one entity left outstanding on a grid that had already found the rest,
    and equally for one that recovered all forty.  Read it beside
    ``entity_recall_after``, which is the number on the absolute scale.
add_precision
    Of the rows the barrier added, the fraction that match a gold entity.
    Low precision with high recall means the barrier is padding the grid.
gold_rows_dropped
    Gold entities that were matched before the barrier and are not after.
    Every one is an answer the run can no longer reach.
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple

from .gold import GoldTable
from .matching import match_rows
from .vocabulary import DEFAULT_VOCAB, TableVocabulary


def _rows(snapshot: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return list((snapshot or {}).get("rows") or [])


def diff_stages(
    prev: Optional[Dict[str, Any]], curr: Optional[Dict[str, Any]],
) -> Dict[str, List[Dict[str, Any]]]:
    """Rows added and dropped between two consecutive snapshots, by ``row_id``.

    A row whose id is in *curr* but not *prev* was added; the reverse was
    dropped.  Ids are never reused within a query -- ``_new_row`` only ever
    increments -- so an add and a drop can never be confused for each other.
    """
    prev_rows, curr_rows = _rows(prev), _rows(curr)
    prev_ids = {r.get("row_id") for r in prev_rows}
    curr_ids = {r.get("row_id") for r in curr_rows}
    return {
        "added": [r for r in curr_rows if r.get("row_id") not in prev_ids],
        "dropped": [r for r in prev_rows if r.get("row_id") not in curr_ids],
    }


def _matched_gold(gold: GoldTable, rows: Sequence[Dict[str, Any]]) -> Dict[int, int]:
    return match_rows(gold.entities, rows)


def score_barrier(
    gold: GoldTable,
    prev: Optional[Dict[str, Any]],
    curr: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Score one barrier: what it recovered, what it padded, what it destroyed."""
    prev_rows, curr_rows = _rows(prev), _rows(curr)
    delta = diff_stages(prev, curr)

    before = _matched_gold(gold, prev_rows)
    after = _matched_gold(gold, curr_rows)

    missing_before = set(range(len(gold.entities))) - set(before)
    recovered = missing_before & set(after)
    lost = set(before) - set(after)

    # Which of the added rows landed on a gold entity.  ``after`` maps gold
    # index -> row index in curr_rows, so invert it and look the added rows up
    # by position rather than by name (two rows may normalise alike).
    added_ids = {r.get("row_id") for r in delta["added"]}
    matched_row_idx = set(after.values())
    added_hits = sum(
        1 for idx, row in enumerate(curr_rows)
        if row.get("row_id") in added_ids and idx in matched_row_idx
    )
    n_added = len(delta["added"])

    n_gold = len(gold.entities)
    return {
        "n_added": n_added,
        "n_dropped": len(delta["dropped"]),
        "n_missing_before": len(missing_before),
        "n_recovered": len(recovered),
        "n_gold": n_gold,
        "n_matched_before": len(before),
        "n_matched_after": len(after),
        # The absolute scale: the same quantity ``_score_stage`` reports for a
        # stage, so a barrier's before/after can be compared to init and final
        # directly.
        "entity_recall_before": (len(before) / n_gold) if n_gold else 0.0,
        "entity_recall_after": (len(after) / n_gold) if n_gold else 0.0,
        # Undefined when nothing was missing: a barrier with nothing left to
        # find cannot be scored on finding it, and counting it as 0.0 would
        # drag the mean down for doing its job.
        "recovery_rate": (len(recovered) / len(missing_before)) if missing_before else None,
        "add_precision": (added_hits / n_added) if n_added else None,
        "gold_rows_dropped": len(lost),
        "rows_before": len(prev_rows),
        "rows_after": len(curr_rows),
    }


def barrier_pairs(
    snapshots: Sequence[Dict[str, Any]],
    vocab: TableVocabulary = DEFAULT_VOCAB,
) -> List[Tuple[str, Dict[str, Any], Dict[str, Any]]]:
    """``(stage, before, after)`` for every revision barrier in a snapshot sequence.

    Under GridFill a barrier is a ``modify_round_*`` snapshot; the grid it acted
    on is whatever snapshot immediately precedes it (the sweep that just
    finished).  A leading barrier with nothing before it is skipped rather than
    diffed against an empty grid, which would score every row as an add.

    Only the *selector* is agent-specific.  :func:`score_barrier` and
    :func:`diff_stages` are vocabulary-free and score whatever pairs they are
    handed.  An agent whose schedule contains no revision barrier yields no
    pairs, so the ``barriers`` block is correctly absent for it -- the metrics
    here are defined against a barrier that revisited an already-populated grid,
    and a row-discovery step is not that however similar the diff looks.

    Args:
        snapshots: One query's snapshots, oldest first.
        vocab: Decides which stage names count as revision barriers.
    """
    pairs: List[Tuple[str, Dict[str, Any], Dict[str, Any]]] = []
    for i, snap in enumerate(snapshots):
        stage = str(snap.get("stage") or "")
        if vocab.is_revision_stage(stage) and i > 0:
            pairs.append((stage, snapshots[i - 1], snap))
    return pairs


def _mean(values: Sequence[Optional[float]]) -> Optional[float]:
    """Mean over the defined values only, or None when none are defined."""
    present = [v for v in values if v is not None]
    return (sum(present) / len(present)) if present else None


def aggregate_barriers(per_barrier: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Pool every barrier of every query into one block.

    Rates are pooled over the barriers where they are defined (macro), and each
    is *also* reported micro -- pooled over entities rather than over barriers
    -- because a barrier facing one missing entity and one facing forty should
    not weigh the same in a number about how much gets recovered.
    """
    if not per_barrier:
        return {}

    total_missing = sum(b["n_missing_before"] for b in per_barrier)
    total_recovered = sum(b["n_recovered"] for b in per_barrier)
    total_gold = sum(b["n_gold"] for b in per_barrier)

    return {
        "num_barriers": len(per_barrier),
        "entity_recall_before_macro": _mean([b["entity_recall_before"] for b in per_barrier]),
        "entity_recall_after_macro": _mean([b["entity_recall_after"] for b in per_barrier]),
        "entity_recall_before_micro": (
            sum(b["n_matched_before"] for b in per_barrier) / total_gold) if total_gold else None,
        "entity_recall_after_micro": (
            sum(b["n_matched_after"] for b in per_barrier) / total_gold) if total_gold else None,
        "recovery_rate_macro": _mean([b["recovery_rate"] for b in per_barrier]),
        "recovery_rate_micro": (total_recovered / total_missing) if total_missing else None,
        "add_precision": _mean([b["add_precision"] for b in per_barrier]),
        "rows_added": float(sum(b["n_added"] for b in per_barrier)) / len(per_barrier),
        "rows_dropped": float(sum(b["n_dropped"] for b in per_barrier)) / len(per_barrier),
        "gold_rows_dropped": sum(b["gold_rows_dropped"] for b in per_barrier),
        "num_barriers_that_changed_nothing": sum(
            1 for b in per_barrier if not b["n_added"] and not b["n_dropped"]
        ),
    }
