"""Compare a filled cell's value against the gold value for its entity.

Entity and column structure tells you whether ``init_grid`` enumerated the right
things.  It says nothing about whether ``fill_cell`` then established the right
*value* -- a grid can have perfect rows and be wrong in every cell.  This module
supplies that missing half.

It was deliberately left out when the table evaluator was first written,
because "did it get the value right" is not one question: TRQA's gold is
typed, and each type is right in a different way.  So each datatype gets its
own comparator rather than a single fuzzy string match:

``Quantity``        float; compared numerically with a relative tolerance, so
                    ``39.8`` and ``"39.8 years"`` agree and a unit suffix is
                    not scored as a miss.  Reuses TRQA's own
                    :func:`~evaluation.generation.trqa_match.soft_exact_match`,
                    so a cell is judged by the same rule as a final answer.
``WikibaseItem``    list of entity names; compared as normalised sets, using
                    the same :func:`~evaluation.table.matching.normalize_entity`
                    that pairs rows, so naming conventions are folded the same
                    way in both places.
``Time``            ISO date string.  The primary verdict is the *year*: the
                    gold pins day-precision facts to ``-01-01`` (a city founded
                    in ``1021`` is stored ``1021-01-01``), so demanding the full
                    date would score a correct answer wrong.  The exact-date
                    rate is reported alongside.
``GlobeCoordinate`` ``[lat, lon]``; compared by absolute degree distance, since
                    coordinates for the same place legitimately differ by the
                    precision of the source.

Every comparator returns ``None`` rather than ``False`` when the prediction is
absent or unparseable in a way that makes the comparison meaningless.  A cell
nobody filled is not a wrong answer, and pooling the two would make the
accuracy number a mix of "got it wrong" and "did not try".
"""

import math
import re
from typing import Any, Dict, List, Optional, Tuple

from ..generation.trqa_match import extract_number_from_string, soft_exact_match
from .matching import normalize_entity

#: Relative tolerance (percent) at which a Quantity cell counts as correct.
#: TRQA's own report starts its tolerance ladder at 1%; a cell is a single
#: retrieved fact rather than an aggregate over many, so the tightest rung is
#: the right default here.
QUANTITY_TOLERANCE_PCT: float = 1.0

#: Degrees within which a GlobeCoordinate counts as correct.  One degree of
#: latitude is ~111 km -- loose enough for two sources to disagree about where
#: the centre of a region is, tight enough to separate neighbouring regions.
COORDINATE_TOLERANCE_DEG: float = 1.0

_YEAR_RE = re.compile(r"(-?\d{3,4})")
_ISO_RE = re.compile(r"(-?\d{3,4})-(\d{1,2})-(\d{1,2})")


def _as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    return list(value) if isinstance(value, (list, tuple, set)) else [value]


# ---------------------------------------------------------------------------
# Per-datatype comparators
# ---------------------------------------------------------------------------

def _cmp_quantity(gold: Any, pred: Any) -> Optional[Dict[str, bool]]:
    if extract_number_from_string(str(pred)) is None:
        return None
    match = soft_exact_match(pred, gold, tolerance_pct=QUANTITY_TOLERANCE_PCT)
    return {"correct": bool(match.get("soft_match")), "exact": bool(match.get("exact_match"))}


#: Punctuation an agent uses to join several items into one cell.  ``and`` is
#: deliberately not a separator: "Trinidad and Tobago" is one entity, and the
#: joined-string case is already covered by the punctuation forms.
_ITEM_SPLIT_RE = re.compile(r"[;,/|\n]")


def _pred_items(pred: Any) -> set:
    """Normalised candidate items from a prediction, however it was written.

    Two fifths of the gold ``WikibaseItem`` values are lists, and an agent
    writing one into a single cell joins it -- ``"Ohio River, Wabash River"``.
    Comparing that blob whole matches neither item, so the split forms are
    added as candidates too.

    The unsplit string is kept as a candidate as well, and that ordering
    matters: two gold items really do contain a comma ("Washington, D.C."), and
    they still match on the whole form even though the split would shred them.
    """
    out = set()
    for value in _as_list(pred):
        text = str(value)
        whole = normalize_entity(text)
        if whole:
            out.add(whole)
        for part in _ITEM_SPLIT_RE.split(text):
            normalized = normalize_entity(part)
            if normalized:
                out.add(normalized)
    return out


def _cmp_wikibase_item(gold: Any, pred: Any) -> Optional[Dict[str, bool]]:
    # The gold list is authoritative and is never split: an item that contains
    # a separator is a name, not two names.
    gold_set = {normalize_entity(g) for g in _as_list(gold)}
    gold_set.discard("")
    pred_set = _pred_items(pred)
    if not pred_set or not gold_set:
        return None
    return {
        # One of the gold items named is the answer to "which river borders
        # Illinois": the gold list is the set of acceptable answers, not a set
        # the cell must reproduce in full.
        "correct": bool(gold_set & pred_set),
        # The stricter tier is "named every one of them", not set equality --
        # ``pred_set`` deliberately carries both the joined and the split forms,
        # so it is a candidate set rather than a claim about cardinality.
        "exact": gold_set <= pred_set,
        # How much of the gold set the cell actually recovered.  The two tiers
        # above are "any" and "all", and a COUNT question that filters on one
        # particular item falls between them: a cell listing Moscow, Leningrad
        # and Plyos against a gold of Moscow, Saint Petersburg and
        # Novo-Ogaryovo is `correct` on Moscow alone, while the item the
        # question actually selects on is the one it missed.  Recall over the
        # gold set shows that as 1/3.  Only recall is defined here -- precision
        # is not, because ``pred_set`` holds both the joined and split forms of
        # every value and so has no meaningful cardinality.
        "recall": len(gold_set & pred_set) / len(gold_set),
    }


def _parse_year(value: Any) -> Optional[int]:
    m = _YEAR_RE.search(str(value))
    return int(m.group(1)) if m else None


def _parse_date(value: Any) -> Optional[Tuple[int, int, int]]:
    m = _ISO_RE.search(str(value))
    return (int(m.group(1)), int(m.group(2)), int(m.group(3))) if m else None


def _cmp_time(gold: Any, pred: Any) -> Optional[Dict[str, bool]]:
    gold_year, pred_year = _parse_year(gold), _parse_year(pred)
    if pred_year is None or gold_year is None:
        return None
    gold_date, pred_date = _parse_date(gold), _parse_date(pred)
    return {
        "correct": gold_year == pred_year,
        "exact": bool(gold_date and pred_date and gold_date == pred_date),
    }


def _parse_coords(value: Any) -> Optional[Tuple[float, float]]:
    nums = [float(n.replace(",", "")) for n in
            re.findall(r"[-+]?\d+(?:\.\d+)?", str(value))]
    return (nums[0], nums[1]) if len(nums) >= 2 else None


def _cmp_globe_coordinate(gold: Any, pred: Any) -> Optional[Dict[str, bool]]:
    gold_pt, pred_pt = _parse_coords(gold), _parse_coords(pred)
    if gold_pt is None or pred_pt is None:
        return None
    dist = math.hypot(gold_pt[0] - pred_pt[0], gold_pt[1] - pred_pt[1])
    return {
        "correct": dist <= COORDINATE_TOLERANCE_DEG,
        "exact": dist <= 0.01,
    }


_COMPARATORS = {
    "Quantity": _cmp_quantity,
    "WikibaseItem": _cmp_wikibase_item,
    "Time": _cmp_time,
    "GlobeCoordinate": _cmp_globe_coordinate,
}


def compare_value(datatype: str, gold: Any, pred: Any) -> Optional[Dict[str, float]]:
    """``{"correct", "exact", "recall"}`` for one cell, or None when not comparable.

    An unknown datatype falls back to the ``WikibaseItem`` comparator (a
    normalised string comparison), which is the least assuming of the four --
    so a datatype TRQA adds later is scored conservatively rather than skipped.

    ``recall`` is how much of a set-valued gold the cell recovered.  Only the
    ``WikibaseItem`` comparator computes one; for the single-valued datatypes a
    cell is all of its gold or none of it, so it defaults to ``correct`` and the
    metric stays comparable across datatypes rather than being absent for three
    of the four.
    """
    if pred is None or (isinstance(pred, str) and not pred.strip()):
        return None
    verdict = _COMPARATORS.get(datatype, _cmp_wikibase_item)(gold, pred)
    if verdict is not None:
        verdict.setdefault("recall", float(verdict["correct"]))
    return verdict


# ---------------------------------------------------------------------------
# Scoring a grid
# ---------------------------------------------------------------------------

def score_cell_values(
    gold,
    snapshot: Dict[str, Any],
    matched_col_id: Optional[str],
    row_matches: Dict[int, int],
) -> Dict[str, Any]:
    """Score the gold column's cells of one grid snapshot.

    Only the column matched to the gold property is scored -- the gold records
    one property, and a grid legitimately carries others it cannot account for
    (see :mod:`evaluation.table.gold`).  Only rows matched to a gold entity are
    scored, because an unmatched row has no gold value to compare against.

    Two denominators, because they answer different questions:

    ``value_precision`` over the cells that were actually filled -- when the
    agent commits to a value, how often is it right?
    ``value_recall`` over every gold entity -- what fraction of the answer
    table did the run establish correctly?  A row never enumerated and a cell
    never filled both count against this one, which is what makes it the
    end-to-end number.

    Args:
        gold:           The query's :class:`~evaluation.table.gold.GoldTable`.
        snapshot:       One grid snapshot.
        matched_col_id: Column matched to the gold property, from ``match_column``.
        row_matches:    ``{gold_index: row_index}`` from ``match_rows``.
    """
    if not matched_col_id or not row_matches:
        return {}

    rows = snapshot.get("rows") or []
    scored = correct = exact = not_comparable = 0
    item_recall = 0.0

    for gold_idx, row_idx in row_matches.items():
        if row_idx >= len(rows):
            continue
        entity = gold.entities[gold_idx]
        cell = ((rows[row_idx].get("cells") or {}).get(matched_col_id)) or {}
        if cell.get("status") != "filled":
            continue
        verdict = compare_value(gold.datatype, gold.values.get(entity), cell.get("value"))
        if verdict is None:
            not_comparable += 1
            continue
        scored += 1
        correct += bool(verdict["correct"])
        exact += bool(verdict["exact"])
        item_recall += float(verdict["recall"])

    if not scored:
        return {"n_cells_scored": 0, "n_not_comparable": not_comparable}

    # ``n_gold_rows`` is deliberately not returned: ``_score_stage`` already
    # carries it, and a second copy here is a second thing to keep in step.
    n_gold = len(gold.entities)
    return {
        "n_cells_scored": scored,
        "n_not_comparable": not_comparable,
        "value_precision": correct / scored,
        "value_precision_exact": exact / scored,
        # Between the two tiers above.  ``value_precision`` asks whether the
        # cell named *any* of its gold items and ``value_precision_exact``
        # whether it named *all* of them; on a set-valued property -- 78% of
        # wiki2 -- most real failures sit in between, where the cell recovered
        # part of the set and missed the part the question selects on.
        "value_item_recall": item_recall / scored,
        "value_recall": (correct / n_gold) if n_gold else 0.0,
        "n_correct": correct,
    }
