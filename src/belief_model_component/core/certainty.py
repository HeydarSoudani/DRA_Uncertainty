"""The certainty score s_t of a belief state: one number in [0, 1] for how
sure the belief is, read from the statuses of its criteria and their
candidates during the search (s_t, after step t).  s_T, compared with the
correctness y of the agent's final answer, is the final scorer's
(``final_score``), which judges how much each criterion counts for the
answer; this function of the final statuses is kept next to it as a
baseline (``certainty_formula``), with its answer-anchored form
(``answer_certainty``).

A fixed function, nothing learned, the same for every query shape:

1. per criterion a score q_k: the status of the criterion (its highest
   candidate) as a probability (``core.types.STATUS_PROBABILITY``: unlikely
   0.10, unknown 0.25, possible 0.50, probable 0.70, almost certain 0.90,
   certain 0.97).  The status of a candidate of a one-answer criterion
   already counts its rivals (two tied candidates are each at most
   "possible"), so the score needs no share between them;
2. s_t = the mean q_k (``CERTAINTY_AGGREGATOR``) over the criteria the
   updater judges (``counted``): the expected share of criteria that hold,
   so a change on any criterion moves it.

An aggregate is left out: its status is derived from its inputs
(``BeliefState._derive_aggregates``, the lowest of them), so counting it
would count its weakest input twice.  In ``answer_certainty`` a computed
aggregate counts (the agent's answer is its candidate there, so a wrong
total must weigh); a target does not.

The beliefs are read as saved (``BeliefState.belief_lists``: per
criterion ``{id, status, selected, candidates}``), so the same code scores a
live belief and a saved one.
"""

import math
from typing import Any, Callable, Dict, List, Optional, Sequence

from ..text.matching import alnum
from .types import AGGREGATE, UNKNOWN, UNLIKELY, status_probability


def geometric_mean(scores: Sequence[float]) -> float:
    """The geometric mean of *scores* in [0, 1] (0 when one is 0)."""
    if any(q <= 0.0 for q in scores):
        return 0.0
    return math.exp(sum(math.log(q) for q in scores) / len(scores))


# The aggregators of the criterion scores: the certainty score's (the mean,
# ``CERTAINTY_AGGREGATOR``) and the others, reported next to it in the teacher
# scores (information only).
AGGREGATORS: Dict[str, Callable[[Sequence[float]], float]] = {
    "min": min,
    "mean": lambda q: sum(q) / len(q),
    "geometric_mean": geometric_mean,
    "product": lambda q: math.prod(q),
}
CERTAINTY_AGGREGATOR = "mean"


def counted(beliefs: List[Dict[str, Any]], kinds: List[str]) -> List[Dict[str, Any]]:
    """The beliefs of the criteria s_t counts: every criterion of *kinds*
    but an aggregate (its status is derived from its inputs)."""
    return [b for b, kind in zip(beliefs, kinds) if kind != AGGREGATE]


def criterion_score(belief: Dict[str, Any]) -> float:
    """q_k of one criterion (its saved belief): its status as a probability."""
    return status_probability(belief.get("status") or UNKNOWN)


def criterion_scores(beliefs: List[Dict[str, Any]]) -> List[float]:
    """q_k of every criterion, in order."""
    return [criterion_score(s) for s in beliefs]


def certainty(beliefs: List[Dict[str, Any]], kinds: List[str],
              aggregator: Optional[str] = None) -> Optional[float]:
    """s_t of a belief (per criterion its saved belief, the criteria of
    *kinds*): the *aggregator* (default ``CERTAINTY_AGGREGATOR``) of the
    counted criteria's scores; None when no criterion counts."""
    scores = criterion_scores(counted(beliefs, kinds))
    return AGGREGATORS[aggregator or CERTAINTY_AGGREGATOR](scores) if scores else None


def answer_certainty(beliefs: List[Dict[str, Any]], kinds: List[str], answer_candidates: List[Dict[str, Any]],
                     ruled_out: Optional[Dict[str, List[str]]] = None,
                     aggregator: Optional[str] = None,
                     computed: Optional[List[bool]] = None) -> Optional[float]:
    """s_T anchored on the agent's answer (a baseline of the final scorer):
    q_k is the status probability of the answer's candidate the scorer
    names on criterion k (``final_score``: ``{id, candidate}``):
    "unlikely" when it is a name a passage ruled out (*ruled_out*, per
    criterion id, ``BeliefState.ruled_out_lists``; the scorer prompt reads
    a removed name so); when it names none or one that is no candidate (a
    proposal), the status of a name no passage supports: "unknown" when
    the criterion lists no candidate, else "unlikely" (a listed candidate
    has support, or is derived from inputs that do).  On a computed
    aggregate (*computed*, per criterion), an answer not listed is a new
    computation, judged as a candidate of the aggregate: the lowest q of
    its inputs, the criteria before it (whether a passage rules out or
    states a missing input is the scorer's to judge).  A target aggregate
    is left out (its status is derived from its inputs).  The
    *aggregator* defaults to ``CERTAINTY_AGGREGATOR``; None when no
    criterion counts."""
    named = {c.get("id"): alnum(c.get("candidate") or "") for c in answer_candidates}
    ruled_out = ruled_out or {}
    computed = computed or [False] * len(beliefs)
    scores: List[float] = []
    for belief, kind, is_computed_k in zip(beliefs, kinds, computed):
        if kind == AGGREGATE and not is_computed_k:
            continue
        cid = belief.get("id")
        key = named.get(cid)
        listed = belief.get("candidates") or []
        status = next((x["status"] for x in listed if key and alnum(x["name"]) == key), None)
        if status is None:
            removed = bool(key) and key in {alnum(n) for n in ruled_out.get(cid) or []}
            if is_computed_k and not removed:
                scores.append(min(scores) if scores else status_probability(UNKNOWN))
                continue
            status = UNLIKELY if removed or listed else UNKNOWN
        scores.append(status_probability(status))
    return AGGREGATORS[aggregator or CERTAINTY_AGGREGATOR](scores) if scores else None
