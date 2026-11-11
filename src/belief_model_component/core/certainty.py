"""The certainty score s_t of a belief state: one number in [0, 1] for how
sure the belief is, read from the statuses of its criteria and their
candidates during the search (s_t, after step t).  On a short-answer
query s_T, compared with the correctness y of the agent's final answer, is
the final scorer's (``final_score``); this function of the final statuses
is kept next to it as a baseline (``certainty_formula``), with its
answer-anchored form (``answer_certainty``).

A fixed function, nothing learned:

1. per criterion a score q_k: the status of the criterion (its highest
   candidate) as a probability (``core.types.status_probability``: (level + 2)
   / 6, unlikely 1/6, unknown 1/3, possible 1/2, certain 1).  The status of
   a candidate of a one-answer criterion already counts its rivals (two
   tied candidates are each at most "possible"), so the score needs no
   share between them;
2. s_t = the geometric mean of the q_k (``geometric_mean``): one weak or
   contradicted criterion pulls it down (but not to 0, since no status
   scores 0), a query with many criteria (a set
   query's members) is not pushed to 0 as by a product.

Left out (``EXCLUDED_KINDS``): the rest criterion of a multi-member
query, which stands for the members nobody listed, so it has no gold and
no fixed answer; and the target criterion of a multi-clue query, whose
statuses are the lowest of the clues' (``core.types.DERIVED_KINDS``), so
counting it would count the weakest clue twice.

The beliefs are read as saved (``BeliefState.belief_lists``: per
criterion ``{id, status, selected, candidates}``), so the same code scores a
live belief and a saved one.
"""

import math
from typing import Any, Callable, Dict, List, Optional, Sequence

from ..text.matching import alnum
from .types import REST, TARGET, UNKNOWN, UNLIKELY, status_probability

# Criteria left out of the certainty score.
EXCLUDED_KINDS = (REST, TARGET)


def geometric_mean(scores: Sequence[float]) -> float:
    """The geometric mean of *scores* in [0, 1] (0 when one is 0)."""
    if any(q <= 0.0 for q in scores):
        return 0.0
    return math.exp(sum(math.log(q) for q in scores) / len(scores))


# Other aggregators of the criterion scores, reported next to the
# geometric mean in the teacher scores (information only).
AGGREGATORS: Dict[str, Callable[[Sequence[float]], float]] = {
    "geometric_mean": geometric_mean,
    "mean": lambda q: sum(q) / len(q),
    "product": lambda q: math.prod(q),
    "min": min,
}
CERTAINTY_AGGREGATOR = "geometric_mean"


def criterion_score(belief: Dict[str, Any]) -> float:
    """q_k of one criterion (its saved belief): its status as a probability."""
    return status_probability(belief.get("status") or UNKNOWN)


def criterion_scores(beliefs: List[Dict[str, Any]], kinds: List[str]) -> List[float]:
    """q_k of every criterion the score counts, in order."""
    return [criterion_score(s) for s, kind in zip(beliefs, kinds) if kind not in EXCLUDED_KINDS]


def certainty(beliefs: List[Dict[str, Any]], kinds: List[str],
              aggregator: str = CERTAINTY_AGGREGATOR) -> Optional[float]:
    """s_t of a belief (per criterion its saved belief, the criteria of
    *kinds*); None when no criterion counts."""
    scores = criterion_scores(beliefs, kinds)
    return AGGREGATORS[aggregator](scores) if scores else None


def answer_certainty(beliefs: List[Dict[str, Any]], kinds: List[str], answer_candidates: List[Dict[str, Any]],
                     ruled_out: Optional[Dict[str, List[str]]] = None,
                     aggregator: str = CERTAINTY_AGGREGATOR) -> Optional[float]:
    """s_T anchored on the agent's answer (a baseline of the final scorer):
    q_k is the status probability of the answer's candidate the scorer
    names on criterion k (``final_score``: ``{id, candidate}``):
    "unlikely" when it is a name a passage ruled out (*ruled_out*, per
    criterion id, ``BeliefState.ruled_out_lists``; the scorer prompt reads
    a removed name so), "unknown" when it names none or one that is no
    candidate (a proposal); None when no criterion counts."""
    named = {c.get("id"): alnum(c.get("candidate") or "") for c in answer_candidates}
    ruled_out = ruled_out or {}
    scores = []
    for belief, kind in zip(beliefs, kinds):
        if kind in EXCLUDED_KINDS:
            continue
        cid = belief.get("id")
        key = named.get(cid)
        status = next((x["status"] for x in belief.get("candidates") or [] if key and alnum(x["name"]) == key),
                      None)
        if status is None:
            status = UNLIKELY if key and key in {alnum(n) for n in ruled_out.get(cid) or []} else UNKNOWN
        scores.append(status_probability(status))
    return AGGREGATORS[aggregator](scores) if scores else None
