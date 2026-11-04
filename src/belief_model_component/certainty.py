"""The certainty score s_t of a belief state: one number in [0, 1] for how
sure the belief is that the agent's answer is correct, compared with the
correctness y of the agent's final answer (s_T, after the final call on
the answer turn) or read during the search (s_t, after step t).

Three fixed steps, nothing learned:

1. each nugget's status as a probability (``types.status_probability``:
   (level + 2) / 6, doubtful 0, unknown 1/3, certain 1);
2. per criterion a score q_k: for the answer criterion (the last
   criterion of a multi-hop query) after the final call, the probability
   of the nugget the answer turn supports (the belief in the answer the
   agent gave), the unknown probability when no nugget has it; for every
   other criterion, its most likely nugget's probability; a criterion
   without a nugget scores as unknown.  The rest and aggregate criteria of
   a set query are left out (``EXCLUDED_KINDS``): the rest criterion has
   no gold yet, the aggregate's nuggets rest on the agent's turns only, so
   they are always "unknown";
3. s_t = the geometric mean of the q_k (``geometric_mean``): one weak or
   contradicted criterion pulls it down, a query with many criteria (a set
   query's members) is not pushed to 0 as by a product.

The nuggets are read as saved (``BeliefState.nugget_lists``: per
criterion ``[{value, status, evidence: [{label, stance, value, ...}]}]``),
so the same code scores a live belief and a saved one.
"""

import math
from typing import Any, Callable, Dict, List, Optional, Sequence

from .types import AGGREGATE, MULTI_HOP, REST, SUPPORTS, status_probability

# Criteria left out of the certainty score.
EXCLUDED_KINDS = (REST, AGGREGATE)

UNKNOWN_P = status_probability("unknown")


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


def answer_criterion(query_shape: str, num_criteria: int) -> Optional[int]:
    """The index of the criterion whose answer is the query's answer (the
    last hop of a multi-hop query), or None."""
    return num_criteria - 1 if query_shape == MULTI_HOP and num_criteria else None


def _supported_by(nugget: Dict[str, Any], label: str) -> bool:
    return any(e.get("label") == label and e.get("stance") == SUPPORTS for e in nugget.get("evidence") or [])


def criterion_score(nuggets: List[Dict[str, Any]], answer_label: Optional[str] = None) -> float:
    """q_k of one criterion's *nuggets*: with *answer_label* (the answer
    turn's "t<n>"), the probability of the nugget that turn supports, else
    of its most likely nugget; unknown when there is none."""
    if answer_label is not None:
        nuggets = [n for n in nuggets if _supported_by(n, answer_label)]
    return max((status_probability(n["status"]) for n in nuggets), default=UNKNOWN_P)


def criterion_scores(nuggets: List[List[Dict[str, Any]]], kinds: List[str], query_shape: str,
                     answer_label: Optional[str] = None) -> List[float]:
    """q_k of every criterion the score counts, in order."""
    answer = answer_criterion(query_shape, len(kinds)) if answer_label is not None else None
    return [criterion_score(ns, answer_label if k == answer else None)
            for k, (ns, kind) in enumerate(zip(nuggets, kinds)) if kind not in EXCLUDED_KINDS]


def certainty(nuggets: List[List[Dict[str, Any]]], kinds: List[str], query_shape: str,
              answer_label: Optional[str] = None, aggregator: str = CERTAINTY_AGGREGATOR) -> Optional[float]:
    """s_t of a belief (per criterion its *nuggets*, the criteria of
    *kinds*); *answer_label* the answer turn's label after the final call
    (None during the search).  None when no criterion counts."""
    scores = criterion_scores(nuggets, kinds, query_shape, answer_label)
    return AGGREGATORS[aggregator](scores) if scores else None
