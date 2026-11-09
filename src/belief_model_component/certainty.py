"""The certainty score s_t of a belief state: one number in [0, 1] for how
sure the belief is, read from the statuses of its criteria and their
candidates, and compared with the correctness y of the agent's final
answer (s_T, the belief after the answer turn) or read during the search
(s_t, after step t).

A fixed function, nothing learned:

1. per criterion a score q_k: the status of the criterion (its highest
   candidate) as a probability (``types.status_probability``: (level + 2)
   / 6, unlikely 1/6, unknown 1/3, possible 1/2, certain 1).  The status of
   a candidate of a one-answer criterion already counts its rivals (two
   tied candidates are each at most "possible"), so the score needs no
   share between them;
2. s_t = the geometric mean of the q_k (``geometric_mean``): one weak or
   contradicted criterion pulls it down (but not to 0, since no status
   scores 0), a query with many criteria (a set
   query's members) is not pushed to 0 as by a product.

The rest criterion of a set query is left out (``EXCLUDED_KINDS``): it
stands for the members nobody listed, so it has no gold and no fixed
answer.

The beliefs are read as saved (``BeliefState.belief_lists``: per
criterion ``{id, status, selected, candidates}``), so the same code scores a
live belief and a saved one.
"""

import math
from typing import Any, Callable, Dict, List, Optional, Sequence

from .types import REST, UNKNOWN, status_probability

# Criteria left out of the certainty score.
EXCLUDED_KINDS = (REST,)


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
