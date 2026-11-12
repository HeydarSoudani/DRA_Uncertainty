"""The certainty score s_t of a belief state: one number in [0, 1] for how
sure the belief is, read from the statuses of its criteria and their
candidates during the search (s_t, after step t).  On a short-answer
query s_T, compared with the correctness y of the agent's final answer, is
the final scorer's (``final_score``); this function of the final statuses
is kept next to it as a baseline (``certainty_formula``), with its
answer-anchored form (``answer_certainty``).

A fixed function, nothing learned:

1. per criterion a score q_k: the status of the criterion (its highest
   candidate) as a probability (``core.types.STATUS_PROBABILITY``: unlikely
   0.10, unknown 0.25, possible 0.50, probable 0.70, almost certain 0.90,
   certain 0.97).  The status of a candidate of a one-answer criterion
   already counts its rivals (two tied candidates are each at most
   "possible"), so the score needs no share between them;
2. a short-answer query: s_t = the lowest q_k (``min``, the weakest
   link).  The answer is correct only when every criterion it relies on
   holds, and min is the upper bound of that conjunction (Frechet), exact
   when the criteria hang together as here (they describe one entity);
   it reads only the order of the statuses, so its ranking (AUROC) does
   not depend on the table's values.  A report (every criterion
   an aspect): s_t = the mean q_k (``mean``), the expected share of
   aspects covered, since one thin aspect does not make a report wrong
   (``aggregator_name``).

Every criterion counts.  An aggregate's status is the lowest of its
inputs' (``BeliefState._derive_aggregates``), so under ``min`` it adds
only what its inputs do not show: a target joins the clues (a name must
be listed on every one), a computed answer a passage speaks against is
"unlikely".  An aspect's status is its own (``BeliefState.aspect_status``).

The beliefs are read as saved (``BeliefState.belief_lists``: per
criterion ``{id, status, selected, candidates}``), so the same code scores a
live belief and a saved one.
"""

import math
from typing import Any, Callable, Dict, List, Optional, Sequence

from ..text.matching import alnum
from .types import ASPECT, UNKNOWN, UNLIKELY, status_probability


def geometric_mean(scores: Sequence[float]) -> float:
    """The geometric mean of *scores* in [0, 1] (0 when one is 0)."""
    if any(q <= 0.0 for q in scores):
        return 0.0
    return math.exp(sum(math.log(q) for q in scores) / len(scores))


# The aggregators of the criterion scores: the certainty score's (min for a
# short-answer query, mean for a report, ``aggregator_name``) and the others,
# reported next to it in the teacher scores (information only).
AGGREGATORS: Dict[str, Callable[[Sequence[float]], float]] = {
    "min": min,
    "mean": lambda q: sum(q) / len(q),
    "geometric_mean": geometric_mean,
    "product": lambda q: math.prod(q),
}
CERTAINTY_AGGREGATOR = "min"
REPORT_AGGREGATOR = "mean"


def aggregator_name(kinds: List[str]) -> str:
    """The aggregator of the certainty score for criteria of *kinds*:
    ``REPORT_AGGREGATOR`` when every criterion is an aspect (a report),
    else ``CERTAINTY_AGGREGATOR``."""
    return REPORT_AGGREGATOR if kinds and all(k == ASPECT for k in kinds) else CERTAINTY_AGGREGATOR


def criterion_score(belief: Dict[str, Any]) -> float:
    """q_k of one criterion (its saved belief): its status as a probability."""
    return status_probability(belief.get("status") or UNKNOWN)


def criterion_scores(beliefs: List[Dict[str, Any]]) -> List[float]:
    """q_k of every criterion, in order."""
    return [criterion_score(s) for s in beliefs]


def certainty(beliefs: List[Dict[str, Any]], kinds: List[str],
              aggregator: Optional[str] = None) -> Optional[float]:
    """s_t of a belief (per criterion its saved belief, the criteria of
    *kinds*), by *aggregator* (default ``aggregator_name``); None when no
    criterion counts."""
    scores = criterion_scores(beliefs)
    return AGGREGATORS[aggregator or aggregator_name(kinds)](scores) if scores else None


def answer_certainty(beliefs: List[Dict[str, Any]], kinds: List[str], answer_candidates: List[Dict[str, Any]],
                     ruled_out: Optional[Dict[str, List[str]]] = None,
                     aggregator: Optional[str] = None) -> Optional[float]:
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
        cid = belief.get("id")
        key = named.get(cid)
        status = next((x["status"] for x in belief.get("candidates") or [] if key and alnum(x["name"]) == key),
                      None)
        if status is None:
            status = UNLIKELY if key and key in {alnum(n) for n in ruled_out.get(cid) or []} else UNKNOWN
        scores.append(status_probability(status))
    return AGGREGATORS[aggregator or aggregator_name(kinds)](scores) if scores else None
