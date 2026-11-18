"""The certainty score s_t of a belief state: one number in [0, 1] for how
sure the belief is, read from the statuses of its beliefs during the
search (s_t, after step t).  s_T, compared with the correctness y of the
agent's final answer, is the final scorer's (``final_score``), which judges
how much each belief counts for the answer; this function of the final
statuses is kept next to it as a baseline (``certainty_formula``).

A fixed function, nothing learned, the same for every query shape:

1. per belief a score q_k: its WEP as a probability
   (``core.types.STATUS_PROBABILITY``, the midpoint of each word's range).
   The update judges a one-answer WEP from all the belief's passages,
   rivals included, so the score needs no share between them;
2. s_t = the mean q_k over the beliefs the updater judges (``counted``):
   the expected share of beliefs that hold, so a change on any belief
   moves it.

An aggregate is left out: its status is derived from its inputs
(``BeliefState._derive_aggregates``, the lowest of them), so counting it
would count its weakest input twice.

The beliefs are read as saved (``BeliefState.belief_lists``: per
belief ``{id, status, ...}``), so the same code scores a live belief
and a saved one.
"""

from typing import Any, Dict, List, Optional

from .types import AGGREGATE, UNKNOWN, status_probability


def counted(beliefs: List[Dict[str, Any]], kinds: List[str]) -> List[Dict[str, Any]]:
    """The states of the beliefs s_t counts: every belief of *kinds*
    but an aggregate (its status is derived from its inputs)."""
    return [b for b, kind in zip(beliefs, kinds) if kind != AGGREGATE]


def status_score(belief: Dict[str, Any]) -> float:
    """q_k of one belief (its saved state): its status as a probability."""
    return status_probability(belief.get("status") or UNKNOWN)


def certainty(beliefs: List[Dict[str, Any]], kinds: List[str]) -> Optional[float]:
    """s_t of the belief states (one saved state per belief, the beliefs
    of *kinds*): the mean score of the counted beliefs; None when no
    belief counts."""
    scores = [status_score(b) for b in counted(beliefs, kinds)]
    return sum(scores) / len(scores) if scores else None
