"""The belief as text: each criterion with its candidates, and the
``<belief_state>`` tag that ``--belief-mode inform`` injects.

The belief state keeps per criterion its candidates (``types.Candidate``);
what is shown of one is its name, status and basis, one line each, highest
status first, then "passage" before "reasoning", then the order added
(``BeliefState.ordered``).  The belief update prompt lists
every criterion with its candidates (``format_belief``)::

    c2 (one answer): the engineer who was born in the town where the Varn Folk Museum was founded
      "Henrik Dahl": certain (passage)
      "Nils Berg": unlikely (reasoning)
    c3 (one answer): the lighthouse that was designed by that engineer
      (no candidate)

At the end of every search iteration the agent reads one tag, each
criterion with the same candidate lines, its own id "k<n>" in place of the
criterion id::

    <belief_state step="3">
      <criteria>
        <k1>the engineer who was born in the town where the Varn Folk Museum was founded
          "Henrik Dahl": certain (passage)
          "Nils Berg": unlikely (reasoning)
        </k1>
        <k2>the lighthouse that was designed by that engineer</k2>
      </criteria>
    </belief_state>

The tag is descriptive only, and carries no source labels: the reasons of
the updates are saved, never shown.  Nothing gold-based (qrels, recall)
reaches the agent.  With no criteria there is no tag.
"""

import re
from html import escape
from typing import Any, Dict, List, Optional

from .types import NO_DOC_KINDS, Candidate, exclusive

# A tag the model wrote itself; removed before its text enters the context.
BELIEF_STATE_RE = re.compile(r"<belief_state\b[^>]*>.*?</belief_state>\s*", re.DOTALL)

# The line of a criterion without candidates.
NO_CANDIDATE = "(no candidate)"


def strip_belief_state(text: str) -> str:
    """Remove every ``<belief_state>`` tag from *text*."""
    return BELIEF_STATE_RE.sub("", text) if text and "<belief_state" in text else text


def criterion_head(state: Any, k: int) -> str:
    """Criterion *k* as the belief update reads it: its id, its type
    ("computed" when no document states it, else "one answer" or "several
    answers") and its text."""
    c = state.criteria[k]
    kind = "computed" if c.kind in NO_DOC_KINDS else "one answer" if exclusive(c.kind) else "several answers"
    return f"{c.id} ({kind}): {c.text}"


def candidate_line(c: Candidate) -> str:
    """One candidate as shown: ``"<name>": <status> (<basis>)``."""
    return f'"{c.name}": {c.status} ({c.basis})'


def format_belief(state: Any) -> str:
    """Every criterion of *state* with its candidates, highest status
    first, or ``NO_CANDIDATE``."""
    lines = []
    for k in range(len(state.criteria)):
        lines.append(criterion_head(state, k))
        cands = state.ordered(k)
        if cands:
            lines.extend(f"  {candidate_line(c)}" for c in cands)
        else:
            lines.append(f"  {NO_CANDIDATE}")
    return "\n".join(lines)


def render_belief_state(record: Dict[str, Any], state: Any) -> Optional[str]:
    """The ``<belief_state>`` tag of one step *record* (``BeliefModel.observe``)
    with the belief *state* after the step; None without a state or
    criteria."""
    if state is None or not state.criteria:
        return None
    lines: List[str] = [f'<belief_state step="{record["iteration"]}">', "  <criteria>"]
    for i, c in enumerate(state.criteria, 1):
        text = escape(c.text, quote=False)
        cands = state.ordered(i - 1)
        if not cands:
            lines.append(f"    <k{i}>{text}</k{i}>")
            continue
        lines.append(f"    <k{i}>{text}")
        lines.extend(f"      {escape(candidate_line(x), quote=False)}" for x in cands)
        lines.append(f"    </k{i}>")
    lines.append("  </criteria>")
    lines.append("</belief_state>")
    return "\n".join(lines)
