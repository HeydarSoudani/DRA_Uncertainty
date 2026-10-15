"""The ``<belief_state>`` tag that ``--belief-mode inform`` injects.

At the end of every search iteration the agent reads one tag built from the
iteration's step record::

    <belief_state step="3">
      <criteria>
        <k1 status="certain" attempts="1">born in the 1960s</k1>
        <k2 status="possible" attempts="2">won a regional award</k2>
        <k3 status="unknown" attempts="0">the films they directed</k3>
      </criteria>
      <signals doc_novelty="0.40" criteria_delta="+2" query_novelty="0.81"/>
    </belief_state>

The tag is descriptive only: per criterion its belief status and attempts, then
the step's signals.  Only the fields in ``SIGNAL_FIELDS``, the belief state and
the attempts (``criteria_attempts_after``; left out when null) are read, so
gold-based signals (new-item precision, new-item graded recall, relevant
counts) never reach the agent; they stay in ``belief/{qid}.jsonl`` for
analysis.  A null signal is left out; an element with nothing to show is left
out; with nothing to show at all there is no tag.
"""

import re
from html import escape
from typing import Any, Dict, List, Optional

from .types import Criterion

# Document side of x_t (nu^D, Delta^D) and query side (nu^q); the attempts
# a are shown per criterion.
SIGNAL_FIELDS = ("doc_novelty", "criteria_delta", "query_novelty")

# A tag the model wrote itself; removed before its text enters the context.
BELIEF_STATE_RE = re.compile(r"<belief_state\b[^>]*>.*?</belief_state>\s*", re.DOTALL)


def strip_belief_state(text: str) -> str:
    """Remove every ``<belief_state>`` tag from *text*."""
    return BELIEF_STATE_RE.sub("", text) if text and "<belief_state" in text else text


def _format(field: str, value: Any) -> str:
    if field == "criteria_delta":
        return f"{value:+d}"
    return f"{value:.2f}"


def render_belief_state(record: Dict[str, Any], criteria: List[Criterion]) -> Optional[str]:
    """The ``<belief_state>`` tag of one step *record* (``BeliefModel.observe``);
    None when it would be empty."""
    lines = [f'<belief_state step="{record["iteration"]}">']

    statuses = record.get("belief_after")
    if criteria and statuses and len(statuses) == len(criteria):
        attempts = record.get("criteria_attempts_after")
        if not attempts or len(attempts) != len(criteria):
            attempts = [None] * len(criteria)
        lines.append("  <criteria>")
        for i, (c, status, n) in enumerate(zip(criteria, statuses, attempts), 1):
            attrs = f'status="{status}"' + (f' attempts="{n}"' if n is not None else "")
            lines.append(f'    <k{i} {attrs}>{escape(c.text, quote=False)}</k{i}>')
        lines.append("  </criteria>")

    signals = " ".join(
        f'{f}="{_format(f, record[f])}"' for f in SIGNAL_FIELDS if record.get(f) is not None
    )
    if signals:
        lines.append(f"  <signals {signals}/>")

    if len(lines) == 1:
        return None
    lines.append("</belief_state>")
    return "\n".join(lines)
