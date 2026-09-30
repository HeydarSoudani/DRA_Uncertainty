"""The ``<certainty>`` tag that ``--uncertainty-estimator-mode inform`` injects.

At the end of every search iteration the agent reads one tag built from the
iteration's step record::

    <certainty step="3">
      <criteria covered="1" partial="1" not_covered="1">
        <k1 status="covered" attempts="1">born in the 1960s</k1>
        <k2 status="partial" attempts="2">won a regional award</k2>
        <k3 status="not_covered" attempts="0">studied in Lisbon</k3>
      </criteria>
      <retrieval_signals doc_novelty="0.40" criteria_delta="+1"/>
      <reasoning_signals query_novelty="0.81"/>
    </certainty>

Only the fields in ``RETRIEVAL_FIELDS`` and ``REASONING_FIELDS``, the
criteria state and the attempts (``criteria_attempts_after``; left out when
null) are read, so gold-based signals (new-item recall, relevant
counts) never reach the agent; they stay in ``uncertainty/{qid}.jsonl`` for
analysis.  A null signal is left out; an element with nothing to show is left
out; with nothing to show at all there is no tag.
"""

import re
from html import escape
from typing import Any, Dict, List, Optional

from .types import FULLY_COVERED, PARTIALLY_COVERED, UNCOVERED, Criterion

# Document side of x_t (nu^D, Delta^D) and query side (nu^q; the attempts
# a are shown per criterion).
RETRIEVAL_FIELDS = ("doc_novelty", "criteria_delta")
REASONING_FIELDS = ("query_novelty",)

_STATUS_LABEL = {FULLY_COVERED: "covered", PARTIALLY_COVERED: "partial", UNCOVERED: "not_covered"}

# A tag the model wrote itself; removed before its text enters the context.
CERTAINTY_RE = re.compile(r"<certainty\b[^>]*>.*?</certainty>\s*", re.DOTALL)


def strip_certainty(text: str) -> str:
    """Remove every ``<certainty>`` tag from *text*."""
    return CERTAINTY_RE.sub("", text) if text and "<certainty" in text else text


def _format(field: str, value: Any) -> str:
    if field == "criteria_delta":
        return f"{value:+d}"
    return f"{value:.2f}"


def _signals(name: str, fields, record: Dict[str, Any]) -> Optional[str]:
    attrs = " ".join(
        f'{f}="{_format(f, record[f])}"' for f in fields if record.get(f) is not None
    )
    return f"  <{name} {attrs}/>" if attrs else None


def render_certainty(record: Dict[str, Any], criteria: List[Criterion]) -> Optional[str]:
    """The ``<certainty>`` tag of one step *record* (``UncertaintyEstimator.observe``);
    None when it would be empty."""
    lines = [f'<certainty step="{record["iteration"]}">']

    statuses = record.get("criteria_state_after")
    if criteria and statuses and len(statuses) == len(criteria):
        labels = [_STATUS_LABEL[s] for s in statuses]
        attempts = record.get("criteria_attempts_after")
        if not attempts or len(attempts) != len(criteria):
            attempts = [None] * len(criteria)
        lines.append(
            f'  <criteria covered="{labels.count("covered")}" partial="{labels.count("partial")}" '
            f'not_covered="{labels.count("not_covered")}">'
        )
        for i, (c, label, n) in enumerate(zip(criteria, labels, attempts), 1):
            attrs = f'status="{label}"' + (f' attempts="{n}"' if n is not None else "")
            lines.append(f'    <k{i} {attrs}>{escape(c.text, quote=False)}</k{i}>')
        lines.append("  </criteria>")

    for name, fields in (("retrieval_signals", RETRIEVAL_FIELDS), ("reasoning_signals", REASONING_FIELDS)):
        line = _signals(name, fields, record)
        if line:
            lines.append(line)

    if len(lines) == 1:
        return None
    lines.append("</certainty>")
    return "\n".join(lines)
