"""The reason of a candidate update: its opening and the labels it cites.

The belief update writes, per candidate that is new, removed or whose
status changes, one sentence in this form (``prompts/belief_update/system.txt``)::

    Was <current status>: <why its support gives the new status>.
    Was kept: <...>                  (a listed fact of an aspect)
    Was proposed: <...>              (a name the agent proposed)
    New: <...>                       (a candidate only a passage names)

The reason is saved with the update and never shown to the next one.  Code
reads from it only the opening (``parse_reason``: was the candidate listed,
and with which status, or proposed) and the source labels it cites ("d<n>" a passage,
"t<n>" a turn: ``reason_labels``).
"""

import re
from typing import List, NamedTuple, Optional

from .types import KEPT, is_kept, parse_status

# Quotes the model may write around a name.
_QUOTE = "\"“”"

_WAS_RE = re.compile(r"^\s*Was\s+(?P<status>[A-Za-z][A-Za-z _-]*?)\s*:", re.IGNORECASE)
_PROPOSED = "proposed"
_NEW_RE = re.compile(r"^\s*New\s*:", re.IGNORECASE)
_QUOTED_RE = re.compile(rf"[{_QUOTE}]([^{_QUOTE}]+)[{_QUOTE}]")
# A source label ("d4", "t2"; "D4" read as "d4").
_LABEL_RE = re.compile(r"(?<![A-Za-z0-9])([dt]\d+)(?![A-Za-z0-9])", re.IGNORECASE)


class ReasonOpening(NamedTuple):
    new: bool                 # "New:"
    previous: Optional[str]   # the status after "Was" ("kept" for a fact; None for "New:", "Was proposed:", or a word that is no status)
    proposed: bool = False    # "Was proposed:"


def parse_reason(text: str) -> Optional[ReasonOpening]:
    """The opening of a reason, or None when it has neither form."""
    text = text or ""
    if _NEW_RE.match(text):
        return ReasonOpening(True, None)
    m = _WAS_RE.match(text)
    if m:
        if m.group("status").strip().lower() == _PROPOSED:
            return ReasonOpening(False, None, True)
        word = m.group("status")
        return ReasonOpening(False, KEPT if is_kept(word) else parse_status(word))
    return None


def reason_labels(text: str) -> List[str]:
    """The source labels the reason cites outside quoted names, in order,
    each once, lowercased."""
    unquoted = _QUOTED_RE.sub('""', text or "")
    return list(dict.fromkeys(lb.lower() for lb in _LABEL_RE.findall(unquoted)))
