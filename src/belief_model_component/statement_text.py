"""The statement of a criterion as text: its fixed opening and its labels.

The belief update writes each statement in one form
(``prompts/belief_update_system.txt``)::

    It is <status> that <id> is covered, by "<candidate>"[ and "<candidate>"]: <the observations the status rests on>. <each other candidate and why it loses or ties>.
    It is unknown whether <id> is covered.

Code reads from the text only what it needs: the status and the selected
candidate(s) of the opening (``parse_opening``), the source labels it cites
("d<n>" a passage, "t<n>" a turn: ``cited_labels``) and the labels of its
support part, the text before the first sentence that starts with a quoted
candidate (``support_labels``).  ``with_status`` rewrites the status word
of the opening (the turn-only cap).
"""

import re
from typing import List, NamedTuple, Optional

from .types import UNKNOWN, parse_status

# Quotes the model may write around a candidate.
_QUOTE = "\"“”"

_OPENING_RE = re.compile(
    r"^\s*It\s+is\s+(?P<status>[A-Za-z][A-Za-z _-]*?)\s+that\s+(?P<id>[A-Za-z]+\d+)\s+is\s+covered\b",
    re.IGNORECASE,
)
_UNKNOWN_RE = re.compile(r"^\s*It\s+is\s+unknown\s+whether\s+(?P<id>[A-Za-z]+\d+)\s+is\s+covered\b", re.IGNORECASE)
# The selected candidates after the opening: ', by "A"[, "B"][ and "C"]'.
_BY_RE = re.compile(
    rf"^\s*,?\s*by\s+(?P<by>[{_QUOTE}][^{_QUOTE}]+[{_QUOTE}](?:\s*(?:,\s*and|,|and)\s*[{_QUOTE}][^{_QUOTE}]+[{_QUOTE}])*)",
    re.IGNORECASE,
)
_CANDIDATE_RE = re.compile(rf"[{_QUOTE}]([^{_QUOTE}]+)[{_QUOTE}]")
# A source label ("d4", "t2"; "D4" read as "d4").
_LABEL_RE = re.compile(r"(?<![A-Za-z0-9])([dt]\d+)(?![A-Za-z0-9])", re.IGNORECASE)
# The start of an other-candidate sentence: a quote after a sentence end.
_OTHER_RE = re.compile(rf"(?<=[.;!?])\s+(?=[{_QUOTE}])")


class Opening(NamedTuple):
    status: str
    criterion: str
    selected: List[str]


def unknown_text(criterion_id: str) -> str:
    """The statement of a criterion no observation gives a candidate."""
    return f"It is unknown whether {criterion_id} is covered."


def parse_opening(text: str) -> Optional[Opening]:
    """The status, criterion id and selected candidates of the statement's
    opening, or None when it has not the fixed form (or names no known
    status)."""
    text = text or ""
    m = _UNKNOWN_RE.match(text)
    if m:
        return Opening(UNKNOWN, m.group("id"), [])
    m = _OPENING_RE.match(text)
    if not m:
        return None
    status = parse_status(m.group("status"))
    if status is None:
        return None
    by = _BY_RE.match(text[m.end():])
    selected = [c.strip() for c in _CANDIDATE_RE.findall(by.group("by"))] if by else []
    return Opening(status, m.group("id"), [c for c in selected if c])


def with_status(text: str, status: str) -> str:
    """*text* with the status word of its opening replaced by *status*."""
    m = _OPENING_RE.match(text or "")
    if not m:
        return text
    return text[:m.start("status")] + status + text[m.end("status"):]


def _unquoted(text: str) -> str:
    """*text* with every quoted candidate blanked, so a label-like word in
    a candidate's name ("Model d2 engine") is not read as a citation."""
    return _CANDIDATE_RE.sub('""', text or "")


def cited_labels(text: str) -> List[str]:
    """The source labels the text cites outside its quoted candidates, in
    order, each once, lowercased."""
    return list(dict.fromkeys(lb.lower() for lb in _LABEL_RE.findall(_unquoted(text))))


def support_part(text: str) -> str:
    """The text before the first other-candidate sentence: one that starts
    with a quoted candidate, as the prompt's form writes it (a sentence on
    other candidates that starts otherwise, e.g. 'Both "Y" and "Z" (d2)
    lose.', is read as support)."""
    return _OTHER_RE.split(text or "", maxsplit=1)[0]


def support_labels(text: str) -> List[str]:
    """The labels of the support part (``support_part``)."""
    return cited_labels(support_part(text))


def rename_labels(text: str, rename: dict) -> str:
    """*text* with every cited label (outside its quoted candidates)
    replaced by ``rename[label]`` (labels without an entry kept)."""
    if not rename:
        return text
    out, last = [], 0
    for m in _CANDIDATE_RE.finditer(text):
        out.append(_LABEL_RE.sub(lambda x: rename.get(x.group(1).lower(), x.group(1)), text[last:m.start()]))
        out.append(m.group(0))
        last = m.end()
    out.append(_LABEL_RE.sub(lambda x: rename.get(x.group(1).lower(), x.group(1)), text[last:]))
    return "".join(out)
