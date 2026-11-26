"""Passages and turns of a belief update call: span checks against the
text shown, passage titles, and the labels a reply writes ("d4", "t2")."""

import re
from typing import Any, List, NamedTuple, Optional, Tuple

from .matching import alnum, alnum_index


def span_in_passage(text: str, span: str, index: Optional[Tuple[str, List[int]]] = None) -> Optional[str]:
    """The passage's own text matching *span* (case, whitespace and
    punctuation ignored, so words the passage runs together still match),
    or None when *span* does not occur in *text*.  *index* is
    ``alnum_index(text)`` when the caller has it (several spans of one
    passage)."""
    target = alnum(span)
    if not target:
        return None
    keys, pos = index if index is not None else alnum_index(text)
    j = keys.find(target)
    if j < 0:
        return None
    return text[pos[j]:pos[j + len(target) - 1] + 1]


def split_span(span: str) -> List[str]:
    """The pieces of a cited span that joins sentences with an ellipsis."""
    return [p.strip() for p in re.split(r"\.\.\.|…", span) if alnum(p)]


def _label(value: Any, prefix: str) -> str:
    """A label as the reply writes it ("d4", "[d4]", 4) with its *prefix*
    ("d4")."""
    text = str(value if value is not None else "").strip().strip("[]").strip().lower()
    return f"{prefix}{text}" if text.isdigit() else text


def doc_label(value: Any) -> str:
    """A passage label as the reply writes it ("d4", "[d4]", 4) as "d4"."""
    return _label(value, "d")


class Passage(NamedTuple):
    """One passage of a call: its label, document id and text."""
    label: str
    doc_id: str
    text: str


# One turn of a call: its step (its label is "t<step>") and its text.
Turn = Tuple[int, str]
