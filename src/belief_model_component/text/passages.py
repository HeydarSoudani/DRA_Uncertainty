"""Passages and turns of a belief update call: span checks against the
text shown, passage titles, and the labels a reply writes ("d4", "t2")."""

import re
from typing import Any, List, NamedTuple, Optional, Tuple

from utils.text_utils import cut_chars

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


# A document's front matter (``---\ntitle: ...\n---``, BrowseComp-Plus).
_FRONT_MATTER = re.compile(r"\A---\s*\n(.*?)\n---\s*(?:\n|\Z)", re.S)
_FRONT_TITLE = re.compile(r"^title:\s*(.+)$", re.M)


def passage_title(text: str, title: str = "", max_chars: int = 120) -> str:
    """A document's *title* (its metadata), else the title of its front
    matter, else the first line of *text* with a letter or digit (a wiki
    page's title or a headline); cut at *max_chars* with ``...``."""
    line = " ".join(str(title or "").split())
    if not line:
        front = _FRONT_MATTER.match(text)
        found = _FRONT_TITLE.search(front.group(1)) if front else None
        body = text[front.end():] if front else text
        line = (found.group(1).strip().strip("'\"") if found else
                next((ln.strip() for ln in body.splitlines() if alnum(ln)), ""))
    return cut_chars(line, max_chars)


def _label(value: Any, prefix: str) -> str:
    """A label as the reply writes it ("d4", "[d4]", 4) with its *prefix*
    ("d4")."""
    text = str(value if value is not None else "").strip().strip("[]").strip().lower()
    return f"{prefix}{text}" if text.isdigit() else text


def doc_label(value: Any) -> str:
    """A passage label as the reply writes it ("d4", "[d4]", 4) as "d4"."""
    return _label(value, "d")


_LABEL_SEP_RE = re.compile(r"[,;\s]+")


def reply_labels(value: Any) -> List[str]:
    """The passage labels of a reply's ``support`` / ``against`` list (a
    single label read as a list of one, a string of several, "d1, d2",
    split)."""
    items = value if isinstance(value, list) else [value] if value not in (None, "") else []
    pieces = [p for v in items for p in (_LABEL_SEP_RE.split(v) if isinstance(v, str) else [v])]
    return [lb for lb in (doc_label(v) for v in pieces) if lb]


class Passage(NamedTuple):
    """One passage of a call: its label, document id and text."""
    label: str
    doc_id: str
    text: str


# One turn of a call: its step (its label is "t<step>") and its text.
Turn = Tuple[int, str]
