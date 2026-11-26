"""The References block of a generation.

Every generation ends with a ``## References`` block of ``[N] doc_id`` lines
(written by CPM-Report itself, by ``utils.citations.with_references_section``
for the other agents).  Two cuts of it, kept apart because the answer
judges' saved verdicts are keyed on their input text:

    strip_references   the text before the first appended ``## References``
                       block (the answer judges, the verbal confidence)
    split_references   the report before its last References heading of any
                       level, and that block's ``[N] doc_id`` lines (Auto-ARGUE)
"""

import re
from typing import Dict, Tuple

#: Header of the references block appended to every generation.
REFERENCES_MARKER = "\n\n## References\n"

_REFERENCES_HEADING_RE = re.compile(r"^#{1,6}\s*References\s*$", re.MULTILINE)
_REFERENCE_LINE_RE = re.compile(r"^\[(\d+)\]\s+([^\s*][^\s]*)\s*$")


def strip_references(text: str) -> str:
    """*text* without the appended ``## References`` block."""
    idx = text.find(REFERENCES_MARKER)
    if idx != -1:
        text = text[:idx].rstrip()
    return text.strip()


def split_references(generation: str) -> Tuple[str, Dict[int, str]]:
    """``(body, {N: doc_id})``: the report without its last References block,
    and the block's ``[N] doc_id`` lines."""
    headings = list(_REFERENCES_HEADING_RE.finditer(generation))
    if not headings:
        return generation, {}
    start = headings[-1].start()
    refs: Dict[int, str] = {}
    for line in generation[headings[-1].end():].splitlines():
        m = _REFERENCE_LINE_RE.match(line.strip())
        if m:
            refs.setdefault(int(m.group(1)), m.group(2))
    return generation[:start], refs
