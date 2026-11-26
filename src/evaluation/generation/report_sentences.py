"""A report split into sentences, each with the documents it cites.

Auto-ARGUE judges a report sentence by sentence, every sentence carrying the
ids of the documents it cites (the NeuCLIR / RAGTIME run format).  Our
reports are markdown with ``[N]`` markers and a References block of
``[N] doc_id`` lines, so :func:`report_sentences` rebuilds that format from
``generation/{qid}.md`` alone:

* the References block gives ``N -> doc_id`` (``references.split_references``);
  markers without a reference line are dropped (counted in ``unmapped_markers``);
* headings, rules and empty lines are dropped, list markers and emphasis
  are removed, and every list item, table row or paragraph is split into
  sentences;
* a sentence keeps the markers inside it and those right after its final
  punctuation (``claim.[1][2] Next``); the markers are removed from its text;
* a bracketed number of :data:`MIN_NON_MARKER` or more that is not a
  reference (a year: ``in [2019]``) is text, not a marker.
"""

import re
from typing import Dict, List, Tuple

from .references import split_references

# [1], [1, 2], [1; 2]
_MARKER_RE = re.compile(r"\[(\d+(?:\s*[,;]\s*\d+)*)\]")
_MARKERS_RE = r"(?:\s*\[\d+(?:\s*[,;]\s*\d+)*\])*"
# Sentence end: final punctuation, closing quotes or brackets, the markers
# after it, then the next sentence's capital (or digit, quote, bracket) or
# the end of the text.
_SENTENCE_END_RE = re.compile(
    r"[.!?][\"'”’)]*(?P<markers>" + _MARKERS_RE + r")(?=\s+[\"“‘(\[]?[A-Z0-9]|\s*$)")
# A period that ends an abbreviation or an initial, not a sentence.
_ABBREVIATION_RE = re.compile(
    r"(?:\b[A-Z]|\b(?:e\.g|i\.e|etc|vs|approx|Mr|Mrs|Ms|Dr|Prof|St|Jr|Sr|Inc|Ltd|Co|No|Fig|Jan|Feb|Mar|Apr"
    r"|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec))\.$")
_HEADING_RE = re.compile(r"^\s*#{1,6}\s")
_RULE_RE = re.compile(r"^\s*(?:[-*_]\s*){3,}$|^\s*\|?\s*:?-{3,}")
_LIST_ITEM_RE = re.compile(r"^\s*(?:[-*+•]|\d+[.)])\s+")
_EMPHASIS_RE = re.compile(r"\*\*|__|`")
_TABLE_ROW_RE = re.compile(r"^\s*\|")
#: A bracketed number from this on is text (a year) unless it is a reference.
MIN_NON_MARKER = 1000


def _units(body: str) -> List[str]:
    """The paragraphs and list items of a markdown body, as plain text."""
    units: List[str] = []
    paragraph: List[str] = []

    def flush() -> None:
        if paragraph:
            units.append(" ".join(paragraph))
            paragraph.clear()

    for line in body.splitlines():
        if not line.strip() or _HEADING_RE.match(line) or _RULE_RE.match(line):
            flush()
            continue
        row = bool(_TABLE_ROW_RE.match(line))
        line = _EMPHASIS_RE.sub("", line).replace("|", " ").strip()
        if row or _LIST_ITEM_RE.match(line):
            flush()
            units.append(_LIST_ITEM_RE.sub("", line))
        else:
            paragraph.append(line)
    flush()
    return units


def _split_sentences(unit: str) -> List[str]:
    sentences, start = [], 0
    for m in _SENTENCE_END_RE.finditer(unit):
        if _ABBREVIATION_RE.search(unit[start:m.start() + 1]):
            continue
        sentences.append(unit[start:m.end()])
        start = m.end()
    sentences.append(unit[start:])
    return [s.strip() for s in sentences if s.strip()]


def _is_marker(group: str, refs: Dict[int, str]) -> bool:
    """Whether a bracketed group (``1, 2``) is a citation marker."""
    numbers = [int(n) for n in re.split(r"\s*[,;]\s*", group)]
    return any(n in refs or n < MIN_NON_MARKER for n in numbers)


def _clean(sentence: str, refs: Dict[int, str]) -> str:
    text = _MARKER_RE.sub(lambda m: "" if _is_marker(m.group(1), refs) else m.group(0), sentence)
    text = re.sub(r"\s+([.,;:!?])", r"\1", text)
    return re.sub(r"\s{2,}", " ", text).strip()


def report_sentences(generation: str) -> Tuple[List[Tuple[str, List[str]]], int]:
    """``([(sentence, cited doc ids), ...], unmapped_markers)`` of a report.

    Doc ids keep their first-citation order within a sentence.  A sentence
    without a letter or digit (a stray marker) is dropped.
    """
    body, refs = split_references(generation)
    sentences: List[Tuple[str, List[str]]] = []
    unmapped = 0
    for unit in _units(body):
        for raw in _split_sentences(unit):
            cited: List[str] = []
            for group in _MARKER_RE.findall(raw):
                if not _is_marker(group, refs):
                    continue
                for n in re.split(r"\s*[,;]\s*", group):
                    doc = refs.get(int(n))
                    if doc is None:
                        unmapped += 1
                    elif doc not in cited:
                        cited.append(doc)
            text = _clean(raw, refs)
            if re.search(r"\w", text):
                sentences.append((text, cited))
    return sentences, unmapped
