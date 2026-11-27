"""The checks of stage data_prepare every dataset runs: a repeated question
(``duplicate``), an empty answer, a non-answer or one the question states
(``answer_invalid``), no gold page (``no_gold``); and the text matching the
datasets' own checks use."""

import html
import re
from typing import Dict, List, Optional

from ..annotation.names import normalize
from .funnel import Check

# ── Text matching ───────────────────────────────────────────────────────────

_THOUSANDS = re.compile(r"(?<=\d),(?=\d{3}\b)")
_ORDINAL = re.compile(r"\b(\d+)(?:st|nd|rd|th)\b")
# Function words, left out of a name's content words.
STOPWORDS = frozenset("a an the of and or in on at for to by with from s".split())


def match_text(text: str) -> str:
    """*text* as the checks match it: HTML entities decoded, thousands
    separators removed ("1,148" -> "1148"), then lowercase ASCII-folded
    words (``annotation.names.normalize``)."""
    return normalize(_THOUSANDS.sub("", html.unescape(text)))


def has_phrase(matched: str, phrase: str) -> bool:
    """*phrase* (``match_text`` applied) occurs as whole words in
    *matched* (a ``match_text`` result)."""
    p = match_text(phrase)
    return bool(p) and f" {p} " in f" {matched} "


def words(text: str) -> List[str]:
    """The words of *text* (``match_text``) with ordinal suffixes dropped
    ("428th" -> "428") and a plural "s" dropped from words of more than
    three letters ("functions" -> "function")."""
    out = []
    for w in _ORDINAL.sub(r"\1", match_text(text)).split():
        out.append(w[:-1] if len(w) > 3 and w.endswith("s") and not w.isdigit() else w)
    return out


def coverage(name: str, page_words: set) -> float:
    """The share of *name*'s content words (``words`` without
    ``STOPWORDS``; all its words when it has no other) found in
    *page_words*."""
    ws = words(name)
    content = [w for w in ws if w not in STOPWORDS] or ws
    return sum(w in page_words for w in content) / len(content) if content else 0.0


# ── Shared checks ───────────────────────────────────────────────────────────

def question_key(text: str) -> str:
    """A question as repeated questions are matched: lower case, words only."""
    return re.sub(r"\W+", " ", text.lower()).strip()


def duplicate_check() -> Check:
    """A question whose text (``question_key``) an earlier record
    has: the first one is kept."""
    seen: Dict[str, str] = {}

    def check(record: dict) -> Optional[List[dict]]:
        key = question_key(record["question"])
        if key in seen:
            return [{"status": "duplicate", "duplicate_of": seen[key]}]
        seen[key] = record["id"]
        return None
    return check


# An answer that says the source holds no answer ("The contributors are
# not specified in the given text"), written by a dataset's generator.
_NON_ANSWER = re.compile(
    r"\b(?:(?:is|are|was|were) not (?:specified|mentioned|provided|given|stated|known|available|identified|found)"
    r"|(?:not|cannot|can ?not|could not) be (?:determined|found|identified|answered)"
    r"|no (?:information|answer|data) (?:is |was )?(?:available|provided|given|found)"
    r"|(?:in|from) the (?:given|provided) (?:text|context|information|passages?))\b", re.IGNORECASE)


def answer_check(record: dict) -> Optional[List[dict]]:
    """An empty answer, a non-answer (``_NON_ANSWER``: it says the source
    gives none), or one the question states as whole words.  Short and
    numeric answers ("2", "1915") are not looked for in the question: a
    number or a short word in a question is rarely its answer."""
    answer = match_text(record["answer"])
    if not answer:
        return [{"status": "answer_invalid", "why": "empty"}]
    if _NON_ANSWER.search(record["answer"]):
        return [{"status": "answer_invalid", "why": "non_answer"}]
    compact = answer.replace(" ", "")
    if len(compact) >= 4 and not compact.isdigit() and has_phrase(match_text(record["question"]), record["answer"]):
        return [{"status": "answer_invalid", "why": "in_question"}]
    return None


def no_gold_check(record: dict) -> Optional[List[dict]]:
    return None if record["gold"] else [{"status": "no_gold"}]
