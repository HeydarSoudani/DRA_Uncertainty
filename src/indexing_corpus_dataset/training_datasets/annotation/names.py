"""Name matching of the gold annotation: how an entity name is found in a
text (a question, a belief text, a page head)."""

import re
import unicodedata
from typing import List


def _dedup(items) -> list:
    return list(dict.fromkeys(items))


def normalize(text: str) -> str:
    """Lowercase ASCII-folded words, punctuation removed, spaces collapsed."""
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


def strip_parenthetical(title: str) -> str:
    """A title without its disambiguation ("Mercury (planet)" -> "Mercury")."""
    return re.sub(r"\s*\([^)]*\)\s*$", "", title).strip()


def entity_names(entity: str) -> List[str]:
    """The names of an entity to match: as given, and without a trailing
    parenthetical."""
    return _dedup(n for n in (entity.strip(), strip_parenthetical(entity)) if n)


def mentions(text: str, name: str, others: List[str]) -> bool:
    """*text* contains *name* as whole words outside the longer names of
    *others* that contain it ("Groningen" inside "University of Groningen"
    does not count)."""
    n = normalize(name)
    if not n:
        return False
    padded = f" {normalize(text)} "
    for other in others:
        o = normalize(other)
        if n in o and o != n:
            padded = padded.replace(f" {o} ", " | ")
    return f" {n} " in padded
