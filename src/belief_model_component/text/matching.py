"""Text matching of the belief model: values and spans are compared on
their letters and digits only (case, whitespace and punctuation ignored)."""

from typing import Any, List, Tuple


def squash(text: Any) -> str:
    """*text* as a string with its whitespace runs made single spaces and
    its ends stripped ("" for None)."""
    return " ".join(str(text or "").split())


def alnum_index(text: str) -> Tuple[str, List[int]]:
    """*text* lowercased with everything but letters and digits removed, and
    the position in *text* of each character kept (to map a match back to
    the text's own words)."""
    keys, pos = [], []
    for i, ch in enumerate(text):
        for k in ch.lower():
            if k.isalnum():
                keys.append(k)
                pos.append(i)
    return "".join(keys), pos


def alnum(text: str) -> str:
    """*text* lowercased, letters and digits only."""
    return "".join(ch for ch in text.lower() if ch.isalnum())


def _unspaced(ch: str) -> bool:
    """Whether *ch* is of a script written without spaces between words
    (CJK ideographs, kana), where any character may start a word."""
    o = ord(ch)
    return 0x3040 <= o <= 0x30FF or 0x3400 <= o <= 0x9FFF or 0xF900 <= o <= 0xFAFF


def _word_edge(text: str, i: int, inner: str) -> bool:
    """Whether a match whose edge character is *inner* ends at a word
    boundary when *text*[i] is the character just outside it."""
    return not 0 <= i < len(text) or not text[i].isalnum() or _unspaced(text[i]) or _unspaced(inner)


def contains_words(text: str, name: str) -> bool:
    """Whether *name* occurs in *text* as whole words: compared on letters
    and digits only (``alnum``), and the match must start and end at a word
    boundary of *text*, so "12" is not found in "1912" nor "Ann" in "Anne"."""
    key = alnum(name)
    if not key or not text:
        return False
    keys, pos = alnum_index(text)
    j = keys.find(key)
    while j >= 0:
        a, b = pos[j], pos[j + len(key) - 1]
        if _word_edge(text, a - 1, text[a]) and _word_edge(text, b + 1, text[b]):
            return True
        j = keys.find(key, j + 1)
    return False
