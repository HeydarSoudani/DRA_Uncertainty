"""Documents as the agents, the belief model and the run files read them.

A document is a dict (``doc_id``/``id``, ``title`` or ``metadata.title``,
``relevant_text``/``text``/``contents``/``snippet``).  Pure Python, except the
cut (:func:`cut_passage`), which loads one tokenizer file with ``tokenizers``
on first use.
"""

import re
import threading
from typing import Any, Dict, List, Optional


def doc_id(doc) -> str:
    """The canonical id of a doc dict (``doc_id``, else ``id``) or a bare id
    string; "" when neither is set."""
    if isinstance(doc, str):
        return doc
    return doc.get("doc_id") or doc.get("id") or ""


def doc_title(doc: Dict[str, Any], default: str = "Untitled Document") -> str:
    """Extract document title, trying metadata then top-level."""
    metadata = doc.get("metadata")
    if isinstance(metadata, dict):
        title = metadata.get("title")
        if title:
            return str(title)
    return doc.get("title") or default


#: Tokens of one document shown to an agent and to the belief model
#: (``max_passage_tokens`` of a run, one shared setting): both read the same
#: text, the document's first tokens, cut by :func:`cut_passage`.  As in
#: BrowseComp-Plus (search results cut to their first 512 tokens), counted
#: with one fixed tokenizer, whatever the agent's model: the student belief
#: model's (its ``tokenizer.json``, read from the Hugging Face cache).
MAX_PASSAGE_TOKENS = 512
PASSAGE_TOKENIZER = "Qwen/Qwen3.5-4B"

# A first window of the text holds the first max_tokens tokens when
# tokenized alone (a token spans at least about one character); kept
# this far from the window's end, a token is not changed by the text
# after it.
_WINDOW_CHARS_PER_TOKEN = 8
_WINDOW_MARGIN_CHARS = 64

_passage_tokenizer = None
_passage_tokenizer_lock = threading.Lock()


def _tokenizer():
    """The tokenizer of the cut (``PASSAGE_TOKENIZER``), loaded once."""
    global _passage_tokenizer
    with _passage_tokenizer_lock:
        if _passage_tokenizer is None:
            from huggingface_hub import hf_hub_download
            from tokenizers import Tokenizer
            _passage_tokenizer = Tokenizer.from_file(hf_hub_download(PASSAGE_TOKENIZER, "tokenizer.json"))
        return _passage_tokenizer


def token_ends(text: str) -> List[int]:
    """The character offset where each token of *text* ends.  Encoding holds
    the lock too (the tokenizer is shared by the belief threads)."""
    tokenizer = _tokenizer()
    with _passage_tokenizer_lock:
        encoded = tokenizer.encode(text, add_special_tokens=False)
    return [end for _, end in encoded.offsets]


def cut_passage(text: str, max_tokens: Optional[int]) -> str:
    """*text* cut to its first *max_tokens* tokens (``PASSAGE_TOKENIZER``),
    with no marker; None keeps it whole.  The cut is a prefix of *text*."""
    if max_tokens is None or not text:
        return text
    if max_tokens < 1:
        raise ValueError(f"max_tokens must be at least 1 (None keeps the text whole), got {max_tokens}")
    # The window grows (doubles) until its tokens hold the cut far enough from
    # its end, or it is the whole text: a long document is never tokenized whole.
    window_chars = max_tokens * _WINDOW_CHARS_PER_TOKEN
    while True:
        window = text[:window_chars]
        ends = token_ends(window)
        if len(window) == len(text) or (
                len(ends) > max_tokens and ends[max_tokens - 1] <= len(window) - _WINDOW_MARGIN_CHARS):
            break
        window_chars *= 2
    return text[:ends[max_tokens - 1]] if len(ends) > max_tokens else text


def cut_chars(text: str, max_chars: Optional[int]) -> str:
    """*text* cut to *max_chars* characters, ending in ``...`` when cut
    (None keeps it whole): a short display text, e.g. a title."""
    return text[:max_chars] + "..." if max_chars is not None and len(text) > max_chars else text


def doc_text(doc: Dict[str, Any], max_tokens: Optional[int] = MAX_PASSAGE_TOKENS) -> str:
    """Extract document text, cut to its first *max_tokens* tokens
    (``None`` = full text)."""
    text = (
        doc.get("relevant_text")
        or doc.get("text")
        or doc.get("contents")
        or doc.get("snippet")
        or ""
    )
    return cut_passage(str(text).strip(), max_tokens)


def unique_docs(docs: List[Dict[str, Any]], *, keep_unidentified: bool = True) -> List[Dict[str, Any]]:
    """*docs* in order without repeated ids (``doc_id``), first occurrence
    kept.  Docs without an id are all kept, or all dropped with
    ``keep_unidentified=False`` (e.g. when every doc needs a label)."""
    seen = set()
    out = []
    for doc in docs:
        key = doc_id(doc)
        if not key:
            if keep_unidentified:
                out.append(doc)
            continue
        if key in seen:
            continue
        seen.add(key)
        out.append(doc)
    return out


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
                next((ln.strip() for ln in body.splitlines() if any(ch.isalnum() for ch in ln.lower())), ""))
    return cut_chars(line, max_chars)
