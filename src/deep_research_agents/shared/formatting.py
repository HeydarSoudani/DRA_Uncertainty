"""Text the agents show and print: verbose loop lines, documents formatted
for the prompt, numbered citations, and the evidence summary of a reasoning path."""

import json
import re
from typing import Any, Callable, Dict, List, Optional, Tuple

from utils.documents import MAX_PASSAGE_TOKENS, doc_text, doc_title


# ===========================================================================
# Printing: verbose terminal output for agent loops
# ===========================================================================

def verbose_print(
    iter_num: int,
    component: str,
    message: str,
    *,
    agent_name: str = "",
    sub_iter: Optional[int] = None,
) -> None:
    """Print a consistently formatted verbose line.

    Format: ``[AgentName] [Iter N] [component]: message``
    """
    prefix = f"[{agent_name}] " if agent_name else ""
    iter_label = f"[Iter {iter_num}.{sub_iter}]" if sub_iter is not None else f"[Iter {iter_num}]"
    if len(message) > 2000:
        message = message[:2000] + "..."
    print(f"{prefix}{iter_label} [{component}]: {message}")


def verbose_print_search_results(
    iter_num: int,
    docs: List[Dict[str, Any]],
    *,
    agent_name: str = "",
    sub_iter: Optional[int] = None,
    top_n: int = 5,
    max_context_len: int = 100,
) -> None:
    """Print top search results in a compact format."""
    prefix = f"[{agent_name}] " if agent_name else ""
    iter_label = f"[Iter {iter_num}.{sub_iter}]" if sub_iter is not None else f"[Iter {iter_num}]"

    if not docs:
        print(f"{prefix}{iter_label} [ret result]: (no docs retrieved)")
        return

    for rank, doc in enumerate(docs[:top_n], 1):
        doc_id = doc.get("doc_id") or doc.get("id", "?")
        title = doc.get("title") or doc.get("metadata", {}).get("title", "") or "Untitled"
        context = doc.get("relevant_text") or doc.get("text") or doc.get("contents", "")
        if len(context) > max_context_len:
            context = context[:max_context_len] + "..."
        print(f"{prefix}{iter_label} [ret result]: [{rank}] {doc_id}, {title}, {context.replace(chr(10), ' ').strip()}")

    remaining = len(docs) - top_n
    if remaining > 0:
        print(f"{prefix}{iter_label} [ret result]: ... and {remaining} more docs")


# ===========================================================================
# Documents: formatting — convert doc dicts to strings for the LLM prompt
# ===========================================================================

def passages2string(passages: List[Dict[str, Any]], max_tokens: Optional[int] = MAX_PASSAGE_TOKENS) -> str:
    """Convert passage dicts to a numbered, human-readable string.

    Format: ``[1] Title\\ntext``

    Used by: ReSearch, SearchR1, SearchO1, and as the default format.
    """
    if not passages:
        return "No documents retrieved."

    parts = []
    for idx, passage in enumerate(passages, start=1):
        parts.append(f"[{idx}] {doc_title(passage)}\n{doc_text(passage, max_tokens)}")

    return "\n\n".join(parts)


def format_as_markdown(docs: List[Dict[str, Any]], top_k: int = 5, max_tokens: Optional[int] = MAX_PASSAGE_TOKENS) -> str:
    """Convert docs to markdown format with ``## Web Results`` header.

    Used by: Tongyi-DR agent.
    """
    if not docs:
        return "No results found."

    parts = []
    for i, doc in enumerate(docs[:top_k], 1):
        title = doc_title(doc)
        text = doc_text(doc, max_tokens)
        if title and text:
            parts.append(f"{i}. [{title}]\n{text}")
        elif text:
            parts.append(f"{i}. {text}")
        elif title:
            parts.append(f"{i}. [{title}]\n(no content available)")
        else:
            parts.append(f"{i}. (empty document)")

    return "## Web Results\n" + "\n\n".join(parts)


def format_as_json(docs: List[Dict[str, Any]], top_k: int = 5, max_tokens: Optional[int] = MAX_PASSAGE_TOKENS) -> str:
    """Convert docs to JSON format with docid/title/snippet.

    Used by: ``BasicAgent._format_search_results``, the default of every
    tool-calling agent (GLM, OSS, ...).
    """
    formatted = []
    for doc in docs[:top_k]:
        formatted.append({
            "docid": doc.get("doc_id", ""),
            "title": doc_title(doc),
            "snippet": doc_text(doc, max_tokens),
        })
    return json.dumps(formatted)


def format_as_snippets(
    docs: List[Dict[str, Any]],
    top_k: int = 5,
    snippet_offset: int = 0,
    max_tokens: Optional[int] = MAX_PASSAGE_TOKENS,
) -> tuple:
    """Convert docs to DR-Tulu ``<snippet id="SN">…</snippet>`` XML format.

    Used by: DrTulu agent.

    Returns:
        (xml_string, list_of_snippet_ids_assigned)
    """
    parts: List[str] = []
    ids: List[str] = []

    for i, doc in enumerate(docs[:top_k]):
        sid = f"S{snippet_offset + i + 1}"
        ids.append(sid)

        title = doc_title(doc, default="")
        url = doc.get("url", doc.get("doc_id", ""))
        body = doc_text(doc, max_tokens)
        content = f"Title: {title}\nURL: {url}\n{body}".strip()

        parts.append(f'<snippet id="{sid}">{content}</snippet>')

    return "\n".join(parts), ids


def number_citations(
    text: str, citation_re: re.Pattern, doc_of: Callable[[str], Optional[str]],
) -> Tuple[str, Dict[int, str]]:
    """Turn an agent's own citations into numbered ``[N]`` markers.

    Every match of *citation_re* is replaced by ``[N]`` markers, one per cited
    document, after the match's ``span`` group when the pattern has one (a
    tag that wraps the cited text).  The match's ``ids`` group holds the
    agent's ids (separated by commas, semicolons or spaces), resolved to doc
    ids by *doc_of*; an id it cannot resolve is dropped.  N counts the
    documents in order of first citation.

    Returns the text and ``{N: doc_id}``, the ``citation_to_doc_id`` that
    :func:`utils.citations.with_references_section` lists.
    """
    number_of: Dict[str, int] = {}

    def replace(m: re.Match) -> str:
        markers: List[str] = []
        for agent_id in re.split(r"[\s,;]+", m.group("ids").strip()):
            did = doc_of(agent_id) if agent_id else None
            if did:
                marker = f"[{number_of.setdefault(did, len(number_of) + 1)}]"
                if marker not in markers:
                    markers.append(marker)
        span = m.groupdict().get("span") or ""
        return span + "".join(markers)

    numbered = citation_re.sub(replace, text)
    return numbered, {n: did for did, n in number_of.items()}


# A bracketed list of document ids as an agent cites them: "[id]", "[[id]]",
# "[id1, id2]" (ids separated by commas, semicolons or spaces).
_DOCID_CITATION_RE = re.compile(r"\[\[?\s*(?P<ids>[^\[\]\n]{1,400}?)\s*\]?\]")
# The shortest id prefix accepted as a citation of the one document it starts.
MIN_DOCID_PREFIX = 6
# The heading of a reference list the agent wrote itself ("## References",
# "**Sources:**").
_OWN_REFERENCES_RE = re.compile(
    r"^[ \t]*(?:#{1,6}[ \t]*|\*\*)?(?:References|Sources|Bibliography|Works Cited)(?:[ \t]*:)?(?:\*\*)?(?:[ \t]*:)?[ \t]*$",
    re.MULTILINE | re.IGNORECASE)
# Bracketed numbers, as the [N] markers a numbered report is read with.
_NUMBER_BRACKET_RE = re.compile(r"\d+(?:\s*[,;]\s*\d+)*")


def number_docid_citations(text: str, doc_ids: List[str]) -> Tuple[str, Dict[int, str]]:
    """Turn bracketed document-id citations into numbered ``[N]`` markers.

    For agents told to cite documents by their docids: a bracket (single or
    double) is a citation when every id in it is one of *doc_ids* or a prefix
    of exactly one of them, at least ``MIN_DOCID_PREFIX`` characters long
    (agents shorten long ids).  N counts the documents in order of first
    citation.  Any other bracket is left as written, except bracketed
    numbers ("[2019]"), which get parentheses so they are never read as
    ``[N]`` markers.  A reference list the agent wrote at the end is
    dropped: the references are built from the returned mapping
    (:func:`utils.citations.with_references_section`).

    Returns the text and ``{N: doc_id}`` (``citation_to_doc_id``).
    """
    own = list(_OWN_REFERENCES_RE.finditer(text))
    if own:
        text = text[:own[-1].start()].rstrip()
    known = list(dict.fromkeys(d for d in doc_ids if d))
    exact = set(known)

    def resolve(agent_id: str) -> Optional[str]:
        if agent_id in exact:
            return agent_id
        if len(agent_id) < MIN_DOCID_PREFIX:
            return None
        hits = [d for d in known if d.startswith(agent_id)]
        return hits[0] if len(hits) == 1 else None

    number_of: Dict[str, int] = {}

    def replace(m: re.Match) -> str:
        ids = [x for x in re.split(r"[\s,;]+", m.group("ids").strip()) if x]
        docs = [resolve(x) for x in ids]
        if not docs or any(d is None for d in docs):
            if _NUMBER_BRACKET_RE.fullmatch(m.group("ids").strip()):
                return f"({m.group('ids').strip()})"
            return m.group(0)
        markers: List[str] = []
        for did in docs:
            marker = f"[{number_of.setdefault(did, len(number_of) + 1)}]"
            if marker not in markers:
                markers.append(marker)
        return "".join(markers)

    numbered = _DOCID_CITATION_RE.sub(replace, text)
    return numbered, {n: did for did, n in number_of.items()}


# ===========================================================================
# Documents: evidence — prune/summarise reasoning-path evidence for answer gen
# ===========================================================================

def reduce_reasoning_path(
    reasoning_path: List[Dict[str, Any]],
    keep_last: int = 3,
    middle_strategy: str = "strip_docs",
) -> List[Dict[str, Any]]:
    """Prune reasoning_path for context-limited answer generation.

    Returns a new list where recent search steps retain full docs and
    older search steps have docs removed or are dropped entirely.

    Args:
        keep_last: Number of most-recent search steps to keep with full docs.
        middle_strategy: ``"strip_docs"`` removes doc content from older
            steps but keeps think/query; ``"drop"`` removes them entirely.
    """
    search_indices = [
        i for i, step in enumerate(reasoning_path)
        if step.get("docs") or step.get("all_docs")
    ]

    if len(search_indices) <= keep_last:
        return list(reasoning_path)

    strip_indices = set(search_indices[:-keep_last])

    result: List[Dict[str, Any]] = []
    for i, step in enumerate(reasoning_path):
        if i in strip_indices:
            if middle_strategy == "drop":
                continue
            stripped = {k: v for k, v in step.items() if k not in ("docs", "all_docs")}
            stripped["docs"] = []
            stripped["_docs_stripped"] = True
            result.append(stripped)
        else:
            result.append(step)

    return result


def build_evidence_summary(
    reasoning_path: List[Dict[str, Any]],
    seen_top_k: int = 5,
    keep_last: int = 3,
    max_tokens: Optional[int] = MAX_PASSAGE_TOKENS,
) -> str:
    """Build windowed evidence summary from reasoning_path.

    Recent search steps (last ``keep_last``) include full document text.
    Older steps include only the reasoning and search query.
    """
    pruned = reduce_reasoning_path(reasoning_path, keep_last=keep_last)
    parts: List[str] = []
    for step in pruned:
        action = step.get("action_type", "")
        has_search = bool(step.get("search_query") or step.get("docs"))
        if not has_search and action != "search":
            continue
        sq = step.get("search_query", step.get("query", ""))
        think = step.get("think", "")
        is_stripped = step.get("_docs_stripped", False)
        if is_stripped:
            entry = f"- Search: {sq}"
            if think:
                entry += f"\n  Reasoning: {think}"
        else:
            docs = step.get("docs", [])
            doc_text = passages2string(docs[:seen_top_k], max_tokens)
            entry = f"- Search: {sq}"
            if think:
                entry += f"\n  Reasoning: {think}"
            entry += f"\n  Results:\n  {doc_text}"
        parts.append(entry)
    return "\n\n".join(parts) if parts else "(no evidence collected)"
