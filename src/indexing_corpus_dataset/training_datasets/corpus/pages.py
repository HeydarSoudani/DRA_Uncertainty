"""The gold page entries of a source record (``record["gold"]``), each with
a ``key`` that identifies the page across questions::

    {"key": "wiki:{lang}:{title}", "kind": "wiki", "lang": str, "title": str}
    {"key": "web:{url}", "kind": "web", "url": str}
    {"key": "doc:{id}", "kind": "doc", "id": str}        (trqa_val: a corpus id)
"""

import re
from typing import Dict, List, Optional
from urllib.parse import unquote, urlsplit, urlunsplit

_WIKI_HOST = re.compile(r"^(?:(?P<lang>[a-z][a-z0-9\-]*)\.)?(?:m\.)?wikipedia\.org$")
# Hosts without a language (www.wikipedia.org, m.wikipedia.org) serve English.
_NO_LANG = (None, "www", "m")


def normalize_title(title: str) -> str:
    """A Wikipedia title as MediaWiki displays it: decoded, spaces for
    underscores, no section anchor, first letter upper case."""
    title = unquote(title).replace("_", " ").split("#")[0]
    title = re.sub(r"\s+", " ", title).strip()
    return title[:1].upper() + title[1:]


def wiki_gold(title: str, lang: str = "en") -> Dict[str, str]:
    title = normalize_title(title)
    return {"key": f"wiki:{lang}:{title}", "kind": "wiki", "lang": lang, "title": title}


def url_gold(url: str) -> Optional[Dict[str, str]]:
    """The gold entry of a URL: a Wikipedia page (any language, mobile links
    included) or a web page; None for an empty or non-HTTP URL."""
    url = url.strip().strip("'\"")
    if not url:
        return None
    if "://" not in url:
        url = "https://" + url
    try:
        parts = urlsplit(url)
        parts.hostname  # noqa: B018 - validates a bracketed host
    except ValueError:
        return None
    if parts.scheme not in ("http", "https") or not parts.netloc:
        return None
    host = parts.netloc.lower().split("@")[-1].split(":")[0]
    match = _WIKI_HOST.match(host)
    if match:
        title = None
        if parts.path.startswith("/wiki/"):
            title = parts.path[len("/wiki/"):]
        else:
            found = re.search(r"(?:^|&)title=([^&]+)", parts.query)
            title = found.group(1) if found else None
        if title:
            lang = match.group("lang")
            return wiki_gold(title, "en" if lang in _NO_LANG else lang)
    url = urlunsplit((parts.scheme.lower(), parts.netloc.lower(), parts.path or "/", parts.query, ""))
    return {"key": f"web:{url}", "kind": "web", "url": url}


def dedup_gold(entries: List[Optional[Dict[str, str]]]) -> List[Dict[str, str]]:
    """*entries* without None and without a repeated key, in order."""
    seen, out = set(), []
    for entry in entries:
        if entry and entry["key"] not in seen:
            seen.add(entry["key"])
            out.append(entry)
    return out
