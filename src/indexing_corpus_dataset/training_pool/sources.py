"""Step 1 loaders: the questions of each source in one record format.

Record (one JSON line per question)::

    {"id": "infoseek_00123", "source": "infoseek" | "webshaper" | "frames",
     "question": str, "answer": str,
     "gold": [{"key": str, "kind": "wiki", "lang": str, "title": str}
              | {"key": str, "kind": "web", "url": str}],
     "facts": [{"text": str, "pages": [title, ...]}],
     "meta": {...}}

``gold`` holds the source's gold pages (InfoSeek: the Wikipedia page of every
tree node; WebShaper: its URLs; FRAMES: its Wikipedia links), each with a
``key`` that identifies the page across questions.  ``facts`` holds the hop
facts the source ships (InfoSeek: the claim of every tree edge, with its two
endpoint pages; WebShaper: the formalization triples; FRAMES: none).
"""

import ast
import csv
import json
import re
import sys
from typing import Callable, Dict, Iterator, List, Optional
from urllib.parse import unquote, urlsplit, urlunsplit

from huggingface_hub import hf_hub_download

# The source files (Hugging Face dataset repos), pinned to a revision: the
# question ids of InfoSeek and FRAMES are row numbers.
INFOSEEK = ("Lk123/InfoSeek", "data/InfoSeek-Hard-18K.jsonl", "137a171bd9e6b31d20d0239d5d73269900fb4540")
WEBSHAPER = ("Alibaba-NLP/WebShaper", "data/main-00000-of-00001.jsonl", "5dd17c63c2d0f86d245c262bfd7bf444200e126f")
FRAMES = ("google/frames-benchmark", "test.tsv", "58d9fb6330f3ab1316d1eca12e5e8ef23dcc22ef")

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


def _dedup_gold(entries: List[Optional[Dict[str, str]]]) -> List[Dict[str, str]]:
    seen, out = set(), []
    for entry in entries:
        if entry and entry["key"] not in seen:
            seen.add(entry["key"])
            out.append(entry)
    return out


def _hf_file(repo: str, filename: str, revision: str) -> str:
    return hf_hub_download(repo_id=repo, filename=filename, repo_type="dataset", revision=revision)


def _iter_jsonl(path: str) -> Iterator[dict]:
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


# ── InfoSeek ────────────────────────────────────────────────────────────────

def _tree_nodes(node: dict) -> Iterator[dict]:
    yield node
    for child in node.get("children") or []:
        yield from _tree_nodes(child)


def _claim_text(claim: str, ids: Dict[str, str]) -> str:
    """A claim ("A was born in B") with its node ids replaced by entities."""
    for node_id, entity in ids.items():
        claim = re.sub(rf"\b{re.escape(node_id)}\b", entity, claim)
    return claim


def load_infoseek() -> Iterator[dict]:
    """InfoSeek-Hard: the root question of each research tree; the answer is
    the root entity."""
    for i, row in enumerate(_iter_jsonl(_hf_file(*INFOSEEK))):
        root = row["root"]
        nodes = list(_tree_nodes(root))
        by_id = {n["id"]: n for n in nodes}
        gold = _dedup_gold([wiki_gold(n["href"]) for n in nodes if n.get("href")])
        facts = []
        for node in nodes:
            for claim in node.get("claims") or []:
                target = by_id.get(claim.get("target_id"))
                ends = [n for n in (node, target) if n]
                ids = {n["id"]: n["entity"] for n in ends}
                facts.append({
                    "text": _claim_text(claim["claim"], ids),
                    "pages": [normalize_title(n["href"]) for n in ends if n.get("href")],
                })
        yield {
            "id": f"infoseek_{i:05d}", "source": "infoseek",
            "question": root["question"].strip(), "answer": root["entity"].strip(),
            "gold": gold, "facts": facts,
            "meta": {"vertices": row.get("vertices")},
        }


# ── WebShaper ───────────────────────────────────────────────────────────────

def load_webshaper() -> Iterator[dict]:
    """WebShaper: questions with their source URLs and formalization triples."""
    for row in _iter_jsonl(_hf_file(*WEBSHAPER)):
        triples = row.get("formalization") or []
        entries = [url_gold(u) for u in row.get("urls") or []]
        yield {
            "id": f"webshaper_{row['id']}", "source": "webshaper",
            "question": row["question"].strip().strip('"').strip(), "answer": str(row["answer"]).strip(),
            "gold": _dedup_gold(entries),
            "facts": [{"text": " | ".join(map(str, t)), "pages": []} for t in triples],
            "meta": {"formalization": triples, "ignored_urls": entries.count(None)},
        }


# ── FRAMES ──────────────────────────────────────────────────────────────────

def _frames_links(raw: str) -> List[str]:
    try:
        links = ast.literal_eval(raw)
        if isinstance(links, (list, tuple)):
            return [str(x) for x in links]
    except (ValueError, SyntaxError):
        pass
    return re.findall(r"https?://[^\s'\",\]]+", raw or "")


def load_frames() -> Iterator[dict]:
    """FRAMES: questions with their Wikipedia links and reasoning types."""
    csv.field_size_limit(sys.maxsize)
    with open(_hf_file(*FRAMES), encoding="utf-8", newline="") as fh:
        for i, row in enumerate(csv.DictReader(fh, delimiter="\t")):
            entries = [url_gold(u) for u in _frames_links(row.get("wiki_links", ""))]
            yield {
                "id": f"frames_{i:04d}", "source": "frames",
                "question": row["Prompt"].strip(), "answer": row["Answer"].strip(),
                "gold": _dedup_gold(entries),
                "facts": [],
                "meta": {"ignored_urls": entries.count(None), "reasoning_types": [t.strip() for t in (row.get("reasoning_types") or "").split("|") if t.strip()]},
            }


# The sources, in merge order (also the order of the questions in the
# output); a duplicate question keeps its first source.
LOADERS: Dict[str, Callable[[], Iterator[dict]]] = {
    "webshaper": load_webshaper,
    "frames": load_frames,
    "infoseek": load_infoseek,
}
