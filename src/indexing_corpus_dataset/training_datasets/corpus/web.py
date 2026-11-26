"""corpus_web: the web gold pages of webshaper, fetched and cut like
corpus_wiki (``wiki.py``), in ``TRAINING_DATASETS_ROOT/corpus/corpus_web.jsonl``
(``web_corpus_path``) until the user merges it into the corpus under
``DATA_ROOT``.

Fetch: each web gold page (a WebShaper URL that is not an English Wikipedia
page; other-language Wikipedia pages are fetched as web pages) live, else
its Wayback Machine snapshot closest to ``web.wayback_timestamp``
(``fetch/web_pages.py``).  One row per fetch in ``web_pages/index.jsonl``
(status, title, words; a transient failure counts its fetches in
``attempts``) and the extracted text of a fetched page in
``web_pages/text.jsonl``; the latest row of a URL wins.  A page whose fetch
failed transiently (transport error, HTTP 429 / 5xx, the archive not
reachable) is fetched again by the next builds, ``MAX_ATTEMPTS`` fetches in
all; then its failure is final.  ``web.refetch_failed`` fetches every failed
page once more (e.g. after a block was lifted); its new row wins.

corpus_web: every fetched page cut into passages of at most
``MAX_PASSAGE_TOKENS`` tokens (``utils.documents.PASSAGE_TOKENIZER``),
whole sentences where a sentence fits, the page title on the first line of
each passage as in corpus_wiki; a passage id is ``web_<url hash>_<k>``.
Written again (atomically) from the fetched pages when they or the cut
changed since it was written (``web_pages/corpus_web.json`` of the raw
folder records what it was written from), so its ids are stable.  It holds the web gold pages of WebShaper only: a web
passage a retriever returns is always some question's gold page.
"""

import hashlib
import json
import logging
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional

from tqdm import tqdm

from indexing_corpus_dataset.shared.layout import corpus_path
from utils.paths import TRAINING_DATASETS_ROOT
from utils.io import atomic_open, atomic_write_text, drop_partial_line, iter_jsonl
from utils.documents import MAX_PASSAGE_TOKENS, PASSAGE_TOKENIZER, token_ends

from ..config import DatasetPaths, WebConfig
from .fetch.http_client import HttpClient, error_text

log = logging.getLogger(__name__)

# The file stem of corpus_web (``web_corpus_path``).
WEB_CORPUS_NAME = "corpus_web"
# The ids of corpus_web's passages (``passage_id``).
WEB_ID_PREFIX = "web_"

# A page that failed transiently is fetched this many times in all; then
# its failure is final.
MAX_ATTEMPTS = 3
# Sentence ends inside a paragraph.
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[\"'“(\[A-Z0-9])")


def web_corpus_path() -> Path:
    """corpus_web: ``TRAINING_DATASETS_ROOT/corpus/corpus_web.jsonl``,
    until the user merges it into the corpus under ``DATA_ROOT``."""
    return corpus_path(TRAINING_DATASETS_ROOT, WEB_CORPUS_NAME)


def failed_transiently(row: dict) -> bool:
    """A failure a retry may fix: transport errors, HTTP 429 / 5xx, or a
    Wayback fallback that could not query the archive."""
    status = row["status"]
    return (status in ("error", "http_429") or status.startswith("http_5")
            or row.get("wayback_status") == "wayback_error")


def is_final(row: Optional[dict]) -> bool:
    """A page with a final outcome: fetched, failed for good, or failed
    transiently ``MAX_ATTEMPTS`` times."""
    return row is not None and (not failed_transiently(row) or row.get("attempts", 1) >= MAX_ATTEMPTS)


class PageStore:
    """``web_pages/index.jsonl`` and ``web_pages/text.jsonl`` (append, the
    latest row of a URL wins)."""

    def __init__(self, paths: DatasetPaths):
        self.paths = paths
        self.rows: Dict[str, dict] = {}
        paths.web_pages.parent.mkdir(parents=True, exist_ok=True)
        for path in (paths.web_pages, paths.web_text):
            drop_partial_line(path)
        if paths.web_pages.exists():
            self.rows = {r["url"]: r for r in iter_jsonl(paths.web_pages)}
        self._lock = threading.Lock()

    def add(self, row: dict, text: Optional[str]) -> None:
        """The text first: a crash in between leaves an unused text row,
        never a fetched row without its text."""
        with self._lock:
            if failed_transiently(row):
                prev = self.rows.get(row["url"])
                row["attempts"] = (prev.get("attempts", 1) if prev and failed_transiently(prev) else 0) + 1
            if text is not None:
                with open(self.paths.web_text, "a", encoding="utf-8", errors="replace") as fh:
                    fh.write(json.dumps({"url": row["url"], "text": text}, ensure_ascii=False) + "\n")
            with open(self.paths.web_pages, "a", encoding="utf-8", errors="replace") as fh:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            self.rows[row["url"]] = row

    def texts(self) -> Dict[str, str]:
        """The text of every fetched page (the latest row of a URL)."""
        if not self.paths.web_text.exists():
            return {}
        return {r["url"]: r["text"] for r in iter_jsonl(self.paths.web_text)}


def to_fetch(cfg: WebConfig, store: PageStore, urls: List[str]) -> List[str]:
    """The pages of *urls* to fetch: those without a final row and, with
    ``cfg.refetch_failed``, every one whose last fetch failed."""
    def due(url: str) -> bool:
        row = store.rows.get(url)
        return not is_final(row) or (cfg.refetch_failed and row["status"] != "ok")

    return [u for u in dict.fromkeys(urls) if due(u)]


def fetch_pages(cfg: WebConfig, store: PageStore, urls: List[str]) -> None:
    """Fetch the pages of *urls* that are due (``to_fetch``) into *store*."""
    todo = to_fetch(cfg, store, urls)
    print(f"[web] {len(set(urls))} web gold pages: {len(set(urls)) - len(todo)} with a final outcome, "
          f"{len(todo)} to fetch" + (" (refetch_failed: the failed ones again)" if cfg.refetch_failed else ""))
    if not todo:
        return
    # Trafilatura and pypdf only when pages are fetched.
    from .fetch.web_pages import WebFetcher
    http = HttpClient(cfg.contact, cfg.timeout, cfg.max_retries)
    wayback = HttpClient(cfg.contact, cfg.timeout, cfg.max_retries, cfg.wayback_rate_per_min) if cfg.wayback else None
    fetcher = WebFetcher(http, wayback, cfg.min_words, cfg.wayback_timestamp)
    # Per-page extraction warnings are noise: the outcome is the page status.
    for name, level in (("trafilatura", logging.CRITICAL), ("pypdf", logging.CRITICAL), ("urllib3", logging.ERROR)):
        logging.getLogger(name).setLevel(level)

    def one(url: str) -> None:
        try:
            result = fetcher.fetch(url)
        except Exception as exc:  # noqa: BLE001 - recorded, retried next run
            result = {"status": "error", "error": error_text(exc)}
        text = result.pop("text", None) if result.get("status") == "ok" else None
        result.pop("text", None)
        store.add({"url": url, **result}, text)

    with ThreadPoolExecutor(max_workers=max(1, cfg.workers)) as pool:
        futures = [pool.submit(one, u) for u in todo]
        for future in tqdm(as_completed(futures), total=len(futures), desc="[web]", dynamic_ncols=True, mininterval=10):
            future.result()
    statuses: Dict[str, int] = {}
    for u in todo:
        statuses[store.rows[u]["status"]] = statuses.get(store.rows[u]["status"], 0) + 1
    print(f"[web] fetched: {dict(sorted(statuses.items()))}")


# ── corpus_web ───────────────────────────────────────────────────────────────

def _pieces(unit: str, max_tokens: int) -> List[str]:
    """*unit* cut into pieces of at most *max_tokens* tokens."""
    ends = token_ends(unit)
    if len(ends) <= max_tokens:
        return [unit]
    out, start = [], 0
    for i in range(max_tokens - 1, len(ends), max_tokens):
        out.append(unit[start:ends[i]].strip())
        start = ends[i]
    if start < len(unit):
        out.append(unit[start:].strip())
    return [p for p in out if p]


def chunk_text(text: str, max_tokens: int = MAX_PASSAGE_TOKENS) -> List[str]:
    """*text* as passages of at most *max_tokens* tokens: its sentences in
    order, as many per passage as fit (a longer sentence cut by tokens)."""
    units = [p for line in text.splitlines() for s in _SENTENCE_END.split(line.strip()) if s.strip()
             for p in _pieces(s.strip(), max_tokens)]
    chunks: List[str] = []
    current: List[str] = []
    size = 0
    for unit in units:
        n = len(token_ends(unit))
        # A joining space is about one token.
        if current and size + 1 + n > max_tokens:
            chunks.append(" ".join(current))
            current, size = [], 0
        size += n + (1 if current else 0)
        current.append(unit)
    if current:
        chunks.append(" ".join(current))
    return chunks


def passage_id(url: str, k: int) -> str:
    return f"{WEB_ID_PREFIX}{hashlib.sha1(url.encode('utf-8')).hexdigest()[:12]}_{k}"


def _signature(store: PageStore) -> dict:
    """What corpus_web is written from: the fetched pages (the sizes of the
    append-only page files) and the cut (this module, the token limit and
    the tokenizer)."""
    paths = store.paths
    return {"index_bytes": paths.web_pages.stat().st_size if paths.web_pages.exists() else 0,
            "text_bytes": paths.web_text.stat().st_size if paths.web_text.exists() else 0,
            "code": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16],
            "max_tokens": MAX_PASSAGE_TOKENS, "tokenizer": PASSAGE_TOKENIZER}


def _read_web_corpus(path: Path) -> Dict[str, List[str]]:
    """The passage ids of each page of corpus_web (URL -> ids, in page order)."""
    out: Dict[str, List[str]] = {}
    for row in iter_jsonl(path):
        out.setdefault(row["url"], []).append(row["id"])
    return out


def write_web_corpus(store: PageStore) -> Dict[str, List[str]]:
    """corpus_web from every fetched page of *store*, replaced atomically
    (kept when it was written from the same pages and cut); returns the
    passage ids of each page (URL -> ids, in page order)."""
    path = web_corpus_path()
    record = store.paths.web_corpus_record
    signature = _signature(store)
    if path.exists() and record.exists():
        written = json.loads(record.read_text(encoding="utf-8"))
        if written.get("signature") == signature and written.get("corpus_bytes") == path.stat().st_size:
            out = _read_web_corpus(path)
            print(f"[web] corpus_web unchanged: {sum(map(len, out.values()))} passages of {len(out)} pages ({path})")
            return out
    texts = store.texts()
    ok = [u for u, r in store.rows.items() if r["status"] == "ok" and u in texts]
    out: Dict[str, List[str]] = {}
    with atomic_open(path, errors="replace") as fh:
        for url in sorted(ok):
            title = store.rows[url].get("title") or url
            ids = []
            for k, chunk in enumerate(chunk_text(texts[url]), 1):
                ids.append(passage_id(url, k))
                fh.write(json.dumps({"id": ids[-1], "contents": f"{title}\n{chunk}", "url": url},
                                    ensure_ascii=False) + "\n")
            out[url] = ids
    atomic_write_text(record, json.dumps({"signature": signature, "corpus_bytes": path.stat().st_size}, indent=2) + "\n")
    print(f"[web] corpus_web: {sum(map(len, out.values()))} passages of {len(out)} pages -> {path}")
    return out
