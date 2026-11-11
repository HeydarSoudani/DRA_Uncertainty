"""The pages stage of webshaper_frames: fetch every gold page of the next
``limit`` questions from its source (Wikipedia API, live web or the Wayback
Machine), then Filter A (``filters.py``): keep a question when every gold
page was fetched or, with web gold pages, at least ``pages.min_page_share``
of them; write the kept questions, one row each with its gold annotation
(``gold/annotations.py``: the skeletons with the fetched documents, the
criterion texts by an LLM, ``gold/criterion_texts.py``) and its split (every
query in ``val``: the dataset is validation only, ``splits.py``), the qrels
of their fetched gold pages in the canonical layout, and the corpus
(``corpus.py``: those pages and their hard negatives).

``PageStore`` (the fetched pages and their text) and the status-line
cadence are shared with the negatives stage.
"""

import hashlib
import json
import logging
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple

from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from utils.io_utils import atomic_write_text, drop_partial_line, iter_jsonl, read_jsonl

from ..config import BuildConfig, DatasetPaths
from ..gold import annotations
from ..gold.criterion_texts import check_writer, fill_texts, require_api_key
from ..splits import split_rows
from ..stats import drop_stats, page_stats, question_stats, read_stats, write_annotation_stats, write_dropped, write_stats
from .corpus import write_corpus
from .fetch.http_client import HttpClient, error_text, failure_status
from .fetch.web_pages import WebFetcher
from .fetch.wikipedia_api import WikipediaClient, html_to_text
from .filters import failed_pages, failed_transiently, filter_a, is_transient, verdict

log = logging.getLogger(__name__)

# Titles per Action API call (its limit).
_RESOLVE_BATCH = 50
# Off a terminal: one status line per this share of the questions (the
# negatives: gold pages), or after this many seconds without one.
LINE_EVERY_SHARE = 0.01
LINE_EVERY_SECONDS = 60.0
# The ``pages`` settings that decide a page's status or a question's verdict:
# an incremental build must use those of the earlier builds.  The worker counts and rates may change between runs.
PAGE_SETTINGS = ("wiki_lang", "min_words", "min_page_share", "wayback", "wayback_timestamp")

class PageStore:
    """``pages/index.jsonl`` (append, latest row per key wins) and
    ``pages/text.jsonl`` (the text of each fetched document, latest row per
    id wins).  A transient failure row counts the fetches of its page in
    ``attempts``.  Text that cannot be encoded (lone surrogates) is written
    with replacement characters."""

    def __init__(self, paths: DatasetPaths):
        self.rows: Dict[str, dict] = {}
        # Page id -> a fetched row of that Wikipedia page (its text is stored).
        self.wiki_docs: Dict[int, dict] = {}
        paths.page_index.parent.mkdir(parents=True, exist_ok=True)
        for path in (paths.page_index, paths.page_text):
            drop_partial_line(path)
        if paths.page_index.exists():
            for row in iter_jsonl(paths.page_index):
                self._remember(row)
        self._lock = threading.Lock()
        self._index = open(paths.page_index, "a", encoding="utf-8", errors="replace")
        self._text = open(paths.page_text, "a", encoding="utf-8", errors="replace")
        self.listener: Optional[Callable[[dict], None]] = None

    def _remember(self, row: dict) -> None:
        self.rows[row["key"]] = row
        if row["status"] == "ok" and row["kind"] == "wiki":
            self.wiki_docs.setdefault(row["pageid"], row)

    def done(self, key: str) -> bool:
        row = self.rows.get(key)
        return row is not None and not is_transient(row)

    def add_text(self, doc_id: str, text: str) -> None:
        """Before the rows of the page: a crash in between leaves an unused
        text row, never a row without its text."""
        with self._lock:
            self._text.write(json.dumps({"id": doc_id, "text": text}, ensure_ascii=False) + "\n")
            self._text.flush()

    def add(self, row: dict) -> None:
        with self._lock:
            if failed_transiently(row):
                prev = self.rows.get(row["key"])
                row["attempts"] = (prev.get("attempts", 1) if prev and failed_transiently(prev) else 0) + 1
            self._remember(row)
            self._index.write(json.dumps(row, ensure_ascii=False) + "\n")
            self._index.flush()
        if self.listener:
            self.listener(row)

    def close(self) -> None:
        self._index.close()
        self._text.close()


class _Progress:
    """Per-question progress of a run: a question finishes when its verdict
    is known (``_verdict``).  A bar on a terminal; otherwise a status line
    every 1% of the questions or every minute."""

    def __init__(self, records: List[dict], store: PageStore, pending: Iterable[str], wiki_http: HttpClient,
                 min_page_share: float):
        self._store = store
        self._min_page_share = min_page_share
        self._records = {r["id"]: r for r in records}
        self._by_key: Dict[str, List[str]] = {}
        for record in records:
            for key in dict.fromkeys(g["key"] for g in record["gold"]):
                self._by_key.setdefault(key, []).append(record["id"])
        pending = set(pending)
        self._waiting = {r["id"]: len({g["key"] for g in r["gold"]} & pending) for r in records}
        self._finished: set = set()
        self.verdicts = {"kept": 0, "dropped": 0, "retry": 0}
        self.pages = {"ok": 0, "failed": 0, "error": 0}
        self.phase = "start"
        self._wiki_http, self._wiki_requests0 = wiki_http, wiki_http.requests
        self._start = self._last_line = time.monotonic()
        self._line_every = max(1, int(len(records) * LINE_EVERY_SHARE))
        self._lock = threading.Lock()
        self._tty = sys.stderr.isatty()
        self._bar = tqdm(total=len(records), desc="[pages]", unit="q", dynamic_ncols=True, disable=not self._tty,
                         bar_format="{desc} {percentage:3.0f}%|{bar}| {n}/{total} [{elapsed}<{remaining}] {postfix}")
        with self._lock:
            for record in records:
                self._check(record["id"])
            self._show(force=True)

    def set_phase(self, phase: str) -> None:
        with self._lock:
            self.phase = phase
            self._show()

    def on_row(self, row: dict) -> None:
        with self._lock:
            if row["status"] == "ok":
                self.pages["ok"] += 1
            else:
                self.pages["error" if is_transient(row) else "failed"] += 1
            for qid in self._by_key.get(row["key"], ()):
                self._waiting[qid] -= 1
                self._check(qid)
            self._show()

    def _check(self, qid: str) -> None:
        if qid in self._finished:
            return
        outcome = verdict(self._records[qid], self._store.rows, self._min_page_share)
        if outcome == "dropped" or (self._waiting[qid] <= 0 and outcome != "pending"):
            self._finished.add(qid)
            self.verdicts[outcome] += 1
            self._bar.update(1)
            if not self._tty and len(self._finished) % self._line_every == 0:
                self._last_line = 0.0

    def _status(self) -> str:
        minutes = max(time.monotonic() - self._start, 1.0) / 60
        wiki_rate = (self._wiki_http.requests - self._wiki_requests0) / minutes
        v, p = self.verdicts, self.pages
        return (f"{self.phase} | kept {v['kept']} · dropped {v['dropped']} · retry {v['retry']}"
                f" | pages ok {p['ok']} · failed {p['failed']} · error {p['error']} | wiki {wiki_rate:.0f} req/min")

    def _show(self, force: bool = False) -> None:
        if self._tty:
            self._bar.set_postfix_str(self._status(), refresh=True)
            return
        now = time.monotonic()
        if not force and now - self._last_line < LINE_EVERY_SECONDS:
            return
        self._last_line = now
        n, total = len(self._finished), self._bar.total
        elapsed = now - self._start
        eta = tqdm.format_interval(elapsed / n * (total - n)) if n else "?"
        print(f"[pages] {n}/{total} questions ({100 * n / max(total, 1):.0f}%)"
              f" [{tqdm.format_interval(elapsed)}<{eta}] {self._status()}", flush=True)

    def close(self) -> None:
        with self._lock:
            self.phase = "done"
            self._show(force=True)
        self._bar.close()


def _resolve_wiki(golds: List[dict], store: PageStore, wiki: WikipediaClient, workers: int,
                  progress: _Progress) -> Dict[int, List[dict]]:
    """Resolve the titles, 50 per call; a row for each title that is missing,
    invalid or could not be resolved; the resolved golds grouped by page id
    (several titles can redirect to one page, fetched once)."""
    titles = list(dict.fromkeys(g["title"] for g in golds))
    batches = [titles[i:i + _RESOLVE_BATCH] for i in range(0, len(titles), _RESOLVE_BATCH)]

    def resolve(batch: List[str]) -> Dict[str, dict]:
        try:
            return wiki.resolve_batch(batch)
        except Exception as exc:  # noqa: BLE001 - recorded, retried on rerun (4xx: final)
            return {title: {"status": failure_status(exc, "error"), "error": error_text(exc)} for title in batch}

    resolved: Dict[str, dict] = {}
    progress.set_phase(f"resolve 0/{len(batches)}")
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        for i, future in enumerate(as_completed([pool.submit(resolve, b) for b in batches]), start=1):
            resolved.update(future.result())
            progress.set_phase(f"resolve {i}/{len(batches)}")

    by_page: Dict[int, List[dict]] = {}
    for gold in golds:
        info = resolved[gold["title"]]
        base = {"key": gold["key"], "kind": "wiki", "lang": gold["lang"], "title": gold["title"]}
        if info["status"] != "ok":
            store.add({**base, **info})
            continue
        by_page.setdefault(info["pageid"], []).append({**base, **info, "url": wiki.page_url(info["canonical_title"])})
    return by_page


def _fetch_pages(by_page: Dict[int, List[dict]], web_golds: List[dict], store: PageStore, wiki: WikipediaClient,
                 fetcher: WebFetcher, cfg: BuildConfig, progress: _Progress) -> None:
    """The Wikipedia pages and the web pages, two thread pools at once.  A
    Wikipedia page fetched by an earlier run under another title (a redirect)
    reuses that text.  A worker that fails unexpectedly records an ``error``
    row (retried on rerun) for its pages without a row of this run, so one
    page never aborts the run."""

    def guarded(fetch: Callable, arg, rows: List[dict]) -> None:
        before = {row["key"]: store.rows.get(row["key"]) for row in rows}
        try:
            fetch(arg)
        except Exception as exc:  # noqa: BLE001 - recorded, retried on rerun
            log.warning("unexpected error on %s: %s", rows[0]["key"], error_text(exc))
            for row in rows:
                if store.rows.get(row["key"]) is before[row["key"]]:
                    store.add({**row, "status": "error", "error": error_text(exc)})

    def fetch_wiki(pageid: int) -> None:
        rows = by_page[pageid]
        known = store.wiki_docs.get(pageid)
        if known is not None:
            for row in rows:
                store.add({**row, "status": "ok", "revid": known["revid"], "fetched_via": known["fetched_via"],
                           "words": known["words"], "doc_id": known["doc_id"]})
            return
        try:
            text = html_to_text(wiki.fetch_html(rows[0]["revid"]))
        except Exception as exc:  # noqa: BLE001 - recorded, retried on rerun (4xx: final)
            for row in rows:
                store.add({**row, "status": failure_status(exc, "fetch_error"), "error": error_text(exc)})
            return
        doc_id = f"wiki_{pageid}"
        store.add_text(doc_id, text)
        for row in rows:
            store.add({**row, "status": "ok", "fetched_via": "api", "words": len(text.split()), "doc_id": doc_id})

    def fetch_web(gold: dict) -> None:
        try:
            result = fetcher.fetch(gold["url"])
        except Exception as exc:  # noqa: BLE001 - recorded, retried on rerun
            result = {"status": "error", "error": error_text(exc)}
        text = result.pop("text", None)
        row = {"key": gold["key"], "kind": "web", "url": gold["url"], **result}
        if result["status"] == "ok":
            row["doc_id"] = f"web_{hashlib.sha1(gold['url'].encode('utf-8')).hexdigest()[:20]}"
            store.add_text(row["doc_id"], text)
        store.add(row)

    progress.set_phase(f"fetch {len(by_page)} wiki + {len(web_golds)} web pages")
    with ThreadPoolExecutor(max_workers=max(1, cfg.pages.wiki_workers)) as wiki_pool, \
            ThreadPoolExecutor(max_workers=max(1, cfg.pages.web_workers)) as web_pool:
        futures = ([wiki_pool.submit(guarded, fetch_wiki, pageid, by_page[pageid]) for pageid in by_page]
                   + [web_pool.submit(guarded, fetch_web, gold, [{"key": gold["key"], "kind": "web", "url": gold["url"]}])
                      for gold in web_golds])
        for future in as_completed(futures):
            future.result()


def _fetch(todo: List[dict], golds: Dict[str, dict], store: PageStore, cfg: BuildConfig) -> None:
    """Fetch the gold pages *golds* (key -> gold) of the questions *todo*
    into *store*: other-language Wikipedia pages get their final row, the
    other Wikipedia titles are resolved, then the pages fetched."""
    min_page_share = cfg.pages.min_page_share
    wiki_golds = [g for g in golds.values() if g["kind"] == "wiki" and g["lang"] == cfg.pages.wiki_lang]
    other_lang = [g for g in golds.values() if g["kind"] == "wiki" and g["lang"] != cfg.pages.wiki_lang]
    web_golds = [g for g in golds.values() if g["kind"] == "web"]
    print(f"[pages] {len(golds)} gold pages to fetch: {len(wiki_golds)} wiki, {len(web_golds)} web, "
          f"{len(other_lang)} other-language wiki (dropped)")

    shared = cfg.shared
    wiki_http = HttpClient(shared.contact, shared.timeout, shared.max_retries, cfg.pages.wiki_rate_per_min)
    web_http = HttpClient(shared.contact, shared.timeout, shared.max_retries)
    wayback_http = (HttpClient(shared.contact, shared.timeout, shared.max_retries, cfg.pages.wayback_rate_per_min)
                    if cfg.pages.wayback else None)
    wiki = WikipediaClient(wiki_http, cfg.pages.wiki_lang)
    fetcher = WebFetcher(web_http, wayback_http, cfg.pages.min_words, cfg.pages.wayback_timestamp)
    # Per-page extraction warnings are noise: the outcome is the page status.
    logging.getLogger("trafilatura").setLevel(logging.CRITICAL)
    logging.getLogger("pypdf").setLevel(logging.CRITICAL)
    logging.getLogger("urllib3").setLevel(logging.ERROR)

    progress = _Progress(todo, store, golds, wiki_http, min_page_share)
    store.listener = progress.on_row
    try:
        with logging_redirect_tqdm():
            for gold in other_lang:
                store.add({"key": gold["key"], "kind": "wiki", "lang": gold["lang"], "title": gold["title"],
                           "status": "other_language"})
            by_page = _resolve_wiki(wiki_golds, store, wiki, cfg.pages.wiki_workers, progress)
            _fetch_pages(by_page, web_golds, store, wiki, fetcher, cfg, progress)
    finally:
        progress.close()
    v = progress.verdicts
    print(f"[pages] this build: {sum(v.values())} of {len(todo)} questions finished: "
          f"kept {v['kept']}, dropped {v['dropped']}, retry next run {v['retry']}")


def _write_dataset(records: List[dict], pages: Dict[str, dict], paths: DatasetPaths,
                   cfg: BuildConfig) -> Tuple[List[dict], int, int]:
    """The rows of the kept questions (``annotations.row``) with their
    criterion texts (``criterion_texts.fill_texts``) and split
    (``splits.split_rows``: every query in ``val``) first, then the files, each replaced
    atomically: the rows (queries), their fetched gold pages (binary qrels)
    in the canonical layout, and the corpus (``corpus.write_corpus``: those
    pages and their hard negatives); returns the rows and the numbers of
    gold pages and hard negatives in the corpus."""
    docs_of = {key: [row["doc_id"]] for key, row in pages.items() if row["status"] == "ok"}
    rows = [annotations.row(r, docs_of) for r in records]
    annotations.check(rows)
    rows = split_rows(paths, cfg.split, fill_texts(cfg.gold_criteria, paths, rows), val_only=True)
    annotations.write(paths.queries, rows)
    atomic_write_text(paths.qrels, "".join(f"{row['id']} 0 {doc_id} 1\n" for row in rows for doc_id in row["doc_ids"]))
    gold_urls: Dict[str, str] = {}
    for record in records:
        for g in record["gold"]:
            if pages[g["key"]]["status"] == "ok":
                gold_urls.setdefault(pages[g["key"]]["doc_id"], pages[g["key"]]["url"])
    return (rows, *write_corpus(paths, gold_urls))


def _check_settings(cfg: BuildConfig, paths: DatasetPaths) -> None:
    """Refuse page settings (``PAGE_SETTINGS``) other than those of the
    earlier builds."""
    recorded = read_stats(paths.stats).get("pages", {}).get("config", {}).get("pages", {})
    pages = cfg.to_dict()["pages"]
    diffs = [f"pages.{k}: {recorded[k]!r} -> {pages[k]!r}" for k in PAGE_SETTINGS
             if k in recorded and recorded[k] != pages[k]]
    if diffs:
        raise SystemExit(f"{paths.root}: the earlier builds used other page settings ({'; '.join(diffs)}); "
                         "their pages and verdicts would mix with this build's. "
                         f"Restore the settings, or remove the built files in {paths.root} (keep "
                         "criteria_bank/, the silver criteria) and build the dataset again")


def run_pages(cfg: BuildConfig, paths: DatasetPaths, limit: Optional[int] = None,
           sources: Optional[List[str]] = None) -> None:
    check_writer(cfg.gold_criteria, paths)
    _check_settings(cfg, paths)
    records = read_jsonl(paths.questions)
    config = {k: v for k, v in cfg.to_dict().items() if k in ("shared", "pages")}
    del config["shared"]["contact"]  # personal: stays in .env
    # The settings first, so a build that stops before its stats still records them.
    write_stats(paths.stats, "pages", {**read_stats(paths.stats).get("pages", {}), "config": config})
    store = PageStore(paths)
    min_page_share = cfg.pages.min_page_share

    def processed(record: dict) -> bool:
        return verdict(record, store.rows, min_page_share) in ("kept", "dropped")

    # ── Resume + limit ──────────────────────────────────────────────────────
    todo = [r for r in records if not processed(r)]
    print(f"[pages] {len(records)} questions: {len(records) - len(todo)} already processed, {len(todo)} remaining")
    if sources:
        todo = [r for r in todo if r["source"] in sources]
        print(f"[pages] this build: sources {', '.join(sources)}, {len(todo)} remaining")
    if limit is not None:
        todo = todo[:limit]
        print(f"[pages] this build: the next {len(todo)} questions (--limit {limit})"
              + ("; no fetch, the dataset files are rewritten" if not todo else ""))
    if todo:
        # Before the fetch: the new questions get their texts at the end of the build.
        require_api_key(cfg.gold_criteria, "the criterion texts of the new questions")
    golds: Dict[str, dict] = {}
    for record in todo:
        for gold in record["gold"]:
            if not store.done(gold["key"]):
                golds.setdefault(gold["key"], gold)

    # ── Fetch ───────────────────────────────────────────────────────────────
    try:
        if todo:
            _fetch(todo, golds, store, cfg)
    finally:
        store.close()

    # ── Dataset: all processed questions ────────────────────────────────────
    done = [r for r in records if processed(r)]
    pages = {g["key"]: store.rows[g["key"]] for r in done for g in r["gold"] if g["key"] in store.rows}
    kept, dropped = filter_a(done, pages, min_page_share)
    partial = [{"id": r["id"], "source": r["source"], "reasons": failed_pages(r, pages)} for r in kept]
    partial = [p for p in partial if p["reasons"]]
    rows, n_docs, n_negatives = _write_dataset(kept, pages, paths, cfg)
    write_dropped(paths.dropped, "pages", dropped)
    write_annotation_stats(paths, cfg.gold_criteria, rows)
    write_stats(paths.stats, "pages", {
        "config": config,
        "questions": {"candidates": len(records), "processed": len(done),
                      "remaining": len(records) - len(done)},
        "pages": page_stats(pages.values()), "kept": question_stats(kept), "corpus_docs": n_docs,
        "corpus_hard_negatives": n_negatives,
        # Kept questions with failed gold pages (no qrels for those), per reason.
        "kept_with_failed_pages": drop_stats(partial),
        "dropped": drop_stats(dropped),
    })
    print(f"[pages] dataset: {len(kept)} kept of {len(done)} processed questions "
          f"({len(partial)} with failed gold pages, {len(records) - len(done)} remaining), "
          f"{n_docs} gold pages + {n_negatives} hard negatives in the corpus -> {paths.root}")
