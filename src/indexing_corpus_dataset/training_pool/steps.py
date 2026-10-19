"""The steps of the training pool build.

    1  merge the sources into one record format, drop duplicate questions and
       the fixed question types (``filters.DROP_FRAMES_REASONING_TYPES``)
    2  fetch every gold page from its source (Wikipedia API, live web or the
       Wayback Machine), then Filter A (``filters.py``): keep a question when
       every gold page was fetched or, with web gold pages, at least
       ``pages.min_page_share`` of them; write the kept questions, the qrels
       of their fetched gold pages and those pages in the canonical layout
    3  stats of the dataset as written (queries, qrels, corpus) per source,
       and consistency checks between the three files

Step 2 is incremental, like ``dra_inference.py``: a question is processed
once Filter A has a final verdict (``filters.verdict``) from the rows of its
gold pages in ``pages/index.jsonl``; a run skips the processed questions and
takes the next ``limit`` of the others (of ``sources`` only, when given;
otherwise in step 1 order).  Transient failures (transport errors, HTTP 429 /
5xx) leave a question unprocessed, so a later run retries it, up to
``filters.MAX_ATTEMPTS`` fetches of the page.  The dataset files are
rewritten after every run from all the processed questions, then split into
sft / rl / val (``splits.py``).  The page settings that decide a question's
verdict (``PAGE_SETTINGS``) must stay those of the earlier runs.
"""

import hashlib
import json
import logging
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple

from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from utils.io_utils import atomic_write_text, iter_jsonl, read_jsonl, write_jsonl

from .config import PoolConfig, PoolPaths
from .filters import failed_pages, failed_transiently, filter_a, is_transient, type_filter, verdict
from .http_client import HttpClient, error_text, failure_status
from .sources import LOADERS
from .splits import check_shares, write_splits
from .stats import (dataset_stats, drop_stats, format_dataset_table, page_stats, question_stats, read_stats,
                    write_stats)
from .web_pages import WebFetcher
from .wikipedia_api import WikipediaClient, html_to_text

log = logging.getLogger(__name__)

# Titles per Action API call (its limit).
_RESOLVE_BATCH = 50
# Off a terminal: one status line per this share of the questions, or after
# this many seconds without one.
_LINE_EVERY_SHARE = 0.01
_LINE_EVERY_SECONDS = 60.0
# The ``pages`` settings that decide a page's status or a question's verdict:
# an incremental run must use those of the earlier runs (``--force`` to
# override).  The worker counts and rates may change between runs.
PAGE_SETTINGS = ("wiki_lang", "min_words", "min_page_share", "wayback", "wayback_timestamp")


def _drop_partial_line(path: Path) -> None:
    """Cut a last line left unfinished by a killed run, so that appended
    rows start on a line of their own (the file is read from its end)."""
    if not path.exists() or path.stat().st_size == 0:
        return
    with open(path, "rb+") as fh:
        end = fh.seek(0, 2)
        fh.seek(end - 1)
        if fh.read(1) == b"\n":
            return
        pos, size = end, 1 << 16
        while pos > 0:
            start = max(0, pos - size)
            fh.seek(start)
            cut = fh.read(pos - start).rfind(b"\n")
            if cut >= 0:
                fh.truncate(start + cut + 1)
                break
            pos = start
        else:
            fh.truncate(0)
    log.warning("%s: removed an unfinished last line", path)


def _write_dropped(paths: PoolPaths, step: int, dropped: List[dict]) -> None:
    """Replace the rows of *step* in ``dropped.jsonl``, the other steps' kept."""
    others = [r for r in read_jsonl(paths.dropped) if r["step"] != step] if paths.dropped.exists() else []
    write_jsonl(paths.dropped, others + [{"step": step, **d} for d in dropped])


# ── Step 1 ──────────────────────────────────────────────────────────────────

def _question_key(text: str) -> str:
    return re.sub(r"\W+", " ", text.lower()).strip()


def _merge() -> Tuple[List[dict], List[dict]]:
    """The questions of every source, the duplicates dropped (the first
    source of a question keeps it)."""
    records, dropped, seen = [], [], {}
    for source, load in LOADERS.items():
        n = 0
        for record in load():
            key = _question_key(record["question"])
            if key in seen:
                dropped.append({"id": record["id"], "source": source, "reasons": ["duplicate"],
                                "duplicate_of": seen[key]})
                continue
            seen[key] = record["id"]
            records.append(record)
            n += 1
        log.info("step 1: %s, %d questions", source, n)
    return records, dropped


def run_step1(paths: PoolPaths) -> None:
    merged, duplicates = _merge()
    kept, by_type = type_filter(merged)
    write_jsonl(paths.questions, kept)
    _write_dropped(paths, 1, duplicates + by_type)
    write_stats(paths.stats, "step1", {"questions_merged": len(merged) + len(duplicates), "kept": question_stats(kept),
                                 "dropped": drop_stats(duplicates + by_type)})
    log.info("step 1: kept %d of %d questions (%d duplicates, %d by type) -> %s",
             len(kept), len(merged) + len(duplicates), len(duplicates), len(by_type), paths.questions)


# ── Step 2 ──────────────────────────────────────────────────────────────────

class _PageStore:
    """``pages/index.jsonl`` (append, latest row per key wins) and
    ``pages/text.jsonl`` (the text of each fetched document, latest row per
    id wins).  A transient failure row counts the fetches of its page in
    ``attempts``.  Text that cannot be encoded (lone surrogates) is written
    with replacement characters."""

    def __init__(self, paths: PoolPaths):
        self.rows: Dict[str, dict] = {}
        # Page id -> a fetched row of that Wikipedia page (its text is stored).
        self.wiki_docs: Dict[int, dict] = {}
        paths.page_index.parent.mkdir(parents=True, exist_ok=True)
        for path in (paths.page_index, paths.page_text):
            _drop_partial_line(path)
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

    def __init__(self, records: List[dict], store: _PageStore, pending: Iterable[str], wiki_http: HttpClient,
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
        self._line_every = max(1, int(len(records) * _LINE_EVERY_SHARE))
        self._lock = threading.Lock()
        self._tty = sys.stderr.isatty()
        self._bar = tqdm(total=len(records), desc="[step 2]", unit="q", dynamic_ncols=True, disable=not self._tty,
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
        if not force and now - self._last_line < _LINE_EVERY_SECONDS:
            return
        self._last_line = now
        n, total = len(self._finished), self._bar.total
        elapsed = now - self._start
        eta = tqdm.format_interval(elapsed / n * (total - n)) if n else "?"
        print(f"[step 2] {n}/{total} questions ({100 * n / max(total, 1):.0f}%)"
              f" [{tqdm.format_interval(elapsed)}<{eta}] {self._status()}", flush=True)

    def close(self) -> None:
        with self._lock:
            self.phase = "done"
            self._show(force=True)
        self._bar.close()


def _resolve_wiki(golds: List[dict], store: _PageStore, wiki: WikipediaClient, workers: int,
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


def _fetch_pages(by_page: Dict[int, List[dict]], web_golds: List[dict], store: _PageStore, wiki: WikipediaClient,
                 fetcher: WebFetcher, cfg: PoolConfig, progress: _Progress) -> None:
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


_ID_PREFIX = b'{"id": "'


def _row_id(line: bytes) -> Optional[str]:
    """The id of a ``pages/text.jsonl`` row, read from its prefix as
    ``add_text`` writes it, without parsing the text (the full row otherwise)."""
    if line.startswith(_ID_PREFIX):
        end = line.find(b'"', len(_ID_PREFIX))
        if end > 0 and b"\\" not in line[len(_ID_PREFIX):end]:
            return line[len(_ID_PREFIX):end].decode("utf-8")
    return json.loads(line)["id"] if line.strip() else None


def _write_dataset(records: List[dict], pages: Dict[str, dict], paths: PoolPaths) -> int:
    """The kept questions (queries), their fetched gold pages (binary qrels)
    and the text of those pages (corpus) in the canonical layout, each file
    replaced atomically; returns the number of documents.  A document's text
    is its last row in ``pages/text.jsonl`` (a page fetched again after a
    crash between its text and its index rows has two)."""
    write_jsonl(paths.queries, ({"id": r["id"], "text": r["question"], "answer": r["answer"], "source": r["source"]}
                                for r in records))
    docs: Dict[str, dict] = {}
    qrels = []
    for record in records:
        fetched = [pages[g["key"]] for g in record["gold"] if pages[g["key"]]["status"] == "ok"]
        for doc_id in dict.fromkeys(page["doc_id"] for page in fetched):
            qrels.append(f"{record['id']} 0 {doc_id} 1\n")
        for page in fetched:
            docs.setdefault(page["doc_id"], page)
    atomic_write_text(paths.qrels, "".join(qrels))

    # Byte offset of the last text row of each document, then those rows only.
    offsets: Dict[str, int] = {}
    if paths.page_text.exists():
        with open(paths.page_text, "rb") as fh:
            offset = 0
            for line in fh:
                doc_id = _row_id(line)
                if doc_id in docs:
                    offsets[doc_id] = offset
                offset += len(line)
    paths.corpus.parent.mkdir(parents=True, exist_ok=True)
    tmp = paths.corpus.with_name(f".{paths.corpus.name}.tmp")
    with open(tmp, "w", encoding="utf-8", errors="replace") as out:
        if offsets:
            with open(paths.page_text, "rb") as fh:
                for doc_id, offset in sorted(offsets.items(), key=lambda item: item[1]):
                    fh.seek(offset)
                    item = json.loads(fh.readline())
                    # {id, contents} as the other corpora (layout.py), plus the
                    # page url; a wiki page's title is the first line of its text.
                    out.write(json.dumps({"id": doc_id, "contents": item["text"], "url": docs[doc_id]["url"]},
                                         ensure_ascii=False) + "\n")
    os.replace(tmp, paths.corpus)
    return len(offsets)


def _check_settings(cfg: PoolConfig, paths: PoolPaths, force: bool) -> None:
    """Refuse page settings (``PAGE_SETTINGS``) other than those of the
    earlier step 2 runs, and split shares other than the recorded ones,
    unless *force*."""
    recorded = read_stats(paths.stats).get("step2", {}).get("config", {}).get("pages", {})
    pages = cfg.to_dict()["pages"]
    diffs = [f"pages.{k}: {recorded[k]!r} -> {pages[k]!r}" for k in PAGE_SETTINGS
             if k in recorded and recorded[k] != pages[k]]
    if diffs and not force:
        raise SystemExit(f"{paths.root}: the earlier step 2 runs used other page settings ({'; '.join(diffs)}); "
                         "their pages and verdicts would mix with this run's. "
                         "Restore the settings, or pass --force")
    check_shares(paths, cfg.split, force)


def run_step2(cfg: PoolConfig, paths: PoolPaths, limit: Optional[int] = None,
              sources: Optional[List[str]] = None, force: bool = False) -> None:
    if not cfg.shared.contact:
        raise SystemExit("step 2 needs shared.contact (e-mail or URL for the User-Agent, "
                         "as the Wikimedia User-Agent policy asks): DRA_CONTACT in .env or shared.contact in the config")
    if not paths.questions.exists():
        raise SystemExit(f"{paths.questions} not found: run step 1 first")
    _check_settings(cfg, paths, force)
    config = cfg.to_dict()
    del config["shared"]["contact"]  # personal: stays in .env
    # The settings first, so a run that stops before its stats still records them.
    write_stats(paths.stats, "step2", {**read_stats(paths.stats).get("step2", {}), "config": config})
    records = read_jsonl(paths.questions)
    store = _PageStore(paths)
    min_page_share = cfg.pages.min_page_share

    def processed(record: dict) -> bool:
        return verdict(record, store.rows, min_page_share) in ("kept", "dropped")

    # ── Resume + limit ──────────────────────────────────────────────────────
    todo = [r for r in records if not processed(r)]
    print(f"[step 2] {len(records)} questions: {len(records) - len(todo)} already processed, {len(todo)} remaining")
    if sources:
        unknown = sorted(set(sources) - set(LOADERS))
        if unknown:
            raise SystemExit(f"unknown source(s) {unknown}; expected some of {list(LOADERS)}")
        todo = [r for r in todo if r["source"] in sources]
        print(f"[step 2] this run: sources {', '.join(sources)}, {len(todo)} remaining")
    if limit is not None:
        todo = todo[:limit]
        print(f"[step 2] this run: the next {len(todo)} questions (--limit {limit})")
    golds: Dict[str, dict] = {}
    for record in todo:
        for gold in record["gold"]:
            if not store.done(gold["key"]):
                golds.setdefault(gold["key"], gold)
    wiki_golds = [g for g in golds.values() if g["kind"] == "wiki" and g["lang"] == cfg.pages.wiki_lang]
    other_lang = [g for g in golds.values() if g["kind"] == "wiki" and g["lang"] != cfg.pages.wiki_lang]
    web_golds = [g for g in golds.values() if g["kind"] == "web"]
    print(f"[step 2] {len(golds)} gold pages to fetch: {len(wiki_golds)} wiki, {len(web_golds)} web, "
          f"{len(other_lang)} other-language wiki (dropped)")

    # ── Fetch ───────────────────────────────────────────────────────────────
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
        store.close()
    v = progress.verdicts
    print(f"[step 2] this run: {sum(v.values())} of {len(todo)} questions finished: "
          f"kept {v['kept']}, dropped {v['dropped']}, retry next run {v['retry']}")

    # ── Dataset: all processed questions ────────────────────────────────────
    done = [r for r in records if processed(r)]
    pages = {g["key"]: store.rows[g["key"]] for r in done for g in r["gold"] if g["key"] in store.rows}
    kept, dropped = filter_a(done, pages, min_page_share)
    partial = [{"id": r["id"], "source": r["source"], "reasons": failed_pages(r, pages)} for r in kept]
    partial = [p for p in partial if p["reasons"]]
    n_docs = _write_dataset(kept, pages, paths)
    _write_dropped(paths, 2, dropped)
    write_stats(paths.stats, "step2", {
        "config": config,
        "questions": {"candidates": len(records), "processed": len(done),
                      "remaining": len(records) - len(done)},
        "pages": page_stats(pages.values()), "kept": question_stats(kept), "corpus_docs": n_docs,
        # Kept questions with failed gold pages (no qrels for those), per reason.
        "kept_with_failed_pages": drop_stats(partial),
        "dropped": drop_stats(dropped),
    })
    print(f"[step 2] dataset: {len(kept)} kept of {len(done)} processed questions "
          f"({len(partial)} with failed gold pages, {len(records) - len(done)} remaining), "
          f"{n_docs} gold pages -> {paths.root}")
    write_splits(paths, cfg.split, force)


# ── Step 3 ──────────────────────────────────────────────────────────────────

def run_step3(paths: PoolPaths) -> None:
    """Stats of the dataset as written by step 2 (queries, qrels, corpus):
    printed as a table and saved as the ``step3`` section of ``stats.json``."""
    for path in (paths.questions, paths.queries, paths.qrels, paths.corpus):
        if not path.exists():
            raise SystemExit(f"{path} not found: run steps 1 and 2 first")
    qrels: Dict[str, List[str]] = {}
    with open(paths.qrels, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                qid, _, doc_id, _ = line.split()
                qrels.setdefault(qid, []).append(doc_id)
    doc_words = {row["id"]: len(row["contents"].split()) for row in iter_jsonl(paths.corpus)}
    dropped = [r for r in iter_jsonl(paths.dropped) if r["step"] == 2] if paths.dropped.exists() else []
    stats = dataset_stats(read_jsonl(paths.questions), read_jsonl(paths.queries), qrels, doc_words, dropped)
    write_stats(paths.stats, "step3", stats)
    print(f"[step 3] dataset {paths.root}\n\n{format_dataset_table(stats)}\n")
