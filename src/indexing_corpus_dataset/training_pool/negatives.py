"""Step 3 of deep_qa: hard negatives of the Wikipedia gold pages, from the
Wikipedia API.  Web gold pages get none.

Per gold page, the sources of ``negatives.sources`` fill its
``negatives.max_per_gold`` slots in order, each asked for
``max_per_gold * candidate_factor`` candidates, best first:

    morelike   the articles of the most similar text (CirrusSearch
               ``morelike:<title>``)
    category   the articles of the gold page's most specific visible
               categories (fewest articles first, at most ``_MAX_CATEGORIES``)

A candidate is skipped when it is gold for any query of the pool, a
disambiguation or list page, titled with the answer of a query of the gold
page (the title with or without its parenthetical), or shorter than
``pages.min_words`` once fetched.  Candidates are fetched as the gold pages
were (Parsoid HTML, ``html_to_text``) and their text appended to
``pages/text.jsonl``; a page that is the negative of several gold pages is
fetched once.

Memory: ``negatives/index.jsonl`` gets one row per gold page once it is done
(``{"gold", "title", "negatives": [{"id", "title", "url", "source", "rank",
"words"}]}``; fewer than ``max_per_gold`` when the candidates run out).  A
run skips the gold pages with a row and takes the next ``limit`` of the
others, in qrels order (gold pages of ``sources`` only, when given); a gold
page added by a later step 2 run comes after the earlier ones.  A transient
failure (transport error, HTTP 429 / 5xx) leaves its gold page without a row,
so a later run retries it.  The corpus and its id files are rewritten after
every run (``corpus.write_corpus``).  The settings (``_settings``) must stay
those of the earlier runs.
"""

import dataclasses
import json
import logging
import re
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Set, Tuple

import requests
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from utils.io_utils import read_jsonl

from .config import PoolConfig, PoolPaths
from .corpus import drop_partial_line, read_negatives, write_corpus
from .http_client import HttpClient, error_text, failure_status, is_permanent
from .stats import negative_stats, read_stats, write_stats
from .steps import PageStore
from .wikipedia_api import WikipediaClient, html_to_text

log = logging.getLogger(__name__)

# Categories asked for members per gold page, the most specific first.
_MAX_CATEGORIES = 5
# Titles of list-like pages (an index of entities, not an entity).
_LIST_PREFIXES = ("list of ", "lists of ", "index of ", "outline of ")
# Off a terminal: one status line per this share of the gold pages, or after
# this many seconds without one.
_LINE_EVERY_SHARE = 0.01
_LINE_EVERY_SECONDS = 60.0


def _norm(text) -> str:
    return re.sub(r"\s+", " ", str(text)).strip().lower()


def _bare(title: str) -> str:
    """*title* without a trailing parenthetical: ``Paris (band)`` -> ``Paris``."""
    return re.sub(r"\s*\([^()]*\)$", "", title)


def _transient(exc: Exception) -> bool:
    """A failure a later run may not meet again: transport error, HTTP 429 /
    5xx, MediaWiki ``maxlag``.  Anything else (an HTTP 4xx, an API error
    reply, a TLS failure, a parse error) is final, so a gold page is not
    retried forever."""
    if isinstance(exc, RuntimeError):
        return "maxlag" in str(exc)
    return (isinstance(exc, requests.RequestException) and not is_permanent(exc)
            and failure_status(exc, "error") == "error")


def _settings(cfg: PoolConfig) -> dict:
    """The settings that decide the negatives of a gold page: an incremental
    run must use those of the earlier runs."""
    return {"wiki_lang": cfg.pages.wiki_lang, "min_words": cfg.pages.min_words,
            **dataclasses.asdict(cfg.negatives)}


class _Memory:
    """``negatives/index.jsonl``: one row per processed gold page, appended."""

    def __init__(self, paths: PoolPaths):
        paths.negatives.parent.mkdir(parents=True, exist_ok=True)
        drop_partial_line(paths.negatives)
        self.rows = read_negatives(paths)
        self._lock = threading.Lock()
        self._fh = open(paths.negatives, "a", encoding="utf-8", errors="replace")

    def add(self, row: dict) -> None:
        with self._lock:
            self._fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            self._fh.flush()
            self.rows[row["gold"]] = row

    def close(self) -> None:
        self._fh.close()


def _gold_pages(paths: PoolPaths, store: PageStore) -> List[dict]:
    """The Wikipedia gold pages of the qrels in qrels order, each once:
    ``{"gold", "pageid", "title", "answers", "sources"}`` (the normalized
    answers and the sources of its queries)."""
    queries = {q["id"]: q for q in read_jsonl(paths.queries)}
    pages = {}
    for row in store.rows.values():
        if row["status"] == "ok" and row["kind"] == "wiki":
            pages.setdefault(row["doc_id"], row)
    golds: Dict[str, dict] = {}
    with open(paths.qrels, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            qid, _, doc_id, _ = line.split()
            if not doc_id.startswith("wiki_"):
                continue
            if doc_id not in golds:
                page = pages[doc_id]
                golds[doc_id] = {"gold": doc_id, "pageid": page["pageid"], "title": page["canonical_title"],
                                 "answers": set(), "sources": set()}
            query = queries[qid]
            golds[doc_id]["answers"].add(_norm(query["answer"]))
            golds[doc_id]["sources"].add(query["source"])
    return list(golds.values())


class _Finder:
    """The hard negatives of one gold page (``find``); thread-safe, shares
    the pages fetched in this run between gold pages."""

    def __init__(self, cfg: PoolConfig, wiki: WikipediaClient, store: PageStore, memory: _Memory,
                 gold_ids: Set[str]):
        self.cfg = cfg.negatives
        self.min_words = cfg.pages.min_words
        self.wiki = wiki
        self.store = store
        self.gold_ids = gold_ids
        # Page id -> {"id", "title", "url", "words"} of a page kept as a
        # negative (text stored); pages too short to keep.
        self.known: Dict[str, dict] = {n["id"]: {k: n[k] for k in ("id", "title", "url", "words")}
                                       for row in memory.rows.values() for n in row["negatives"]}
        self.too_short: Set[str] = set()
        # Candidate outcomes of this run, for the progress lines: fetched,
        # reused (fetched before), too_short, failed, skip_<reason>.
        self.counts: Counter = Counter()
        self._lock = threading.Lock()

    def _count(self, key: str) -> None:
        with self._lock:
            self.counts[key] += 1

    def snapshot(self) -> Counter:
        """A copy of ``counts``, safe to iterate while workers add to it."""
        with self._lock:
            return Counter(self.counts)

    def _candidates(self, source: str, gold: dict, limit: int) -> List[dict]:
        if source == "morelike":
            return self.wiki.more_like(gold["title"], limit)
        categories = sorted((c for c in self.wiki.categories(gold["pageid"]) if c["pages"] >= 2),
                            key=lambda c: (c["pages"], c["title"]))[:_MAX_CATEGORIES]
        members: List[dict] = []
        for category in categories:
            if len(members) >= limit:
                break
            members += self.wiki.category_members(category["title"], limit)
        return members

    def _skip(self, candidate: dict, gold: dict) -> Optional[str]:
        """Why *candidate* cannot be a negative of *gold*, or None."""
        title = _norm(candidate["title"])
        if f"wiki_{candidate['pageid']}" in self.gold_ids:
            return "gold"
        if candidate["disambiguation"]:
            return "disambiguation"
        if title.startswith(_LIST_PREFIXES):
            return "list"
        if title in gold["answers"] or _norm(_bare(candidate["title"])) in gold["answers"]:
            return "answer"
        return None

    def _fetch(self, candidate: dict) -> Optional[dict]:
        """The negative info of *candidate*, its text stored; None when it is
        too short or its fetch failed for good (a transient failure raises)."""
        doc_id = f"wiki_{candidate['pageid']}"
        if doc_id in self.known:
            self._count("reused")
            return self.known[doc_id]
        if doc_id in self.too_short:
            self._count("too_short")
            return None
        try:
            text = html_to_text(self.wiki.fetch_html(candidate["revid"]))
        except Exception as exc:  # noqa: BLE001 - final failures skip the candidate
            if _transient(exc):
                raise
            self._count("failed")
            return None
        words = len(text.split())
        if words < self.min_words:
            self.too_short.add(doc_id)
            self._count("too_short")
            return None
        self._count("fetched")
        self.store.add_text(doc_id, text)
        info = {"id": doc_id, "title": candidate["title"], "url": self.wiki.page_url(candidate["title"]),
                "words": words}
        self.known[doc_id] = info
        return info

    def find(self, gold: dict) -> dict:
        """The memory row of *gold*; a transient failure raises (no row)."""
        limit = self.cfg.max_per_gold * self.cfg.candidate_factor
        kept: List[dict] = []
        seen = {gold["pageid"]}
        for source in self.cfg.sources:
            if len(kept) >= self.cfg.max_per_gold:
                break
            try:
                candidates = self._candidates(source, gold, limit)
            except Exception as exc:  # noqa: BLE001 - a final failure leaves the source empty
                if _transient(exc):
                    raise
                log.warning("%s %s: %s", gold["gold"], source, error_text(exc))
                continue
            for rank, candidate in enumerate(candidates, start=1):
                if len(kept) >= self.cfg.max_per_gold:
                    break
                if candidate["pageid"] in seen:
                    continue
                seen.add(candidate["pageid"])
                reason = self._skip(candidate, gold)
                if reason:
                    self._count(f"skip_{reason}")
                    continue
                info = self._fetch(candidate)
                if info is not None:
                    kept.append({**info, "source": source, "rank": rank})
        return {"gold": gold["gold"], "title": gold["title"], "negatives": kept}


def _n(value: int) -> str:
    return f"{value:,}"


class _Progress:
    """The progress of a run, one update per finished gold page.  On a
    terminal: a bar and three status lines under it (negatives, candidates,
    run); otherwise the same as one log line every 1% of the gold pages or
    every minute.  ``summary`` prints the final status."""

    _BAR = "{desc} {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} gold pages [{elapsed}<{remaining}, {rate_fmt}]"

    def __init__(self, total: int, processed: int, pool: int, cfg: PoolConfig, finder: _Finder, http: HttpClient):
        self.total, self.processed0, self.pool = total, processed, pool
        self.max_per_gold, self.sources = cfg.negatives.max_per_gold, cfg.negatives.sources
        self.finder, self.http, self.requests0 = finder, http, http.requests
        self.done = self.retry = self.negatives = self.short = 0
        self.by_source: Counter = Counter()
        self.start = self.last_line = time.monotonic()
        self.line_every = max(1, int(total * _LINE_EVERY_SHARE))
        self.tty = sys.stderr.isatty()
        self.bars = []
        if self.tty:
            self.bars.append(tqdm(total=total, desc="[step 3]", dynamic_ncols=True, bar_format=self._BAR, unit="gold"))
            self.bars += [tqdm(total=0, position=i, dynamic_ncols=True, bar_format="{desc}", leave=False)
                          for i in (1, 2, 3)]
            self._refresh()

    def _lines(self) -> List[Tuple[str, str]]:
        """(label, text) of the three status lines."""
        c = self.finder.snapshot()
        done = max(self.done, 1)
        minutes = max(time.monotonic() - self.start, 1.0) / 60
        sources = " · ".join(f"{s} {_n(self.by_source[s])}" for s in self.sources)
        skipped = sum(v for k, v in c.items() if k.startswith("skip_"))
        reasons = ", ".join(f"{k[5:]} {_n(v)}" for k, v in sorted(c.items()) if k.startswith("skip_") and v)
        return [
            ("negatives", f"{_n(self.negatives)} ({self.negatives / done:.1f}/gold, {_n(self.short)} gold < "
                          f"{self.max_per_gold}) | {sources}"),
            ("candidates", f"fetched {_n(c['fetched'])} · reused {_n(c['reused'])} · too short "
                           f"{_n(c['too_short'])} · failed {_n(c['failed'])} | skipped {_n(skipped)}"
                           + (f" ({reasons})" if reasons else "")),
            ("run", f"done {_n(self.done)} · retry {_n(self.retry)} | pool {_n(self.processed0 + self.done)}/"
                    f"{_n(self.pool)} gold pages | wiki {(self.http.requests - self.requests0) / minutes:.0f} req/min"),
        ]

    def _aligned(self) -> List[str]:
        return [f"{label:<10} {text}" for label, text in self._lines()]

    def _refresh(self) -> None:
        for bar, line in zip(self.bars[1:], self._aligned()):
            bar.set_description_str("  " + line, refresh=True)

    def update(self, row: Optional[dict]) -> None:
        if row is None:
            self.retry += 1
        else:
            self.done += 1
            self.negatives += len(row["negatives"])
            self.short += len(row["negatives"]) < self.max_per_gold
            self.by_source.update(n["source"] for n in row["negatives"])
        if self.tty:
            self.bars[0].update(1)
            self._refresh()
            return
        n, now = self.done + self.retry, time.monotonic()
        if n % self.line_every == 0 or now - self.last_line >= _LINE_EVERY_SECONDS or n == self.total:
            self.last_line = now
            eta = tqdm.format_interval((now - self.start) / n * (self.total - n))
            print(f"[step 3] {n}/{self.total} gold pages ({100 * n / max(self.total, 1):.0f}%) "
                  f"[{tqdm.format_interval(now - self.start)}<{eta}] "
                  + " || ".join(f"{label} {text}" for label, text in self._lines()), flush=True)

    def close(self) -> None:
        for bar in reversed(self.bars):
            bar.close()

    def summary(self) -> None:
        print(f"[step 3] this run: {self.done} of {self.total} gold pages done, {self.retry} retry next run "
              f"({tqdm.format_interval(time.monotonic() - self.start)})")
        for line in self._aligned():
            print(f"[step 3]   {line}")


def _check_settings(cfg: PoolConfig, paths: PoolPaths) -> dict:
    """The settings of this run (``_settings``); refused when the earlier
    step 3 runs recorded other ones."""
    settings = _settings(cfg)
    recorded = read_stats(paths.stats).get("step3", {}).get("config")
    if recorded is not None and recorded != settings:
        diffs = [f"{k}: {recorded.get(k)!r} -> {settings.get(k)!r}"
                 for k in sorted(set(recorded) | set(settings)) if recorded.get(k) != settings.get(k)]
        raise SystemExit(f"{paths.root}: the earlier step 3 runs used other settings ({'; '.join(diffs)}); "
                         "their negatives would mix with this run's. Restore the settings, or remove "
                         f"{paths.negatives} and the 'step3' section of {paths.stats} to start step 3 over")
    return settings


def run_step3(cfg: PoolConfig, paths: PoolPaths, limit: Optional[int] = None,
              sources: Optional[List[str]] = None) -> None:
    if not cfg.shared.contact:
        raise SystemExit("step 3 needs shared.contact (e-mail or URL for the User-Agent, "
                         "as the Wikimedia User-Agent policy asks): DRA_CONTACT in .env or shared.contact in the config")
    for path in (paths.queries, paths.qrels, paths.page_index):
        if not path.exists():
            raise SystemExit(f"{path} not found: run step 2 first")
    settings = _check_settings(cfg, paths)
    previous = read_stats(paths.stats).get("step3", {})
    # The settings first, so a run that stops before its stats still records
    # them (an earlier section without them is not step 3's).
    write_stats(paths.stats, "step3", {**(previous if "config" in previous else {}), "config": settings})

    store = PageStore(paths)
    memory = _Memory(paths)
    golds = _gold_pages(paths, store)
    with open(paths.qrels, encoding="utf-8") as fh:
        # Every gold page of the pool (wiki and web), in qrels order.
        gold_ids = list(dict.fromkeys(line.split()[2] for line in fh if line.strip()))

    # ── Resume + limit ──────────────────────────────────────────────────────
    todo = [g for g in golds if g["gold"] not in memory.rows]
    already = len(golds) - len(todo)
    print(f"[step 3] {len(golds)} wiki gold pages: {already} already processed, "
          f"{len(todo)} remaining")
    if sources:
        todo = [g for g in todo if g["sources"] & set(sources)]
        print(f"[step 3] this run: sources {', '.join(sources)}, {len(todo)} remaining")
    if limit is not None:
        todo = todo[:limit]
        print(f"[step 3] this run: the next {len(todo)} gold pages (--limit {limit})")

    # ── Find ────────────────────────────────────────────────────────────────
    shared = cfg.shared
    http = HttpClient(shared.contact, shared.timeout, shared.max_retries, cfg.pages.wiki_rate_per_min)
    wiki = WikipediaClient(http, cfg.pages.wiki_lang)
    finder = _Finder(cfg, wiki, store, memory, set(gold_ids))
    logging.getLogger("urllib3").setLevel(logging.ERROR)

    def work(gold: dict) -> Optional[dict]:
        try:
            row = finder.find(gold)
        except Exception as exc:  # noqa: BLE001 - no row: retried on rerun
            log.warning("%s (%s): %s; retried next run", gold["gold"], gold["title"], error_text(exc))
            return None
        memory.add(row)
        return row

    progress = _Progress(len(todo), already, len(golds), cfg, finder, http)
    try:
        with logging_redirect_tqdm(), ThreadPoolExecutor(max_workers=max(1, cfg.pages.wiki_workers)) as pool:
            for future in as_completed([pool.submit(work, g) for g in todo]):
                progress.update(future.result())
    finally:
        progress.close()
        store.close()
        memory.close()
    progress.summary()

    # ── Corpus: all processed gold pages ────────────────────────────────────
    urls = {row["doc_id"]: row["url"] for row in store.rows.values() if row["status"] == "ok"}
    gold_urls = {doc_id: urls[doc_id] for doc_id in gold_ids}
    n_gold, n_negatives = write_corpus(paths, gold_urls)
    processed = {g["gold"]: memory.rows[g["gold"]] for g in golds if g["gold"] in memory.rows}
    write_stats(paths.stats, "step3", {
        "config": settings,
        "gold_pages": {"wiki": len(golds), "processed": len(processed), "remaining": len(golds) - len(processed)},
        "negatives": negative_stats(processed, cfg.negatives.max_per_gold),
        "corpus": {"gold": n_gold, "hard_negatives": n_negatives},
    })
    print(f"[step 3] corpus: {n_gold} gold pages + {n_negatives} hard negatives "
          f"({len(processed)} of {len(golds)} wiki gold pages processed) -> {paths.corpus}")
