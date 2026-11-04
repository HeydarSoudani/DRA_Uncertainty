"""Duplicate documents of the deep_qa corpus: two ids with the same or nearly
the same content.

``DuplicateIndex`` finds, for a text, the indexed documents whose word
5-gram containment with it, |A & B| / min(|A|, |B|), is at least the
threshold (``negatives.dedup_threshold``).  Step 3 keeps a candidate hard
negative only when it matches no document of the corpus (``negatives.py``).

Step 4 reports the duplicates of the corpus as written (report only;
``find_duplicates``):

    duplicate_ids  an id on several rows of ``corpus.jsonl``
    same_url       ids whose urls are one page once canonicalized (scheme,
                   ``www.`` / ``m.``, trailing slash, fragment, tracking
                   parameters, Wayback wrapper)
    exact_text     ids with the same normalized text
    near_text      pairs at or above the threshold (exact_text pairs left
                   out); after step 3 only pairs of gold pages remain

Every group or pair is counted by role (gold / negative, from the id files)
and by kind (wiki / web, from the id prefix), and written with its ids,
urls and titles to ``duplicates.jsonl`` for review.
"""

import hashlib
import re
import zlib
from collections import Counter, defaultdict
from itertools import combinations
from typing import Dict, List, Tuple
from urllib.parse import parse_qsl, unquote, urlencode, urlsplit

import numpy as np

from utils.io_utils import iter_jsonl, write_jsonl

from .annotations import normalize
from .config import PoolPaths

SHINGLE = 5
# Shingle hashes below this bound (1 in 8) are the sample of a document: the
# same hashes in every document, so the sample of A & B is the intersection
# of the samples of A and B.
_SAMPLE_BOUND = np.uint64(1 << 61)
# Pairs whose sampled containment reaches this share of the threshold get
# the exact containment.
_CANDIDATE_SHARE = 0.75

_WAYBACK = re.compile(r"^/web/\d+[a-z_]*/(.+)$")
_TRACKING = re.compile(r"^(utm_.*|fbclid|gclid|mc_cid|mc_eid|ref|ref_src)$")


def canonical_url(url: str) -> str:
    """*url* without what does not change the page: scheme, ``www.`` and
    ``m.`` hosts, port, trailing slash, fragment, tracking parameters (the
    others sorted), a Wayback wrapper."""
    parts = urlsplit(url.strip())
    host = (parts.hostname or "").lower()
    if host == "web.archive.org":
        inner = _WAYBACK.match(parts.path)
        if inner:
            target = inner.group(1) + ("?" + parts.query if parts.query else "")
            return canonical_url(target if "://" in target else "http://" + target)
    for prefix in ("www.", "m."):
        if host.startswith(prefix):
            host = host[len(prefix):]
    host = host.replace(".m.", ".")  # en.m.wikipedia.org
    path = unquote(parts.path).rstrip("/") or "/"
    query = sorted((k, v) for k, v in parse_qsl(parts.query, keep_blank_values=True) if not _TRACKING.match(k))
    return host + path + ("?" + urlencode(query) if query else "")


def _mix(x: np.ndarray) -> np.ndarray:
    """splitmix64 finalizer: uniform hashes, so the sample is uniform."""
    with np.errstate(over="ignore"):
        x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        return x ^ (x >> np.uint64(31))


def shingles(text: str) -> np.ndarray:
    """The sorted unique hashes of the word 5-grams of the normalized *text*
    (deterministic across runs: crc32 per word, a polynomial over the
    window)."""
    words = normalize(text).split()
    if len(words) < SHINGLE:
        return np.empty(0, dtype=np.uint64)
    w = np.fromiter((zlib.crc32(t.encode("utf-8")) for t in words), dtype=np.uint64, count=len(words))
    n = len(words) - SHINGLE + 1
    h = np.zeros(n, dtype=np.uint64)
    with np.errstate(over="ignore"):
        for j in range(SHINGLE):
            h = h * np.uint64(1_000_003) + w[j:j + n]
    return np.unique(_mix(h))


class DuplicateIndex:
    """Documents by their 5-grams: ``matches`` gives the indexed documents
    with a containment of at least *threshold* with a text.  Not
    thread-safe: callers lock around ``matches`` + ``add``."""

    def __init__(self, threshold: float):
        self.threshold = threshold
        self.sets: Dict[str, np.ndarray] = {}
        self._sampled: Dict[str, int] = {}
        self._postings: Dict[int, List[str]] = defaultdict(list)

    def __contains__(self, doc_id: str) -> bool:
        return doc_id in self.sets

    def __len__(self) -> int:
        return len(self.sets)

    def add(self, doc_id: str, grams: np.ndarray) -> None:
        sample = grams[grams < _SAMPLE_BOUND]
        self.sets[doc_id] = grams
        self._sampled[doc_id] = len(sample)
        for value in sample.tolist():
            self._postings[value].append(doc_id)

    def matches(self, grams: np.ndarray) -> List[Tuple[str, float]]:
        """(doc id, containment) of the indexed documents at or above the
        threshold, the highest first."""
        sample = grams[grams < _SAMPLE_BOUND]
        hits: Counter = Counter()
        for value in sample.tolist():
            hits.update(self._postings.get(value, ()))
        out = []
        for doc_id, n in hits.items():
            if n < _CANDIDATE_SHARE * self.threshold * min(len(sample), self._sampled[doc_id]):
                continue
            other = self.sets[doc_id]
            containment = len(np.intersect1d(grams, other, assume_unique=True)) / min(len(grams), len(other))
            if containment >= self.threshold:
                out.append((doc_id, round(containment, 3)))
        return sorted(out, key=lambda m: -m[1])


def read_shingles(paths: PoolPaths, ids: set) -> Dict[str, np.ndarray]:
    """The 5-grams of the documents *ids* from ``pages/text.jsonl`` (the last
    row of a document wins, as in the corpus)."""
    out: Dict[str, np.ndarray] = {}
    if paths.page_text.exists():
        for row in iter_jsonl(paths.page_text):
            if row["id"] in ids:
                out[row["id"]] = shingles(row["text"])
    return out


def _read_ids(path) -> set:
    if not path.exists():
        return set()
    with open(path, encoding="utf-8") as fh:
        return {line.strip() for line in fh if line.strip()}


def find_duplicates(paths: PoolPaths, threshold: float) -> dict:
    """Step 4: the duplicate checks of the corpus; writes
    ``duplicates.jsonl`` and returns the ``duplicates`` section of the step 4
    stats."""
    gold, negative = _read_ids(paths.ids_gold), _read_ids(paths.ids_hard_negative)
    id_rows: Counter = Counter()
    docs: Dict[str, dict] = {}
    by_url: Dict[str, List[str]] = defaultdict(list)
    by_text: Dict[str, List[str]] = defaultdict(list)
    index = DuplicateIndex(threshold)
    near: List[Tuple[str, str, float]] = []
    for row in iter_jsonl(paths.corpus):
        doc_id = row["id"]
        id_rows[doc_id] += 1
        if doc_id in docs:
            continue
        docs[doc_id] = {"url": row.get("url", ""), "title": row["contents"].split("\n", 1)[0][:120]}
        if docs[doc_id]["url"]:
            by_url[canonical_url(docs[doc_id]["url"])].append(doc_id)
        by_text[hashlib.sha1(normalize(row["contents"]).encode("utf-8")).hexdigest()].append(doc_id)
        grams = shingles(row["contents"])
        near += [(other, doc_id, c) for other, c in index.matches(grams)]
        index.add(doc_id, grams)

    def role(d: str) -> str:
        return "gold" if d in gold else "negative" if d in negative else "other"

    records: List[dict] = []
    by_role: Dict[str, Counter] = defaultdict(Counter)
    by_kind: Dict[str, Counter] = defaultdict(Counter)

    def add(check: str, ids: List[str], **extra) -> None:
        records.append({"check": check, "docs": [{"id": d, "role": role(d), **docs[d]} for d in ids], **extra})
        for a, b in combinations(ids, 2):
            by_role[check]["-".join(sorted((role(a), role(b))))] += 1
            by_kind[check]["-".join(sorted((a.split("_", 1)[0], b.split("_", 1)[0])))] += 1

    for doc_id, n in id_rows.items():
        if n > 1:
            add("duplicate_ids", [doc_id], rows=n)
    for url, ids in by_url.items():
        if len(ids) > 1:
            add("same_url", ids, canonical_url=url)
    exact_pairs = set()
    for ids in by_text.values():
        if len(ids) > 1:
            add("exact_text", ids)
            exact_pairs.update(frozenset(p) for p in combinations(ids, 2))
    for a, b, containment in near:
        if frozenset((a, b)) not in exact_pairs:
            add("near_text", [a, b], containment=containment)

    write_jsonl(paths.duplicates, records)
    checks = ("duplicate_ids", "same_url", "exact_text", "near_text")
    counts = Counter(r["check"] for r in records)
    return {
        "settings": {"shingle": SHINGLE, "threshold": threshold},
        "docs": len(docs),
        "groups": {c: counts[c] for c in checks},
        "pairs_by_role": {c: dict(by_role[c]) for c in checks},
        "pairs_by_kind": {c: dict(by_kind[c]) for c in checks},
    }


def format_duplicates(stats: dict) -> str:
    """The ``duplicates`` section as plain-text lines, one per check."""
    lines = [f"duplicates ({stats['docs']} docs; near_text: containment >= {stats['settings']['threshold']} "
             f"of word {stats['settings']['shingle']}-grams; all 0 = no duplicates)"]
    for check, groups in stats["groups"].items():
        roles = ", ".join(f"{k} {v}" for k, v in sorted(stats["pairs_by_role"][check].items())) or "-"
        kinds = ", ".join(f"{k} {v}" for k, v in sorted(stats["pairs_by_kind"][check].items())) or "-"
        unit = "pairs" if check == "near_text" else "groups"
        lines.append(f"  {check:<14} {groups} {unit} | pairs by role: {roles} | by kind: {kinds}")
    return "\n".join(lines)
