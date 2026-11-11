"""Settings of the training dataset build (``shared``, the ``pages`` and
``negatives`` sections of webshaper_frames, the ``split`` section of
infoseek and trqa_validation, and the ``gold_criteria`` section of the
criterion texts) and its paths.  The sources and the dropped question types
are fixed (``sources.py``, ``filters.py``).

The YAML file (``experiments/configs/build_training_dataset.yaml``) has the same
sections and keys as the dataclasses below (the CLI flags ``--limit`` and
``--source`` are not settings).  An unknown section or key is an error.
"""

import dataclasses
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from utils.sectioned_config import load_sectioned
from indexing_corpus_dataset.layout import (ALL_SPLIT, BUILT_DATASETS, TRAINING_DATASETS_ROOT, TRAINING_SPLITS,
                                            default_corpus_path, qrels_base, queries_base)


@dataclass
class SharedConfig:
    # Contact (e-mail or URL) put in the User-Agent of every request, as the
    # Wikimedia User-Agent policy asks; null = the DRA_CONTACT env var (.env).
    # The webshaper_frames build refuses to run without it.
    contact: Optional[str] = None
    timeout: float = 30.0
    max_retries: int = 4


@dataclass
class PagesConfig:
    """webshaper_frames: the gold pages fetched from their source, then
    Filter A."""
    wiki_lang: str = "en"
    # Wikimedia: at most 3 concurrent requests, and 200 requests per minute
    # for an unauthenticated client whose User-Agent has contact information.
    wiki_workers: int = 3
    wiki_rate_per_min: float = 180.0
    web_workers: int = 8
    # Wayback Machine fallback for web pages whose live fetch fails; the
    # archive refuses connections above ~15 requests per minute.
    wayback: bool = True
    wayback_rate_per_min: float = 15.0
    # Snapshot closest to this time (YYYYMMDD): when WebShaper was built.
    wayback_timestamp: str = "20250601"
    # A web page with fewer extracted words counts as dead (soft 404).
    min_words: int = 50
    # Filter A for a question with web gold pages: at least this share of all
    # its gold pages fetched (the failed ones get no qrels); a question with
    # only Wikipedia gold pages needs them all.
    min_page_share: float = 0.8


@dataclass
class NegativesConfig:
    """webshaper_frames: hard negatives of each Wikipedia gold page, from
    the Wikipedia API (``negatives.py``); the request rate and workers are
    those of ``pages``."""
    # Hard negatives kept per Wikipedia gold page.
    max_per_gold: int = 10
    # The candidate sources, in the order they fill the slots of a gold page:
    # ``morelike`` (pages of the most similar text, CirrusSearch) and
    # ``category`` (pages of the gold page's most specific categories).
    sources: List[str] = field(default_factory=lambda: ["morelike", "category"])
    # Candidates asked from each source per slot (filtered or failed
    # candidates leave slots to the next ones).
    candidate_factor: int = 3
    # A candidate is skipped when its word 5-gram containment with a document
    # of the corpus (gold page or kept negative), |A & B| / min(|A|, |B|),
    # is at least this (``duplicates.py``); the duplicate report uses the same.
    dedup_threshold: float = 0.8


@dataclass
class SplitConfig:
    """The shares of the full queries of infoseek and trqa_validation in
    the splits ``sft``, ``rl`` and ``val`` (splits.py), summing to 1;
    webshaper_frames is validation only."""
    sft: float = 0.3
    rl: float = 0.6
    val: float = 0.1

    def shares(self) -> Dict[str, float]:
        return {name: getattr(self, name) for name in TRAINING_SPLITS}


@dataclass
class GoldCriteriaConfig:
    """The criterion texts of the gold annotation (``gold/criterion_texts.py``),
    written by an LLM at the end of every build."""
    # The writer: the teacher belief model of training (native reasoning off).
    model: str = "openrouter/qwen/qwen3.6-27b"
    workers: int = 8
    max_tokens: int = 4096


# The candidate sources of the hard negatives (``negatives.sources``).
NEGATIVE_SOURCES = ("morelike", "category")


SECTIONS = {
    "shared": SharedConfig,
    "pages": PagesConfig,
    "negatives": NegativesConfig,
    "split": SplitConfig,
    "gold_criteria": GoldCriteriaConfig,
}


@dataclass
class BuildConfig:
    shared: SharedConfig = field(default_factory=SharedConfig)
    pages: PagesConfig = field(default_factory=PagesConfig)
    negatives: NegativesConfig = field(default_factory=NegativesConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    gold_criteria: GoldCriteriaConfig = field(default_factory=GoldCriteriaConfig)

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


def require_contact(cfg: BuildConfig, what: str) -> None:
    """Refuse a build stage (*what*) that sends Wikimedia requests without a
    contact."""
    if not cfg.shared.contact:
        raise SystemExit(f"{what} needs shared.contact (e-mail or URL for the User-Agent, "
                         "as the Wikimedia User-Agent policy asks): DRA_CONTACT in .env or shared.contact in the config")


def load_config(path: Optional[str]) -> BuildConfig:
    """The config of *path* (YAML; None = the defaults)."""
    cfg = load_sectioned(BuildConfig(), SECTIONS, path, None)
    if not cfg.shared.contact:
        cfg.shared.contact = os.environ.get("DRA_CONTACT") or None
    shares = cfg.split.shares()
    if any(v < 0 for v in shares.values()) or abs(sum(shares.values()) - 1.0) > 1e-9:
        raise ValueError(f"split: shares must be >= 0 and sum to 1, got {shares}")
    neg = cfg.negatives
    unknown = sorted(set(neg.sources) - set(NEGATIVE_SOURCES))
    if unknown or not neg.sources or len(set(neg.sources)) != len(neg.sources):
        raise ValueError(f"negatives.sources: distinct names of {list(NEGATIVE_SOURCES)}, got {neg.sources}")
    if neg.max_per_gold < 1 or neg.candidate_factor < 1:
        raise ValueError("negatives: max_per_gold and candidate_factor must be >= 1")
    return cfg


class DatasetPaths:
    """The files of a build under ``TRAINING_DATASETS_ROOT/{dataset}``
    (data/training_datasets/{dataset} of the repo): the dataset in the
    canonical layout (``layout.py``; ``queries``, ``qrels`` are split
    ``all``, every query of the dataset, one row per query with its split
    and gold annotation, ``gold/annotations.py``) and, next to it, the build
    files."""

    def __init__(self, dataset: str):
        if dataset not in BUILT_DATASETS:
            raise ValueError(f"unknown training dataset {dataset!r}; expected one of {sorted(BUILT_DATASETS)}")
        self.dataset = dataset
        self.root = TRAINING_DATASETS_ROOT / dataset
        self.split = ALL_SPLIT

    # ── Dataset (canonical layout) ──────────────────────────────────────────

    @property
    def queries(self) -> Path:
        return queries_base(self.root, self.split).with_suffix(".jsonl")

    @property
    def qrels(self) -> Path:
        return qrels_base(self.root, self.split).with_suffix(".txt")

    @property
    def corpus(self) -> Path:
        """The corpus the dataset searches (``layout.default_corpus_path``):
        webshaper_frames' gold pages and hard negatives with their text,
        written by its build; InfoSeek's downloaded wiki-25-512 under
        ``DATA_ROOT/infoseek``; TRQA's wiki_partial for trqa_validation."""
        return default_corpus_path(self.dataset)

    @property
    def ids_gold(self) -> Path:
        """The ids of the corpus documents that are gold for a query, one per
        line (``.txt``: the index builder reads every ``corpus/*.jsonl``)."""
        return self.root / "corpus" / "ids_gold.txt"

    @property
    def ids_hard_negative(self) -> Path:
        """The ids of the hard negatives of the corpus (gold for no query),
        one per line."""
        return self.root / "corpus" / "ids_hard_negative.txt"

    # ── Build files ─────────────────────────────────────────────────────────

    @property
    def texts(self) -> Path:
        """The memory of the criterion texts: one row per query whose texts
        are final, appended as the writer finishes them (``gold/criterion_texts.py``)."""
        return self.root / "criterion_texts.jsonl"

    @property
    def questions(self) -> Path:
        """The candidate questions (webshaper_frames), in the record format
        of ``sources.py``."""
        return self.root / "questions.jsonl"

    @property
    def titles(self) -> Path:
        """infoseek: the chunk ids of each gold page title looked up in the
        corpus (``{"title", "ids"}`` per line, ``ids`` empty when the title
        is not in the corpus), appended as builds look up new titles."""
        return self.root / "corpus_titles.jsonl"

    @property
    def dropped(self) -> Path:
        """One row per dropped question with its stage and reasons."""
        return self.root / "dropped.jsonl"

    @property
    def stats(self) -> Path:
        """One section per build stage (webshaper_frames: ``questions``,
        ``pages``, ``negatives``, ``report``; infoseek: ``questions``,
        ``corpus``; trqa_validation: ``build``), the ``annotations`` section
        (annotation stats and the writer of the texts) and the ``split``
        section (shares and per-split counts, ``splits.py``)."""
        return self.root / "stats.json"

    @property
    def duplicates(self) -> Path:
        """The duplicate documents of the corpus found by the report of
        webshaper_frames, one row per group or pair (``duplicates.py``)."""
        return self.root / "duplicates.jsonl"

    @property
    def page_index(self) -> Path:
        """One row per gold page (webshaper_frames), appended as pages are
        fetched."""
        return self.root / "pages" / "index.jsonl"

    @property
    def page_text(self) -> Path:
        """The extracted text of the fetched pages (``{"id", "text"}``, one row
        per corpus document: gold pages and hard negatives), appended as pages
        are fetched."""
        return self.root / "pages" / "text.jsonl"

    @property
    def negatives(self) -> Path:
        """The memory of the hard negatives: one row per processed Wikipedia
        gold page with its hard negatives, appended as gold pages finish."""
        return self.root / "negatives" / "index.jsonl"
