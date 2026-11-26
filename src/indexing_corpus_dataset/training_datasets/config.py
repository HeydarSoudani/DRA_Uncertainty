"""Settings of the training dataset build and its paths: ``web`` (the web
gold pages of webshaper, fetched by stage data_prepare), ``split``,
``facts`` (the fact LLM of frames and webshaper) and ``gold_passages``
(the judge of the gold passages: infoseek_hard, frames, webshaper), used
by stage data_relevance.  The sources and the filters are fixed in code
(``datasets/``).

The ``data`` section of ``experiments/configs/belief_training.yaml`` has the same
sections and keys as the dataclasses below, except the models of the fact
writer and the judge: the CLI flags ``--fact-model`` and ``--judge-model``
(defaults ``DEFAULT_FACT_MODEL``, ``DEFAULT_JUDGE_MODEL``) set them, as
``--limit`` sets the questions of a run.  An unknown section or key is an
error.
"""

import dataclasses
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from utils.yaml_config import load_sectioned
from indexing_corpus_dataset.shared.layout import (
    ALL_SPLIT, BUILT_DATASETS, TRAINING_SPLITS, dataset_root, default_corpus_path, qrels_base, queries_base,
)
from utils.paths import TRAINING_RAW_ROOT


@dataclass
class WebConfig:
    """webshaper: its web gold pages, fetched live or from the Wayback
    Machine (``corpus/web.py``)."""
    # Contact (e-mail or URL) put in the User-Agent of every request; null =
    # the DRA_CONTACT env var (.env).  The webshaper build refuses to fetch
    # without it.
    contact: Optional[str] = None
    timeout: float = 30.0
    max_retries: int = 4
    workers: int = 8
    # Wayback Machine fallback for pages whose live fetch fails; the archive
    # refuses connections above ~15 requests per minute.
    wayback: bool = True
    wayback_rate_per_min: float = 15.0
    # Snapshot closest to this time (YYYYMMDD): when WebShaper was built.
    wayback_timestamp: str = "20250601"
    # A page with fewer extracted words counts as dead (soft 404).
    min_words: int = 50
    # Fetch again, once per build, every page whose last fetch failed (any
    # status, also those failed for good); a page fetched since keeps it.
    # For one data_prepare run, then false again.
    refetch_failed: bool = False


@dataclass
class SplitConfig:
    """The shares of the full queries of a dataset in the splits ``sft``,
    ``rl`` and ``val`` (``pipeline/splits.py``), summing to 1."""
    sft: float = 0.3
    rl: float = 0.6
    val: float = 0.1

    def shares(self) -> Dict[str, float]:
        return {name: getattr(self, name) for name in TRAINING_SPLITS}


#: The fact LLM: the teacher belief model of training (native reasoning off).
DEFAULT_FACT_MODEL = "openrouter/qwen/qwen3.6-27b"
#: The judge of the gold passages: TRQA's (gpt-4o-mini, temperature 0).
DEFAULT_JUDGE_MODEL = "openrouter/openai/gpt-4o-mini"


@dataclass
class FactsConfig:
    """The fact LLM of frames and webshaper (``gold_passages/facts.py``):
    one call per question (frames: the property of each gold page;
    webshaper: the gold pages of each fact).  Its model is
    ``--fact-model``."""
    model: str = DEFAULT_FACT_MODEL
    workers: int = 8
    max_tokens: int = 4096


@dataclass
class GoldPassagesConfig:
    """The judge of the gold passages (``gold_passages/judge.py``): one call per
    (fact, passage) pair, TRQA's property check.  Its model is
    ``--judge-model``."""
    model: str = DEFAULT_JUDGE_MODEL
    workers: int = 32
    # The reply is one label, or an extracted value and its label.
    max_tokens: int = 128


SECTIONS = {
    "web": WebConfig,
    "split": SplitConfig,
    "facts": FactsConfig,
    "gold_passages": GoldPassagesConfig,
}


@dataclass
class BuildConfig:
    web: WebConfig = field(default_factory=WebConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    facts: FactsConfig = field(default_factory=FactsConfig)
    gold_passages: GoldPassagesConfig = field(default_factory=GoldPassagesConfig)

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


def require_contact(cfg: BuildConfig, what: str) -> None:
    """Refuse a build stage (*what*) that fetches web pages without a
    contact for the User-Agent."""
    if not cfg.web.contact:
        raise SystemExit(f"{what} needs web.contact (e-mail or URL for the User-Agent): "
                         "DRA_CONTACT in .env or web.contact in the config")


# The settings the command line sets (``--fact-model``, ``--judge-model``),
# never the YAML.
FLAG_KEYS = {("facts", "model"): "--fact-model", ("gold_passages", "model"): "--judge-model"}


def load_config(path: Optional[str]) -> BuildConfig:
    """The config of *path* (YAML; None = the defaults)."""
    data = (yaml.safe_load(Path(path).read_text()) or {}) if path else {}
    for (section, key), flag in FLAG_KEYS.items():
        if key in ((data.get(section) or {}) if isinstance(data, dict) else {}):
            raise ValueError(f"{section}.{key} is set by the command line flag {flag}, not in the config")
    cfg = load_sectioned(BuildConfig(), SECTIONS, path, None)
    if not cfg.web.contact:
        cfg.web.contact = os.environ.get("DRA_CONTACT") or None
    shares = cfg.split.shares()
    if any(v < 0 for v in shares.values()) or abs(sum(shares.values()) - 1.0) > 1e-9:
        raise ValueError(f"split: shares must be >= 0 and sum to 1, got {shares}")
    return cfg


class DatasetPaths:
    """The files of a training dataset.

    Its build folder, ``TRAINING_RAW_ROOT/{dataset}``
    (data/training_datasets_raw/{dataset}): the source as loaded (written
    by stage data_prepare, again when the code that makes it changes), the caches of what was looked up or
    fetched for it, the prepared questions (stage data_prepare), the LLM
    memories (stage data_relevance), the dropped questions and the build
    stats.

    The dataset itself, in the folder shared by every training dataset,
    ``TRAINING_DATASETS_ROOT`` (data/training_datasets, laid out like an
    inference dataset, ``shared/layout.py``): ``queries/{dataset}.jsonl`` (split
    ``all``: every query, one row per query with its split and gold
    annotation, ``annotation/rows.py``) and ``qrels/{dataset}.txt``,
    written by stage data_relevance."""

    def __init__(self, dataset: str):
        if dataset not in BUILT_DATASETS:
            raise ValueError(f"unknown training dataset {dataset!r}; expected one of {sorted(BUILT_DATASETS)}")
        self.dataset = dataset
        self.raw = TRAINING_RAW_ROOT / dataset
        self.root = dataset_root(dataset)
        self.split = ALL_SPLIT

    # ── Source (raw) ────────────────────────────────────────────────────────

    @property
    def raw_questions(self) -> Path:
        """Every question of the source in the record format
        (``pipeline/source.py``), nothing removed."""
        return self.raw / "questions.jsonl"

    @property
    def raw_source(self) -> Path:
        """Where the raw questions come from (repo, file, revision or local
        files) and their count."""
        return self.raw / "source.json"

    @property
    def titles(self) -> Path:
        """The chunk ids of each gold page title looked up in corpus_wiki
        (``{"title", "ids", "key_version"}`` per line, ``ids`` empty when the
        title is not in the corpus), appended as new titles are looked up
        (``corpus/wiki.lookup_titles``)."""
        return self.raw / "corpus_titles.jsonl"

    @property
    def web_pages(self) -> Path:
        """webshaper: one row per fetch of a web gold page (status, title,
        words; the latest row of a URL wins), appended as pages are fetched
        (``corpus/web.py``)."""
        return self.raw / "web_pages" / "index.jsonl"

    @property
    def web_corpus_record(self) -> Path:
        """webshaper: what corpus_web was last written from (the page files'
        sizes, the cut), so an unchanged build keeps it (``corpus/web.py``)."""
        return self.raw / "web_pages" / "corpus_web.json"

    @property
    def web_text(self) -> Path:
        """webshaper: the extracted text of each fetched web page
        (``{"url", "text"}``, the latest row of a URL wins), from which
        corpus_web is written."""
        return self.raw / "web_pages" / "text.jsonl"

    # ── Prepared questions (stage data_prepare) ─────────────────────────────

    @property
    def prepared(self) -> Path:
        """The questions that passed the filters
        (``datasets/{dataset}/prepare.py``): the record of each with its
        ``features`` (the chunk ids of its gold pages and what the filters
        measured), the input of stage data_relevance."""
        return self.raw / "prepared.jsonl"

    # ── Dataset (shared folder of the training datasets) ────────────────────

    @property
    def queries(self) -> Path:
        return queries_base(self.root, self.split, self.dataset).with_suffix(".jsonl")

    @property
    def qrels(self) -> Path:
        return qrels_base(self.root, self.split, self.dataset).with_suffix(".txt")

    @property
    def corpus(self) -> Path:
        """The corpus the dataset searches (``layout.default_corpus_path``):
        corpus_wiki (InfoSeek's wiki-25-512 under ``DATA_ROOT/infoseek``)
        for infoseek_hard, frames and webshaper (plus corpus_web,
        ``corpus/web.web_corpus_path``); TRQA's wiki_partial for trqa_val."""
        return default_corpus_path(self.dataset)

    # ── Build files ─────────────────────────────────────────────────────────

    @property
    def facts(self) -> Path:
        """The memory of the facts (frames, webshaper): one row per question
        whose facts are final, appended as the writer finishes them
        (``gold_passages/facts.py``)."""
        return self.raw / "facts.jsonl"

    @property
    def judgments(self) -> Path:
        """The memory of the judge of the gold passages: one row per judged
        (fact, passage) pair, appended as the judge answers
        (``gold_passages/judge.py``)."""
        return self.raw / "gold_passages.jsonl"

    @property
    def gold_report(self) -> Path:
        """frames, webshaper: per kept question its facts, their relevant
        passages (with the judge's label and extracted value) and the source
        of each gold page's relevant passages, rewritten by each
        data_relevance run (``gold_passages/select.py``)."""
        return self.raw / "gold_passages_report.jsonl"

    @property
    def dropped(self) -> Path:
        """One row per dropped question with its stage and reasons."""
        return self.raw / "dropped.jsonl"

    @property
    def stats(self) -> Path:
        """One section per stage: ``prepare`` and ``corpus`` (stage
        data_prepare; webshaper: also ``web``), ``relevance`` and
        ``gold_passages`` (stage data_relevance) and the ``split``
        section (shares and per-split counts, ``pipeline/splits.py``)."""
        return self.raw / "stats.json"
