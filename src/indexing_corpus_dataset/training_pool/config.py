"""Settings of the training pool build (``shared``, the ``pages`` section
of deep_qa step 2 and the ``split`` section) and its paths.  The sources and the dropped question types of step
1 are fixed (``sources.py``, ``filters.py``).

The YAML file (``experiments/configs/build_training_pool.yaml``) has the same
sections and keys as the dataclasses below; the CLI flags override it.  An
unknown section or key is an error.
"""

import dataclasses
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from indexing_corpus_dataset.layout import POOL_ALL_SPLIT, POOL_SPLITS, TRAINING_POOL_ROOT, corpus_path, qrels_base, queries_base


@dataclass
class SharedConfig:
    # Root of the build; null = TRAINING_POOL_ROOT/{pool} (data/training_pool/{pool} of the repo).
    output_dir: Optional[str] = None
    # Contact (e-mail or URL) put in the User-Agent of every request, as the
    # Wikimedia User-Agent policy asks; null = the DRA_CONTACT env var (.env).
    # Step 2 refuses to run without it.
    contact: Optional[str] = None
    timeout: float = 30.0
    max_retries: int = 4


@dataclass
class PagesConfig:
    """Step 2: the gold pages fetched from their source, then Filter A."""
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
class SplitConfig:
    """The shares of the pool's queries in the splits ``sft``, ``rl`` and
    ``val`` (splits.py), summing to 1."""
    sft: float = 0.3
    rl: float = 0.6
    val: float = 0.1

    def shares(self) -> Dict[str, float]:
        return {name: getattr(self, name) for name in POOL_SPLITS}


SECTIONS = {
    "shared": SharedConfig,
    "pages": PagesConfig,
    "split": SplitConfig,
}


@dataclass
class PoolConfig:
    shared: SharedConfig = field(default_factory=SharedConfig)
    pages: PagesConfig = field(default_factory=PagesConfig)
    split: SplitConfig = field(default_factory=SplitConfig)

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


def load_config(path: Optional[str], overrides: Optional[Dict[str, Any]] = None) -> PoolConfig:
    """The config of *path* (YAML; None = the defaults) with *overrides*
    (``{"section.key": value}``) applied."""
    cfg = PoolConfig()
    data = yaml.safe_load(Path(path).read_text()) if path else {}
    items = [(f"{section}.{key}", value)
             for section, values in (data or {}).items() for key, value in (values or {}).items()]
    items += list((overrides or {}).items())
    for dotted, value in items:
        section, _, key = dotted.partition(".")
        if section not in SECTIONS:
            raise ValueError(f"unknown config section {section!r}; expected one of {list(SECTIONS)}")
        obj = getattr(cfg, section)
        if key not in {f.name for f in dataclasses.fields(obj)}:
            raise ValueError(f"unknown config key {dotted!r}")
        setattr(obj, key, value)
    if not cfg.shared.contact:
        cfg.shared.contact = os.environ.get("DRA_CONTACT") or None
    shares = cfg.split.shares()
    if any(v < 0 for v in shares.values()) or abs(sum(shares.values()) - 1.0) > 1e-9:
        raise ValueError(f"split: shares must be >= 0 and sum to 1, got {shares}")
    return cfg


# The training pools (layout.BUILT_DATASETS): deep_qa is built in steps 1-4,
# trqa_validation is merged from the TRQA validation files (trqa_validation.py);
# both end with the split of their queries into sft / rl / val (splits.py).
POOLS = ("deep_qa", "trqa_validation")


class PoolPaths:
    """The files of a build under ``shared.output_dir`` (null =
    ``TRAINING_POOL_ROOT/{pool}``): the dataset in the canonical layout
    (``layout.py``; ``queries``, ``qrels`` are split ``all``, every query of
    the pool) and, next to it, the step files of deep_qa."""

    def __init__(self, cfg: PoolConfig, pool: str = "deep_qa"):
        self.root = Path(cfg.shared.output_dir) if cfg.shared.output_dir else TRAINING_POOL_ROOT / pool
        self.split = POOL_ALL_SPLIT

    # ── Dataset (canonical layout) ──────────────────────────────────────────

    @property
    def queries(self) -> Path:
        return queries_base(self.root, self.split).with_suffix(".jsonl")

    @property
    def qrels(self) -> Path:
        return qrels_base(self.root, self.split).with_suffix(".txt")

    def split_queries(self, split: str) -> Path:
        return queries_base(self.root, split).with_suffix(".jsonl")

    def split_info(self, split: str) -> Path:
        """The intermediate info of *split* (trqa_validation only)."""
        base = queries_base(self.root, split)
        return base.with_name(f"{base.name}_intermediate_info.jsonl")

    def split_qrels(self, split: str) -> Path:
        return qrels_base(self.root, split).with_suffix(".txt")

    @property
    def corpus(self) -> Path:
        """The gold pages of the kept questions."""
        return corpus_path(self.root)

    # ── Step files ──────────────────────────────────────────────────────────

    @property
    def questions(self) -> Path:
        """The questions of step 1 (the candidates of step 2), in the record
        format of ``sources.py``."""
        return self.root / "questions.jsonl"

    @property
    def dropped(self) -> Path:
        """One row per dropped question with its step and reasons."""
        return self.root / "dropped.jsonl"

    @property
    def stats(self) -> Path:
        """One section per step (``step1``, ``step2``)."""
        return self.root / "stats.json"

    @property
    def page_index(self) -> Path:
        """One row per gold page (step 2), appended as pages are fetched."""
        return self.root / "pages" / "index.jsonl"

    @property
    def page_text(self) -> Path:
        """The extracted text of the fetched pages (``{"id", "text"}``, one row
        per corpus document), appended as pages are fetched."""
        return self.root / "pages" / "text.jsonl"
