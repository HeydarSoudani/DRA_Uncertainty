"""Canonical on-disk dataset layout — the single source of truth for where
dataset files live and how their paths, split ids, and corpus names are built.

Both the readers (``dataset_loaders.py``) and the writers/downloaders
(``download_datasets.py``) import this module, as do external consumers (the
inference CLI, the index builder, and its test).  Centralising the layout here
means the data root, the split-id rule, and the corpus-naming rule each have
exactly one definition.

Canonical layout::

    {data_path}/queries/queries_{split}.jsonl   (or .tsv)
    {data_path}/qrels/qrels_{split}.txt         (or .tsv)
    {data_path}/nuggets/nuggets_{split}.jsonl   (report-generation datasets only)
    {data_path}/corpus/{name}.jsonl

Record schemas:
    queries : {"id": str, "text": str, "answer"?: str}   (NeuCLIR/RAGTIME add topic
              fields, e.g. "request", "title", "limit")
    qrels   : TREC -> "qid 0 docid rel"
    nuggets : {"id": str, "nuggets": [{"id", "question", "answers": [str],
                                       "importance": str | None, "support_docs": [str]}]}
    corpus  : {"id": str, "contents": str}

Per-dataset defaults (split, query key, relevance threshold, task type, encoder
lengths) live in :data:`DATASET_SPECS`, read by the inference CLI, the index
builder and its test.

Deliberately import-light (only the standard library) so any module — including
the low-level ``utils.config`` — can import it without pulling heavy dependencies.
"""

import os
from dataclasses import dataclass
from pathlib import Path


def _load_dra_env_defaults() -> None:
    """Populate ``DRA_*`` env vars from the project-root ``.env`` if unset.

    Every entry point imports this module (inference CLI, downloaders, index
    builder), but the standalone scripts never call ``python-dotenv``, and
    ~/.bashrc is not sourced in non-interactive SLURM jobs.  This tiny,
    dependency-free reader makes ``DRA_DATA_ROOT`` / ``DRA_OUTPUT_ROOT`` set in
    ``.env`` take effect everywhere.  A real environment value always wins, so
    exporting the var (shell or sbatch) still overrides ``.env``.
    """
    env_file = Path(__file__).resolve().parents[2] / ".env"
    if not env_file.is_file():
        return
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if key.startswith("DRA_") and key not in os.environ:
            value = value.split("#", 1)[0].strip().strip('"').strip("'")
            if value:
                os.environ[key] = value


_load_dra_env_defaults()

# The one canonical dataset root.  Defaults to the cluster project location but
# can be overridden with the ``DRA_DATA_ROOT`` env var (no code edit needed) —
# set it in the shell, in an sbatch script, or in the project-root ``.env``.
# Hardcoded default (not derived from ``__file__`` depth) so it is stable
# regardless of where the package is imported from.
DATA_ROOT = Path(os.environ.get(
    "DRA_DATA_ROOT", "/projects/0/prjs0834/heydars/DRA_training/data"
))

# ===========================================================================
# Per-dataset defaults
# ===========================================================================

@dataclass(frozen=True)
class DatasetSpec:
    """Defaults for one dataset, applied wherever the run leaves a value null.

    Attributes:
        dataset_year:        Year of the release; TRQA carries its eval split
                             (test|validation) here.  None = not used.
        subset:              Subset / collection.  None = not used.
        query_key:           Queries-file field used as the query text.
        min_relevance_score: Lowest qrel grade counted as relevant by the
                             binary metrics (Recall@N, new_item_precision).
                             Queries with no qrel at this grade are not run.
        relevance_gains:     Official gain of each qrel grade, for the graded
                             metrics (GradedRecall@N, new_item_graded_recall,
                             NDCG);
                             a grade not listed has gain 0.  Independent of
                             ``min_relevance_score``.
        task:                ``"qa"`` (short answer) or ``"report"`` (a report
                             request).  Report tasks get report prompts and no
                             intermediate answers.
        answer_eval:         Answer-correctness evaluator: ``"numeric_match"``,
                             ``"llm_judge"``, or None (retrieval-only evaluation).
        doc_max_length:      Max tokens per passage at index build, for
                             long-context encoders.
        query_max_length:    Max tokens per query at retrieval, for long-context
                             encoders.
        report_chars:        Target report length in characters (report tasks).
    """
    dataset_year: str | None
    subset: str | None
    query_key: str
    min_relevance_score: int
    relevance_gains: dict[int, int]
    task: str
    answer_eval: str | None
    doc_max_length: int
    query_max_length: int
    report_chars: int | None = None


DATASET_SPECS = {
    # TRQA passages are short; answers are numeric.
    "trqa": DatasetSpec(
        dataset_year="test", subset="wiki2", query_key="text", min_relevance_score=1,
        relevance_gains={1: 1}, task="qa", answer_eval="numeric_match", doc_max_length=512, query_max_length=512,
    ),
    # Long web pages and long multi-clue questions; grades: gold=2, evidence=1.
    # Both are evidence (the owners' qrel_evidence) for the binary metrics; the
    # owners define no gains, so gold (contains the answer) gets gain 2 by choice.
    "browsecomp_plus": DatasetSpec(
        dataset_year=None, subset="test", query_key="text", min_relevance_score=1,
        relevance_gains={1: 1, 2: 2}, task="qa", answer_eval="llm_judge", doc_max_length=4096, query_max_length=8196,
    ),
    # Grades 0/1/3 are already the official points (very valuable 3, somewhat
    # valuable 1); relevant = >=1 as in the track's R@1000.  Only the 59
    # report-generation topics carry ``request``.
    # News docs: median ~350-420 tokens, 1024 covers ~90% whole.
    "neuclir": DatasetSpec(
        dataset_year="2024", subset="news", query_key="request", min_relevance_score=1,
        relevance_gains={1: 1, 3: 3}, task="report", answer_eval=None, doc_max_length=1024, query_max_length=512,
        report_chars=2000,
    ),
    # The NIST qrels keep the raw grades: 3 very valuable, 2 valuable, 1 topical,
    # 0 irrelevant.  Official points 3/1/0/0, so relevant = >=2.  ``text`` is
    # the report request (background + problem statement).
    "ragtime": DatasetSpec(
        dataset_year="2025", subset=None, query_key="text", min_relevance_score=2,
        relevance_gains={2: 1, 3: 3}, task="report", answer_eval=None, doc_max_length=1024, query_max_length=512,
        report_chars=2000,
    ),
}

# Datasets with a local corpus + index under ``DATA_ROOT/{dataset}``.
DATASETS = tuple(DATASET_SPECS)

# BERT-style and SPLADE encoders are capped at 512 positions; the long-context
# encoders (Qwen3-Embedding, AgentIR) take the per-dataset lengths above.
SHORT_ENCODER_MAX_LENGTH = 512


def _long_context_encoder(retriever: str) -> bool:
    return retriever.startswith("qwen3_emb") or retriever == "agentir_4b"


def apply_dataset_defaults(args, fields: tuple[str, ...]) -> None:
    """Fill each of ``fields`` left None on ``args`` from ``args.dataset``'s spec."""
    spec = DATASET_SPECS[args.dataset]
    for field in fields:
        if getattr(args, field, None) is None:
            value = getattr(spec, field)
            setattr(args, field, value)
            if value is not None:
                print(f"Auto-selected {field}: {value}")


def doc_max_length(dataset: str, retriever: str) -> int:
    """Max tokens per passage at index build for this dataset + retriever."""
    if _long_context_encoder(retriever):
        return DATASET_SPECS[dataset].doc_max_length
    return SHORT_ENCODER_MAX_LENGTH


def query_max_length(dataset: str, retriever: str) -> int:
    """Max tokens per query at retrieval for this dataset + retriever."""
    if _long_context_encoder(retriever):
        return DATASET_SPECS[dataset].query_max_length
    return SHORT_ENCODER_MAX_LENGTH

# Per-subset NeuCLIR English-MT corpus file stems (no ``.jsonl`` suffix).
_NEUCLIR_CORPUS_NAMES = {
    "news":      "corpus_en_news",
    "technical": "corpus_en_technical",
}

# RAGTIME1 English corpus file stem: native English docs + Arabic/Russian/
# Chinese docs machine-translated to English.
RAGTIME_CORPUS_NAME = "corpus_en"

# Per-subset TRQA corpus file stems (no ``.jsonl`` suffix).  wiki1 and wiki2
# share the single Wikipedia corpus; ecommerce has its own.  The partial
# Wikipedia corpus is selected via the ``partial`` flag of ``trqa_corpus_name``.
_TRQA_CORPUS_NAMES = {
    "wiki1":     "wiki",
    "wiki2":     "wiki",
    "ecommerce": "ecommerce",
}


# ===========================================================================
# Path builders (suffix-less bases + concrete corpus path)
# ===========================================================================

def queries_base(data_path: Path | str, split: str) -> Path:
    """Return the suffix-less queries path, e.g. ``.../queries/queries_test``."""
    return Path(data_path) / "queries" / f"queries_{split}"


def qrels_base(data_path: Path | str, split: str) -> Path:
    """Return the suffix-less qrels path, e.g. ``.../qrels/qrels_test``."""
    return Path(data_path) / "qrels" / f"qrels_{split}"


def nuggets_base(data_path: Path | str, split: str) -> Path:
    """Return the suffix-less nuggets path, e.g. ``.../nuggets/nuggets_2025``."""
    return Path(data_path) / "nuggets" / f"nuggets_{split}"


def corpus_path(data_path: Path | str, name: str = "corpus") -> Path:
    """Return the corpus file path, e.g. ``.../corpus/corpus.jsonl``."""
    return Path(data_path) / "corpus" / f"{name}.jsonl"


def _resolve_split_file(base: Path, suffixes: tuple[str, ...]) -> Path | None:
    """Return the first existing ``base + suffix`` file, or None."""
    for suffix in suffixes:
        candidate = base.with_suffix(suffix)
        if candidate.exists():
            return candidate
    return None


# ===========================================================================
# Naming conventions: corpus file stem, split id, data-path resolution
# ===========================================================================

def corpus_name(subset: str) -> str:
    """NeuCLIR corpus file stem for a subset, e.g. ``news`` -> ``corpus_en_news``.

    Falls back to ``corpus_en_{subset}`` for subsets not in the explicit map.
    """
    return _NEUCLIR_CORPUS_NAMES.get(subset, f"corpus_en_{subset}")


def trqa_corpus_name(subset: str, partial: bool = True) -> str:
    """TRQA corpus file stem for a subset (no ``.jsonl`` suffix).

    ``wiki1``/``wiki2`` -> ``wiki_partial`` (the canonical default; ``wiki``
    only when ``partial=False``), ``ecommerce`` -> ``ecommerce``.  Unknown
    subsets fall back to themselves.
    """
    name = _TRQA_CORPUS_NAMES.get(subset, subset)
    if partial and name == "wiki":
        return "wiki_partial"
    return name


def resolve_split_id(
    dataset: str,
    dataset_year: str | None = None,
    subset: str | None = None,
) -> str:
    """Build the on-disk split identifier from (dataset, year, subset).

    Mirrors the conventions used across the pipeline:
      neuclir         -> "{year}_{subset}"  (falls back to subset/year)
      trqa            -> "{subset}_{eval}"  (eval split carried in dataset_year)
      ragtime         -> "{year}"           (single task; subset unused)
      browsecomp_plus -> subset or "test"
      other           -> subset or "set1"
    """
    if dataset == "neuclir":
        if dataset_year and subset:
            return f"{dataset_year}_{subset}"
        return subset or dataset_year or "2024_news"
    if dataset == "trqa":
        # TRQA splits combine the collection (subset: wiki1/wiki2/ecommerce)
        # with the evaluation split (test/validation), carried in dataset_year
        # to reuse the existing two-dimensional arg plumbing.  This matches the
        # HuggingFace split names, e.g. "wiki1_test".
        eval_split = dataset_year or "test"
        return f"{subset}_{eval_split}" if subset else eval_split
    if dataset == "ragtime":
        return dataset_year or "2025"
    if dataset == "browsecomp_plus":
        return subset or "test"
    return subset or "set1"


def resolve_data_path(dataset: str, project_root: Path | str | None = None) -> str:
    """Resolve the dataset root directory.

    Uses the canonical ``DATA_ROOT/{dataset}`` location, falling back to the
    repo-local ``data/{dataset}/`` for legacy setups.
    """
    candidate = DATA_ROOT / dataset
    if candidate.exists():
        return str(candidate)
    base = Path(project_root) if project_root is not None else Path.cwd()
    return str((base / "data" / dataset).resolve())


# ===========================================================================
# Index-build path derivation (corpus + index dir from dataset/subset)
# ===========================================================================

def default_corpus_path(
    dataset: str, subset: str | None = None, *, partial: bool = True
) -> Path:
    """Canonical full-corpus JSONL path for a dataset/subset under ``DATA_ROOT``.

    The corpus file stem encodes the subset (NeuCLIR ``corpus_en_news``, TRQA
    ``wiki_partial``/``ecommerce``), so an index built from this path is already
    named per dataset/subset by ``Index_Builder``.  Used by both the production
    index builder and its test to avoid passing ``--corpus-path`` by hand.
    """
    if dataset == "neuclir":
        return DATA_ROOT / "neuclir" / "corpus" / f"{corpus_name(subset)}.jsonl"
    if dataset == "trqa":
        return DATA_ROOT / "trqa" / "corpus" / f"{trqa_corpus_name(subset, partial)}.jsonl"
    if dataset == "ragtime":
        return DATA_ROOT / "ragtime" / "corpus" / f"{RAGTIME_CORPUS_NAME}.jsonl"
    if dataset == "browsecomp_plus":
        return DATA_ROOT / "browsecomp_plus" / "corpus" / "corpus.jsonl"
    # Unknown dataset: best-effort canonical location.
    return DATA_ROOT / dataset / "corpus" / "corpus.jsonl"


def default_index_dir(dataset: str) -> Path:
    """Canonical index output directory for a dataset, ``DATA_ROOT/{dataset}/indices``.

    Sibling of the ``corpus/`` directory, matching the production layout.
    """
    return DATA_ROOT / dataset / "indices"
