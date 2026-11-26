"""Canonical on-disk dataset layout — the single source of truth for where
dataset files live and how their paths, split ids, and corpus names are built.

Both the readers (``dataset_loaders.py``) and the writers/downloaders
(``download_datasets.py``) import this module, as do external consumers (the
inference CLI, the index builder, and its test).  Centralising the layout here
means the split-id rule and the corpus-naming rule each have exactly one
definition (the data root has its own, in ``utils.paths``).

Canonical layout::

    {data_path}/queries/queries_{split}.jsonl   (or .tsv)
    {data_path}/qrels/qrels_{split}.txt         (or .tsv)
    {data_path}/nuggets/nuggets_{split}.jsonl   (report-generation datasets only)
    {data_path}/corpus/{name}.jsonl

The training datasets (``BUILT_DATASETS``) share one ``data_path``,
``TRAINING_DATASETS_ROOT``, with one file per dataset holding every split
(``queries/{dataset}.jsonl``, ``qrels/{dataset}.txt``; the loaders take
the dataset name).

Record schemas:
    queries : {"id": str, "text": str, "answer"?: str}   (NeuCLIR/RAGTIME add topic
              fields, e.g. "request", "title", "limit")
    qrels   : TREC -> "qid 0 docid rel"
    nuggets : {"id": str, "nuggets": [{"id", "question", "answers": [str],
                                       "importance": str | None, "support_docs": [str],
                                       "aggregator"?: "AND" | "OR",
                                       "answer_docs"?: {answer: [str]}}]}
              (``aggregator`` and ``answer_docs``, the supporting documents of
              each answer, where the source gives them: NeuCLIR)
    corpus  : {"id": str, "contents": str}   (the title, if any, is the first
              line of ``contents``; InfoSeek's
              wiki-25-512 numbers its chunks, read as strings)

Per-dataset defaults (split, query key, relevance threshold, task type, encoder
lengths) live in :data:`DATASET_SPECS`, read by the inference CLI, the index
builder and its test.

Deliberately import-light (the standard library and ``utils.paths``, which
holds the data and output roots) so any module can import it without pulling
heavy dependencies.
"""

from dataclasses import dataclass
from pathlib import Path

from utils.paths import BELIEF_ROOT, DATA_ROOT, TRAINING_DATASETS_ROOT

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
                             binary metrics (Recall@N, new_item_precision,
                             recall_so_far), the gold of the trajectory.
                             Queries with no qrel at this grade are not run.
        task:                ``"qa"`` (short answer) or ``"report"`` (a report
                             request).  Report tasks get report prompts and no
                             intermediate answers.
        query_shape:         Shape of the dataset's queries, for the belief
                             extraction: ``"multi_clue"`` (a list of clues
                             on one entity, the target: one belief per
                             clue, then the target, then the asked answer
                             when it is not the target), ``"multi_hop"`` (one entity
                             reached through a chain of other entities: one
                             belief per hop), ``"multi_member"`` (a value over every
                             member of a set) or ``"multi_aspect"`` (a report
                             on several aspects of a topic).
        max_beliefs:         Cap on the number of beliefs the belief
                             extractor lists per query, or None (uncapped).
                             Gold beliefs are never capped: a dataset
                             whose extracted beliefs are its gold
                             (``gold_beliefs == "question"``) has None; for
                             the others it caps the silver extraction (the
                             members of a TRQA set).
        gold_beliefs:        Where the dataset's gold beliefs come from:
                             ``"question"`` (the beliefs the extractor
                             takes from the question are the gold: every
                             clue of a BrowseComp-Plus, infoseek_hard or
                             webshaper question, every requirement of a
                             report request; they are banked as gold) or
                             ``"annotation"`` (from the dataset's
                             annotation: the members of trqa_val's rows,
                             none for frames, ``gold_beliefs_path``; TRQA's intermediate
                             info, made on demand into its gold bank,
                             ``belief_model_component/init/trqa_gold.py``).  A query of
                             an annotation dataset without gold beliefs
                             gets silver ones, extracted
                             from the question by an LLM.
        belief_choice:       ``belief_source`` picks between the gold and
                             the silver beliefs of every query (TRQA);
                             else it has no effect: the dataset's
                             queries use their gold, a query without it
                             silver.
        answer_eval:         Answer-correctness evaluator: ``"numeric_match"``,
                             ``"llm_judge"``, or None (retrieval-only evaluation).
        doc_max_length:      Max tokens per passage at index build, for
                             long-context encoders.
        query_max_length:    Max tokens per query at retrieval, for long-context
                             encoders.
        report_eval:         Report evaluator: ``"argue"`` (Auto-ARGUE against
                             the dataset's nuggets, ``evaluation.generation.argue``)
                             or None (reports are not graded).
    """
    dataset_year: str | None
    subset: str | None
    query_key: str
    min_relevance_score: int
    task: str
    query_shape: str
    max_beliefs: int | None
    gold_beliefs: str
    belief_choice: bool
    answer_eval: str | None
    doc_max_length: int
    query_max_length: int
    report_eval: str | None = None


DATASET_SPECS = {
    # TRQA passages are short; answers are numeric.  Up to 52 beliefs, as
    # the gold: one per member (up to 50), the rest and the aggregate
    # (training_datasets/annotation/rows.py).
    "trqa": DatasetSpec(
        dataset_year="test", subset="wiki2", query_key="text", min_relevance_score=1,
        task="qa", query_shape="multi_member", max_beliefs=52, gold_beliefs="annotation", belief_choice=True, answer_eval="numeric_match", doc_max_length=512, query_max_length=512,
    ),
    # Long web pages and long multi-clue questions; grades: gold=2, evidence=1.
    # Both are evidence (the owners' qrel_evidence) for the binary metrics.
    "browsecomp_plus": DatasetSpec(
        dataset_year=None, subset="test", query_key="text", min_relevance_score=1,
        task="qa", query_shape="multi_clue", max_beliefs=None, gold_beliefs="question", belief_choice=False, answer_eval="llm_judge", doc_max_length=4096, query_max_length=8196,
    ),
    # Grades 0/1/3 (very valuable 3, somewhat valuable 1).  The trajectory
    # gold and the end-of-run Recall@N are >=3 (the belief state tracks the
    # very valuable docs).
    # Only the 59 report-generation topics carry ``request``; 49 have a
    # grade-3 doc.
    # News docs: median ~350-420 tokens, 1024 covers ~90% whole.
    "neuclir": DatasetSpec(
        dataset_year="2024", subset="news", query_key="request", min_relevance_score=3,
        task="report", query_shape="multi_aspect", max_beliefs=None, gold_beliefs="question", belief_choice=False, answer_eval=None, doc_max_length=1024, query_max_length=512,
        report_eval="argue",
    ),
    # The NIST qrels keep the raw grades: 3 very valuable, 2 valuable, 1 topical,
    # 0 irrelevant.  The trajectory gold and the
    # end-of-run Recall@N are >=3, as for NeuCLIR (24 of the 30 judged
    # requests have a grade-3 doc).  ``text`` is
    # the report request (background + problem statement).
    "ragtime": DatasetSpec(
        dataset_year="2025", subset=None, query_key="text", min_relevance_score=3,
        task="report", query_shape="multi_aspect", max_beliefs=None, gold_beliefs="question", belief_choice=False, answer_eval=None, doc_max_length=1024, query_max_length=512,
        report_eval="argue",
    ),
    # Training datasets built by this project (experiments/training/train_belief_model.py
    # --stage data_prepare, then data_relevance), under TRAINING_DATASETS_ROOT; ``subset`` is the
    # split (TRAINING_SPLITS).
    # infoseek_hard: InfoSeek-Hard questions over the original release's
    # corpus, wiki-25-512 (Wikipedia 2025 in chunks of 512 tokens, kept under
    # DATA_ROOT/infoseek with its indices, ``_CORPUS_IN_DATA_ROOT``); binary
    # qrels: per gold page the chunk its claims are stated in (InfoSeek
    # writes them from the first 1024 tokens of a page: chunk 1 or 2, an LLM
    # judges chunk 2; chunk 1 for a page without claims).  Multi-clue, as
    # BrowseComp-Plus: the beliefs are the extractor's, from the question.
    "infoseek_hard": DatasetSpec(
        dataset_year=None, subset="sft", query_key="text", min_relevance_score=1,
        task="qa", query_shape="multi_clue", max_beliefs=None, gold_beliefs="question", belief_choice=False, answer_eval="llm_judge", doc_max_length=512, query_max_length=8192,
    ),
    # trqa_val: the TRQA wiki1 + wiki2 validation queries, qrels and
    # intermediate info merged; TRQA's settings, corpus and indices.
    "trqa_val": DatasetSpec(
        dataset_year=None, subset="sft", query_key="text", min_relevance_score=1,
        task="qa", query_shape="multi_member", max_beliefs=52, gold_beliefs="annotation", belief_choice=True, answer_eval="numeric_match", doc_max_length=512, query_max_length=512,
    ),
    # frames and webshaper: FRAMES and WebShaper questions over the corpus of
    # infoseek_hard (corpus_wiki, plus corpus_web: WebShaper's web gold pages
    # in chunks of 512 tokens, merged by the user for the index).  Binary
    # qrels from an LLM judge of each (fact, passage) pair: a passage is
    # relevant when it states a fact of the question; a gold page without a
    # judged relevant passage gets its first passage.
    # frames has no gold beliefs (silver multi-hop ones from the
    # extractor); webshaper is multi-clue, as infoseek_hard.
    "frames": DatasetSpec(
        dataset_year=None, subset="sft", query_key="text", min_relevance_score=1,
        task="qa", query_shape="multi_hop", max_beliefs=8, gold_beliefs="annotation", belief_choice=False, answer_eval="llm_judge", doc_max_length=512, query_max_length=8192,
    ),
    "webshaper": DatasetSpec(
        dataset_year=None, subset="sft", query_key="text", min_relevance_score=1,
        task="qa", query_shape="multi_clue", max_beliefs=None, gold_beliefs="question", belief_choice=False, answer_eval="llm_judge", doc_max_length=512, query_max_length=8192,
    ),
}

# Datasets built by this project, kept under ``TRAINING_DATASETS_ROOT``
# (the others are downloaded inputs under ``DATA_ROOT``).
BUILT_DATASETS = frozenset({"infoseek_hard", "trqa_val", "frames", "webshaper"})

# Splits of a training dataset: ``all`` holds every query of the dataset as
# built, one row per query with its ``split`` (training_datasets/annotation/rows.py);
# ``sft``, ``rl`` and ``val`` are views of it, the rows with that split and
# their qrels (``is_split_view``; training_datasets/pipeline/splits.py), for belief
# model training.
ALL_SPLIT = "all"
TRAINING_SPLITS = ("sft", "rl", "val")

# Built datasets that search another dataset's corpus and indices.
_SHARED_CORPUS = {"trqa_val": "trqa"}

# Built datasets whose corpus is a downloaded one, kept with its indices
# under ``DATA_ROOT/{folder}``: (folder, corpus file stem).  The folder is
# named after the corpus's release, not the dataset: InfoSeek's
# (corpus_wiki; webshaper's web gold pages are
# ``TRAINING_DATASETS_ROOT/corpus/corpus_web.jsonl`` until the user merges
# them into it, ``training_datasets/corpus/``).
_CORPUS_IN_DATA_ROOT = {name: ("infoseek", "wiki-25-512") for name in ("infoseek_hard", "frames", "webshaper")}

# Datasets with a local corpus + index under ``DATA_ROOT/{dataset}``.
DATASETS = tuple(DATASET_SPECS)


def build_dataset_dir_name(dataset: str, split: str, query_key: str, retriever: str) -> str:
    """Name of the per-dataset output directory: ``{dataset}_{split}[_{query_key}]_{retriever}``.

    The query key is part of the name only when it is not the dataset's
    default (layout.DATASET_SPECS), e.g. ``neuclir_2024_news_qwen3_emb_4b`` for
    the default ``request`` and ``neuclir_2024_news_topic_title_qwen3_emb_4b``.
    """
    qk_part = f"_{query_key}" if query_key != DATASET_SPECS[dataset].query_key else ""
    return f"{dataset}_{split}{qk_part}_{retriever}"


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

def dataset_root(dataset: str) -> Path:
    """Root folder of a dataset: ``TRAINING_DATASETS_ROOT`` (shared) for
    the built datasets, ``DATA_ROOT/{dataset}`` for the downloaded ones."""
    return TRAINING_DATASETS_ROOT if dataset in BUILT_DATASETS else DATA_ROOT / dataset


def corpus_root(dataset: str) -> Path:
    """Folder of a dataset's corpus and indices: ``DATA_ROOT/{folder}``
    for a built dataset with a downloaded corpus (``_CORPUS_IN_DATA_ROOT``),
    else its ``dataset_root`` (a dataset searching another's corpus:
    that one's)."""
    dataset = _SHARED_CORPUS.get(dataset, dataset)
    if dataset in _CORPUS_IN_DATA_ROOT:
        return DATA_ROOT / _CORPUS_IN_DATA_ROOT[dataset][0]
    return dataset_root(dataset)


def queries_base(data_path: Path | str, split: str, dataset: str | None = None) -> Path:
    """Return the suffix-less queries path, e.g. ``.../queries/queries_test``;
    for a training *dataset* (``BUILT_DATASETS``) its one file of every
    split, ``.../queries/{dataset}``."""
    return Path(data_path) / "queries" / (dataset if dataset in BUILT_DATASETS else f"queries_{split}")


def qrels_base(data_path: Path | str, split: str, dataset: str | None = None) -> Path:
    """Return the suffix-less qrels path, e.g. ``.../qrels/qrels_test``;
    for a training *dataset* ``.../qrels/{dataset}`` (``queries_base``)."""
    return Path(data_path) / "qrels" / (dataset if dataset in BUILT_DATASETS else f"qrels_{split}")


def nuggets_base(data_path: Path | str, split: str) -> Path:
    """Return the suffix-less nuggets path, e.g. ``.../nuggets/nuggets_2025``."""
    return Path(data_path) / "nuggets" / f"nuggets_{split}"


# The two kinds of beliefs (``belief_source`` of dra_inference and of
# belief model training): gold (from the question or the dataset's
# annotation, ``DatasetSpec.gold_beliefs``) and silver (extracted from
# the question by an LLM where the question does not state its structure:
# the members of a TRQA set).
GOLD_BELIEFS, SILVER_BELIEFS = "gold", "silver"
BELIEF_SOURCES = (GOLD_BELIEFS, SILVER_BELIEFS)
# Where the gold beliefs come from (``DatasetSpec.gold_beliefs``).
QUESTION_GOLD, ANNOTATION_GOLD = "question", "annotation"


def extraction_kind(dataset: str) -> str:
    """The kind of the beliefs the extractor (the run's ``belief_model``) lists for
    *dataset*: gold where the gold beliefs are the question's own
    (``QUESTION_GOLD``), else silver."""
    return GOLD_BELIEFS if DATASET_SPECS[dataset].gold_beliefs == QUESTION_GOLD else SILVER_BELIEFS


def belief_bank_path(dataset: str, split: str, kind: str | None = None) -> Path:
    """Return the bank of the beliefs the extractor lists for a split
    (*kind*: ``extraction_kind`` by default), e.g.
    ``BELIEF_ROOT/ragtime/belief_bank/gold/beliefs_2025.jsonl`` or
    ``BELIEF_ROOT/trqa/belief_bank/silver/beliefs_wiki2_test.jsonl``.
    A training dataset has one bank for all its splits, in the folder of
    the training datasets, e.g.
    ``BELIEF_ROOT/training_datasets/belief_bank/gold/infoseek_hard.jsonl``
    or ``BELIEF_ROOT/training_datasets/belief_bank/silver/trqa_val.jsonl``.
    The belief model reads and adds to them (``belief_model_component.init.factory``)."""
    kind = kind or extraction_kind(dataset)
    if kind not in BELIEF_SOURCES:
        raise ValueError(f"unknown belief kind {kind!r}; expected one of {BELIEF_SOURCES}")
    if dataset in BUILT_DATASETS:
        return BELIEF_ROOT / TRAINING_DATASETS_ROOT.name / "belief_bank" / kind / f"{dataset}.jsonl"
    return BELIEF_ROOT / dataset / "belief_bank" / kind / f"beliefs_{split}.jsonl"


def is_split_view(dataset: str | None, split: str) -> bool:
    """Whether *split* of *dataset* is a view of a training dataset's
    split ``all`` (``sft``, ``rl`` or ``val`` of a dataset of
    ``BUILT_DATASETS``): its queries are the rows of the dataset's
    queries file with that ``split``, its qrels theirs."""
    return split in TRAINING_SPLITS and dataset in BUILT_DATASETS


def gold_beliefs_path(dataset: str) -> Path:
    """Return the rows of a training dataset: its queries file, one row
    per query (every split), e.g.
    ``TRAINING_DATASETS_ROOT/queries/trqa_val.jsonl``
    (``training_datasets/annotation/rows.py``); the gold beliefs of a
    dataset with ``gold_beliefs == "annotation"`` are made from its
    members (those of a
    ``"question"`` one are the extractor's, ``belief_bank_path``).  TRQA's gold beliefs are made
    on demand into its gold bank (``belief_bank_path(..., "gold")``,
    ``belief_model_component/init/trqa_gold.py``); the other datasets' are the extractor's."""
    if dataset not in BUILT_DATASETS:
        raise ValueError(f"gold belief files exist for the training datasets only ({sorted(BUILT_DATASETS)}), "
                         f"not {dataset!r}")
    return queries_base(dataset_root(dataset), ALL_SPLIT, dataset).with_suffix(".jsonl")


def effective_belief_source(requested: str, dataset: str) -> str:
    """The belief source a run of *dataset* uses: the *requested* one on
    a dataset with a choice (``DatasetSpec.belief_choice``, TRQA), else
    gold (each query's gold beliefs, silver for a query without them)."""
    if requested not in BELIEF_SOURCES:
        raise ValueError(f"unknown belief source {requested!r}; expected one of {BELIEF_SOURCES}")
    return requested if DATASET_SPECS[dataset].belief_choice else GOLD_BELIEFS


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
      training datasets -> subset or the dataset's default split (TRAINING_SPLITS, ALL_SPLIT)
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
    if dataset in BUILT_DATASETS:
        return subset or DATASET_SPECS[dataset].subset
    return subset or "set1"


# ===========================================================================
# Index-build path derivation (corpus + index dir from dataset/subset)
# ===========================================================================

def default_corpus_path(
    dataset: str, subset: str | None = None, *, partial: bool = True
) -> Path:
    """Canonical full-corpus JSONL path for a dataset/subset (:func:`dataset_root`).

    The corpus file stem encodes the subset (NeuCLIR ``corpus_en_news``, TRQA
    ``wiki_partial``/``ecommerce``), so an index built from this path is already
    named per dataset/subset by ``Index_Builder``.  Used by both the production
    index builder and its test to avoid passing ``--corpus-path`` by hand.
    """
    if dataset == "neuclir":
        return DATA_ROOT / "neuclir" / "corpus" / f"{corpus_name(subset)}.jsonl"
    if dataset in _SHARED_CORPUS:
        return default_corpus_path(_SHARED_CORPUS[dataset], DATASET_SPECS[_SHARED_CORPUS[dataset]].subset, partial=partial)
    if dataset in _CORPUS_IN_DATA_ROOT:
        return corpus_path(corpus_root(dataset), _CORPUS_IN_DATA_ROOT[dataset][1])
    if dataset == "trqa":
        return DATA_ROOT / "trqa" / "corpus" / f"{trqa_corpus_name(subset, partial)}.jsonl"
    if dataset == "ragtime":
        return DATA_ROOT / "ragtime" / "corpus" / f"{RAGTIME_CORPUS_NAME}.jsonl"
    if dataset == "browsecomp_plus":
        return DATA_ROOT / "browsecomp_plus" / "corpus" / "corpus.jsonl"
    # An unknown dataset (best-effort canonical location).
    return corpus_path(dataset_root(dataset))


def default_index_dir(dataset: str) -> Path:
    """Canonical index output directory for a dataset, ``{dataset_root}/indices``.

    Sibling of the ``corpus/`` directory, matching the production layout
    (``corpus_root``): a dataset that searches another dataset's corpus uses
    that dataset's indices, InfoSeek's those under ``DATA_ROOT/infoseek``.
    """
    return corpus_root(dataset) / "indices"
