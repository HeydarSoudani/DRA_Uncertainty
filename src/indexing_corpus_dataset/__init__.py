"""Corpus, dataset, and index tooling for retrieval research.

Provides the canonical on-disk dataset layout and the readers/writers around it:

    layout.py             canonical paths, data root, split-id + corpus-name rules
    download_datasets.py  fetch TRQA / BrowseComp-Plus / NeuCLIR / RAGTIME into the canonical layout
    dataset_loaders.py    read queries / qrels / answers / nuggets from that layout
    index_builder.py      build BM25 / SPLADE / dense indexes over a corpus

Canonical layout::

    {data_path}/queries/queries_{split}.jsonl
    {data_path}/qrels/qrels_{split}.txt
    {data_path}/nuggets/nuggets_{split}.jsonl
    {data_path}/corpus/{name}.jsonl
"""

from .layout import (
    DATA_ROOT,
    queries_base,
    qrels_base,
    nuggets_base,
    corpus_path,
    corpus_name,
    resolve_split_id,
    resolve_data_path,
    default_corpus_path,
    default_index_dir,
)
from .dataset_loaders import (
    load_queries,
    load_qrels,
    graded_qrels,
    load_query_answers,
    load_nuggets,
    load_split,
)

__all__ = [
    "DATA_ROOT",
    "load_queries",
    "load_qrels",
    "graded_qrels",
    "load_query_answers",
    "load_nuggets",
    "load_split",
    "resolve_split_id",
    "resolve_data_path",
    "queries_base",
    "qrels_base",
    "nuggets_base",
    "corpus_path",
    "corpus_name",
    "default_corpus_path",
    "default_index_dir",
]
