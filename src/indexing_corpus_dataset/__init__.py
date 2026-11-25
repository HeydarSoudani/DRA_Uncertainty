"""Corpus, dataset, and index tooling for retrieval research.

Provides the canonical on-disk dataset layout and the readers around it,
imported across the repo (``shared/``):

    shared/layout.py           canonical paths, split-id + corpus-name rules (the data
                               root itself: ``utils.paths``)
    shared/dataset_loaders.py  read queries / qrels / answers / nuggets from that layout
    shared/doc_lookup.py       fetch corpus documents by id (byte-offset index)

and one subpackage per job:

    download/             fetch TRQA / BrowseComp-Plus / NeuCLIR / RAGTIME into the canonical layout
    indexing/             build BM25 / SPLADE / dense indexes over a corpus (configs in
                          experiments/configs/index_build*.yaml)
    training_datasets/    build the training datasets and their gold annotation

Canonical layout::

    {data_path}/queries/queries_{split}.jsonl
    {data_path}/qrels/qrels_{split}.txt
    {data_path}/nuggets/nuggets_{split}.jsonl
    {data_path}/corpus/{name}.jsonl
"""

from utils.paths import DATA_ROOT

from .shared.layout import (
    queries_base,
    qrels_base,
    nuggets_base,
    corpus_path,
    corpus_name,
    resolve_split_id,
    default_corpus_path,
    default_index_dir,
)
from .shared.dataset_loaders import (
    load_queries,
    load_qrels,
    load_nuggets,
    load_query_intermediate_info,
    load_split,
)

__all__ = [
    "DATA_ROOT",
    "load_queries",
    "load_qrels",
    "load_nuggets",
    "load_query_intermediate_info",
    "load_split",
    "resolve_split_id",
    "queries_base",
    "qrels_base",
    "nuggets_base",
    "corpus_path",
    "corpus_name",
    "default_corpus_path",
    "default_index_dir",
]
