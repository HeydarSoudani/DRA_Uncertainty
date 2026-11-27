"""The training datasets: one package each under ``datasets/``, with one
module per stage, imported when the stage runs:

    source      ``load()`` the source's records (``pipeline/source.py``),
                ``info()`` where they come from
    prepare     ``run(cfg, paths, funnel)`` the checks of stage data_prepare
                on the funnel (``pipeline/prepare.py``); returns the
                dataset's own counts of the ``prepare`` stats section
    relevance   ``build(cfg, paths, processed)`` the rows and qrels of stage
                data_relevance (``pipeline/relevance.py``, a ``Built``);
                ``FACTS`` the dataset's fact LLM (a ``FactSpec``, None
                without one), ``JUDGE`` whether it calls the judge of
                the gold passages and ``CHUNK_WINDOW`` the passages of a
                page the judge reads (its first ones; None without a
                judge)

A new dataset is a package with these modules, listed in ``DATASETS`` and
in ``layout.BUILT_DATASETS`` (its folders and corpus)."""

import importlib
from types import ModuleType

from indexing_corpus_dataset.shared.layout import BUILT_DATASETS

DATASETS = ("infoseek_hard", "frames", "webshaper", "trqa_val")

if set(DATASETS) != BUILT_DATASETS:
    raise ImportError(f"training datasets {sorted(DATASETS)} differ from layout.BUILT_DATASETS "
                      f"{sorted(BUILT_DATASETS)}")


def _module(dataset: str, stage: str) -> ModuleType:
    if dataset not in DATASETS:
        raise ValueError(f"unknown training dataset {dataset!r}; expected one of {sorted(DATASETS)}")
    return importlib.import_module(f"{__package__}.datasets.{dataset}.{stage}")


def source(dataset: str) -> ModuleType:
    return _module(dataset, "source")


def prepare(dataset: str) -> ModuleType:
    return _module(dataset, "prepare")


def relevance(dataset: str) -> ModuleType:
    return _module(dataset, "relevance")
