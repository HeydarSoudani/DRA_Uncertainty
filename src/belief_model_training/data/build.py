"""Stages data_prepare and data_relevance: a training dataset
(``indexing_corpus_dataset.training_datasets``), one call per dataset.

data_prepare: the source's questions (written to
data/training_datasets_raw/{dataset}, again when the code that makes them
changes), filtered with no LLM call into
data/training_datasets_raw/{dataset}/prepared.jsonl.  data_relevance: the gold
annotation, qrels and split of the prepared questions, with the LLM calls
(the fact writer and the judge: ``--fact-model``, ``--judge-model``);
incremental: ``limit`` processes the next N prepared questions not
processed yet (0: none, the dataset files are rewritten; None: all of
them).  The settings are the ``data`` section of the training config.  One
stage of a dataset runs at a time (a lock file in its folder).  See
``experiments/training/README.md``.
"""

import fcntl
import logging
from pathlib import Path
from typing import Optional

from ..shared.config import TrainingConfig


def _lock(root: Path):
    """An exclusive lock on ``{root}/.build.lock`` (the dataset's build
    folder), held until the process exits; exits when another stage of the
    dataset holds it."""
    root.mkdir(parents=True, exist_ok=True)
    fh = open(root / ".build.lock", "w")
    try:
        fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit(f"another stage of {root} is running (lock {root / '.build.lock'}); wait for it to finish")
    return fh


def _paths(dataset: str):
    from indexing_corpus_dataset.training_datasets.config import DatasetPaths

    try:
        return DatasetPaths(dataset)
    except ValueError as e:
        raise SystemExit(f"--dataset {dataset}: {e}")


def prepare_dataset(cfg: TrainingConfig, dataset: str) -> None:
    from indexing_corpus_dataset.training_datasets import prepare

    paths = _paths(dataset)
    lock = _lock(paths.raw)  # noqa: F841 - held until exit
    print(f"[stage data_prepare] {dataset}: build folder {paths.raw}")
    prepare(cfg.build_config(), paths)


def relevance_dataset(cfg: TrainingConfig, dataset: str, limit: Optional[int], judge_model: str,
                      fact_model: str) -> None:
    """*judge_model*, *fact_model*: the judge of the gold passages and the
    fact writer (``--judge-model``, ``--fact-model``)."""
    from indexing_corpus_dataset.training_datasets import build_relevance

    # LiteLLM and its HTTP client log every call of the writer and the judge at INFO.
    for name in ("LiteLLM", "httpx"):
        logging.getLogger(name).setLevel(logging.WARNING)
    paths = _paths(dataset)
    lock = _lock(paths.raw)  # noqa: F841 - held until exit
    print(f"[stage data_relevance] {dataset}: build folder {paths.raw}, dataset {paths.queries}, {paths.qrels}")
    build = cfg.build_config()
    build.gold_passages.model = judge_model
    build.facts.model = fact_model
    build_relevance(build, paths, limit)
