"""Stage data_prepare: the questions of a training dataset worth its gold
annotation (``prepare``), with no LLM call.

    raw       the source's records (``datasets/{dataset}/source.py``), nothing
              removed, written to ``DatasetPaths.raw_questions``
              (data/training_datasets_raw/{dataset}) and read from there
              afterwards; written again when the code that makes them
              changed (``source_code_hash`` in ``source.json``)
    filters   the checks of the dataset, in order
              (``datasets/{dataset}/prepare.py``; the shared ones in
              ``checks.py``, the funnel in ``funnel.py``); a question fails
              at its first failed check, recorded in ``dropped.jsonl``
              (stage ``prepare``) with the filter and the reasons
    prepared  the questions that pass every check, each record with its
              ``features`` (the chunk ids of its gold pages and what the
              checks measured), to ``DatasetPaths.prepared``
              (data/training_datasets_raw/{dataset}/prepared.jsonl), the
              input of stage data_relevance (``relevance.py``)

The ``prepare`` section of ``stats.json`` holds the funnel (per filter the
questions in and the drops per reason), the hash of the prepared file and
the dataset's own counts.  A rerun writes the same output from the raw
file and the caches (titles looked up, web pages fetched): it costs nothing
but the corpus reads.
"""

import hashlib
import json
from pathlib import Path
from typing import List

from utils.io import atomic_write_text, read_jsonl, write_jsonl

from .. import registry
from ..config import BuildConfig, DatasetPaths
from .funnel import Funnel
from .stats import write_dropped, write_stats

# The stage of the questions dropped here, in ``dropped.jsonl``.
STAGE = "prepare"


# The package folder and the modules besides ``datasets/{dataset}/`` that
# make the records of a source (``source_code_hash``).
_PACKAGE = Path(__file__).resolve().parents[1]
_SHARED_SOURCE_CODE = ("pipeline/source.py", "corpus/pages.py", "annotation/names.py", "annotation/rows.py")


def file_hash(path) -> str:
    """The first 16 hex digits of the SHA-256 of a file's bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def source_code_hash(dataset: str) -> str:
    """The hash of the code that turns the source into records: the
    dataset's package and the shared modules it builds them with.  A
    change of any of them (a change of the annotation, the gold entries)
    rebuilds the raw file (``load_raw``); one of the package's other
    modules rebuilds it too, at no cost but the source read."""
    files = sorted((_PACKAGE / "datasets" / dataset).glob("*.py")) + [_PACKAGE / p for p in _SHARED_SOURCE_CODE]
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.relative_to(_PACKAGE).as_posix().encode("utf-8") + b"\0" + path.read_bytes() + b"\0")
    return digest.hexdigest()[:16]


def load_raw(paths: DatasetPaths) -> List[dict]:
    """The source's records: read from the raw folder, (re)written there
    first when it does not hold them or holds them from other code
    (``source_code_hash``, recorded in ``source.json``)."""
    code = source_code_hash(paths.dataset)
    recorded = (json.loads(paths.raw_source.read_text(encoding="utf-8")).get("code")
                if paths.raw_source.exists() else None)
    if not paths.raw_questions.exists() or recorded != code:
        if paths.raw_questions.exists():
            print(f"[prepare] the source code changed ({recorded} -> {code}): rebuilding {paths.raw_questions}")
        source = registry.source(paths.dataset)
        records = list(source.load())
        write_jsonl(paths.raw_questions, records)
        atomic_write_text(paths.raw_source, json.dumps({**source.info(), "questions": len(records), "code": code},
                                                       indent=2, ensure_ascii=False) + "\n")
        print(f"[prepare] {len(records)} source questions -> {paths.raw_questions}")
    return read_jsonl(paths.raw_questions)


def prepare(cfg: BuildConfig, paths: DatasetPaths) -> None:
    records = load_raw(paths)
    funnel = Funnel(records)
    extra = registry.prepare(paths.dataset).run(cfg, paths, funnel)

    write_jsonl(paths.prepared, funnel.kept)
    write_dropped(paths.dropped, STAGE, funnel.dropped)
    write_stats(paths.stats, STAGE, {"raw": len(records), "kept": len(funnel.kept), "pending": len(funnel.pending),
                                     "dropped": len(funnel.dropped), "filters": funnel.steps,
                                     "prepared_hash": file_hash(paths.prepared), **extra})
    print(f"[prepare] {paths.dataset}: {len(funnel.kept)} of {len(records)} questions kept, "
          f"{len(funnel.dropped)} dropped" + (f", {len(funnel.pending)} pending (rerun)" if funnel.pending else "")
          + f" -> {paths.prepared}")
    print(funnel.table())
