"""Read dataset queries, qrels, answers, and nuggets from the canonical layout.

The on-disk layout, path builders, split-id rule, and record schemas live in
:mod:`indexing_corpus_dataset.layout` (the single source of truth, shared with
``download_datasets.py``).  This module is the *reader* half: given a
``data_path`` and a split, it parses queries / qrels / answers.

All functions take an explicit ``data_path`` so they work regardless of the
current working directory.  Split ``sft``, ``rl`` or ``val`` of a training
pool is a view of its split ``all`` (``layout.is_pool_view``): the rows of
``queries_all.jsonl`` with that ``split`` and their qrels.
"""

from pathlib import Path
import csv
import json

# Layout vocabulary (single source of truth).  ``resolve_split_id`` and
# ``resolve_data_path`` are re-exported for the
# ``from ...dataset_loaders import resolve_split_id`` call sites.
from .layout import (  # noqa: F401
    POOL_ALL_SPLIT,
    is_pool,
    is_pool_view,
    queries_base,
    qrels_base,
    nuggets_base,
    resolve_split_id,
    resolve_data_path,
    _resolve_split_file,
)


# ===========================================================================
# Training pool views
# ===========================================================================

# Pool queries file -> ((mtime_ns, size), {split: ids}): the queries, qrels
# and intermediate info of a view read the file once.
_VIEW_IDS: dict[Path, tuple[tuple[int, int], dict[str, set[str]]]] = {}


def _pool_view_ids(data_path: Path | str, split: str) -> set[str] | None:
    """The ids of the queries of pool view *split* (``layout.is_pool_view``),
    or None when *split* is no view."""
    if not is_pool_view(data_path, split):
        return None
    file_path = queries_base(data_path, POOL_ALL_SPLIT).with_suffix(".jsonl")
    if not file_path.exists():
        return set()
    stat = file_path.stat()
    version = (stat.st_mtime_ns, stat.st_size)
    cached = _VIEW_IDS.get(file_path)
    if cached is None or cached[0] != version:
        by_split: dict[str, set[str]] = {}
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    by_split.setdefault(row.get("split"), set()).add(str(row["id"]))
        cached = _VIEW_IDS[file_path] = (version, by_split)
    return cached[1].get(split, set())


def _file_split(data_path: Path | str, split: str) -> str:
    """The split whose files hold *split*: ``all`` for a pool view."""
    return POOL_ALL_SPLIT if is_pool_view(data_path, split) else split


# ===========================================================================
# Queries + answers (single-pass)
# ===========================================================================

def _load_queries_and_answers(
    data_path: Path | str, split: str, query_key: str = "text"
) -> tuple[dict[str, str], dict[str, str]]:
    """Parse the queries file once, returning (queries, answers).

    ``answers`` is populated only from JSONL records that carry an ``"answer"``
    field (e.g. BrowseComp-Plus); it is empty for TSV files or datasets without
    answers.
    """
    view = _pool_view_ids(data_path, split)
    file_path = _resolve_split_file(queries_base(data_path, _file_split(data_path, split)), (".tsv", ".jsonl"))
    if file_path is None:
        print(f"Error reading queries file: No valid file found for "
              f"{data_path}/queries/queries_{split}")
        return {}, {}

    queries: dict[str, str] = {}
    answers: dict[str, str] = {}
    n_records = 0
    try:
        if file_path.suffix == ".jsonl":
            # Supports standard {"id","text"} and NeuCLIR {"topic_id","topic_title",...}.
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    qid = obj.get("id") or obj.get("topic_id")
                    if not qid or (view is not None and str(qid) not in view):
                        continue
                    n_records += 1
                    text = obj.get(query_key)
                    if text:
                        queries[qid] = text
                    answer = obj.get("answer")
                    if answer:
                        answers[qid] = answer
            if len(queries) < n_records:
                # e.g. NeuCLIR: only the report-generation topics carry "request".
                print(f"{len(queries)} of {n_records} queries have '{query_key}'; "
                      f"the rest are skipped")
        else:  # .tsv
            with open(file_path, "r", encoding="utf-8") as f:
                reader = csv.reader(f, delimiter="\t")
                for rows in reader:
                    if len(rows) >= 2:
                        queries[rows[0]] = rows[1]
    except Exception as e:
        print(f"Error reading queries file {file_path}: {e}")

    return queries, answers


def load_queries(
    data_path: Path | str, data_set: str = "set1", query_key: str = "text"
) -> dict[str, str]:
    """Load queries from a dataset directory.

    Args:
        data_path:  Path to dataset directory.
        data_set:   Split identifier (e.g. "test", "2024_news").
        query_key:  Key in each JSONL record to use as the query text.
                    The per-dataset default is in layout.DATASET_SPECS
                    ("request" for neuclir, "text" for the others).
    """
    return _load_queries_and_answers(data_path, data_set, query_key)[0]


# ===========================================================================
# Qrels
# ===========================================================================

def load_qrels(
    data_path: Path | str, data_set: str = "set1", min_relevance_score: int | None = None
) -> dict[str, dict[str, int]]:
    """Load qrels from a dataset directory.

    Args:
        data_path: Path to dataset directory.
        data_set: Split identifier (default: "set1").
        min_relevance_score: If set, only passages with relevance
            >= min_relevance_score are kept.

    Returns:
        ``{query_id: {doc_id: relevance_score}}``
    """
    view = _pool_view_ids(data_path, data_set)
    file_path = _resolve_split_file(qrels_base(data_path, _file_split(data_path, data_set)), (".tsv", ".txt"))
    if file_path is None:
        print(f"Error reading qrels file: No valid file found for "
              f"{data_path}/qrels/qrels_{data_set}")
        return {}

    qrels: dict[str, dict[str, int]] = {}
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            if file_path.suffix == ".txt":
                # TREC qrels: "query_id 0 doc_id relevance" (space-separated)
                rows = (line.split() for line in f)
                parsed = ((r[0], r[2], int(r[3])) for r in rows if len(r) >= 4)
            else:  # .tsv: "query_id \t doc_id \t relevance"
                reader = csv.reader(f, delimiter="\t")
                parsed = ((r[0], r[1], int(r[2])) for r in reader if len(r) >= 3)

            for qid, docid, rel in parsed:
                if min_relevance_score is not None and rel < min_relevance_score:
                    continue
                if view is not None and qid not in view:
                    continue
                qrels.setdefault(qid, {})[docid] = rel
    except Exception as e:
        print(f"Error reading qrels file {file_path}: {e}")
        return {}

    return qrels


# ===========================================================================
# Nuggets
# ===========================================================================

def load_nuggets(data_path: Path | str, data_set: str) -> dict[str, list[dict]]:
    """Load report-generation nuggets from a dataset directory.

    Returns ``{query_id: [nugget, ...]}`` where each nugget is
    ``{"id", "question", "answers", "importance", "support_docs"}``; empty for
    datasets without a nuggets file.
    """
    file_path = nuggets_base(data_path, data_set).with_suffix(".jsonl")
    if not file_path.exists():
        return {}
    nuggets: dict[str, list[dict]] = {}
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                nuggets[rec["id"]] = rec["nuggets"]
    return nuggets


def load_query_intermediate_info(data_path: Path | str, data_set: str) -> dict[str, dict]:
    """Load the per-query intermediate information (TRQA).

    Reads ``queries/queries_{split}_intermediate_info.jsonl`` and returns
    ``{query_id: record}``, where a record is ``{"qid", "entity_values":
    [{"entity", "value", "passage_ids"}], "property": {"label", ...}, "aggregation",
    "answer"}``.  A training pool has no such file: its query rows hold the
    information (``members``, ``property``, ``aggregation``;
    ``training_pool/annotations.py``), returned in the same record format
    for the rows with members.  Empty for datasets without either.
    """
    if is_pool(data_path):
        return _pool_intermediate_info(data_path, data_set)
    base = queries_base(data_path, data_set)
    file_path = base.with_name(f"{base.name}_intermediate_info.jsonl")
    if not file_path.exists():
        return {}
    info: dict[str, dict] = {}
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                info[str(rec["qid"])] = rec
    return info


def _pool_intermediate_info(data_path: Path | str, data_set: str) -> dict[str, dict]:
    """The intermediate information of the query rows of a training pool
    (``load_query_intermediate_info``)."""
    view = _pool_view_ids(data_path, data_set)
    file_path = queries_base(data_path, _file_split(data_path, data_set)).with_suffix(".jsonl")
    if not file_path.exists():
        return {}
    info: dict[str, dict] = {}
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if not row.get("members"):
                continue
            qid = str(row["id"])
            if view is not None and qid not in view:
                continue
            info[qid] = {
                "qid": qid,
                "entity_values": [{"entity": m["entity"], "value": m["value"], "passage_ids": m["doc_ids"]}
                                  for m in row["members"]],
                "property": {"label": row.get("property")},
                "aggregation": row.get("aggregation"),
                "answer": row.get("answer"),
            }
    return info


# ===========================================================================
# One-pass convenience loader
# ===========================================================================

def load_split(
    data_path: Path | str,
    data_set: str = "set1",
    *,
    query_key: str = "text",
    min_relevance_score: int | None = None,
    only_with_qrels: bool = False,
) -> tuple[dict[str, str], dict[str, dict[str, int]], dict[str, str]]:
    """Load queries, qrels, and answers for a split in a single call.

    The queries file is read only once (answers come from the same records).

    Args:
        data_path:           Path to dataset directory.
        data_set:            Split identifier (e.g. "test", "2024_news").
        query_key:           Key in each JSONL record to use as the query text.
        min_relevance_score: Minimum qrel score to keep (None = keep all).
        only_with_qrels:     If True, drop queries that have no qrels entry.

    Returns:
        ``(queries, qrels, answers)``.
    """
    queries, answers = _load_queries_and_answers(data_path, data_set, query_key)
    qrels = load_qrels(data_path, data_set, min_relevance_score)

    if only_with_qrels:
        queries = {qid: q for qid, q in queries.items() if qid in qrels}

    return queries, qrels, answers
