"""Per-criterion gold of a run's queries: the gold answers and documents of
each criterion, for the belief update evaluation
(``evaluation.belief.evaluator``).

Where the gold rows are (the training datasets' row format,
``training_datasets/gold/annotations.py``: ``criteria[]`` with ``id, text, answers,
doc_ids, kind``, plus ``entity_names`` and ``source``):

- a training dataset (infoseek, webshaper_frames, trqa_validation): its queries file
  (``layout.gold_criteria_path``), one row per query;
- TRQA: its gold bank (``layout.criteria_bank_path(..., "gold")``), the row
  kept in each line's ``bank_only`` (``training_datasets.gold.trqa``); a query may
  have several lines (other writers), so every row is returned;
- any other dataset: none (its criteria are the question's clues, with no
  answer per criterion).

A row is the gold of a run's query only when its criterion texts equal the
criteria the run used, in order (``match_row``): the run then used these
gold criteria, and the ids line up.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from indexing_corpus_dataset.layout import BUILT_DATASETS, criteria_bank_path, gold_criteria_path

#: Datasets whose queries have a gold answer and documents per criterion.
GOLD_DATASETS = ("trqa",) + tuple(sorted(BUILT_DATASETS))


def _rows(path: Path, bank: bool) -> Dict[str, List[Dict[str, Any]]]:
    out: Dict[str, List[Dict[str, Any]]] = {}
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        row = ((rec.get("bank_only") or {}).get("row") if bank else rec)
        if row and row.get("criteria"):
            out.setdefault(str(row["id"] if not bank else rec["query_id"]), []).append(row)
    return out


def load_gold_rows(dataset: str, split: Optional[str]) -> Dict[str, List[Dict[str, Any]]]:
    """``{query_id: [gold row, ...]}`` of *dataset* (*split*: the file-level
    split id, e.g. ``wiki2_test``, for TRQA's bank); empty for a dataset
    without per-criterion gold."""
    if dataset in BUILT_DATASETS:
        return _rows(gold_criteria_path(dataset), bank=False)
    if dataset == "trqa" and split:
        return _rows(criteria_bank_path(dataset, split, "gold"), bank=True)
    return {}


def match_row(rows: List[Dict[str, Any]], criteria: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """The row of *rows* whose criterion texts equal the run's *criteria*
    (``[{id, text, kind}]``, the belief file's meta line), in order, or None."""
    texts = [c.get("text") for c in criteria]
    for row in rows:
        if [c.get("text") for c in row["criteria"]] == texts:
            return row
    return None
