"""Per-belief gold of a run's queries: the gold answers and documents of
each belief, for the belief update evaluation
(``evaluation.belief.evaluator``).

Where the gold rows are (the training datasets' row format,
``training_datasets/gold/annotations.py``: ``beliefs[]`` with ``id, text, answers,
doc_ids, kind``, plus ``entity_names`` and ``source``):

- a training dataset (infoseek, webshaper_frames, trqa_validation): its queries file
  (``layout.gold_beliefs_path``), one row per query;
- TRQA: its gold bank (``layout.belief_bank_path(..., "gold")``), the row
  kept in each line's ``bank_only`` (``training_datasets.gold.trqa``); a query may
  have several lines (other writers), so every row is returned;
- any other dataset: none (its beliefs are the question's clues, with no
  answer per belief).

A row is the gold of a run's query only when its belief texts equal the
beliefs the run used, in order (``match_row``): the run then used these
gold beliefs, and the ids line up.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from indexing_corpus_dataset.layout import BUILT_DATASETS, belief_bank_path, gold_beliefs_path

#: Datasets whose queries have a gold answer and documents per belief.
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
        if row and row.get("beliefs"):
            out.setdefault(str(row["id"] if not bank else rec["query_id"]), []).append(row)
    return out


def load_gold_rows(dataset: str, split: Optional[str]) -> Dict[str, List[Dict[str, Any]]]:
    """``{query_id: [gold row, ...]}`` of *dataset* (*split*: the file-level
    split id, e.g. ``wiki2_test``, for TRQA's bank); empty for a dataset
    without per-belief gold."""
    if dataset in BUILT_DATASETS:
        return _rows(gold_beliefs_path(dataset), bank=False)
    if dataset == "trqa" and split:
        return _rows(belief_bank_path(dataset, split, "gold"), bank=True)
    return {}


def match_row(rows: List[Dict[str, Any]], beliefs: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """The row of *rows* whose belief texts equal the run's *beliefs*
    (``[{id, text, kind}]``, the belief file's meta line), in order, or None."""
    texts = [c.get("text") for c in beliefs]
    for row in rows:
        if [c.get("text") for c in row["beliefs"]] == texts:
            return row
    return None
