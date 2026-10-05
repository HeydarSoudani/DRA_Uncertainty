"""Per-query gold units.

The dataset's ``criteria_gold`` (``layout.DATASET_SPECS``) names the gold:

* ``"entities"`` (TRQA): the entities of the query's set, from the
  intermediate information.  One unit per entity; ``text`` is its name.
* ``"nuggets"`` (NeuCLIR, RAGTIME): the nugget questions.  Nuggets that share
  a question (one per answer in the NeuCLIR bank) are one unit; it is
  ``vital`` when any of them is, and its support documents are pooled.
* None (BrowseComp-Plus): no gold; the units are a reference criteria list
  (:func:`load_reference_criteria`), e.g. the clues extracted by an earlier
  prompt.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from indexing_corpus_dataset.dataset_loaders import load_nuggets, load_query_intermediate_info
from indexing_corpus_dataset.layout import DATASET_SPECS


@dataclass
class GoldUnit:
    """One gold unit of a query.

    ``importance`` is ``vital`` / ``okay`` for NeuCLIR nuggets, else None;
    ``num_support_docs`` is None when the gold has no support documents.
    """
    id: str
    text: str
    importance: Optional[str] = None
    num_support_docs: Optional[int] = None

    def to_dict(self) -> Dict:
        return {"id": self.id, "text": self.text, "importance": self.importance,
                "num_support_docs": self.num_support_docs}


def _entity_units(data_path: Path, split: str) -> Dict[str, List[GoldUnit]]:
    units = {}
    for qid, rec in load_query_intermediate_info(data_path, split).items():
        names = []
        for ev in rec.get("entity_values") or []:
            name = str(ev.get("entity", "")).strip()
            if name and name not in names:
                names.append(name)
        units[qid] = [GoldUnit(id=f"e{i + 1}", text=n) for i, n in enumerate(names)]
    return units


def _nugget_units(data_path: Path, split: str) -> Dict[str, List[GoldUnit]]:
    units = {}
    for qid, nuggets in load_nuggets(data_path, split).items():
        grouped: Dict[str, Dict] = {}
        for n in nuggets:
            question = str(n.get("question", "")).strip()
            if not question:
                continue
            g = grouped.setdefault(question, {"importance": None, "docs": set(), "has_docs": False})
            if n.get("importance") == "vital" or (n.get("importance") and g["importance"] is None):
                g["importance"] = n["importance"]
            if n.get("support_docs"):
                g["has_docs"] = True
                g["docs"].update(n["support_docs"])
        if grouped:
            units[qid] = [
                GoldUnit(id=f"n{i + 1}", text=q, importance=g["importance"],
                         num_support_docs=len(g["docs"]) if g["has_docs"] else None)
                for i, (q, g) in enumerate(grouped.items())
            ]
    return units


def load_reference_criteria(path: Path | str) -> Dict[str, List[Dict]]:
    """Criteria lists ``{query_id: [{"id", "text", "kind"}, ...]}`` from either
    a ``criteria.jsonl`` written by ``python -m evaluation.criteria`` or a
    run's ``uncertainty/`` directory (the meta line of each file)."""
    path = Path(path)
    criteria: Dict[str, List[Dict]] = {}
    files = sorted(path.glob("*.jsonl")) if path.is_dir() else [path]
    for f in files:
        with open(f, "r", encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                rec = json.loads(line)
                if rec.get("record", "meta") == "meta" and isinstance(rec.get("criteria"), list):
                    criteria[str(rec["query_id"])] = rec["criteria"]
                if path.is_dir():
                    break  # one meta line per uncertainty file
    return criteria


def load_gold_units(
    dataset: str,
    data_path: Path | str,
    split: str,
    reference: Optional[Path | str] = None,
) -> Dict[str, List[GoldUnit]]:
    """``{query_id: [GoldUnit, ...]}`` for the dataset's ``criteria_gold``;
    with *reference* the units are that criteria list instead (required when
    the dataset has no gold)."""
    if reference is not None:
        return {
            qid: [GoldUnit(id=f"r{i + 1}", text=str(c["text"])) for i, c in enumerate(items)]
            for qid, items in load_reference_criteria(reference).items() if items
        }
    kind = DATASET_SPECS[dataset].criteria_gold
    if kind == "entities":
        return _entity_units(Path(data_path), split)
    if kind == "nuggets":
        return _nugget_units(Path(data_path), split)
    raise ValueError(f"{dataset} has no criteria gold; give a reference criteria list")
