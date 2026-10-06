"""Per-query gold units.

The dataset's ``criteria_gold`` (``layout.DATASET_SPECS``) names the gold:

* ``"entities"`` (TRQA): the entities of the query's set, from the
  intermediate information.  One unit per entity; ``text`` is its name.
* ``"nuggets"`` (NeuCLIR, RAGTIME): the nugget questions of the Auto-ARGUE
  nugget banks the report evaluation scores (``answer.argue.build_nugget_banks``):
  nuggets that share a question are one unit, ``vital`` when any of them is,
  with their gold answers and AND/OR aggregator; answers without documents
  and questions without answers are dropped, and so is a query left with
  none.  Both evaluations thus score the same nuggets of the same queries.
* None (BrowseComp-Plus): no gold; the units are a reference criteria list
  (:func:`load_reference_criteria`), e.g. the clues extracted by an earlier
  prompt.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from indexing_corpus_dataset.dataset_loaders import load_nuggets, load_query_intermediate_info
from indexing_corpus_dataset.layout import DATASET_SPECS

from ..answer.argue import build_nugget_banks
from ..common import read_jsonl
from ..uncertainty import load_uncertainty_meta


@dataclass
class GoldUnit:
    """One gold unit of a query.

    ``importance`` is ``vital`` / ``okay`` for NeuCLIR nuggets, else None.
    A nugget question also has its gold ``answers`` and ``aggregator``
    (``OR``: one answer answers it; ``AND``: all of them).
    """
    id: str
    text: str
    importance: Optional[str] = None
    answers: List[str] = field(default_factory=list)
    aggregator: Optional[str] = None


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
    for qid, bank in build_nugget_banks(load_nuggets(data_path, split), {}).items():
        units[qid] = [
            GoldUnit(id=f"n{i + 1}", text=nq.question, importance=nq.importance,
                     answers=[a.answer for a in (nq.answers or {}).values()],
                     aggregator=getattr(nq.aggregator_type, "value", nq.aggregator_type) or "OR")
            for i, nq in enumerate(bank.nuggets_as_list() or [])
        ]
    return units


def load_reference_criteria(path: Path | str) -> Dict[str, List[Dict]]:
    """Criteria lists ``{query_id: [{"id", "text", "kind"}, ...]}`` from either
    a ``criteria.jsonl`` written by ``python -m evaluation.criteria`` or a
    run's ``uncertainty/`` directory (the meta line of each file)."""
    path = Path(path)
    records = load_uncertainty_meta(path).values() if path.is_dir() else read_jsonl(path)
    return {str(r["query_id"]): r["criteria"] for r in records
            if r.get("record", "meta") == "meta" and isinstance(r.get("criteria"), list)}


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
