"""The gold table a question's answer is aggregated from.

One dataset record -> one :class:`GoldTable`: the entity set the agent's grid is
expected to enumerate (its rows) and the single property it is expected to
establish about each of them (one of its columns).

The shape comes from TRQA's ``queries_{split}_intermediate_info.jsonl``, read by
``indexing_corpus_dataset.load_intermediate_info``.  Two properties of that file
drive how the metrics read:

* ``entity_values`` is the **pre-filter universe** -- every entity of the
  question's class, not only the ones satisfying its condition.  A question over
  "states whose mean age is at least Virginia's" still lists all 50 states.  So
  entity recall measures whether the planner enumerated the universe, which is what
  GridFill's design asks of it; it is not an accuracy ceiling, and an agent that
  filters during planning can answer correctly while scoring below 1.0.
* ``property`` names only the **aggregated** property.  A question that also
  filters on a second property does not record that one, so the agent
  legitimately needs columns this gold cannot account for -- which is why there
  is column *recall* here but deliberately no column precision.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .matching import normalize_entity


@dataclass
class GoldTable:
    """The gold rows and single gold column for one query."""

    qid: str
    entities: List[str] = field(default_factory=list)
    values: Dict[str, Any] = field(default_factory=dict)
    property_label: str = ""
    property_description: str = ""
    datatype: str = ""
    aggregation: str = ""

    @classmethod
    def from_record(cls, record: Dict[str, Any]) -> Optional["GoldTable"]:
        """Build from one ``intermediate_info`` record, or None if unusable.

        A record with no entities has no rows to score against and is dropped
        rather than counted as a zero, which would silently depress recall.

        Entities that normalise to nothing are dropped for the same reason.
        The gold contains a couple of these -- one is literally named ``"( )"``
        -- and no agent can enumerate a name with no content, so holding a run
        responsible for it would put recall permanently out of reach.
        """
        entity_values = record.get("entity_values") or []
        entities: List[str] = []
        values: Dict[str, Any] = {}
        for item in entity_values:
            if not isinstance(item, dict):
                continue
            entity = str(item.get("entity") or "").strip()
            if not entity or entity in values or not normalize_entity(entity):
                continue
            entities.append(entity)
            values[entity] = item.get("value")

        if not entities:
            return None

        prop = record.get("property") or {}
        return cls(
            qid=str(record.get("qid") or record.get("id") or ""),
            entities=entities,
            values=values,
            property_label=str(prop.get("label") or ""),
            property_description=str(prop.get("description") or ""),
            datatype=str(prop.get("datatype") or ""),
            aggregation=str(record.get("aggregation") or ""),
        )


def load_gold_tables(records: Dict[str, Dict[str, Any]]) -> Dict[str, GoldTable]:
    """Convert ``{qid: record}`` into ``{qid: GoldTable}``, dropping unusable ones."""
    tables: Dict[str, GoldTable] = {}
    for qid, record in (records or {}).items():
        gold = GoldTable.from_record(record)
        if gold is not None:
            tables[qid] = gold
    return tables
