"""The rows of a training dataset: one per query, in its queries file
(``queries/{dataset}.jsonl`` of ``TRAINING_DATASETS_ROOT``), with the query,
its answer, its split and its gold passages (stage data_relevance).  The
rows hold no beliefs: the belief model makes its gold beliefs when it first
needs them, as at inference (``belief_model_component/init``): from the
question (infoseek_hard, webshaper, frames: multi-clue), or, for a
multi-member row (trqa_val), from the row's members
(``init/trqa_gold.py``).

Every row has the same keys, whatever the source; what the source does not
state is an empty value (``[]`` or ``None``), never a missing key.

Query row::

    {"id", "source",
     "split": "sft" | "rl" | "val" | None,       (``pipeline/splits.py``)
     "text", "answer",          the question and its final answer
     "doc_ids": [...],          the query's gold passages (its qrels,
                                binary)
     "members": [{"entity", "value", "doc_ids"}],   multi-member queries; else []
     "entity_names": [...],     every entity name of the source structure
                                (InfoSeek's tree, TRQA's members;
                                ``names.mentions`` skips an answer inside a
                                longer one); else []
     "property": str | None,    the property of a multi-member query; else None
     "aggregation": str | None} how a multi-member query's answer follows from
                                the members' values (TRQA, e.g.
                                ``AVG_TOP_K``); else None

A multi-member row (trqa_val, ``member_row``) has TRQA's
``entity_values`` as ``members``, each with its value and documents
(``passage_ids``), named by the Wikipedia title of its first document
("San Fernando, La Union", not TRQA's label "San Fernando"; the label stays
among ``entity_names``).  The split sets ``split``
(``pipeline/relevance.finish``).

The prepared records keep their source skeleton (``annotation``: InfoSeek's
tree, WebShaper's variables, data_prepare); this stage does not use it
beyond its entity names.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

from utils.io_utils import write_jsonl

from .names import entity_names

# The keys of a query row and of a member, in their order.
QUERY_KEYS = ("id", "source", "split", "text", "answer", "doc_ids", "members", "entity_names", "property",
              "aggregation")
MEMBER_KEYS = ("entity", "value", "doc_ids")
# The kinds of a source skeleton's beliefs (data_prepare, ``annotation``)
# and of the beliefs made from a multi-member row
# (belief_model_component.core.types): an aggregate one has no documents
# (it is computed).
CLOSED, AGGREGATE = "closed", "aggregate"


def problems(row: dict) -> List[str]:
    """What breaks the row format in *row* (``[]``: none)."""
    out = []
    if tuple(row) != QUERY_KEYS:
        out.append(f"query keys {sorted(row)}")
    for m in row.get("members") or []:
        if tuple(m) != MEMBER_KEYS:
            out.append(f"member keys {sorted(m)}")
    return out


def check(rows: List[dict]) -> None:
    """Refuse a row that breaks the row format (``problems``)."""
    for r in rows:
        bad = problems(r)
        if bad:
            raise SystemExit(f"gold annotation of {r['id']}: {'; '.join(bad)}")


def write(path: Path, rows: List[dict]) -> None:
    """Write the rows of the dataset (``check``ed)."""
    check(rows)
    write_jsonl(path, rows)


def _dedup(items) -> list:
    return list(dict.fromkeys(items))


def _row(qid: str, source: str, text: str, answer: str, doc_ids: List[str], members: List[dict],
         names: List[str], prop: Optional[str], aggregation: Optional[str]) -> dict:
    return {"id": qid, "source": source, "split": None, "text": text, "answer": answer, "doc_ids": doc_ids,
            "members": members, "entity_names": names, "property": prop, "aggregation": aggregation}


# ── Rows ─────────────────────────────────────────────────────────────────────

def row(record: dict, doc_ids: List[str]) -> dict:
    """The row of a kept question with its gold passages *doc_ids*."""
    names = (record.get("annotation") or {}).get("entity_names") or []
    return _row(record["id"], record["source"], record["question"], record["answer"], _dedup(doc_ids), [],
                list(names), None, None)


def value_texts(value: Any) -> List[str]:
    """A gold value as answer texts: one per item of a list, a whole float
    without its ".0"."""
    values = value if isinstance(value, list) else [value]
    out = []
    for v in values:
        if v is None or str(v).strip() == "":
            continue
        out.append(str(int(v)) if isinstance(v, float) and v.is_integer() else str(v))
    return out


def member_row(qid: str, source: str, query: str, answer: str, info: dict, qrels: List[str],
               titles: Dict[str, str]) -> dict:
    """The row of a TRQA query from its intermediate info
    (``entity_values``, ``property``, ``aggregation``): its members with
    their values and documents (``passage_ids``).  A member is named by the
    title of its first document (*titles*: doc id -> page title), or by
    TRQA's label when it has none."""
    labels, members = [], []
    for v in info.get("entity_values") or []:
        doc_ids = _dedup(v.get("passage_ids") or [])
        label = str(v["entity"])
        labels.append(label)
        members.append({"entity": titles.get(doc_ids[0], label) if doc_ids else label, "value": v.get("value"),
                        "doc_ids": doc_ids})
    prop = (info.get("property") or {}).get("label") or None
    names = _dedup(n for m, label in zip(members, labels) for name in (m["entity"], label)
                   for n in entity_names(name))
    return _row(qid, source, query, answer, qrels, members, names, prop, info.get("aggregation"))
