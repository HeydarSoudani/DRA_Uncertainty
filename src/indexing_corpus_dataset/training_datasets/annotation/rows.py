"""The rows of a training dataset: one per query, in its queries file
(``queries/{dataset}.jsonl`` of ``TRAINING_DATASETS_ROOT``), with the query, its split and its gold
annotation: its beliefs with their gold answers and gold documents, as far
as the source states them.

Every row has the same keys, whatever the source; what the source does not
state is an empty value (``[]`` or ``None``), never a missing key.

Query row::

    {"id", "source",
     "split": "sft" | "rl" | "val" | None,       (``pipeline/splits.py``)
     "text", "answer",          the question and its final answer
     "query_shape": "multi_hop" | "multi_member",
     "level": "full" | "partial" | "query",      (``level``)
     "doc_ids": [...],          the query's gold documents (its qrels;
                                frames, webshaper: those of grade 2)
     "beliefs": [belief],   [] when the source has no structure
     "members": [{"entity", "value", "doc_ids"}],   multi-member queries; else []
     "entity_names": [...],     every entity name of the gold structure
                                (InfoSeek's tree, TRQA's members;
                                ``names.mentions`` skips an answer inside a
                                longer one); else []
     "property": str | None,    the property of a multi-member query; else None
     "aggregation": str | None, how a multi-member query's answer follows from
                                the members' values (TRQA, e.g.
                                ``AVG_TOP_K``); else None
     "texts_ok": bool | None,   the belief texts (belief_texts.py):
                                true written (or no beliefs), false
                                failed for good (a leak after the retry,
                                no text for a belief), None not written
                                yet (retried next run)
     "errors": [...]}           why the texts failed

The writer of the texts and the hash of its prompts are recorded once, in
the ``annotations`` section of the dataset's ``stats.json`` (and per query in
``belief_texts.jsonl``, the writer's memory).

Belief::

    {"id": "B1",
     "text": str | None,        written by the writer (belief_texts.py)
     "answers": [...], "doc_ids": [...],
     "kind": "closed" | "aggregate",
     "relations": [...]}        the source structure the writer
                                reads (hop relations with placeholders E1,
                                E2, ... for the beliefs, by position)

What each source states:

    infoseek     one belief per entity of the research tree the question
                 does not name, children before parents, the answer last;
                 answers = the entity, doc_ids = the gold chunks
                 (wiki-25-512) of its own Wikipedia page and of the claims
                 it is an end of (``datasets/infoseek_hard/``)
    webshaper    one belief per variable of the formalization, the
                 farthest from the answer first, then the answer ("?");
                 answers = the variable's name (named by the fact writer,
                 ``datasets/webshaper/facts.py``; the answer: the final
                 answer), doc_ids = the gold passages of the triples it
                 occurs in (``gold_passages/``)
    frames       no beliefs (final answer and qrels only)
    trqa_wiki2   multi_member: one belief per member, "<member>: <property>"
                 (answers = its value, doc_ids = its documents,
                 ``entity_values[].passage_ids``), then the aggregate
                 belief (how the
                 answer follows from the values; answers = the final answer,
                 no doc_ids: it is computed); members = TRQA's
                 ``entity_values``, its documents as doc_ids, each named by
                 the Wikipedia title of its first document ("San Fernando,
                 La Union", not TRQA's label "San Fernando"; the label
                 stays among ``entity_names``)
    trqa_wiki1   the same

The source loaders (``datasets/{dataset}/source.py``) build the skeletons
(beliefs with the source id they stand for, ``ref``: InfoSeek's node id,
WebShaper's variable), the build gives the documents and answers of each
ref (``row``); the trqa_val build writes its rows directly
(``multi_member_row``).  Then an LLM writes the belief texts
(``belief_texts.fill_texts``) and the split sets ``split``
(``pipeline/relevance.finish``).
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

from utils.io_utils import write_jsonl

from .names import entity_names

# The keys of a query row and of a belief, in their order.
QUERY_KEYS = ("id", "source", "split", "text", "answer", "query_shape", "level", "doc_ids", "beliefs", "members",
              "entity_names", "property", "aggregation", "texts_ok", "errors")
BELIEF_KEYS = ("id", "text", "answers", "doc_ids", "kind", "relations")
# The kinds of belief (belief_model_component.core.types): an aggregate
# one has no documents (it is computed).
CLOSED, AGGREGATE = "closed", "aggregate"
KINDS = (CLOSED, AGGREGATE)
MEMBER_KEYS = ("entity", "value", "doc_ids")
FULL = "full"
LEVELS = (FULL, "partial", "query")


def has_gold(c: dict) -> bool:
    """A belief has its gold: answers and documents; an aggregate one
    answers only (no document states a computed value)."""
    if c["kind"] == AGGREGATE:
        return bool(c["answers"])
    return bool(c["answers"] and c["doc_ids"])


def level(beliefs: List[dict]) -> str:
    """``full``: every belief has its gold (``has_gold``); ``partial``:
    some belief lacks it; ``query``: no beliefs."""
    if not beliefs:
        return "query"
    return "full" if all(has_gold(c) for c in beliefs) else "partial"


def problems(row: dict) -> List[str]:
    """What breaks the row format in *row* (``[]``: none)."""
    out = []
    if tuple(row) != QUERY_KEYS:
        out.append(f"query keys {sorted(row)}")
    for c in row.get("beliefs") or []:
        if tuple(c) != BELIEF_KEYS:
            out.append(f"belief keys {sorted(c)}")
    for m in row.get("members") or []:
        if tuple(m) != MEMBER_KEYS:
            out.append(f"member keys {sorted(m)}")
    if row.get("level") not in LEVELS or ("beliefs" in row and row["level"] != level(row["beliefs"])):
        out.append(f"level {row.get('level')!r}")
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


def _belief(cid: str, kind: str, relations: List[str], answers: List[str], doc_ids: List[str]) -> dict:
    return {"id": cid, "text": None, "answers": answers, "doc_ids": doc_ids, "kind": kind, "relations": relations}


def _row(qid: str, source: str, text: str, answer: str, shape: str, doc_ids: List[str], beliefs: List[dict],
         members: List[dict], names: List[str], prop: Optional[str], aggregation: Optional[str]) -> dict:
    return {"id": qid, "source": source, "split": None, "text": text, "answer": answer, "query_shape": shape,
            "level": level(beliefs), "doc_ids": doc_ids, "beliefs": beliefs, "members": members,
            "entity_names": names, "property": prop, "aggregation": aggregation, "texts_ok": None, "errors": []}


# ── Rows ─────────────────────────────────────────────────────────────────────

def row(record: dict, doc_ids: List[str], ref_docs: Dict[str, List[str]],
        ref_answers: Optional[Dict[str, List[str]]] = None) -> dict:
    """The annotation row of a kept question: its gold documents *doc_ids*
    and its skeleton (``record["annotation"]``) with the documents of each
    belief's ``ref`` (*ref_docs*) and, for a belief without the source's
    answers, those of *ref_answers*; a ref without documents or answers
    gives none."""
    ref_answers = ref_answers or {}
    skel = record["annotation"]
    beliefs = [_belief(c["id"], c["kind"], c["relations"], c["answers"] or _dedup(ref_answers.get(c["ref"], [])),
                       _dedup(ref_docs.get(c["ref"], [])))
               for c in skel["beliefs"]]
    return _row(record["id"], record["source"], record["question"], record["answer"], skel["query_shape"],
                _dedup(doc_ids), beliefs, [], skel["entity_names"], None, None)


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


def multi_member_row(qid: str, source: str, query: str, answer: str, info: dict, qrels: List[str],
            titles: Dict[str, str]) -> dict:
    """The annotation row of a TRQA query from its intermediate info
    (``entity_values``, ``property``, ``aggregation``): one belief per
    member with its value and documents (``passage_ids``), then the
    aggregate belief with the final answer.  A member
    is named by the title of its first document (*titles*: doc id -> page
    title), or by TRQA's label when it has none."""
    labels, members = [], []
    for v in info.get("entity_values") or []:
        doc_ids = _dedup(v.get("passage_ids") or [])
        label = str(v["entity"])
        labels.append(label)
        members.append({"entity": titles.get(doc_ids[0], label) if doc_ids else label, "value": v.get("value"),
                        "doc_ids": doc_ids})
    beliefs = []
    if members:
        beliefs = [_belief(f"B{k}", CLOSED, [], value_texts(m["value"]), m["doc_ids"])
                    for k, m in enumerate(members, 1)]
        n = len(beliefs)
        # The final answer as the query states it, and a whole float without its ".0".
        answers = _dedup([answer, *value_texts(info.get("answer"))]) if answer else value_texts(info.get("answer"))
        beliefs.append(_belief(f"B{n + 1}", AGGREGATE, [], answers, []))
    prop = (info.get("property") or {}).get("label") or None
    names = _dedup(n for m, label in zip(members, labels) for name in (m["entity"], label)
                   for n in entity_names(name))
    return _row(qid, source, query, answer, "multi_member", qrels, beliefs, members, names, prop, info.get("aggregation"))
