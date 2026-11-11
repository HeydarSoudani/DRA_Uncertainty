"""The rows of a training dataset: one per query, in its queries file
(``queries/queries_all.jsonl``), with the query, its split and its gold
annotation: its criteria with their gold answers and gold documents, as far
as the source states them.

Every row has the same keys, whatever the source; what the source does not
state is an empty value (``[]`` or ``None``), never a missing key.

Query row::

    {"id", "source",
     "split": "sft" | "rl" | "val" | None,       (``splits.py``)
     "text", "answer",          the question and its final answer
     "query_shape": "multi_hop" | "multi_member",
     "level": "full" | "partial" | "query",      (``level``)
     "doc_ids": [...],          the query's qrels
     "criteria": [criterion],   [] when the source has no structure
     "members": [{"entity", "value", "doc_ids"}],   multi-member queries; else []
     "entity_names": [...],     every entity name of the gold structure
                                (InfoSeek's tree, TRQA's members;
                                ``mentions`` skips an answer inside a
                                longer one); else []
     "property": str | None,    the property of a multi-member query; else None
     "aggregation": str | None, how a multi-member query's answer follows from
                                the members' values (TRQA, e.g.
                                ``AVG_TOP_K``); else None
     "texts_ok": bool | None,   the criterion texts (criterion_texts.py):
                                true written (or no criteria), false
                                failed for good (a leak after the retry,
                                no text for a criterion), None not written
                                yet (retried next run)
     "errors": [...]}           why the texts failed

The writer of the texts and the hash of its prompts are recorded once, in
the ``annotations`` section of the dataset's ``stats.json`` (and per query in
``criterion_texts.jsonl``, the writer's memory).

Criterion::

    {"id": "c1",
     "text": str | None,        written by the writer (criterion_texts.py)
     "answers": [...], "doc_ids": [...],
     "kind": "closed" | "rest" | "aggregate",
     "relations": [...]}        the source structure the writer
                                reads (hop relations with placeholders E1,
                                E2, ... for the criteria, by position)

What each source states:

    infoseek     one criterion per entity of the research tree the question
                 does not name, children before parents, the answer last;
                 answers = the entity, doc_ids = the first corpus chunk
                 (wiki-25-512) of its Wikipedia page and of the pages whose
                 claims link to it (the claim was written from the start of
                 the expanded node's page)
    webshaper    one criterion per variable of the formalization, the
                 farthest from the answer first, then the answer ("?");
                 only the answer has answers (the final answer); no doc_ids
                 (the URLs are the query's)
    frames       no criteria (final answer and qrels only)
    trqa_wiki2   multi_member: one criterion per member, "<member>: <property>"
                 (answers = its value, doc_ids = its documents,
                 ``entity_values[].passage_ids``), then the rest criterion
                 "any other member: <property>" (answers and doc_ids empty:
                 every member has a criterion; training may hide some
                 members under it), then the aggregate criterion (how the
                 answer follows from the values; answers = the final answer,
                 no doc_ids: it is computed); members = TRQA's
                 ``entity_values``, its documents as doc_ids, each named by
                 the Wikipedia title of its first document ("San Fernando,
                 La Union", not TRQA's label "San Fernando"; the label
                 stays among ``entity_names``)
    trqa_wiki1   the same

The loaders build the skeletons (criteria with the page keys of their
documents, ``gold_keys``), the build resolves the keys to documents
(``row``: webshaper_frames the fetched page, infoseek the first corpus
chunk of the page); the trqa_validation build writes its rows directly
(``multi_member_row``).  Each then has an LLM write the criterion texts
(``criterion_texts.fill_texts``), and the split sets ``split``.
"""

import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from utils.io_utils import write_jsonl

# The keys of a query row and of a criterion, in their order.
QUERY_KEYS = ("id", "source", "split", "text", "answer", "query_shape", "level", "doc_ids", "criteria", "members",
              "entity_names", "property", "aggregation", "texts_ok", "errors")
CRITERION_KEYS = ("id", "text", "answers", "doc_ids", "kind", "relations")
# The kinds of criterion (belief_model_component.core.types): a rest criterion
# has no gold of its own, an aggregate one no documents (it is computed).
CLOSED, REST, AGGREGATE = "closed", "rest", "aggregate"
KINDS = (CLOSED, REST, AGGREGATE)
MEMBER_KEYS = ("entity", "value", "doc_ids")
FULL = "full"
LEVELS = (FULL, "partial", "query")


def has_gold(c: dict) -> bool:
    """A criterion has its gold: answers and documents; a rest criterion
    none (its members have their own criteria), an aggregate one answers
    only (no document states a computed value)."""
    if c["kind"] == REST:
        return True
    if c["kind"] == AGGREGATE:
        return bool(c["answers"])
    return bool(c["answers"] and c["doc_ids"])


def level(criteria: List[dict]) -> str:
    """``full``: every criterion has its gold (``has_gold``); ``partial``:
    some criterion lacks it; ``query``: no criteria."""
    if not criteria:
        return "query"
    return "full" if all(has_gold(c) for c in criteria) else "partial"


def problems(row: dict) -> List[str]:
    """What breaks the row format in *row* (``[]``: none)."""
    out = []
    if tuple(row) != QUERY_KEYS:
        out.append(f"query keys {sorted(row)}")
    for c in row.get("criteria") or []:
        if tuple(c) != CRITERION_KEYS:
            out.append(f"criterion keys {sorted(c)}")
    for m in row.get("members") or []:
        if tuple(m) != MEMBER_KEYS:
            out.append(f"member keys {sorted(m)}")
    if row.get("level") not in LEVELS or ("criteria" in row and row["level"] != level(row["criteria"])):
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


def _criterion(cid: str, kind: str, relations: List[str], answers: List[str], doc_ids: List[str]) -> dict:
    return {"id": cid, "text": None, "answers": answers, "doc_ids": doc_ids, "kind": kind, "relations": relations}


def _row(qid: str, source: str, text: str, answer: str, shape: str, doc_ids: List[str], criteria: List[dict],
         members: List[dict], names: List[str], prop: Optional[str], aggregation: Optional[str]) -> dict:
    return {"id": qid, "source": source, "split": None, "text": text, "answer": answer, "query_shape": shape,
            "level": level(criteria), "doc_ids": doc_ids, "criteria": criteria, "members": members,
            "entity_names": names, "property": prop, "aggregation": aggregation, "texts_ok": None, "errors": []}


# ── Matching ─────────────────────────────────────────────────────────────────

def normalize(text: str) -> str:
    """Lowercase ASCII-folded words, punctuation removed, spaces collapsed."""
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


def strip_parenthetical(title: str) -> str:
    """A title without its disambiguation ("Mercury (planet)" -> "Mercury")."""
    return re.sub(r"\s*\([^)]*\)\s*$", "", title).strip()


def entity_names(entity: str) -> List[str]:
    """The names of an entity to match: as given, and without a trailing
    parenthetical."""
    return _dedup(n for n in (entity.strip(), strip_parenthetical(entity)) if n)


def mentions(text: str, name: str, others: List[str]) -> bool:
    """*text* contains *name* as whole words outside the longer names of
    *others* that contain it ("Groningen" inside "University of Groningen"
    does not count)."""
    n = normalize(name)
    if not n:
        return False
    padded = f" {normalize(text)} "
    for other in others:
        o = normalize(other)
        if n in o and o != n:
            padded = padded.replace(f" {o} ", " | ")
    return f" {n} " in padded


# ── InfoSeek ─────────────────────────────────────────────────────────────────

def tree_nodes(node: dict) -> Iterator[dict]:
    """The nodes of a research tree, *node* first."""
    yield node
    for child in node.get("children") or []:
        yield from tree_nodes(child)


def _post_order(node: dict) -> Iterator[dict]:
    for child in node.get("children") or []:
        yield from _post_order(child)
    yield node


def is_entity(node: dict) -> bool:
    """A tree node that stands for an entity (an attribute node, such as a
    population figure, has none)."""
    return str(node.get("entity") or "").strip() not in ("", "None")


def named(question: str, node: dict, nodes) -> bool:
    """The question names *node*: one of its names occurs in the question
    outside the names of other nodes that contain it."""
    others = [name for o in nodes if o is not node and is_entity(o) for name in entity_names(o["entity"])]
    return any(mentions(question, own, others) for own in entity_names(node["entity"]))


def infoseek_skeleton(question: str, root: dict, gold_key) -> dict:
    """The hops of a research tree: the nodes the question does not name
    (the root, the answer, always), children before parents.  Each hop's
    relations name the other hops by placeholder (E1, E2, ...) and the named
    entities by name; *gold_key* maps a node's ``href`` to its page key.  A
    hop's pages are its own and those of the nodes whose claims link to it:
    InfoSeek writes a claim from the page of the node it expands (the first
    1024 tokens), so the link to a hop is stated on the linking node's page."""
    nodes = {n["id"]: n for n in tree_nodes(root)}
    linked_from: Dict[str, List[dict]] = {}
    for n in nodes.values():
        for claim in n.get("claims") or []:
            if claim.get("target_id") in nodes and claim["target_id"] != n["id"]:
                linked_from.setdefault(claim["target_id"], []).append(n)
    hops = [n for n in _post_order(root) if n is root or (is_entity(n) and not named(question, n, nodes.values()))]
    label = {n["id"]: f"E{k}" for k, n in enumerate(hops, 1)}
    relations: Dict[str, List[str]] = {n["id"]: [] for n in hops}
    for n in nodes.values():
        for claim in n.get("claims") or []:
            target = claim.get("target_id")
            # Both ends in one pass: a name put in may contain the other id
            # as a word ("Robert E. Lee", node E).  Node ids are capital
            # letters, so an id after a quote or a period, or before a period
            # and a letter, is text ("an 'A' rating", "B.A."), not the node.
            ends = [node_id for node_id in (n["id"], target) if node_id in nodes]
            text = re.sub(rf"(?<![\w'\".])({'|'.join(map(re.escape, ends))})(?!\w|\.\w)",
                          lambda m: label.get(m.group(1)) or nodes[m.group(1)]["entity"], claim["claim"])
            # A relation describes the hops it links.
            for node_id in (n["id"], target):
                if node_id in relations and text not in relations[node_id]:
                    relations[node_id].append(text)
    def keys(n: dict) -> List[str]:
        return _dedup(gold_key(m["href"]) for m in [n, *linked_from.get(n["id"], [])] if m.get("href"))

    criteria = [{"id": f"c{k}", "kind": CLOSED, "relations": relations[n["id"]],
                 "answers": entity_names(n["entity"]), "gold_keys": keys(n)}
                for k, n in enumerate(hops, 1)]
    names = [name for n in nodes.values() if is_entity(n) for name in entity_names(n["entity"])]
    return {"query_shape": "multi_hop", "criteria": criteria, "entity_names": names}


# ── WebShaper ────────────────────────────────────────────────────────────────

_ANSWER = "?"
_VARIABLE = re.compile(r"V@[A-Za-z0-9_]+")
_PACKED = " & "


def _unpack(triple: list) -> List[List[str]]:
    """A triple as its terms; a packed one (``[s, "p1 & p2", "o1 & o2"]``,
    predicates and objects aligned) as one triple per predicate."""
    terms = [str(t).strip() for t in triple]
    if len(terms) == 3 and _PACKED in terms[1]:
        preds, objs = terms[1].split(_PACKED), terms[2].split(_PACKED)
        if len(preds) == len(objs):
            return [[terms[0], p.strip(), o.strip()] for p, o in zip(preds, objs)]
    return [terms]


def _term_nodes(terms: List[str]) -> List[str]:
    """The variables (anywhere in a term) and the answer of a triple."""
    found = [v for t in terms for v in _VARIABLE.findall(t)]
    if _ANSWER in terms:
        found.append(_ANSWER)
    return list(dict.fromkeys(found))


def webshaper_skeleton(triples: List[list], answer: str) -> dict:
    """One criterion per variable (``V@...``) of the formalization, then the
    answer (``?``, its occurrences one node).  The variables are ordered
    farthest from the answer first (in the graph of the triples), ties by
    first occurrence; a variable not linked to the answer comes first.  Each
    criterion's relations are the triples it occurs in, the variables and
    the answer as placeholders, the constants (``C@...``) as text."""
    triples = [t for triple in triples for t in _unpack(triple)]
    order: List[str] = []
    edges: Dict[str, set] = {}
    for terms in triples:
        nodes = _term_nodes(terms)
        for t in nodes:
            if t not in edges:
                edges[t] = set()
                if t != _ANSWER:
                    order.append(t)
            edges[t].update(n for n in nodes if n != t)
    dist = {_ANSWER: 0}
    frontier = [_ANSWER]
    while frontier:
        nxt = []
        for node in frontier:
            for other in edges.get(node, ()):
                if other not in dist:
                    dist[other] = dist[node] + 1
                    nxt.append(other)
        frontier = nxt
    hops = sorted(order, key=lambda v: (-dist.get(v, len(order) + 1), order.index(v))) + [_ANSWER]
    label = {v: f"E{k}" for k, v in enumerate(hops, 1)}

    def term(t: str) -> str:
        if t == _ANSWER:
            return label[_ANSWER]
        t = _VARIABLE.sub(lambda m: label.get(m.group(0), m.group(0)), t)
        return t.replace("C@", "").replace("_", " ").strip()

    relations: Dict[str, List[str]] = {v: [] for v in hops}
    for terms in triples:
        text = " ".join(x for x in (term(t) for t in terms) if x)
        for node in _term_nodes(terms):
            if text not in relations[node]:
                relations[node].append(text)
    criteria = [{"id": f"c{k}", "kind": CLOSED, "relations": relations[v],
                 "answers": [answer] if v == _ANSWER and answer else [], "gold_keys": []}
                for k, v in enumerate(hops, 1)]
    return {"query_shape": "multi_hop", "criteria": criteria, "entity_names": []}


def no_skeleton() -> dict:
    """A source without structure (FRAMES): the query only."""
    return {"query_shape": "multi_hop", "criteria": [], "entity_names": []}


# ── Rows ─────────────────────────────────────────────────────────────────────

def row(record: dict, docs_of: Dict[str, List[str]]) -> dict:
    """The annotation row of a kept question: its skeleton
    (``record["annotation"]``) with the page keys resolved to their
    documents, *docs_of* (page key -> doc ids: webshaper_frames the fetched
    page, infoseek its first corpus chunk); a key without documents gives none."""
    def docs(keys) -> List[str]:
        return _dedup(d for k in keys for d in docs_of.get(k, ()))

    skel = record["annotation"]
    criteria = [_criterion(c["id"], c["kind"], c["relations"], c["answers"], docs(c["gold_keys"]))
                for c in skel["criteria"]]
    return _row(record["id"], record["source"], record["question"], record["answer"], skel["query_shape"],
                docs(g["key"] for g in record["gold"]), criteria, [], skel["entity_names"], None, None)


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
    (``entity_values``, ``property``, ``aggregation``): one criterion per
    member with its value and documents (``passage_ids``), the rest
    criterion, and the aggregate criterion with the final answer.  A member
    is named by the title of its first document (*titles*: doc id -> page
    title), or by TRQA's label when it has none."""
    labels, members = [], []
    for v in info.get("entity_values") or []:
        doc_ids = _dedup(v.get("passage_ids") or [])
        label = str(v["entity"])
        labels.append(label)
        members.append({"entity": titles.get(doc_ids[0], label) if doc_ids else label, "value": v.get("value"),
                        "doc_ids": doc_ids})
    criteria = []
    if members:
        criteria = [_criterion(f"c{k}", CLOSED, [], value_texts(m["value"]), m["doc_ids"])
                    for k, m in enumerate(members, 1)]
        n = len(criteria)
        criteria.append(_criterion(f"c{n + 1}", REST, [], [], []))
        # The final answer as the query states it, and a whole float without its ".0".
        answers = _dedup([answer, *value_texts(info.get("answer"))]) if answer else value_texts(info.get("answer"))
        criteria.append(_criterion(f"c{n + 2}", AGGREGATE, [], answers, []))
    prop = (info.get("property") or {}).get("label") or None
    names = _dedup(n for m, label in zip(members, labels) for name in (m["entity"], label)
                   for n in entity_names(name))
    return _row(qid, source, query, answer, "multi_member", qrels, criteria, members, names, prop, info.get("aggregation"))
