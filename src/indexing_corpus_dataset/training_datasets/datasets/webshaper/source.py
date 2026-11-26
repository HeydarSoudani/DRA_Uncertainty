"""webshaper's source: WebShaper (``Alibaba-NLP/WebShaper``), questions with
their source URLs and formalization triples; the skeleton is one belief per
variable (``skeleton``)."""

from typing import Dict, Iterator, List

from utils.io import iter_jsonl

from ...annotation.rows import CLOSED
from ...corpus.pages import dedup_gold, url_gold
from ...pipeline.source import hf_file, hf_info
from .formalization import ANSWER, VARIABLE, term_nodes, unpack

WEBSHAPER = ("Alibaba-NLP/WebShaper", "data/main-00000-of-00001.jsonl", "5dd17c63c2d0f86d245c262bfd7bf444200e126f")


def info() -> dict:
    return hf_info(WEBSHAPER)


def load() -> Iterator[dict]:
    for row in iter_jsonl(hf_file(WEBSHAPER)):
        triples = row.get("formalization") or []
        entries = [url_gold(u) for u in row.get("urls") or []]
        answer = str(row["answer"]).strip()
        yield {
            "id": f"webshaper_{row['id']}", "source": "webshaper",
            "question": row["question"].strip().strip('"').strip(), "answer": answer,
            "gold": dedup_gold(entries),
            "annotation": skeleton(triples, answer),
            "meta": {"ignored_urls": entries.count(None), "formalization": triples},
        }


def skeleton(triples: List[list], answer: str) -> dict:
    """One belief per variable (``V@...``) of the formalization, then the
    answer (``?``, its occurrences one node).  The variables are ordered
    farthest from the answer first (in the graph of the triples), ties by
    first occurrence; a variable not linked to the answer comes first.  Each
    belief's relations are the triples it occurs in, the variables and
    the answer as placeholders, the constants (``C@...``) as text; its
    ``ref`` is the variable (``?`` the answer)."""
    triples = [t for triple in triples for t in unpack(triple)]
    order: List[str] = []
    edges: Dict[str, set] = {}
    for terms in triples:
        nodes = term_nodes(terms)
        for t in nodes:
            if t not in edges:
                edges[t] = set()
                if t != ANSWER:
                    order.append(t)
            edges[t].update(n for n in nodes if n != t)
    dist = {ANSWER: 0}
    frontier = [ANSWER]
    while frontier:
        nxt = []
        for node in frontier:
            for other in edges.get(node, ()):
                if other not in dist:
                    dist[other] = dist[node] + 1
                    nxt.append(other)
        frontier = nxt
    hops = sorted(order, key=lambda v: (-dist.get(v, len(order) + 1), order.index(v))) + [ANSWER]
    label = {v: f"E{k}" for k, v in enumerate(hops, 1)}

    def term(t: str) -> str:
        if t == ANSWER:
            return label[ANSWER]
        t = VARIABLE.sub(lambda m: label.get(m.group(0), m.group(0)), t)
        return t.replace("C@", "").replace("_", " ").strip()

    relations: Dict[str, List[str]] = {v: [] for v in hops}
    for terms in triples:
        text = " ".join(x for x in (term(t) for t in terms) if x)
        for node in term_nodes(terms):
            if text not in relations[node]:
                relations[node].append(text)
    beliefs = [{"id": f"B{k}", "kind": CLOSED, "relations": relations[v],
                "answers": [answer] if v == ANSWER and answer else [], "ref": v}
               for k, v in enumerate(hops, 1)]
    return {"beliefs": beliefs, "entity_names": []}
