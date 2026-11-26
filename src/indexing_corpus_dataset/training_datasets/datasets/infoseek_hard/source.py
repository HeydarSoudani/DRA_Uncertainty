"""infoseek_hard's source: InfoSeek-Hard (``Lk123/InfoSeek``), the root
question of each research tree (its node ids made unique,
``tree.unique_ids``); the answer is the root entity, the gold pages the
Wikipedia page of every tree node, the skeleton the hops of the tree
(``skeleton``)."""

from typing import Dict, Iterator, List

from utils.io_utils import iter_jsonl

from ...annotation.names import entity_names
from ...annotation.rows import CLOSED
from ...corpus.pages import dedup_gold, wiki_gold
from ...pipeline.source import hf_file, hf_info
from .tree import claim_text, is_entity, named, post_order, tree_nodes, unique_ids

# Pinned to a revision: the question ids are row numbers.
INFOSEEK = ("Lk123/InfoSeek", "data/InfoSeek-Hard-18K.jsonl", "137a171bd9e6b31d20d0239d5d73269900fb4540")


def info() -> dict:
    return hf_info(INFOSEEK)


def load() -> Iterator[dict]:
    for i, row in enumerate(iter_jsonl(hf_file(INFOSEEK))):
        root = unique_ids(row["root"])
        gold = dedup_gold([wiki_gold(n["href"]) for n in tree_nodes(root) if n.get("href")])
        question = root["question"].strip()
        yield {
            "id": f"infoseek_{i:05d}", "source": "infoseek",
            "question": question, "answer": root["entity"].strip(),
            "gold": gold,
            "annotation": skeleton(question, root),
            "meta": {"root": root},
        }


def skeleton(question: str, root: dict) -> dict:
    """The hops of a research tree: the nodes the question does not name
    (the root, the answer, always), children before parents, each with its
    node id as ``ref``.  Each hop's relations name the other hops by
    placeholder (E1, E2, ...) and the named entities by name."""
    nodes = {n["id"]: n for n in tree_nodes(root)}
    hops = [n for n in post_order(root) if n is root or (is_entity(n) and not named(question, n, nodes.values()))]
    label = {n["id"]: f"E{k}" for k, n in enumerate(hops, 1)}
    relations: Dict[str, List[str]] = {n["id"]: [] for n in hops}
    for n in nodes.values():
        for claim in n.get("claims") or []:
            target = claim.get("target_id")
            ends = [node_id for node_id in (n["id"], target) if node_id in nodes]
            text = claim_text(claim["claim"], ends, lambda i: label.get(i) or nodes[i]["entity"])
            # A relation describes the hops it links.
            for node_id in (n["id"], target):
                if node_id in relations and text not in relations[node_id]:
                    relations[node_id].append(text)
    beliefs = [{"id": f"B{k}", "kind": CLOSED, "relations": relations[n["id"]],
                "answers": entity_names(n["entity"]), "ref": n["id"]}
               for k, n in enumerate(hops, 1)]
    names = [name for n in nodes.values() if is_entity(n) for name in entity_names(n["entity"])]
    return {"beliefs": beliefs, "entity_names": names}
