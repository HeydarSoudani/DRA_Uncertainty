"""infoseek_hard's source: InfoSeek-Hard (``Lk123/InfoSeek``), the root
question of each research tree (its node ids made unique,
``tree.unique_ids``); the answer is the root entity, the gold pages the
Wikipedia page of every tree node; ``annotation`` keeps the entity names
of the tree (``entity_names_of``)."""

from typing import Iterator, List

from utils.io import iter_jsonl

from ...annotation.names import entity_names
from ...corpus.pages import dedup_gold, wiki_gold
from ...pipeline.source import hf_file, hf_info
from .tree import is_entity, tree_nodes, unique_ids

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
            "annotation": {"entity_names": entity_names_of(root)},
            "meta": {"root": root},
        }


def entity_names_of(root: dict) -> List[str]:
    """Every entity name of a research tree (``names.entity_names`` of each
    entity node)."""
    return [name for n in tree_nodes(root) if is_entity(n) for name in entity_names(n["entity"])]
