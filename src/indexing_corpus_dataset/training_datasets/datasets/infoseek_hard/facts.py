"""infoseek_hard's facts (``gold_passages/facts.py``), no LLM: one per claim
of the research tree, the claim with its node ids written as the
entities; the entity is the claim's own node (its page, where InfoSeek
wrote the claim from), the value the target node's entity; an attribute
claim ("A has won seven Goya Awards": its target is no entity) states the
whole fact, its value is the claim."""

from typing import List

from ...gold_passages.facts import fact
from .tree import claim_text, is_entity, tree_nodes


def claim_facts(record: dict, page_key) -> List[dict]:
    """One fact per claim of the record's research tree
    (``record["meta"]["root"]``) on a node with a page (``href``,
    *page_key* maps it to the gold page key)."""
    nodes = {n["id"]: n for n in tree_nodes(record["meta"]["root"])}
    facts = []
    for n in nodes.values():
        if not n.get("href") or not is_entity(n):
            continue
        for claim in n.get("claims") or []:
            target = nodes.get(claim.get("target_id"))
            ends = [i for i in dict.fromkeys((n["id"], claim.get("target_id"))) if i in nodes]
            ends = [i for i in ends if is_entity(nodes[i])]
            text = claim_text(claim["claim"], ends, lambda i: str(nodes[i]["entity"]).strip())
            value = str(target["entity"]).strip() if target is not None and is_entity(target) else text
            facts.append(fact(len(facts) + 1, str(n["entity"]).strip(), None, value, text,
                              [page_key(n["href"])], ends))
    return facts
