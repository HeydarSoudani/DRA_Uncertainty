"""InfoSeek's research tree (``record["meta"]["root"]``): its nodes, the
entity ones, the nodes a question names and the claims as text."""

import copy
import re
from collections import Counter, defaultdict
from typing import Iterator, List

from ...annotation.names import entity_names, mentions


def tree_nodes(node: dict) -> Iterator[dict]:
    """The nodes of a research tree, *node* first."""
    yield node
    for child in node.get("children") or []:
        yield from tree_nodes(child)


def unique_ids(root: dict) -> dict:
    """*root* (a copy) with one id per node.  InfoSeek gives an attribute
    node (entity ``None``) the id of an entity node at times ("C" both
    Paris Masters and an attribute of the tournament A).  A claim whose
    target id is repeated names the one child of the claim's node with
    that id; the node that keeps the id is the entity one, the others get
    ``{id}#{k}`` and their claims follow them.  A tree with a claim no
    child resolves is returned as it is (``prepare.tree_check`` drops it)."""
    root = copy.deepcopy(root)
    nodes = list(tree_nodes(root))
    count = Counter(n.get("id") for n in nodes)
    if all(k == 1 for k in count.values()):
        return root
    bound = []
    for n in nodes:
        for claim in n.get("claims") or []:
            target = claim.get("target_id")
            if count[target] < 2:
                continue
            kids = [c for c in n.get("children") or [] if c.get("id") == target]
            if len(kids) != 1:
                return root
            bound.append((claim, kids[0]))
    by_id = defaultdict(list)
    for n in nodes:
        by_id[n.get("id")].append(n)
    taken = set(count)
    for node_id, group in by_id.items():
        if len(group) < 2:
            continue
        keep = next((n for n in group if is_entity(n)), group[0])
        k = 0
        for n in group:
            if n is keep:
                continue
            k += 1
            while f"{node_id}#{k}" in taken:
                k += 1
            n["id"] = f"{node_id}#{k}"
            taken.add(n["id"])
    for claim, node in bound:
        claim["target_id"] = node["id"]
    return root


def is_entity(node: dict) -> bool:
    """A tree node that stands for an entity (an attribute node, such as a
    population figure, has none)."""
    return str(node.get("entity") or "").strip() not in ("", "None")


def named(question: str, node: dict, nodes) -> bool:
    """The question names *node*: one of its names occurs in the question
    outside the names of other nodes that contain it."""
    others = [name for o in nodes if o is not node and is_entity(o) for name in entity_names(o["entity"])]
    return any(mentions(question, own, others) for own in entity_names(node["entity"]))


def claim_text(claim: str, ends: List[str], name) -> str:
    """A claim of a research tree with each node id of *ends* (both ends of
    the claim, in the tree) written as ``name(id)``.  Both ends in one
    pass: a name put in may contain the other id as a word ("Robert E.
    Lee", node E).  Node ids are capital letters, so an id after a quote or
    a period, or before a period and a letter, is text ("an 'A' rating",
    "B.A."), not the node."""
    if not ends:
        return claim
    return re.sub(rf"(?<![\w'\".])({'|'.join(map(re.escape, ends))})(?!\w|\.\w)",
                  lambda m: name(m.group(1)), claim)
