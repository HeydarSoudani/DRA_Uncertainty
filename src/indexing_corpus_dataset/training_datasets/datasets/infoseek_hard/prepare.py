"""infoseek_hard: the checks of stage data_prepare, in order.

    duplicate          a repeated question (its first row kept)
    hardness           the research tree's (depth, max branching) is not
                       one of ``HARD_CELLS`` (``not_hard``)
    tree               a node without an href, a repeated node id that
                       ``tree.unique_ids`` (source) could not resolve, or a
                       claim whose target is not in the tree
                       (``tree_invalid``)
    answer             empty, or stated by the question (``answer_invalid``)
    hop_named          the question names an entity node that has children
                       (a hop between the answer and the leaves:
                       ``hop_named``); the leaves are named by design
    pages              a gold page (the page of a node) not in corpus_wiki,
                       or whose first chunk has no text (``page_not_in_corpus``,
                       ``page_empty``)
    claim_target       a claim whose target cannot be found in the first
                       two chunks of its page (``claim_unreachable``;
                       ``ClaimTargets``)

Depth counts the nodes of the longest root-to-leaf path, root included;
branching is the most children of one node.  Both count every child: an
entity clue (a node with its own page) or an attribute clue (entity
``None``, a fact of its parent).
"""

import re
from collections import Counter
from typing import Dict, List, Optional, Tuple

from ...annotation.names import entity_names
from ...corpus.pages import normalize_title, wiki_gold
from ...corpus.wiki import corpus_stats
from ...corpus.wiki_pages import WikiPages
from ...pipeline.checks import answer_check, coverage, duplicate_check, words
from ...pipeline.funnel import Funnel
from ...pipeline.stats import write_stats
from .tree import is_entity, named, tree_nodes

# The (depth, max branching) cells kept: the deep chains (depth 5 and 6)
# and every tree both deep and wide (depth 4 from branching 3); 3022
# questions of InfoSeek-Hard-18K once repeated questions are removed.
HARD_CELLS = frozenset({(6, 1), (5, 1), (5, 2), (5, 3), (4, 3), (4, 4), (4, 5), (3, 6), (2, 6)})

# A number in a claim: digits with separators inside ("1,148", "3.5").
_NUMBER = re.compile(r"\d+(?:[.,]\d+)*")
# An entity target is found when this share of the content words of one of
# its names is on the page: InfoSeek names a target by the title of its
# page ("Poland national football team", "Caroline of Ansbach"), the page
# that states the claim by its own words ("the Poland national team",
# "Queen Caroline").
MIN_COVERAGE = 0.5


def depth(node: dict) -> int:
    return 1 + max((depth(c) for c in node.get("children") or []), default=0)


def branching(root: dict) -> int:
    return max(len(n.get("children") or []) for n in tree_nodes(root))


def cell(record: dict) -> Tuple[int, int]:
    root = record["meta"]["root"]
    return depth(root), branching(root)


def _pages(record: dict) -> List[dict]:
    return record["gold"]


def hardness_check(record: dict) -> Optional[List[dict]]:
    d, b = cell(record)
    return None if (d, b) in HARD_CELLS else [{"status": "not_hard", "depth": d, "branching": b}]


def tree_check(record: dict) -> Optional[List[dict]]:
    nodes = list(tree_nodes(record["meta"]["root"]))
    ids = [n.get("id") for n in nodes]
    reasons = []
    if len(set(ids)) != len(ids):
        reasons.append({"status": "tree_invalid", "why": "repeated node id"})
    for n in nodes:
        if not str(n.get("href") or "").strip():
            reasons.append({"status": "tree_invalid", "why": "node without href", "node": n.get("id")})
        for claim in n.get("claims") or []:
            if claim.get("target_id") not in ids:
                reasons.append({"status": "tree_invalid", "why": "claim target not in the tree",
                                "node": n.get("id"), "claim": claim.get("claim")})
    return reasons or None


def hop_named_check(record: dict) -> Optional[List[dict]]:
    question = record["question"]
    root = record["meta"]["root"]
    nodes = list(tree_nodes(root))
    hops = [n for n in nodes if n is not root and is_entity(n) and n.get("children")]
    reasons = [{"status": "hop_named", "node": n["id"], "entity": n["entity"]}
               for n in hops if named(question, n, nodes)]
    return reasons or None


def target_forms(target: dict) -> List[str]:
    """The names of a claim's target entity looked for on the page: the
    entity, and the title of its page (decoded), each also without its
    disambiguation."""
    forms = entity_names(str(target["entity"]))
    if target.get("href"):
        forms += entity_names(normalize_title(target["href"]))
    return list(dict.fromkeys(forms))


def claim_numbers(claim: str) -> List[str]:
    return list(dict.fromkeys(_NUMBER.findall(claim)))


class ClaimTargets:
    """Each claim of a tree checked on the first two chunks of the page of
    the node that states it (InfoSeek writes a node's claims from the first
    1024 tokens of its page), as words (``pipeline.checks.words``: case, accents,
    punctuation, ordinal suffixes and a plural "s" ignored): an entity
    target is found when ``MIN_COVERAGE`` of the content words of one of
    its names (``target_forms``) are on the page, an attribute target
    (entity ``None``) when every number of the claim is.  A claim on an
    attribute without a number is not checked.  Counts per kind in
    ``counts``."""

    def __init__(self, pages: WikiPages):
        self.pages = pages
        self.counts = Counter()

    def check(self, record: dict) -> Optional[List[dict]]:
        nodes = {n["id"]: n for n in tree_nodes(record["meta"]["root"])}
        reasons = []
        for n in nodes.values():
            if not is_entity(n):
                continue
            page = wiki_gold(n["href"])
            head = set(words(self.pages.head(page["key"])))
            for claim in n.get("claims") or []:
                target = nodes[claim["target_id"]]
                text = str(claim.get("claim") or "")
                if is_entity(target):
                    forms = target_forms(target)
                    ok = max(coverage(f, head) for f in forms) >= MIN_COVERAGE
                    self.counts["entity"] += 1
                    miss = {"forms": forms}
                else:
                    numbers = claim_numbers(text)
                    if not numbers:
                        self.counts["attribute_without_number"] += 1
                        continue
                    ok = all(all(w in head for w in words(x)) for x in numbers)
                    self.counts["attribute"] += 1
                    miss = {"numbers": numbers}
                if not ok:
                    self.counts["missed"] += 1
                    reasons.append({"status": "claim_unreachable", "claim": text, "page": page["title"],
                                    "target": target["entity"], **miss})
        return reasons or None


def run(cfg, paths, funnel: Funnel) -> dict:
    before = Counter(cell(r) for r in funnel.kept)
    funnel.apply("duplicate", duplicate_check())
    cells_unique = Counter(cell(r) for r in funnel.kept)
    funnel.apply("hardness", hardness_check)
    funnel.apply("tree", tree_check)
    funnel.apply("answer", answer_check)
    funnel.apply("hop_named", hop_named_check)
    pages = WikiPages(paths, funnel.kept, _pages)
    funnel.apply("pages", pages.check(_pages))
    claims = ClaimTargets(pages)
    funnel.apply("claim_target", claims.check)

    kept_cells = Counter(cell(r) for r in funnel.kept)
    for record in funnel.kept:
        d, b = cell(record)
        record["features"] = {"depth": d, "branching": b, "nodes": len(list(tree_nodes(record["meta"]["root"]))),
                              "page_chunks": pages.features(record, _pages)}

    def cells(counts: Counter) -> Dict[str, int]:
        return {f"{d},{b}": counts[(d, b)] for d, b in sorted(HARD_CELLS, key=lambda c: (-c[0], c[1]))}

    write_stats(paths.stats, "corpus", corpus_stats(paths, pages.looked_up, pages.titles))
    return {"hardness": {"cells": "depth,branching", "source": cells(before), "unique": cells(cells_unique),
                         "kept": cells(kept_cells)},
            "claims": dict(claims.counts)}
