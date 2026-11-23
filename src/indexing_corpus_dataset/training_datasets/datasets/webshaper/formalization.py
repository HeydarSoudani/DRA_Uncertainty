"""WebShaper's formalization (``record["meta"]["formalization"]``): triples
of variables (``V@...``), constants (``C@...``) and the answer (``?``)."""

import re
from typing import List

ANSWER = "?"
VARIABLE = re.compile(r"V@[A-Za-z0-9_]+")
_PACKED = " & "


def unpack(triple: list) -> List[List[str]]:
    """A triple as its terms; a packed one (``[s, "p1 & p2", "o1 & o2"]``,
    predicates and objects aligned) as one triple per predicate."""
    terms = [str(t).strip() for t in triple]
    if len(terms) == 3 and _PACKED in terms[1]:
        preds, objs = terms[1].split(_PACKED), terms[2].split(_PACKED)
        if len(preds) == len(objs):
            return [[terms[0], p.strip(), o.strip()] for p, o in zip(preds, objs)]
    return [terms]


def term_nodes(terms: List[str]) -> List[str]:
    """The variables (anywhere in a term) and the answer of a triple."""
    found = [v for t in terms for v in VARIABLE.findall(t)]
    if ANSWER in terms:
        found.append(ANSWER)
    return list(dict.fromkeys(found))


def triples_of(record: dict) -> List[List[str]]:
    """The unpacked triples of a record."""
    return [t for triple in record["meta"].get("formalization") or [] for t in unpack(triple)]
