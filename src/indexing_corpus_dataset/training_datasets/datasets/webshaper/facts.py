"""webshaper's fact writer (``gold_passages/facts.py``): one fact per triple
of the formalization; one LLM call per question (``prompts/facts_*.txt``)
names each variable (when the question, answer, triples, page titles or
URLs show its entity; else a description from the triples) and assigns
each triple the gold pages likely to state it; the fact is the triple
with its terms written out, the value the object (None for a variable
without a name)."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ...gold_passages.facts import BLANK, FactSpec, fact, read_prompts
from .formalization import ANSWER, VARIABLE, term_nodes, triples_of


def _term(term: str, names: Dict[str, Optional[str]], descriptions: Dict[str, str], answer: str,
          as_value: bool) -> Optional[str]:
    """A triple term written out: the answer, a variable by its name (or,
    as a subject, its description; as a value, None without a name), a
    constant without its ``C@`` prefix."""
    if term == ANSWER:
        return answer
    unnamed = False

    def var(m) -> str:
        nonlocal unnamed
        name = names.get(m.group(0))
        if name:
            return name
        unnamed = True
        return descriptions.get(m.group(0)) or "an entity"

    text = VARIABLE.sub(var, term).replace("C@", "").replace("_", " ").strip()
    if as_value and unnamed:
        return None
    return text


def parse(record: dict, pages: List[dict], data: Dict[str, Any]) -> Tuple[List[dict], Dict[str, str], List[str]]:
    """The facts of a reply (one per unpacked triple) and the names of its
    variables, or the problems."""
    triples = triples_of(record)
    variables = list(dict.fromkeys(v for t in triples for v in term_nodes(t) if v != ANSWER))
    problems = []
    names: Dict[str, Optional[str]] = {}
    descriptions: Dict[str, str] = {}
    for item in data.get("variables") or []:
        if isinstance(item, dict) and item.get("variable") in variables:
            name = item.get("name")
            names[item["variable"]] = str(name).strip() if name is not None and str(name).strip() else None
            descriptions[item["variable"]] = str(item.get("description") or "").strip()
    missing = [v for v in variables if v not in names]
    if missing:
        problems.append(f"no entry for variable(s) {missing}")
    assigned: Dict[int, List[int]] = {}
    for item in data.get("triples") or []:
        try:
            k = int(item.get("triple"))
            got = [int(p) for p in item.get("pages") or []]
        except (TypeError, ValueError, AttributeError):
            problems.append(f"an invalid triple entry: {item}")
            continue
        if not 1 <= k <= len(triples) or any(not 1 <= p <= len(pages) for p in got):
            problems.append(f"triple {k} or its pages {got} are not numbers of the lists")
            continue
        assigned[k] = list(dict.fromkeys(got))
    missing = [k for k in range(1, len(triples) + 1) if k not in assigned]
    if missing:
        problems.append(f"no entry for triple(s) {missing}")
    if problems:
        return [], {}, problems
    facts = []
    for k, terms in enumerate(triples, 1):
        if len(terms) != 3:
            continue
        subject, prop, obj = terms
        entity = _term(subject, names, descriptions, record["answer"], as_value=False)
        value = _term(obj, names, descriptions, record["answer"], as_value=True)
        statement = f"{entity} {prop} {value if value is not None else BLANK}"
        facts.append(fact(len(facts) + 1, entity, prop, value, statement,
                          [pages[p - 1]["key"] for p in assigned[k]], term_nodes(terms)))
    return facts, {v: n for v, n in names.items() if n}, []


def fields(record: dict) -> Dict[str, str]:
    """The numbered triples of the user prompt."""
    return {"triples": "\n".join(f"{k}. " + " | ".join(t) for k, t in enumerate(triples_of(record), 1))}


SPEC = FactSpec("webshaper", *read_prompts(Path(__file__).resolve().parent / "prompts"), parse=parse, fields=fields)
