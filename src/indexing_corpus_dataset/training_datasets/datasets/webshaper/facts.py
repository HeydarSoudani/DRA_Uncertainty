"""webshaper's facts (``gold_passages/facts.py``): one fact per triple of the
formalization, written by code from what the dataset states; one LLM call
per question (``prompts/facts_*.txt``) only maps each fact to the gold
pages likely to state it, from the pages' titles.

A fact holds no name or value the dataset does not state:

    subject    a constant, the answer, or a variable described by its other
               triples whose other term is known ("the entity that took
               place on 16 August 1986"; "an entity" without one)
    value      the object: a constant or the answer (known: the judge
               checks it), a variable (unknown: the judge extracts it,
               ``value_of`` the variable)

An identity triple (a variable and the answer joined by a copula,
``COPULAS``: "V@X | has name | ?") says the variable is the answer: the
variable is written as the answer everywhere and the triple itself is no
fact.  A repeated triple (or one written out as the same fact as an
earlier one) counts once; a triple that is not three terms is
skipped, and so is one with no known term (an unnamed subject and an
unknown value: "an entity has work ___" would fit any passage stating some
work)."""

from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from ...gold_passages.facts import BLANK, FactSpec, fact, read_prompts
from .formalization import ANSWER, VARIABLE, term_nodes, triples_of

# The predicates that make a variable the answer ("V@X | has name | ?").
COPULAS = ("is", "is named", "is called", "has name", "has title", "equals")
# The most triples a variable's description is made of.
MAX_DESCRIPTION_PARTS = 3
UNNAMED = "an entity"


def _pred(p: str) -> str:
    return " ".join(p.replace("_", " ").split())


def _is_var(term: str) -> bool:
    return VARIABLE.fullmatch(term) is not None


def _is_identity(terms: List[str]) -> bool:
    if len(terms) != 3:
        return False
    s, p, o = terms
    return _pred(p).lower() in COPULAS and ANSWER in (s, o) and (_is_var(s) or _is_var(o))


def answer_variables(triples: List[List[str]]) -> Set[str]:
    """The variables an identity triple makes the answer."""
    return {t for terms in triples if _is_identity(terms) for t in (terms[0], terms[2]) if _is_var(t)}


class _Writer:
    """The terms of a record's triples written out."""

    def __init__(self, record: dict, triples: List[List[str]]):
        self.answer = str(record["answer"]).strip()
        self.triples = triples
        self.same = answer_variables(triples)

    def known(self, term: str) -> Optional[str]:
        """A term the dataset states written out (a constant, the answer, a
        variable that is the answer, a constant holding such variables);
        None for a term with another variable."""
        if term == ANSWER or term in self.same:
            return self.answer
        if any(v not in self.same for v in VARIABLE.findall(term)):
            return None
        text = VARIABLE.sub(lambda m: self.answer, term)
        return text.replace("C@", "").replace("_", " ").strip()

    def subject(self, term: str, k: int) -> str:
        """The subject of triple *k*: the term written out, a variable by
        its description."""
        text = self.known(term)
        if text is not None:
            return text
        if _is_var(term):
            return self.description(term, k)
        return VARIABLE.sub(lambda m: self.answer if m.group(0) in self.same else UNNAMED,
                            term).replace("C@", "").replace("_", " ").strip()

    def description(self, var: str, k: int) -> str:
        """*var* described by its other triples whose other term is known:
        first those it is the subject of ("took place on 16 August 1986"),
        then those it is the object of ("Jeff Lemire wrote"); at most
        ``MAX_DESCRIPTION_PARTS``."""
        own, other = [], []
        for j, terms in enumerate(self.triples):
            if j == k or len(terms) != 3 or _is_identity(terms):
                continue
            s, p, o = terms
            if s == var and o != var and self.known(o) is not None:
                own.append(f"{_pred(p)} {self.known(o)}")
            elif o == var and s != var and self.known(s) is not None:
                other.append(f"{self.known(s)} {_pred(p)}")
        parts = (own + other)[:MAX_DESCRIPTION_PARTS]
        return "the entity that " + " and that ".join(parts) if parts else UNNAMED


def build_facts(record: dict) -> Tuple[List[dict], Dict[str, int]]:
    """The facts of a record (pages not assigned yet) and the counts of its
    triples (``triples``, ``duplicate``, ``identity``, ``skipped``,
    ``no_known_term``)."""
    given = triples_of(record)
    # Without its repeats, so no triple describes its own subject.
    triples = [list(t) for t in dict.fromkeys(tuple(t) for t in given)]
    w = _Writer(record, triples)
    counts = Counter(triples=len(given), duplicate=len(given) - len(triples))
    facts = []
    seen = set()
    for k, terms in enumerate(triples):
        if _is_identity(terms):
            counts["identity"] += 1
            continue
        if len(terms) != 3 or not all(terms):
            counts["skipped"] += 1
            continue
        s, p, o = terms
        subject, value = w.subject(s, k), w.known(o)
        if subject == UNNAMED and value is None:
            counts["no_known_term"] += 1
            continue
        statement = f"{subject} {_pred(p)} {value if value is not None else BLANK}"
        if statement in seen:
            counts["duplicate"] += 1
            continue
        seen.add(statement)
        facts.append(fact(len(facts) + 1, subject, _pred(p), value, statement, [], term_nodes(terms),
                          value_of=o if value is None and _is_var(o) else None))
    return facts, dict(counts)


def parse(record: dict, pages: List[dict], data: Dict[str, Any]) -> Tuple[List[dict], List[str]]:
    """The facts of a record with the pages the reply maps them to, or the
    problems.  A fact may get no page (it is then judged on every page)."""
    facts, _ = build_facts(record)
    problems = []
    assigned: Dict[int, List[int]] = {}
    for item in data.get("facts") or []:
        try:
            k = int(item.get("fact"))
            got = [int(p) for p in item.get("pages") or []]
        except (TypeError, ValueError, AttributeError):
            problems.append(f"an invalid fact entry: {item}")
            continue
        if not 1 <= k <= len(facts) or any(not 1 <= p <= len(pages) for p in got):
            problems.append(f"fact {k} or its pages {got} are not numbers of the lists")
            continue
        assigned[k] = list(dict.fromkeys(got))
    missing = [k for k in range(1, len(facts) + 1) if k not in assigned]
    if missing:
        problems.append(f"no entry for fact(s) {missing}")
    if problems:
        return [], problems
    for k, f in enumerate(facts, 1):
        f["pages"] = [pages[p - 1]["key"] for p in assigned[k]]
    return facts, []


def fields(record: dict) -> Dict[str, str]:
    """The numbered facts of the user prompt."""
    facts, _ = build_facts(record)
    return {"facts": "\n".join(f"{k}. {f['statement']}" for k, f in enumerate(facts, 1))}


SPEC = FactSpec("webshaper", *read_prompts(Path(__file__).resolve().parent / "prompts"), parse=parse, fields=fields)
