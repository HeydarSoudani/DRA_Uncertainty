"""webshaper's facts (``gold_passages/facts.py``): one fact per triple of the
formalization, written by code from what the dataset states; one LLM call
per question (``prompts/facts_*.txt``) only maps each fact to the gold
pages likely to state it, from the pages' titles.

A fact is TRQA's statement of a property, ``the [p] of [e] is [v]``
(``facts.property_statement``), and holds no name or value the dataset
does not state:

    entity     the subject: a constant, the answer, or a variable
               described by one of its other triples whose other term is
               known ("the entity whose [took place on] is [16 August
               1986]", "the entity that is the [written by] of [Carolina
               Sonnet]"; its type, when a triple gives one, in place of
               "entity": "the book whose ..."); "an entity" without one
    property   the predicate
    value      the object: a constant or the answer (known: the judge
               checks it), a variable (unknown: the judge extracts it,
               ``value_of`` the variable), or a constant holding a variable
               (unknown too: its known part written out, the variable a
               blank, "[German school and ___]", ``partial_value``)

An identity triple (a variable and the answer joined by a copula,
``COPULAS``: "V@X | has name | ?") says the variable is the answer: the
variable is written as the answer everywhere and the triple itself is no
fact.  A type triple (``TYPE_PREDICATES`` with a constant: "V@B | is a |
C@book") is no fact either: almost any passage about the entity states
it; it names the variable's type in its descriptions.  Skipped too: a
repeated triple (or one written out as the same fact as an earlier one),
one that is not three terms, one with a variable or a constant inside its
predicate (``term_in_predicate``: "won_by_5_wickets_at_V@Y"), and one with
no known term (an unnamed subject and an unknown value: "an entity has
work ___" would fit any passage stating some work)."""

import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from ...gold_passages.facts import BLANK, FactSpec, fact, property_statement, read_prompts
from ...pipeline.checks import STOPWORDS
from .formalization import ANSWER, VARIABLE, term_nodes, triples_of

# The predicates that make a variable the answer ("V@X | has name | ?").
COPULAS = ("is", "is named", "is called", "has name", "has title", "equals")
# The predicates that give an entity's type ("V@B | is a | C@book").
TYPE_PREDICATES = ("is a", "is an", "has type", "is a type of")
UNNAMED = "an entity"
# The counts of a record's triples (``build_facts``).
TRIPLE_COUNTS = ("triples", "duplicate", "identity", "type", "skipped", "term_in_predicate", "no_known_term",
                 "partial_value")


def _pred(p: str) -> str:
    return " ".join(p.replace("_", " ").split())


def _is_var(term: str) -> bool:
    return VARIABLE.fullmatch(term) is not None


def _is_identity(terms: List[str]) -> bool:
    if len(terms) != 3:
        return False
    s, p, o = terms
    return _pred(p).lower() in COPULAS and ANSWER in (s, o) and (_is_var(s) or _is_var(o))


def _is_type(terms: List[str]) -> bool:
    return len(terms) == 3 and _pred(terms[1]).lower() in TYPE_PREDICATES and terms[2].startswith("C@")


def _term_in_predicate(p: str) -> bool:
    return "C@" in p or VARIABLE.search(p) is not None


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

    def unknown_vars(self, term: str) -> List[str]:
        """The variables of *term* that are not the answer."""
        return [v for v in dict.fromkeys(VARIABLE.findall(term)) if v not in self.same]

    def partial(self, term: str) -> Optional[str]:
        """A constant holding a variable that is not the answer written
        out with that variable as a blank ("C@German school and V@Y" ->
        "German school and ___"); None for any other term, and for one
        whose known part has no content word ("___ with ___")."""
        if _is_var(term) or term == ANSWER or not self.unknown_vars(term):
            return None
        out, end = [], 0
        for m in VARIABLE.finditer(term):
            out.append(term[end:m.start()].replace("C@", "").replace("_", " "))
            out.append(self.answer if m.group(0) in self.same else BLANK)
            end = m.end()
        out.append(term[end:].replace("C@", "").replace("_", " "))
        text = " ".join("".join(out).split())
        known = re.findall(r"\w+", text.replace(BLANK, " "))
        return text if any(w.lower() not in STOPWORDS for w in known) else None

    def subject(self, term: str, k: int) -> Optional[str]:
        """The subject of triple *k*: the term written out, a variable by
        its description; None for an unnamed one (``UNNAMED``)."""
        text = self.known(term)
        if text is not None:
            return text
        if _is_var(term):
            return self.description(term, k)
        return None

    def _usable(self, j: int, k: int) -> bool:
        terms = self.triples[j]
        return (j != k and len(terms) == 3 and not _is_identity(terms) and not _term_in_predicate(terms[1]))

    def description(self, var: str, k: int) -> Optional[str]:
        """*var* described by one of its other triples whose other term is
        known, the most specific one (the longest known term: a title or a
        name before a year; a triple *var* is the subject of first on a
        tie): "the entity whose [took place on] is [16 August 1986]", "the
        entity that is the [wrote] of [Jeff Lemire]"; its type in place of
        "entity" when a type triple gives one; None without such a
        triple."""
        noun = "entity"
        parts: List[Tuple[int, int, str]] = []
        for j, (s, p, o) in ((j, t) for j, t in enumerate(self.triples) if self._usable(j, k)):
            if s == var and _is_type(self.triples[j]):
                noun = noun if noun != "entity" else self.known(o)
            elif s == var and o != var and self.known(o) is not None:
                parts.append((len(self.known(o)), 1, f"the {{noun}} whose [{_pred(p)}] is [{self.known(o)}]"))
            elif o == var and s != var and self.known(s) is not None:
                parts.append((len(self.known(s)), 0, f"the {{noun}} that is the [{_pred(p)}] of [{self.known(s)}]"))
        return max(parts, key=lambda x: (x[0], x[1]))[2].format(noun=noun) if parts else None


def build_facts(record: dict) -> Tuple[List[dict], Dict[str, int]]:
    """The facts of a record (pages not assigned yet) and the counts of its
    triples (``TRIPLE_COUNTS``)."""
    given = triples_of(record)
    # Without its repeats, so no triple describes its own subject.
    triples = [list(t) for t in dict.fromkeys(tuple(t) for t in given)]
    w = _Writer(record, triples)
    counts = Counter({key: 0 for key in TRIPLE_COUNTS})
    counts.update(triples=len(given), duplicate=len(given) - len(triples))
    facts = []
    seen = set()
    for k, terms in enumerate(triples):
        if _is_identity(terms):
            counts["identity"] += 1
            continue
        if len(terms) != 3 or not all(terms):
            counts["skipped"] += 1
            continue
        if _term_in_predicate(terms[1]):
            counts["term_in_predicate"] += 1
            continue
        if _is_type(terms):
            counts["type"] += 1
            continue
        s, p, o = terms
        subject, value, partial = w.subject(s, k), w.known(o), w.partial(o)
        if subject is None and value is None and partial is None:
            counts["no_known_term"] += 1
            continue
        subject = subject or UNNAMED
        statement = property_statement(subject, _pred(p), value if value is not None else partial)
        if statement in seen:
            counts["duplicate"] += 1
            continue
        seen.add(statement)
        counts["partial_value"] += partial is not None
        blanks = [o] if _is_var(o) else w.unknown_vars(o) if partial is not None else []
        facts.append(fact(len(facts) + 1, subject, _pred(p), value, statement, [], term_nodes(terms),
                          value_of=blanks[0] if value is None and len(blanks) == 1 else None))
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
