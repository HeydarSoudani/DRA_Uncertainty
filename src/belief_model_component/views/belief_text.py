"""The belief as text: each belief with its statement and its passages.

The belief state keeps per belief its candidates (``core.types.Candidate``),
its most probable candidate and WEP, and its proposals
(``core.types.Proposal``).  The belief update prompt lists every belief by
its id "B<k>" with its statement (``belief_statement``), built by code from
its WEP and most probable candidate, then one line per passage on a
candidate, + (it states a part for it) or - (it states the candidate fails
a part), with the candidate, the spans and the passage's title in
brackets, most probable candidate first (``BeliefState.ordered``), the
latest ``MAX_SUPPORT_LINES`` passages of each candidate and stance (the
earlier ones counted); then the names the agent proposed that no passage
is on (``format_belief``)::

    B1: It is possible that Osby addresses the town where the Varn Folk Museum was founded.
      + Osby: "the folk museum of the Varn valley opened in Osby in 1921" [Osby]
      + Tirsa: "a crafts museum with objects from the Varn valley opened in Tirsa in 1925" [Tirsa]
    B2: It is unknown which candidate addresses the engineer who was born in B1.
      proposed: Nils Berg

An aggregate shows its statement only: a target names its derived
candidate; a computed answer is always "unknown" (it takes no belief item;
the agent's computed answers are saved, never shown), then the passage
texts on its inputs.  The agent's claims are never shown to the update.
"""

import re
from typing import Any, List, Optional

from ..core.types import ALMOST_NO_CHANCE, AGGREGATE, ASPECT, BELIEF_REF_RE, DOC, UNKNOWN, Candidate, is_computed

# Characters of the spans of one passage line.
MAX_SUPPORT_CHARS = 300
# Passage lines shown per candidate and stance (the latest ones; the
# earlier ones counted in one line before them), so a long search does
# not grow the prompt without bound.
MAX_SUPPORT_LINES = 5
# Passage lines shown on a computed aggregate's inputs (one per passage,
# the latest ones; the others are counted).
MAX_INPUT_LINES = 40


def belief_text(state: Any, k: int) -> str:
    """Belief *k*'s text as the update reads it: its references to other
    beliefs by their "B<k>" ids, no closing full stop."""
    n = len(state.beliefs)

    def _ref(m: "re.Match") -> str:
        return f"B{m.group(1)}" if 1 <= int(m.group(1)) <= n else m.group(0)

    return BELIEF_REF_RE.sub(_ref, state.beliefs[k].text.strip()).rstrip(".")


def belief_statement(state: Any, k: int) -> str:
    """The statement of belief *k*: "It is <wep> that <most probable
    candidate> addresses <belief>.", on an aspect "It is <wep> that the
    kept facts address <belief>." ("There is almost no chance that ..."
    for that word, as the WEP definitions read); with "unknown" (or no candidate to
    name, or a computed answer) "It is unknown which candidate addresses
    <belief>." and on an aspect "It is unknown how well the facts address
    <belief>."."""
    text, status = belief_text(state, k), state.status(k)
    if state.beliefs[k].kind == ASPECT:
        if status == UNKNOWN:
            return f"It is unknown how well the facts address {text}."
        return f"{_opening(status)} that the kept facts address {text}."
    c = state.beliefs[k]
    if c.kind != AGGREGATE:
        top = state.most_probable(k)
    else:
        top = None if is_computed(c.kind, c.text) else next(iter(state.ordered(k)), None)
    if status == UNKNOWN or top is None:
        return f"It is unknown which candidate addresses {text}."
    return f"{_opening(status)} that {top.name} addresses {text}."


def _opening(status: str) -> str:
    """"It is <wep>", or "There is almost no chance"."""
    return f"There is {status}" if status == ALMOST_NO_CHANCE else f"It is {status}"


def _spans_text(spans: List[str]) -> str:
    text = " ... ".join(" ".join(s.split()) for s in spans)
    return text if len(text) <= MAX_SUPPORT_CHARS else text[:MAX_SUPPORT_CHARS - 3] + "..."


def support_lines(c: Candidate) -> List[str]:
    """The passages assigned to candidate *c*, + then -, one line each:
    ``+ <candidate>: "<spans>" [<title>]``; per stance the latest
    ``MAX_SUPPORT_LINES``, after ``+ <candidate>: (N more passages)``."""
    lines = []
    for against in (False, True):
        sign, supports = "-" if against else "+", c.supports(against)
        hidden = len(supports) - MAX_SUPPORT_LINES
        if hidden > 0:
            lines.append(f"{sign} {c.name}: ({hidden} more passage{'s' if hidden > 1 else ''})")
            supports = supports[hidden:]
        lines += [f'{sign} {c.name}: "{_spans_text(x.spans)}" [{x.title}]' for x in supports]
    return lines


def input_lines(state: Any, k: int) -> List[str]:
    """A computed aggregate's passage observations (what passages state on
    its inputs, such as a member meeting the set's condition): a
    heading, then one line per passage, ``"<spans>" [<title>]``, at most
    ``MAX_INPUT_LINES`` (the latest; "+N more sources" before them).
    Empty for other beliefs or without one."""
    c = state.beliefs[k]
    if not is_computed(c.kind, c.text):
        return []
    obs = [o for o in state.trace[k] if o.source == DOC and o.verified_spans]
    if not obs:
        return []
    lines = ["passages on its inputs:"]
    hidden = len(obs) - MAX_INPUT_LINES
    if hidden > 0:
        lines.append(f"  {hidden} more source" + ("s" if hidden > 1 else ""))
        obs = obs[hidden:]
    lines.extend(f'  "{_spans_text(o.verified_spans)}" [{state.title(o.label)}]' for o in obs)
    return lines


def proposals_line(state: Any, k: int, label: str = "proposed") -> Optional[str]:
    names = [p.name for p in state.proposals[k]]
    return f"{label}: " + ", ".join(names) if names else None


def format_belief(state: Any) -> str:
    """Every belief of *state* (the belief update's and the final scorer's
    view): its id and statement, the passages assigned to its candidates
    (``support_lines``, ``BeliefState.ordered``), then its proposals.  An
    aggregate shows no passage lines (a computed one then the passages on
    its inputs, ``input_lines``)."""
    lines = []
    for k, c in enumerate(state.beliefs):
        lines.append(f"{c.id}: {belief_statement(state, k)}")
        if c.kind != AGGREGATE:
            for x in state.ordered(k):
                lines.extend(f"  {line}" for line in support_lines(x))
        proposed = proposals_line(state, k)
        if proposed:
            lines.append(f"  {proposed}")
        lines.extend(f"  {line}" for line in input_lines(state, k))
    return "\n".join(lines)
