"""The belief as text: each criterion with its candidates, and the
``<belief_state>`` tag that ``--belief-mode inform`` injects.

The belief state keeps per criterion its candidates (``core.types.Candidate``)
and its proposals (``core.types.Proposal``).  The belief update prompt lists
every criterion with its candidates, highest status first
(``BeliefState.ordered``), each with its support: one line per passage
observation that supports it (``+``) or speaks against it (``-``), its
spans and the passage's title in brackets (labels such as "d2" are local to
one call, so they are never shown again); then the names the agent
proposed that no passage supports yet; an aspect criterion (a report) shows
its coverage first, and a computed criterion then the passage texts on its
definition, such as whether a member meets the set's condition
(``format_belief``)::

    c1 (several answers): the American theoretical physicist who shared his Nobel Prize with a Nobel Laureate from Pakistan
      "Sheldon Glashow": certain
        + "Sheldon Lee Glashow ... is an American theoretical physicist" [Sheldon Glashow]
      "Steven Weinberg": probable
        + "He shared the 1979 Nobel Prize in Physics with Abdus Salam and Sheldon Glashow" [Abdus Salam]
      proposed: "Murray Gell-Mann"
    c2 (one answer): the Nobel Laureate from Pakistan who shared his Nobel Prize with the physicist of c1
      (no candidate)

The agent's claims are never shown to the update.  At the end of every
search iteration the agent reads one tag, each criterion with its
candidates and statuses (no support) and its proposals as unverified, its
own id "k<n>" in place of the criterion id::

    <belief_state step="3">
      <criteria>
        <k1>the town where the Varn Folk Museum was founded
          "Osby": certain
        </k1>
        <k2>the engineer who was born in the town of k1
          "Henrik Dahl": certain
          unverified: "Nils Berg"
        </k2>
        <k3>the lighthouse that was designed by the engineer of k2</k3>
      </criteria>
    </belief_state>

The tag is descriptive only, and carries no source labels: the reasons of
the updates are saved, never shown.  Nothing gold-based (qrels, recall)
reaches the agent.  With no criteria there is no tag.
"""

import re
from html import escape
from typing import Any, Dict, List, Optional

from ..core.types import ASPECT, CRITERION_REF_RE, DERIVED_KINDS, DOC, NO_DOC_KINDS, Candidate, Support, exclusive

# A tag the model wrote itself; removed before its text enters the context.
BELIEF_STATE_RE = re.compile(r"<belief_state\b[^>]*>.*?</belief_state>\s*", re.DOTALL)

# The line of a criterion without candidates.
NO_CANDIDATE = "(no candidate)"

# Support lines shown per candidate and side (supports / against); the
# others are counted ("+N more sources").
MAX_SUPPORT_LINES = 3
# Characters of the spans of one support line.
MAX_SUPPORT_CHARS = 300
# Passage lines shown on a computed criterion's definition (one per passage,
# the latest ones; the others are counted).
MAX_DEFINITION_LINES = 40


def strip_belief_state(text: str) -> str:
    """Remove every ``<belief_state>`` tag from *text*."""
    return BELIEF_STATE_RE.sub("", text) if text and "<belief_state" in text else text


def criterion_type(kind: str) -> str:
    """The type of a criterion of *kind* as the belief update reads it."""
    if kind in DERIVED_KINDS:
        return "target"
    if kind == ASPECT:
        return "aspect"
    return "computed" if kind in NO_DOC_KINDS else "one answer" if exclusive(kind) else "several answers"


def criterion_head(state: Any, k: int) -> str:
    """Criterion *k* as the belief update reads it: its id, its type and its text."""
    c = state.criteria[k]
    return f"{c.id} ({criterion_type(c.kind)}): {c.text}"


def candidate_line(c: Candidate) -> str:
    """One candidate as shown: ``"<name>": <status>``."""
    return f'"{c.name}": {c.status}'


def _spans_text(spans: List[str]) -> str:
    text = " ... ".join(" ".join(s.split()) for s in spans)
    return text if len(text) <= MAX_SUPPORT_CHARS else text[:MAX_SUPPORT_CHARS - 3] + "..."


def support_lines(c: Candidate) -> List[str]:
    """The support of candidate *c*: per side (supports, then against) one
    line per passage observation, ``+ "<spans>" [<title>]``, the same spans
    of another passage counted on the line ("; +1 source"), at most
    ``MAX_SUPPORT_LINES`` lines per side ("+N more sources" after)."""
    lines: List[str] = []
    for against, sign in ((False, "+"), (True, "-")):
        groups: Dict[str, List[Support]] = {}
        for x in c.supports(against):
            groups.setdefault(_spans_text(x.spans), []).append(x)
        items = list(groups.items())
        for text, same in items[:MAX_SUPPORT_LINES]:
            more = f"; +{len(same) - 1} source" + ("s" if len(same) > 2 else "") if len(same) > 1 else ""
            lines.append(f'{sign} "{text}" [{same[0].title}]{more}')
        rest = sum(len(same) for _, same in items[MAX_SUPPORT_LINES:])
        if rest:
            lines.append(f"{sign} {rest} more source" + ("s" if rest > 1 else ""))
    return lines


def coverage_line(state: Any, k: int) -> Optional[str]:
    """An aspect criterion's coverage (``BeliefState.coverage``), else None."""
    return f"coverage: {state.coverage[k]}" if state.criteria[k].kind == ASPECT else None


def definition_lines(state: Any, k: int) -> List[str]:
    """A computed criterion's passage observations (what passages state on
    its definition, such as a member meeting the set's condition): a
    heading, then one line per passage, ``"<spans>" [<title>]``, at most
    ``MAX_DEFINITION_LINES`` (the latest; "+N more sources" before them).
    Empty for other criteria or without one."""
    if state.criteria[k].kind not in NO_DOC_KINDS:
        return []
    obs = [o for o in state.trace[k] if o.source == DOC and o.verified_spans]
    if not obs:
        return []
    lines = ["passages on its definition:"]
    hidden = len(obs) - MAX_DEFINITION_LINES
    if hidden > 0:
        lines.append(f"  {hidden} more source" + ("s" if hidden > 1 else ""))
        obs = obs[hidden:]
    lines.extend(f'  "{_spans_text(o.verified_spans)}" [{state.title(o.label)}]' for o in obs)
    return lines


def proposals_line(state: Any, k: int, label: str = "proposed") -> Optional[str]:
    names = [p.name for p in state.proposals[k]]
    return f"{label}: " + ", ".join(f'"{n}"' for n in names) if names else None


def format_belief(state: Any, removed: bool = False) -> str:
    """Every criterion of *state* with its candidates and their support,
    highest status first, or ``NO_CANDIDATE``, then its proposals; with
    *removed* (the final scorer's view), then the names the belief ruled
    out (``BeliefState.ruled_out``).  A target criterion's candidates are
    shown without support lines: their support is the clues', shown there;
    a computed criterion's too: its passages are listed once, as the
    passages on its definition (``definition_lines``).  An aspect shows its
    coverage first (``coverage_line``)."""
    lines = []
    for k in range(len(state.criteria)):
        lines.append(criterion_head(state, k))
        coverage = coverage_line(state, k)
        if coverage:
            lines.append(f"  {coverage}")
        cands = state.ordered(k)
        derived = state.criteria[k].kind in DERIVED_KINDS or state.criteria[k].kind in NO_DOC_KINDS
        if not cands:
            lines.append(f"  {NO_CANDIDATE}")
        for c in cands:
            lines.append(f"  {candidate_line(c)}")
            if not derived:
                lines.extend(f"    {line}" for line in support_lines(c))
        proposed = proposals_line(state, k)
        if proposed:
            lines.append(f"  {proposed}")
        ruled_out = state.ruled_out(k) if removed else []
        if ruled_out:
            lines.append("  removed: " + ", ".join(f'"{n}"' for n in ruled_out))
        lines.extend(f"  {line}" for line in definition_lines(state, k))
    return "\n".join(lines)


def render_belief_state(record: Dict[str, Any], state: Any) -> Optional[str]:
    """The ``<belief_state>`` tag of one step *record* (``BeliefModel.observe``)
    with the belief *state* after the step; None without a state or
    criteria."""
    if state is None or not state.criteria:
        return None
    lines: List[str] = [f'<belief_state step="{record["iteration"]}">', "  <criteria>"]
    for i, c in enumerate(state.criteria, 1):
        text = escape(CRITERION_REF_RE.sub(r"k\1", c.text), quote=False)
        coverage = coverage_line(state, i - 1)
        body = ([escape(coverage, quote=False)] if coverage else []) + \
               [escape(candidate_line(x), quote=False) for x in state.ordered(i - 1)]
        proposed = proposals_line(state, i - 1, "unverified")
        if proposed:
            body.append(escape(proposed, quote=False))
        if not body:
            lines.append(f"    <k{i}>{text}</k{i}>")
            continue
        lines.append(f"    <k{i}>{text}")
        lines.extend(f"      {line}" for line in body)
        lines.append(f"    </k{i}>")
    lines.append("  </criteria>")
    lines.append("</belief_state>")
    return "\n".join(lines)
