"""The belief as text: Agent-BRACE style nugget sentences, and the
``<belief_state>`` tag that ``--belief-mode inform`` injects.

The belief state keeps nuggets and their evidence as data
(``criteria.BeliefState``); the belief update prompt and the agent read
each nugget as one sentence (``nugget_sentence``)::

    It is probable that the answer is "Henrik Dahl", given [d1] [t2].
    It is doubtful that the answer is "Nils Berg", given [t1]; against [d3].
    It is unknown whether the answer is "Lake Brenna: 12 km²", given [t3].

A criterion with one answer (``types.exclusive``) reads "the answer is";
one with several reads '"<value>" is one of the answers'; a report aspect
reads '"<fact>" holds'.  "given" lists the sources that support the value,
"against" the other sources of the nugget's evidence (a contradiction, or
a source of another value).  The belief update prompt also lists each
nugget's evidence under its sentence (``format_belief``)::

    c2 (one answer): the engineer who was born in Osby
      - It is doubtful that the answer is "Nils Berg", given [t1]; against [d3].
          [t1] supports "Nils Berg"
          [d3] contradicts "Nils Berg": "Nils Berg ... was a railway engineer born in Tirsa"

At the end of every search iteration the agent reads one tag::

    <belief_state step="3">
      <criteria>
        <k1>the engineer who was born in Osby
          It is certain that the answer is "Henrik Dahl", given [d1] [t2].
          It is doubtful that the answer is "Nils Berg", given [t1]; against [d3].
        </k1>
        <k2>the lighthouse that the engineer designed</k2>
      </criteria>
      <sources>[d1] Henrik Dahl | [d3] Nils Berg (engineer)</sources>
    </belief_state>

The tag is descriptive only: per criterion its nuggets, then the titles of
the documents the nuggets cite ([tN] is the agent's own turn of step N).
An agent that labels its passages itself (the belief-aware agent's "dN")
passes its labels, and the tag cites each document with the agent's label
(``tag_labels``); the belief update keeps its own labels.
Nothing gold-based (qrels, recall) reaches the agent.  With no criteria
there is no tag.
"""

import re
from html import escape
from typing import Any, Dict, List, Optional

from .types import ASPECT, NO_DOC_KINDS, SUPPORTS, Evidence, Nugget, alnum, exclusive

# A tag the model wrote itself; removed before its text enters the context.
BELIEF_STATE_RE = re.compile(r"<belief_state\b[^>]*>.*?</belief_state>\s*", re.DOTALL)

# How a criterion without nuggets reads in the belief update prompt.
NO_NUGGETS = "(none)"

# Evidence entries listed per nugget in the belief update prompt: the
# latest; the earlier ones stay in the belief (and in the nugget's labels).
MAX_EVIDENCE_SHOWN = 5


def strip_belief_state(text: str) -> str:
    """Remove every ``<belief_state>`` tag from *text*."""
    return BELIEF_STATE_RE.sub("", text) if text and "<belief_state" in text else text


def nugget_sentence(nugget: Nugget, kind: str, rename: Optional[Dict[str, str]] = None) -> str:
    """One nugget of a criterion of *kind* as an Agent-BRACE sentence:
    "given" the sources that support its value, "against" the others.
    *rename* maps a source label to the label shown (the inform tag shows
    the agent's own passage labels); a label not in it is shown as is."""
    value, status = nugget.value, nugget.status or "unknown"
    if kind == ASPECT:
        claim = f'"{value}" holds'
    elif exclusive(kind):
        claim = f'the answer is "{value}"'
    else:
        claim = f'"{value}" is one of the answers'
    head = f"It is unknown whether {claim}" if status == "unknown" else f"It is {status} that {claim}"
    key = alnum(value)
    support = {e.label for e in nugget.evidence if e.stance == SUPPORTS and alnum(e.value) == key}
    given = [lb for lb in nugget.labels if lb in support]
    against = [lb for lb in nugget.labels if lb not in support]
    rename = rename or {}
    parts = []
    if given:
        parts.append("given " + " ".join(f"[{rename.get(e, e)}]" for e in given))
    if against:
        parts.append("against " + " ".join(f"[{rename.get(e, e)}]" for e in against))
    return head + (", " + "; ".join(parts) if parts else "") + "."


def evidence_line(e: Evidence) -> str:
    """One evidence entry as the belief update reads it: its source, stance
    and value, and a document's verified spans."""
    line = f'[{e.label}] {e.stance} "{e.value}"'
    return line + (": " + " | ".join(f'"{s}"' for s in e.spans) if e.spans else "")


def criterion_sentences(state: Any, k: int, rename: Optional[Dict[str, str]] = None) -> List[str]:
    """The nugget sentences of criterion *k* of *state* (a ``BeliefState``)."""
    c = state.criteria[k]
    return [nugget_sentence(n, c.kind, rename) for n in state.nuggets[k]]


def criterion_head(state: Any, k: int) -> str:
    """Criterion *k* as the belief update reads it: its id, how many
    answers it has, "computed" when no document states it, and its text."""
    c = state.criteria[k]
    marks = ["one answer" if exclusive(c.kind) else "several answers"]
    if c.kind in NO_DOC_KINDS:
        marks.append("computed")
    return f"{c.id} ({', '.join(marks)}): {c.text}"


def format_belief(state: Any) -> str:
    """Every criterion of *state* with its current belief: per nugget its
    sentence and its evidence (the latest ``MAX_EVIDENCE_SHOWN``)."""
    lines = []
    for k, nuggets in enumerate(state.nuggets):
        lines.append(criterion_head(state, k))
        if not nuggets:
            lines.append(f"  current belief: {NO_NUGGETS}")
            continue
        for n, sentence in zip(nuggets, criterion_sentences(state, k)):
            lines.append(f"  - {sentence}")
            hidden = len(n.evidence) - MAX_EVIDENCE_SHOWN
            if hidden > 0:
                lines.append(f"      ({hidden} earlier sources not listed)")
            lines.extend(f"      {evidence_line(e)}" for e in n.evidence[-MAX_EVIDENCE_SHOWN:])
    return "\n".join(lines)


def _number(label: str) -> int:
    digits = re.sub(r"\D", "", label)
    return int(digits) if digits else 0


def tag_labels(state: Any, agent_labels: Optional[Dict[str, str]]) -> Dict[str, str]:
    """Per document label of *state*, the label the tag shows: the agent's
    own label of that document (*agent_labels*: document id -> the label
    the agent shows it with, e.g. "d3") when it has one; else its own label,
    prefixed "b" when the agent uses that label for another passage."""
    if not agent_labels:
        return {}
    used = set(agent_labels.values())
    out = {}
    for key, label in state.doc_keys().items():
        own = agent_labels.get(key)
        out[label] = own if own else (f"b{label}" if label in used else label)
    return out


def render_belief_state(record: Dict[str, Any], state: Any,
                        agent_labels: Optional[Dict[str, str]] = None) -> Optional[str]:
    """The ``<belief_state>`` tag of one step *record* (``BeliefModel.observe``)
    with the belief *state* after the step; None without a state or
    criteria.  With *agent_labels* (document id -> the agent's own passage
    label), each document is cited with the agent's label (``tag_labels``),
    so a [dN] of the tag is the agent's [dN]."""
    if state is None or not state.criteria:
        return None
    rename = tag_labels(state, agent_labels)
    lines = [f'<belief_state step="{record["iteration"]}">', "  <criteria>"]
    cited: List[str] = []
    for i, c in enumerate(state.criteria, 1):
        text = escape(c.text, quote=False)
        sentences = criterion_sentences(state, i - 1, rename)
        if not sentences:
            lines.append(f"    <k{i}>{text}</k{i}>")
            continue
        lines.append(f"    <k{i}>{text}")
        lines.extend(f"      {escape(s, quote=False)}" for s in sentences)
        lines.append(f"    </k{i}>")
        cited.extend(lb for nug in state.nuggets[i - 1] for lb in nug.labels if lb.startswith("d"))
    lines.append("  </criteria>")
    titles = state.doc_titles()
    shown = sorted(((rename.get(e, e), titles[e]) for e in dict.fromkeys(cited) if titles.get(e)),
                   key=lambda pair: _number(pair[0]))
    if shown:
        lines.append("  <sources>" + " | ".join(f"[{label}] {escape(title, quote=False)}" for label, title in shown)
                     + "</sources>")
    lines.append("</belief_state>")
    return "\n".join(lines)
