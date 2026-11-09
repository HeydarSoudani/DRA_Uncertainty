"""The belief as text: one statement per criterion, and the
``<belief_state>`` tag that ``--belief-mode inform`` injects.

The belief state keeps per criterion one statement (``types.Statement``);
what is shown of it is its text only, as the belief update wrote it
(``statement_text`` describes the form)::

    It is certain that c2 is covered, by "Henrik Dahl": d2 states that he was an engineer born in Osby, and d5 states that the Varn Folk Museum was founded in Osby. "Nils Berg" (t1) is ruled out: d4 states that he was born in Tirsa.
    It is unknown whether c3 is covered.

The belief update prompt lists every criterion with its statement
(``format_belief``)::

    c2 (one answer): the engineer who was born in the town where the Varn Folk Museum was founded
      It is certain that c2 is covered, by "Henrik Dahl": ...

At the end of every search iteration the agent reads one tag, each
criterion with its full statement, its own id "k<n>" in place of the
criterion id::

    <belief_state step="3">
      <criteria>
        <k1>the engineer who was born in the town where the Varn Folk Museum was founded
          It is certain that k1 is covered, by "Henrik Dahl": d2 states ...
        </k1>
        <k2>the lighthouse that was designed by that engineer</k2>
      </criteria>
      <sources>[d2] Henrik Dahl | [d4] Nils Berg (engineer) | [d5] Osby</sources>
    </belief_state>

The tag is descriptive only: per criterion its statement, then the titles
of the documents the statements cite ([tN] is the agent's own turn of step
N).  An agent that labels its passages itself (the belief-aware agent's
"dN") passes its labels, and the tag cites each document with the agent's
label (``tag_labels``, applied to the text by
``statement_text.rename_labels``); the belief update keeps its own labels.
Nothing gold-based (qrels, recall) reaches the agent.  With no criteria
there is no tag.
"""

import re
from html import escape
from typing import Any, Dict, List, Optional

from .statement_text import cited_labels, rename_labels
from .types import NO_DOC_KINDS, UNKNOWN, exclusive

# A tag the model wrote itself; removed before its text enters the context.
BELIEF_STATE_RE = re.compile(r"<belief_state\b[^>]*>.*?</belief_state>\s*", re.DOTALL)


def strip_belief_state(text: str) -> str:
    """Remove every ``<belief_state>`` tag from *text*."""
    return BELIEF_STATE_RE.sub("", text) if text and "<belief_state" in text else text


def criterion_head(state: Any, k: int) -> str:
    """Criterion *k* as the belief update reads it: its id, how many
    answers it has, "computed" when no document states it, and its text."""
    c = state.criteria[k]
    marks = ["one answer" if exclusive(c.kind) else "several answers"]
    if c.kind in NO_DOC_KINDS:
        marks.append("computed")
    return f"{c.id} ({', '.join(marks)}): {c.text}"


def format_belief(state: Any) -> str:
    """Every criterion of *state* with its current statement text."""
    lines = []
    for k, text in enumerate(state.statement_texts()):
        lines.append(criterion_head(state, k))
        lines.append(f"  {text}")
    return "\n".join(lines)


def _rename_criteria(text: str, names: Dict[str, str]) -> str:
    """*text* with every criterion id it names (its own in the opening, any
    other it refers to) shown as the tag's name of it (*names*: id -> "k<n>")."""
    if not names:
        return text
    ids = sorted(names, key=len, reverse=True)
    pattern = rf"(?<![A-Za-z0-9])({'|'.join(map(re.escape, ids))})(?![A-Za-z0-9])"
    return re.sub(pattern, lambda m: names[m.group(1)], text)


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
    names = {c.id: f"k{i}" for i, c in enumerate(state.criteria, 1)}
    lines = [f'<belief_state step="{record["iteration"]}">', "  <criteria>"]
    cited: List[str] = []
    for i, (c, s) in enumerate(zip(state.criteria, state.statements), 1):
        text = escape(c.text, quote=False)
        if s.status == UNKNOWN and not s.selected and not s.cited:
            lines.append(f"    <k{i}>{text}</k{i}>")
            continue
        shown = rename_labels(_rename_criteria(s.text, names), rename)
        lines.append(f"    <k{i}>{text}")
        lines.append(f"      {escape(shown, quote=False)}")
        lines.append(f"    </k{i}>")
        cited.extend(lb for lb in cited_labels(s.text) if lb.startswith("d"))
    lines.append("  </criteria>")
    titles = state.doc_titles()
    shown_titles = sorted(((rename.get(e, e), titles[e]) for e in dict.fromkeys(cited) if titles.get(e)),
                          key=lambda pair: _number(pair[0]))
    if shown_titles:
        lines.append("  <sources>" + " | ".join(f"[{label}] {escape(title, quote=False)}"
                                                for label, title in shown_titles) + "</sources>")
    lines.append("</belief_state>")
    return "\n".join(lines)
