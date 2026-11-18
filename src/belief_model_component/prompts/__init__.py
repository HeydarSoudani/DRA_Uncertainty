"""Prompts of the belief model.

One folder per belief model call (stage), each laid out the same way:

    <stage>/system.txt    the core system prompt, shared by every query shape
    <stage>/user.txt      the user template
    <stage>/<shape>.txt   the parts of one query shape, as ``---name---``
                          sections: rules, example

The stages are ``belief_init`` (the beliefs of a query), ``belief_update``
(the new belief state after a turn) and ``belief_score`` (the WEP that
the final answer is correct; short-answer shapes only, so a shape without a
file there has no scorer).  ``shared/wep_definitions.txt`` holds the words
of estimative probability of each belief type, used by the update and the
scorer.

A section ``name`` fills the placeholder ``{shape_<name>}`` of the core
(``{example}`` for the example) and ``{wep_definitions}`` takes the WEP
definitions of the shape's belief types (``SHAPE_WEP_TYPES``).  Shape
rules are written as ``- `` lines, numbered after the core's last rule
when the core numbers its rules, else kept as bullets.
A placeholder alone on its line whose text is empty is removed with its
line.  Every prompt is built at import, which fails on a section the core
has no placeholder for, an unknown section or an unknown placeholder.
"""

import re
from pathlib import Path
from typing import Optional

from ..core.types import MULTI_ASPECT, MULTI_CLUE, MULTI_HOP, MULTI_MEMBER, QUERY_SHAPES

_PROMPTS_DIR = Path(__file__).resolve().parent

SHAPE_SECTIONS = ("rules", "example")

_SECTION_RE = re.compile(r"^---(\w+)---$", re.MULTILINE)
_PLACEHOLDER_RE = re.compile(r"\{(shape_\w+|wep_definitions|example)\}")
_KNOWN_PLACEHOLDERS = {f"shape_{s}" for s in SHAPE_SECTIONS if s != "example"} | {"wep_definitions", "example"}
_RULE_NUMBER_RE = re.compile(r"^(\d+)\. ", re.MULTILINE)


def _read(name: str) -> str:
    return (_PROMPTS_DIR / name).read_text(encoding="utf-8").strip()


def _sections(name: str) -> dict:
    """The ``---name---`` sections of the file *name*; text before the
    first one is an error."""
    parts = _SECTION_RE.split(_read(name))
    if parts[0].strip():
        raise ValueError(f"{name}: text before the first ---section---")
    # Blank lines only: a field list keeps its indentation.
    return {k: body.strip("\n").rstrip() for k, body in zip(parts[1::2], parts[2::2])}


# The words of estimative probability of each belief type, meaning only,
# and the belief types of each query shape: "one answer" for the
# short-answer shapes, "aspect" for a report.
WEP_DEFINITIONS = _sections("shared/wep_definitions.txt")
SHAPE_WEP_TYPES = {
    MULTI_CLUE: ("one_answer",),
    MULTI_HOP: ("one_answer",),
    MULTI_MEMBER: ("one_answer",),
    MULTI_ASPECT: ("aspect",),
}
_missing = {t for types in SHAPE_WEP_TYPES.values() for t in types} - set(WEP_DEFINITIONS)
if _missing:
    raise ValueError(f"shared/wep_definitions.txt: no section for {sorted(_missing)}")


def _number_rules(core: str, rules: str) -> str:
    """The shape's ``- `` rules numbered after the last rule number of
    *core* before ``{shape_rules}``; kept as ``- `` bullets when the core's
    rules there are bullets (no numbered rule before it)."""
    numbers = _RULE_NUMBER_RE.findall(core[:core.index("{shape_rules}")])
    if not numbers:
        return rules
    first = int(numbers[-1]) + 1
    lines = rules.split("\n")
    n = first
    for i, line in enumerate(lines):
        if line.startswith("- "):
            lines[i] = f"{n}. {line[2:]}"
            n += 1
    return "\n".join(lines)


def _compose(stage: str, shape: str) -> str:
    """The system prompt of *stage* for a *shape* query."""
    core = _read(f"{stage}/system.txt")
    used = set(_PLACEHOLDER_RE.findall(core))
    unknown = used - _KNOWN_PLACEHOLDERS
    if unknown:
        raise ValueError(f"{stage}/system.txt: unknown placeholders {sorted(unknown)}")
    fills = {"wep_definitions": "\n\n".join(WEP_DEFINITIONS[t] for t in SHAPE_WEP_TYPES[shape])}
    for name, body in _sections(f"{stage}/{shape}.txt").items():
        key = "example" if name == "example" else f"shape_{name}"
        if name not in SHAPE_SECTIONS or key not in used:
            raise ValueError(f"{stage}/{shape}.txt: section ---{name}--- has no place in {stage}/system.txt")
        fills[key] = _number_rules(core, body) if name == "rules" else body
    for key in used:
        body = fills.get(key, "")
        if not body:
            core = core.replace("{" + key + "}\n", "")
        core = core.replace("{" + key + "}", body)
    left = _PLACEHOLDER_RE.findall(core)
    if left:
        raise ValueError(f"{stage} prompt of {shape}: unfilled {left}")
    return core


def _systems(stage: str) -> dict:
    """The system prompt of *stage* for every shape with a file there."""
    return {shape: _compose(stage, shape) for shape in QUERY_SHAPES
            if (_PROMPTS_DIR / stage / f"{shape}.txt").exists()}


BELIEF_INIT_SYSTEMS = _systems("belief_init")
BELIEF_UPDATE_SYSTEMS = _systems("belief_update")
BELIEF_SCORE_SYSTEMS = _systems("belief_score")
for _stage, _systems_of in (("belief_init", BELIEF_INIT_SYSTEMS), ("belief_update", BELIEF_UPDATE_SYSTEMS)):
    if set(_systems_of) != set(QUERY_SHAPES):
        raise ValueError(f"{_stage}: no prompt for {sorted(set(QUERY_SHAPES) - set(_systems_of))}")


# The user templates.  The update reads the query, the previous belief
# state (``views.belief_text.format_belief``), the agent's turns since the
# last update, each "[t<step>]" on its own line before its text, and the
# passages its last search returned, each "[d<n>]" before its text (whose
# first line is the title); no turn or no passage reads ``NO_TEXT``.  The
# scorer reads the query, the belief before the answer turn (``format_belief``)
# and the agent's final turn, "[t<N>]" with its final answer at the end
# (``update.final_score.scorer_answer_text``).
BELIEF_INIT_USER = _read("belief_init/user.txt")
BELIEF_UPDATE_USER = _read("belief_update/user.txt")
BELIEF_SCORE_USER = _read("belief_score/user.txt")

NO_TEXT = "(none)"
# One turn and one passage of the update's user message.
TURN_TEMPLATE = "[{label}]\n{text}"
PASSAGE_TEMPLATE = "[{label}] {text}"

# The initializer's instruction, the same for every shape: the cap on the
# number of beliefs is a run setting (None: uncapped, as for gold
# beliefs, layout.DatasetSpec.max_beliefs None), stated to a multi-hop
# query only (a chain keeps the hops closest to the answer); a multi-clue,
# multi-member or report query lists every clue, member or requirement the
# query states (code still keeps a multi-member list within the cap,
# ``init.parsing.member_layout``).
_BELIEF_INIT_INSTRUCTION = "List the beliefs of this query."
_CAPPED_BELIEF_INIT_INSTRUCTION = "List the beliefs of this query, at most {max_beliefs}."


def belief_init_user(shape: str, query: str, max_beliefs: Optional[int]) -> str:
    """The initializer's user message for a *shape* query, within
    *max_beliefs* (None: uncapped; stated to a multi-hop query only)."""
    if max_beliefs is None or shape != MULTI_HOP:
        instruction = _BELIEF_INIT_INSTRUCTION
    else:
        instruction = _CAPPED_BELIEF_INIT_INSTRUCTION.format(max_beliefs=max_beliefs)
    return BELIEF_INIT_USER.format(query=query, instruction=instruction)
