"""Prompts of the belief model.

System prompts are one folder per stage next to this module: the
criteria extractor (``criteria_init/``: the shared core ``system.txt``, its
per-shape parts in ``<shape>.txt``), the belief updater (``belief_update/``:
the shared core ``system.txt``, the status definitions of each criterion
type in ``statuses.txt``, and per query shape ``<shape>.txt`` with its
``terms`` (what its criteria's parts and candidates are), its own
``rules`` (numbered after the core's), its ``output`` notes and its
``example``) and the final scorer (``final_score/``: ``system.txt`` with the
shape's ``terms`` and status definitions from ``belief_update/``, never its
rules or output notes, and its own per-shape example in ``examples.txt``;
short-answer shapes only).  User prompts are in ``user_prompts`` and
re-exported here.
"""

import re
from pathlib import Path

from ..core.types import MULTI_ASPECT, MULTI_CLUE, MULTI_HOP, MULTI_MEMBER, QUERY_SHAPES
from .user_prompts import (
    BELIEF_UPDATE_USER_TEMPLATE,
    FINAL_SCORE_USER_TEMPLATE,
    NO_TEXT,
    PASSAGE_TEMPLATE,
    TURN_TEMPLATE,
    criteria_init_user,
    intermediate_answer_instruction,
)

_PROMPTS_DIR = Path(__file__).resolve().parent


def _read(name: str) -> str:
    return (_PROMPTS_DIR / name).read_text(encoding="utf-8").strip()


_SECTION_RE = re.compile(r"^---(\w+)---$", re.MULTILINE)


def _sections(name: str) -> dict:
    """The ``---name---`` sections of the file *name*, and the text before
    the first one as ``""``."""
    parts = _SECTION_RE.split(_read(name))
    # Blank lines only: a field list keeps its indentation.
    return {"": parts[0].strip(), **{k: body.strip("\n").rstrip() for k, body in zip(parts[1::2], parts[2::2])}}


def _compose(stage: str, shape: str) -> str:
    """The shared core ``{stage}/system.txt`` with the sections of one query
    shape (``{stage}/{shape}.txt``): the text before the first ``---name---``
    line fills ``{shape_context}``, the ``example`` section ``{example}`` and
    every other section ``{shape_<name>}``.  A placeholder alone on its line
    whose section is empty or missing is removed with its line."""
    sections = {("shape_context" if k == "" else "example" if k == "example" else f"shape_{k}"): body
                     for k, body in _sections(f"{stage}/{shape}.txt").items()}
    text = _read(f"{stage}/system.txt")
    for key, body in sections.items():
        if not body:
            text = text.replace("{" + key + "}\n", "")
        text = text.replace("{" + key + "}", body)
    return re.sub(r"^\{shape_\w+\}\n", "", text, flags=re.MULTILINE)


# The status definitions of each criterion type (``belief_update/statuses.txt``,
# meaning only) and the criterion types each query shape has: "one answer"
# for the short-answer shapes, "aspect" for a report, and the aggregate where
# the shape has one, as a target (multi-clue: the entity that meets every
# clue) or a computation (multi-member).  The input names both "aggregate"
# (``views.tag.criterion_type``).
CRITERION_STATUSES = {k: v for k, v in _sections("belief_update/statuses.txt").items() if k}
SHAPE_CRITERION_TYPES = {
    MULTI_CLUE: ("one_answer", "target"),
    MULTI_HOP: ("one_answer",),
    MULTI_MEMBER: ("one_answer", "computed"),
    MULTI_ASPECT: ("aspect",),
}

# The belief update's sections of each query shape (``belief_update/<shape>.txt``):
# ``terms``, ``rules``, ``output`` and ``example``.
SHAPE_SECTIONS = {shape: _sections(f"belief_update/{shape}.txt") for shape in QUERY_SHAPES}

_PLACEHOLDER_RE = re.compile(r"\{(?:shape_\w+|status_definitions|example)\}")


def _call_system(stage: str, shape: str, example: str) -> str:
    """The system prompt of a belief model call (*stage*: ``belief_update``
    or ``final_score``) for a *shape* query: the stage's core with the
    shape's terms, the status definitions of its criterion types (shared
    by both stages), its rules and output notes (placeholders only the
    update has) and *example*.  A placeholder alone on its line whose text
    is empty is removed with its line."""
    sections = SHAPE_SECTIONS[shape]
    fills = {
        "shape_terms": sections.get("terms", ""),
        "status_definitions": "\n\n".join(CRITERION_STATUSES[t] for t in SHAPE_CRITERION_TYPES[shape]),
        "shape_rules": sections.get("rules", ""),
        "shape_output": sections.get("output", ""),
        "example": example,
    }
    text = _read(f"{stage}/system.txt")
    for key, body in fills.items():
        if not body:
            text = text.replace("{" + key + "}\n", "")
        text = text.replace("{" + key + "}", body)
    left = _PLACEHOLDER_RE.findall(text)
    if left:
        raise ValueError(f"{stage} prompt of {shape}: unfilled {left}")
    return text


# One criteria-extraction prompt per query shape (layout.DATASET_SPECS);
# the cap on the number of criteria is the extractor's input
# (``criteria_init_user``).  One belief update prompt per query shape.
CRITERIA_INIT_SYSTEMS = {shape: _compose("criteria_init", shape) for shape in QUERY_SHAPES}
BELIEF_UPDATE_SYSTEMS = {shape: _call_system("belief_update", shape, SHAPE_SECTIONS[shape]["example"])
                         for shape in QUERY_SHAPES}

# One worked example of the final scorer per query shape
# (``final_score/examples.txt``); a shape without one (a report) has no
# final scorer.
FINAL_SCORE_EXAMPLES = {k: v for k, v in _sections("final_score/examples.txt").items() if k and v}


FINAL_SCORE_SYSTEMS = {shape: _call_system("final_score", shape, example) for shape, example in FINAL_SCORE_EXAMPLES.items()}
