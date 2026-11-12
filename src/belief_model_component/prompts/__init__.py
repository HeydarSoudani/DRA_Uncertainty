"""Prompts of the belief model.

System prompts are one folder per stage next to this module: the
criteria extractor (``criteria_init/``: the shared core ``system.txt``, its
per-shape parts in ``<shape>.txt``)
and the belief updater (``belief_update/``: the shared core ``system.txt``,
how each shape's criteria are read in ``query_criteria.txt``, the status definitions of
each criterion type in ``criterion_statuses.txt`` and the per-shape example in ``examples.txt``) and the final scorer (``final_score/``: ``system.txt``
with the same ``query_criteria.txt`` and its own per-shape example in
``examples.txt``; short-answer shapes only).  User prompts are in
``user_prompts`` and re-exported here.
"""

import re
from pathlib import Path
from typing import Dict

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


# How the belief updater and the final scorer read the criteria of each query
# shape (``belief_update/query_criteria.txt``): what their parts and answers are.
QUERY_CRITERIA = {k: v for k, v in _sections("belief_update/query_criteria.txt").items() if k}

# The status definitions of each criterion type (``belief_update/criterion_statuses.txt``)
# and the criterion types each query shape has (``criteria.parsing.criterion_kinds``):
# one status set per query type ("one answer" for the short-answer shapes,
# "aspect" for a report), and the aggregate's rule where the shape has one
# (the target of a multi-clue query, the computation of a multi-member one).
CRITERION_STATUSES = {k: v for k, v in _sections("belief_update/criterion_statuses.txt").items() if k}
SHAPE_CRITERION_TYPES = {
    MULTI_CLUE: ("one_answer", "aggregate"),
    MULTI_HOP: ("one_answer",),
    MULTI_MEMBER: ("one_answer", "aggregate"),
    MULTI_ASPECT: ("aspect",),
}

# One worked example of the belief update per query shape (``belief_update/examples.txt``).
BELIEF_UPDATE_EXAMPLES = {k: v for k, v in _sections("belief_update/examples.txt").items() if k}


def _call_system(stage: str, shape: str, examples: Dict[str, str]) -> str:
    """The system prompt of a belief model call (*stage*: ``belief_update``
    or ``final_score``) for a *shape* query: the stage's core with the
    shape's ``query_criteria``, the status definitions of its criterion types (shared
    by both stages) and the stage's example of the shape."""
    fills = {
        "query_criteria": QUERY_CRITERIA[shape],
        "status_definitions": "\n\n".join(CRITERION_STATUSES[t] for t in SHAPE_CRITERION_TYPES[shape]),
        "example": examples[shape],
    }
    text = _read(f"{stage}/system.txt")
    for key, body in fills.items():
        if not body:
            text = text.replace("{" + key + "}\n", "")
        text = text.replace("{" + key + "}", body)
    return text


# One criteria-extraction prompt per query shape (layout.DATASET_SPECS);
# the cap on the number of criteria is the extractor's input
# (``criteria_init_user``).  One belief update prompt per query shape.
CRITERIA_INIT_SYSTEMS = {shape: _compose("criteria_init", shape) for shape in QUERY_SHAPES}
BELIEF_UPDATE_SYSTEMS = {shape: _call_system("belief_update", shape, BELIEF_UPDATE_EXAMPLES) for shape in QUERY_SHAPES}

# One worked example of the final scorer per query shape
# (``final_score/examples.txt``); a shape without one (a report) has no
# final scorer.
FINAL_SCORE_EXAMPLES = {k: v for k, v in _sections("final_score/examples.txt").items() if k and v}


FINAL_SCORE_SYSTEMS = {shape: _call_system("final_score", shape, FINAL_SCORE_EXAMPLES) for shape in FINAL_SCORE_EXAMPLES}
