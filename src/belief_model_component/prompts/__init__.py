"""Prompts of the belief model.

System prompts are one folder per stage next to this module: the
criteria extractor (``criteria_init/``: the shared core ``system.txt``, its
per-shape parts in ``<shape>.txt``)
and the belief updater (``belief_update/``: the shared core ``system.txt``,
the per-shape context in ``criterion_shapes.txt``, the status definitions of
each criterion type in ``criterion_statuses.txt`` and the per-shape example
in ``examples.txt``).  User prompts are in
``user_prompts`` and re-exported here.
"""

import re
from pathlib import Path

from ..types import MULTI_ASPECT, MULTI_HOP, QUERY_SHAPES, SET, SINGLE_TARGET
from .user_prompts import (
    BELIEF_UPDATE_USER_TEMPLATE,
    NO_TEXT,
    PASSAGE_TEMPLATE,
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


# The query shapes as the belief updater reads them (``belief_update/criterion_shapes.txt``): what a
# criterion is and what its answers are.
CRITERION_SHAPES = {k: v for k, v in _sections("belief_update/criterion_shapes.txt").items() if k}

# The status definitions of each criterion type (``belief_update/criterion_statuses.txt``)
# and the criterion types each query shape has (``criteria._kinds``):
# closed criteria are "one answer", rest and aspect criteria "several
# answers", the aggregate of a set query "computed".
CRITERION_STATUSES = {k: v for k, v in _sections("belief_update/criterion_statuses.txt").items() if k}
SHAPE_CRITERION_TYPES = {
    SINGLE_TARGET: ("one_answer",),
    MULTI_HOP: ("one_answer",),
    SET: ("one_answer", "several_answers", "computed"),
    MULTI_ASPECT: ("several_answers",),
}

# One worked example of the belief update per query shape (``belief_update/examples.txt``).
BELIEF_UPDATE_EXAMPLES = {k: v for k, v in _sections("belief_update/examples.txt").items() if k}


def _belief_update_system(shape: str) -> str:
    """The belief update prompt of a *shape* query: the shared core with the
    shape's context, the status definitions of its criterion types and its
    example."""
    fills = {
        "shape_context": CRITERION_SHAPES[shape],
        "status_definitions": "\n\n".join(CRITERION_STATUSES[t] for t in SHAPE_CRITERION_TYPES[shape]),
        "example": BELIEF_UPDATE_EXAMPLES[shape],
    }
    text = _read("belief_update/system.txt")
    for key, body in fills.items():
        text = text.replace("{" + key + "}", body)
    return text


# One criteria-extraction prompt per query shape (layout.DATASET_SPECS);
# the cap on the number of criteria is the extractor's input
# (``criteria_init_user``).  One belief update prompt per query shape.
CRITERIA_INIT_SYSTEMS = {shape: _compose("criteria_init", shape) for shape in QUERY_SHAPES}
BELIEF_UPDATE_SYSTEMS = {shape: _belief_update_system(shape) for shape in QUERY_SHAPES}
