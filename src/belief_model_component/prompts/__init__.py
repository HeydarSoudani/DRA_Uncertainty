"""Prompts of the belief model.

System prompts are the ``*_system.txt`` files next to this module: the
criteria extractor (its per-shape parts in ``criteria_init/<shape>.txt``)
and the belief updater (its per-shape context in ``nugget_shapes.txt``).  User prompts are in
``user_prompts`` and re-exported here.
"""

import re
from pathlib import Path

from ..types import QUERY_SHAPES
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
    """The shared core ``{stage}_system.txt`` with the sections of one query
    shape (``{stage}/{shape}.txt``): the text before the first ``---name---``
    line fills ``{shape_context}``, the ``example`` section ``{example}`` and
    every other section ``{shape_<name>}``.  A placeholder alone on its line
    whose section is empty or missing is removed with its line."""
    sections = {("shape_context" if k == "" else "example" if k == "example" else f"shape_{k}"): body
                     for k, body in _sections(f"{stage}/{shape}.txt").items()}
    text = _read(f"{stage}_system.txt")
    for key, body in sections.items():
        if not body:
            text = text.replace("{" + key + "}\n", "")
        text = text.replace("{" + key + "}", body)
    return re.sub(r"^\{shape_\w+\}\n", "", text, flags=re.MULTILINE)


# The query shapes as the belief updater reads them (``nugget_shapes.txt``): what a
# criterion is and what its answers are.
NUGGET_SHAPES = {k: v for k, v in _sections("nugget_shapes.txt").items() if k}


# One criteria-extraction prompt per query shape (layout.DATASET_SPECS);
# the cap on the number of criteria is the extractor's input
# (``criteria_init_user``).  One belief update prompt per query shape.
CRITERIA_INIT_SYSTEMS = {shape: _compose("criteria_init", shape) for shape in QUERY_SHAPES}
BELIEF_UPDATE_SYSTEMS = {
    shape: _read("belief_update_system.txt").replace("{shape_context}", NUGGET_SHAPES[shape])
    for shape in QUERY_SHAPES
}
