"""Prompts of the belief model.

System prompts are the .txt files next to this module; user prompts are in
``user_prompts`` and re-exported here.
"""

import re
from pathlib import Path

from ..types import QUERY_SHAPES
from .user_prompts import (
    criteria_init_user,
    BELIEF_UPDATE_PASSAGE_TEMPLATE,
    BELIEF_UPDATE_USER_TEMPLATE,
    QUERY_TARGETING_USER_TEMPLATE,
    intermediate_answer_instruction,
)

_PROMPTS_DIR = Path(__file__).resolve().parent


def _read(name: str) -> str:
    return (_PROMPTS_DIR / name).read_text(encoding="utf-8").strip()


_SECTION_RE = re.compile(r"^---(\w+)---$", re.MULTILINE)


def _compose(stage: str, shape: str) -> str:
    """The shared core ``{stage}_system.txt`` with the sections of one query
    shape (``{stage}_{shape}.txt``): the text before the first ``---name---``
    line fills ``{shape_context}``, the ``example`` section ``{example}`` and
    every other section ``{shape_<name>}``.  A placeholder alone on its line
    whose section is empty or missing is removed with its line."""
    parts = _SECTION_RE.split(_read(f"{stage}_{shape}.txt"))
    sections = {"shape_context": parts[0].strip()}
    for name, body in zip(parts[1::2], parts[2::2]):
        # Blank lines only: a field list keeps its indentation.
        sections["example" if name == "example" else f"shape_{name}"] = body.strip("\n").rstrip()
    text = _read(f"{stage}_system.txt")
    for key, body in sections.items():
        if not body:
            text = text.replace("{" + key + "}\n", "")
        text = text.replace("{" + key + "}", body)
    return re.sub(r"^\{shape_\w+\}\n", "", text, flags=re.MULTILINE)


# One criteria-extraction prompt and one belief-update prompt per query
# shape (layout.DATASET_SPECS); the cap on the number of criteria is the
# extractor's input (``criteria_init_user``).
CRITERIA_INIT_SYSTEMS = {shape: _compose("criteria_init", shape) for shape in QUERY_SHAPES}
BELIEF_UPDATE_SYSTEMS = {shape: _compose("belief_update", shape) for shape in QUERY_SHAPES}
QUERY_TARGETING_SYSTEM = _read("query_targeting_system.txt")
