"""Prompts of the belief model.

System prompts are the .txt files next to this module; user prompts are in
``user_prompts`` and re-exported here.
"""

import re
from pathlib import Path

from ..types import MULTI_ASPECT, QUERY_SHAPES
from .user_prompts import (
    criteria_init_user,
    CRITERIA_UPDATE_CANDIDATES_INSTRUCTION,
    CRITERIA_UPDATE_INSTRUCTION,
    CRITERIA_UPDATE_NO_PASSAGES,
    CRITERIA_UPDATE_PASSAGE_TEMPLATE,
    CRITERIA_UPDATE_TURN_TEMPLATE,
    CRITERIA_UPDATE_USER_TEMPLATE,
    QUERY_TARGETING_USER_TEMPLATE,
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


# The answer-candidate parts of the criteria-update prompt (the reasoner
# side of the belief); a report query (multi-aspect) has no candidates.
_CANDIDATE_PARTS = _sections("criteria_update_candidates.txt")
_CANDIDATES = {
    "candidates_section": _CANDIDATE_PARTS["section"] + "\n\n",
    "candidates_rules": _CANDIDATE_PARTS["rules"] + "\n\n",
    "candidates_output": _CANDIDATE_PARTS["output"] + "\n",
    "candidates_intro_belief": " and answer candidates",
    "candidates_intro_inputs": "the agent's latest turn, and",
    "candidates_intro_task": " and the answer candidates of the criteria from the agent's turn",
    "output_keys": "two keys",
}
_NO_CANDIDATES = {**{k: "" for k in _CANDIDATES}, "candidates_rules": "\n", "candidates_intro_inputs": "and",
                  "output_keys": "one key"}


def _compose(stage: str, shape: str) -> str:
    """The shared core ``{stage}_system.txt`` with the sections of one query
    shape (``{stage}_{shape}.txt``): the text before the first ``---name---``
    line fills ``{shape_context}``, the ``example`` section ``{example}`` and
    every other section ``{shape_<name>}``.  A placeholder alone on its line
    whose section is empty or missing is removed with its line.  The update
    prompt also gets the answer-candidate parts (none for a report query)."""
    sections = {}
    if stage == "criteria_update":
        # First: the candidate parts hold the {shape_candidates} placeholder.
        sections.update(_NO_CANDIDATES if shape == MULTI_ASPECT else _CANDIDATES)
    sections.update({("shape_context" if k == "" else "example" if k == "example" else f"shape_{k}"): body
                     for k, body in _sections(f"{stage}_{shape}.txt").items()})
    text = _read(f"{stage}_system.txt")
    for key, body in sections.items():
        if not body:
            text = text.replace("{" + key + "}\n", "")
        text = text.replace("{" + key + "}", body)
    return re.sub(r"^\{shape_\w+\}\n", "", text, flags=re.MULTILINE)


# One criteria-extraction prompt and one criteria-update prompt per query
# shape (layout.DATASET_SPECS); the cap on the number of criteria is the
# extractor's input (``criteria_init_user``).
CRITERIA_INIT_SYSTEMS = {shape: _compose("criteria_init", shape) for shape in QUERY_SHAPES}
CRITERIA_UPDATE_SYSTEMS = {shape: _compose("criteria_update", shape) for shape in QUERY_SHAPES}
QUERY_TARGETING_SYSTEM = _read("query_targeting_system.txt")
