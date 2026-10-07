"""Prompts of the uncertainty estimator.

System prompts are the .txt files next to this module; user prompts are in
``user_prompts`` and re-exported here.
"""

from pathlib import Path

from ..types import QUERY_SHAPES
from .user_prompts import (
    criteria_init_user,
    CRITERIA_UPDATE_DOC_USER_TEMPLATE,
    CRITERIA_MATCH_QUERY_USER_TEMPLATE,
    intermediate_answer_instruction,
)

_PROMPTS_DIR = Path(__file__).resolve().parent


def _read(name: str) -> str:
    return (_PROMPTS_DIR / name).read_text(encoding="utf-8").strip()


def _compose(stage: str, shape: str) -> str:
    """The shared core ``{stage}_system.txt`` with the context, rules and
    example of one query shape (``{stage}_{shape}.txt``, split by
    ``---rules---`` and ``---example---``)."""
    context, rest = _read(f"{stage}_{shape}.txt").split("---rules---")
    rules, example = rest.split("---example---")
    return (
        _read(f"{stage}_system.txt")
        .replace("{shape_context}", context.strip())
        .replace("{shape_rules}", rules.strip())
        .replace("{example}", example.strip())
    )


# One criteria-extraction prompt and one criteria-update prompt per query
# shape (layout.DATASET_SPECS); the cap on the number of criteria is the
# extractor's input (``criteria_init_user``).
CRITERIA_INIT_SYSTEMS = {shape: _compose("criteria_init", shape) for shape in QUERY_SHAPES}
CRITERIA_UPDATE_DOC_SYSTEMS = {shape: _compose("criteria_update_doc", shape) for shape in QUERY_SHAPES}
CRITERIA_MATCH_QUERY_SYSTEM = _read("criteria_match_query_system.txt")
