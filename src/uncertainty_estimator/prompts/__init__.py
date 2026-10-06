"""Prompts of the uncertainty estimator.

System prompts are the .txt files next to this module; user prompts are in
``user_prompts`` and re-exported here.
"""

from pathlib import Path

from ..types import KINDS, QUERY_SHAPES
from .user_prompts import (
    criteria_init_user,
    CRITERIA_JUDGE_DOC_USER_TEMPLATE,
    CRITERIA_JUDGE_QUERY_USER_TEMPLATE,
    intermediate_answer_instruction,
)

_PROMPTS_DIR = Path(__file__).resolve().parent


def _read(name: str) -> str:
    return (_PROMPTS_DIR / name).read_text(encoding="utf-8").strip()


def _init_system(shape: str) -> str:
    """The shared core with the rules and example of one query shape
    (``criteria_init_{shape}.txt``, rules and example split by ``---example---``)."""
    rules, example = _read(f"criteria_init_{shape}.txt").split("---example---")
    return (
        _read("criteria_init_system.txt")
        .replace("{shape_rules}", rules.strip())
        .replace("{example}", example.strip())
    )


# One criteria-extraction prompt per query shape (layout.DATASET_SPECS); the
# cap on the number of criteria is its input (``criteria_init_user``).
CRITERIA_INIT_SYSTEMS = {shape: _init_system(shape) for shape in QUERY_SHAPES}
CRITERIA_JUDGE_DOC_SYSTEM = _read("criteria_judge_doc_system.txt")
CRITERIA_JUDGE_QUERY_SYSTEM = _read("criteria_judge_query_system.txt")


def coverage_judge_system(kinds) -> str:
    """The coverage judge's system prompt, defining only the criterion kinds
    in *kinds* (a closed-only list never reads about open criteria)."""
    return "\n".join(
        line for line in CRITERIA_JUDGE_DOC_SYSTEM.split("\n")
        if not any(line.startswith(f'- "{k}":') and k not in kinds for k in KINDS)
    )
