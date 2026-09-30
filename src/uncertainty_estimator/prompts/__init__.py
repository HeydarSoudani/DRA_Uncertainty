"""Prompts of the uncertainty estimator.

System prompts are the .txt files next to this module; user prompts are in
``user_prompts`` and re-exported here.
"""

from pathlib import Path

from .user_prompts import (
    CRITERIA_INIT_USER_TEMPLATE,
    CRITERIA_JUDGE_DOC_USER_TEMPLATE,
    CRITERIA_JUDGE_QUERY_USER_TEMPLATE,
    intermediate_answer_instruction,
)

_PROMPTS_DIR = Path(__file__).resolve().parent


def _read(name: str) -> str:
    return (_PROMPTS_DIR / name).read_text(encoding="utf-8").strip()


CRITERIA_INIT_SYSTEM = _read("criteria_init_system.txt")
CRITERIA_JUDGE_DOC_SYSTEM = _read("criteria_judge_doc_system.txt")
CRITERIA_JUDGE_QUERY_SYSTEM = _read("criteria_judge_query_system.txt")
