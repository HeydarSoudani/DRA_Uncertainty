"""Prompts of the uncertainty estimator."""

from pathlib import Path

from deep_research_agents.prompts.answer_prompts import (
    BOXED_FORMAT,
    CPM_EXPLORE_FORMAT,
    DRTULU_FORMAT,
    OSS_FORMAT,
    REACT_FORMAT,
    SELFASK_FORMAT,
    TAG_FORMAT,
    WEBWEAVER_FORMAT,
)

_PROMPTS_DIR = Path(__file__).resolve().parent


def _read(name: str) -> str:
    return (_PROMPTS_DIR / name).read_text(encoding="utf-8").strip()


CRITERIA_INIT_SYSTEM = _read("criteria_init_system.txt")

CRITERIA_INIT_USER_TEMPLATE = """\
Query: {query}

Identify each criterion, clue, or condition stated in the query and list them as criteria. \
List at most {max_criteria} criteria. This list will be FIXED for the entire search.\
"""

CRITERIA_JUDGE_DOC_SYSTEM = _read("criteria_judge_doc_system.txt")

CRITERIA_JUDGE_DOC_USER_TEMPLATE = """\
Query: {query}

Criteria:
{criteria}

Passages:
{passages}

Judge how well each passage covers each criterion.\
"""

CRITERIA_JUDGE_QUERY_SYSTEM = _read("criteria_judge_query_system.txt")

CRITERIA_JUDGE_QUERY_USER_TEMPLATE = """\
Query: {query}

Criteria:
{criteria}

Search queries:
{subqueries}

Judge how strongly each search query targets each criterion.\
"""


# Intermediate answer: asked at the end of every turn; the model may give one
# answer, several (as a list in the agent's answer field) or none.
INTERMEDIATE_ANSWER_INSTRUCTION = (
    "You have now reached the maximum context length you can handle. "
    "You should stop making tool calls and, based on all the information "
    "above, think again and provide what you consider the most likely answer.\n"
    "If several answers are plausible, give all of them as a list in the answer "
    "field, for example [answer 1, answer 2]. "
    "If the evidence is insufficient to answer, respond with 'no candidate'."
)

# Answer format of each agent; others get TAG_FORMAT.
INTERMEDIATE_ANSWER_FORMATS = {
    "searcho1":    BOXED_FORMAT,
    "react":       REACT_FORMAT,
    "drtulu":      DRTULU_FORMAT,
    "selfask":     SELFASK_FORMAT,
    "oss":         OSS_FORMAT,
    "glm":         OSS_FORMAT,
    "cpm_explore": CPM_EXPLORE_FORMAT,
    "tongyi":      TAG_FORMAT,
    "webweaver":   WEBWEAVER_FORMAT,
}


def intermediate_answer_instruction(agentic_model: str) -> str:
    """The instruction plus the agent's answer format."""
    return f"{INTERMEDIATE_ANSWER_INSTRUCTION}\n\n{INTERMEDIATE_ANSWER_FORMATS.get(agentic_model, TAG_FORMAT)}"
