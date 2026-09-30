"""User prompts of the uncertainty estimator (the system prompts are the .txt files next to this module)."""

from deep_research_agents.prompts.answer_prompts import AGENT_ANSWER_FORMATS, TAG_FORMAT

CRITERIA_INIT_USER_TEMPLATE = """\
Query: {query}

Identify each criterion, clue, or condition stated in the query and list them as criteria. \
List at most {max_criteria} criteria. This list will be FIXED for the entire search.\
"""

CRITERIA_JUDGE_DOC_USER_TEMPLATE = """\
Query: {query}

Criteria:
{criteria}

Passages:
{passages}

Judge how well each passage covers each criterion.\
"""

CRITERIA_JUDGE_QUERY_USER_TEMPLATE = """\
Query: {query}

Criteria:
{criteria}

Search queries:
{subqueries}

Judge how strongly each search query targets each criterion.\
"""


# Intermediate answer: a user message appended to the agent's conversation at
# the end of every turn (the agent keeps its own system prompt). The model gives
# one answer, several (as a list in the agent's answer field) or none.
INTERMEDIATE_ANSWER_INSTRUCTION = """\
Pause the search here. Based on all the information above, provide what you consider the most likely answer so far.
If several answers are plausible, list them in the answer as [answer 1, answer 2].
If the evidence is insufficient to answer, respond with 'no candidate'.\
"""


def intermediate_answer_instruction(agentic_model: str) -> str:
    """The instruction plus the agent's answer format."""
    fmt = AGENT_ANSWER_FORMATS.get(agentic_model, TAG_FORMAT)
    return f"{INTERMEDIATE_ANSWER_INSTRUCTION}\n\n{fmt}"
