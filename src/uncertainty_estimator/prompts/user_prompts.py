"""User prompts of the uncertainty estimator (the system prompts are the .txt files next to this module)."""

from deep_research_agents.prompts.answer_prompts import AGENT_ANSWER_FORMATS, TAG_FORMAT

from ..types import MULTI_ASPECT

CRITERIA_INIT_USER_TEMPLATE = """\
Query: {query}

{instruction} This list will be FIXED for the entire search.\
"""

# What the cap limits: every criterion, or (a report request) only the
# criteria the extractor adds to the ones the request states (asked of the
# model; the code keeps a report request's list whole).
_CRITERIA_INIT_INSTRUCTIONS = {
    MULTI_ASPECT: "List the criteria of this query: every criterion the query states, then at most {max_criteria} added criteria.",
}
_DEFAULT_CRITERIA_INIT_INSTRUCTION = "List the criteria of this query, at most {max_criteria}."


def criteria_init_user(shape: str, query: str, max_criteria: int) -> str:
    """The criteria extractor's user message for a *shape* query."""
    instruction = _CRITERIA_INIT_INSTRUCTIONS.get(shape, _DEFAULT_CRITERIA_INIT_INSTRUCTION)
    return CRITERIA_INIT_USER_TEMPLATE.format(
        query=query, instruction=instruction.format(max_criteria=max_criteria))

CRITERIA_UPDATE_DOC_USER_TEMPLATE = """\
Query: {query}

Criteria, with their current status and attached evidence:
{criteria}

New passages:
{passages}

Update the status of the criteria given the attached evidence and the new passages.\
"""

CRITERIA_MATCH_QUERY_USER_TEMPLATE = """\
Query: {query}

Criteria, with their current status and what the search has found so far:
{criteria}

Search queries:
{subqueries}

Score how strongly each search query targets each criterion.\
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
