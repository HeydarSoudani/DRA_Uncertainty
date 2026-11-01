"""User prompts of the belief model (the system prompts are the .txt files next to this module)."""

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

CRITERIA_UPDATE_USER_TEMPLATE = """\
Query: {query}

Criteria, with their current status:
{belief}

{utilization}New passages:
{passages}

{instruction}\
"""

# The section on the rises of the previous step, before the passages; left
# out (empty) when no criterion rose there.
CRITERIA_UPDATE_RISES_TEMPLATE = """\
Criteria that rose in the previous step, with the passages that raised them:
{rises}

The agent's turn after that step:
<agent_turn>
{agent_turn}
</agent_turn>

"""

# The passages section of a call with rises to judge and no new passage.
CRITERIA_UPDATE_NO_PASSAGES = "(none)"

CRITERIA_UPDATE_INSTRUCTION = (
    "Decide which criteria each new passage supports, contradicts or relates to, if any, and give their new status."
)
CRITERIA_UPDATE_RISES_INSTRUCTION = (
    " Then judge, for each criterion that rose in the previous step, whether the agent's turn uses the "
    "information that raised it."
)

# One passage of the belief updater's user message.
CRITERIA_UPDATE_PASSAGE_TEMPLATE = "<passage {n}>\n{text}\n</passage {n}>"

QUERY_TARGETING_USER_TEMPLATE = """\
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


def intermediate_answer_instruction(policy_model: str) -> str:
    """The instruction plus the agent's answer format."""
    fmt = AGENT_ANSWER_FORMATS.get(policy_model, TAG_FORMAT)
    return f"{INTERMEDIATE_ANSWER_INSTRUCTION}\n\n{fmt}"
