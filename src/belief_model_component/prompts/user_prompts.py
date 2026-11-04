"""User prompts of the belief model (the system prompts are the .txt files next to this module)."""

from typing import Optional

from deep_research_agents.prompts.answer_prompts import AGENT_ANSWER_FORMATS, TAG_FORMAT

from ..types import MULTI_ASPECT

CRITERIA_INIT_USER_TEMPLATE = """\
Query: {query}

{instruction} This list will be FIXED for the entire search.\
"""

# The extractor's instruction: a report request lists every criterion it
# states, uncapped; any other query lists its criteria within the cap, or
# uncapped (gold criteria, layout.DatasetSpec.max_criteria None).
_CRITERIA_INIT_INSTRUCTIONS = {
    MULTI_ASPECT: "List the criteria of this query: every criterion the query states.",
}
_DEFAULT_CRITERIA_INIT_INSTRUCTION = "List the criteria of this query."
_CAPPED_CRITERIA_INIT_INSTRUCTION = "List the criteria of this query, at most {max_criteria}."


def criteria_init_user(shape: str, query: str, max_criteria: Optional[int]) -> str:
    """The criteria extractor's user message for a *shape* query, within
    *max_criteria* (None: uncapped; a report request is never capped)."""
    if shape in _CRITERIA_INIT_INSTRUCTIONS:
        instruction = _CRITERIA_INIT_INSTRUCTIONS[shape]
    elif max_criteria is None:
        instruction = _DEFAULT_CRITERIA_INIT_INSTRUCTION
    else:
        instruction = _CAPPED_CRITERIA_INIT_INSTRUCTION.format(max_criteria=max_criteria)
    return CRITERIA_INIT_USER_TEMPLATE.format(query=query, instruction=instruction)


CRITERIA_UPDATE_USER_TEMPLATE = """\
Query: {query}

Criteria, with their current status{candidates}:
{belief}

{agent_turn}New passages:
{passages}

{instruction}\
"""

# The agent's turn of the step, before the passages; left out (empty) for a
# query without answer candidates (multi-aspect) or a call without a turn.
CRITERIA_UPDATE_TURN_TEMPLATE = """\
The agent's latest turn:
<agent_turn>
{agent_turn}
</agent_turn>

"""

# The passages section of a call with an agent turn and no new passage.
CRITERIA_UPDATE_NO_PASSAGES = "(none)"

CRITERIA_UPDATE_INSTRUCTION = (
    "Decide which criteria each new passage supports, contradicts or relates to, if any, and give their new status."
)
CRITERIA_UPDATE_CANDIDATES_INSTRUCTION = (
    " Then give the new answer candidates of each criterion whose candidates change."
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
