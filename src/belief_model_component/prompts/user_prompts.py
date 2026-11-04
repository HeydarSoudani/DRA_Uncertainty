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


# The belief update (``updater``): every criterion as "c1 (one answer):
# <text>" with its current belief (``belief_state.format_belief``), the
# agent's turn labelled with its step, then the new passages labelled
# "d<n>"; a missing turn or no passage reads ``NO_TEXT``.
BELIEF_UPDATE_USER_TEMPLATE = """\
Query: {query}

Criteria, with their current belief:
{criteria}

The agent's turn [{turn_label}]:
<agent_turn>
{agent_turn}
</agent_turn>

New passages:
{passages}

Record what the agent's turn and the new passages state about the answers of the criteria, then write the new belief of every criterion with a new observation.\
"""

NO_TEXT = "(none)"

# One passage of the belief update's user message.
PASSAGE_TEMPLATE = "<passage {label}>\n{text}\n</passage {label}>"

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
