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


# The belief update (``updater``), in the layout of the system prompt's
# example: every criterion as "c1 (one answer): <text>" with its current
# candidates, their support and its proposals (``belief_state.format_belief``),
# the agent's turns since the last update, each "[t<step>]" on its own line
# before its text, then the passages new to the belief, each "[d<n>]" before
# its text (whose first line is the title); no turn or no
# passage reads ``NO_TEXT``.  The answer turn is read as any other turn.
BELIEF_UPDATE_USER_TEMPLATE = """\
Query: {query}

Criteria, with their current candidates:
{criteria}

The agent's reasoning since the last update:
{turns}

Passages:
{passages}

Extract the agent claims of the reasoning and the text of each passage that states a part of a criterion, then give the support and the belief status of every candidate they change.\
"""

NO_TEXT = "(none)"

# The final scorer (``final_score``): the query, the belief before the
# answer turn in the update's layout, the names a passage ruled out
# included (``format_belief(state, removed=True)``), and the agent's answer
# turn (its last reasoning, then "Final answer:" and the answer).
FINAL_SCORE_USER_TEMPLATE = """\
Query: {query}

Criteria, with their candidates:
{criteria}

The agent's final answer:
{answer}

Find the answer's candidate on each criterion, then give the final status.\
"""

# One turn of the belief update's user message.
TURN_TEMPLATE = "[{label}]\n{text}"
# One passage of the belief update's user message.
PASSAGE_TEMPLATE = "[{label}] {text}"


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
