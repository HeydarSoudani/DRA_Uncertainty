"""The agent's own answer(s) after each turn (evaluation only, never shown
to the agent, not evaluated)."""

import logging
from typing import Any, Callable, Dict, Optional

from deep_research_agents.prompts.answer_prompts import extract_answer_candidates, extract_bare_answer
from reasoner_component import REASONING_FALLBACK_PREFIX

from ..prompts import intermediate_answer_instruction

logger = logging.getLogger(__name__)

# (original_query, trajectory, instruction) -> raw model output; the agent's
# ``answer_from_trajectory``.
AnswerFn = Callable[[str, Any, str], str]


class IntermediateAnswerSignal:
    """The agent's own answer(s) at the end of each turn, not evaluated.

    Asks the agent's model, through ``answer_fn``, for its most likely
    answer given the trajectory so far, in the agent's answer format.  The
    model may give one answer, several (a list) or none ("no candidate").
    A reply that is only the answer value (one short line, no tags, e.g. a
    bare ``no candidate``) is accepted as is.

    ``score`` returns ``{"answers": [...], "reasoning": str}``; ``answers``
    is empty when the model gave no answer.  A stated confidence is ignored.
    It returns ``None`` when the reply has no answer format, and raises
    when the call fails.

    Args:
        answer_fn: The agent's ``answer_from_trajectory``.
        policy_model: Agent name; picks the answer format and the parser.
    """

    def __init__(self, answer_fn: AnswerFn, policy_model: str) -> None:
        self._answer_fn = answer_fn
        self._policy_model = policy_model

    def score(self, original_query: str, trajectory: Any) -> Optional[Dict[str, Any]]:
        instruction = intermediate_answer_instruction(self._policy_model)
        raw = self._answer_fn(original_query, trajectory, instruction) or ""
        if raw.startswith(REASONING_FALLBACK_PREFIX):
            raw = raw[len(REASONING_FALLBACK_PREFIX):]
        outputs, format_matched = extract_answer_candidates(raw, expected_format=self._policy_model)
        if not format_matched:
            outputs = extract_bare_answer(raw)
            if outputs is None:
                logger.warning("Intermediate answer without answer format: %r", raw.strip()[:200])
                return None
        answers = [o.candidate for o in outputs if o.candidate.lower() != "no candidate"]
        first = outputs[0] if outputs else None
        return {
            "answers": answers,
            "reasoning": first.reasoning if first else "",
        }
