"""The agent's turn as the belief update reads it.

The turn of search iteration t is the agent's reasoning of that iteration,
written after it read the passages of iteration t-1; its search queries are
not part of it (they stay in the trajectory).  The answer turn is the
reasoning after the last iteration and the final answer.  The belief update
of iteration t reads turn t, with the passages its queries returned, for
what the turn says about each belief (think observations, labelled
"t<t>"); the final update reads the answer turn's reasoning only
(``answer_reasoning``), and the final scorer its final answer only.  The
belief stage passes the two apart; ``split_answer_turn`` recovers them from
a joined answer turn.

The text is built the same way at inference (``BaseAgent`` passes the
iteration's steps) and in belief model training (from the saved
trajectory), so the prompts of both match.
"""

from typing import Any, Dict, Iterable, List, Optional, Tuple

# Reasoning shown per turn: its last characters (the conclusions on the
# passages just read come last).
MAX_REASONING_CHARS = 6000

# The header of the answer turn's final answer (``agent_turn_text``).
FINAL_ANSWER_HEADER = "Final answer:"


def agent_turn_text(thinks: Iterable[Optional[str]], answer: Optional[str] = None) -> str:
    """The turn of one iteration: its reasoning (the non-empty *thinks*, in
    order, the last ``MAX_REASONING_CHARS`` kept); for the answer turn,
    then its final *answer* under ``FINAL_ANSWER_HEADER``."""
    reasoning = "\n\n".join(t.strip() for t in thinks if t and t.strip())
    if len(reasoning) > MAX_REASONING_CHARS:
        reasoning = "..." + reasoning[-MAX_REASONING_CHARS:]
    if answer is None:
        return reasoning
    return "\n\n".join(p for p in (reasoning, f"{FINAL_ANSWER_HEADER}\n{answer.strip()}") if p)


def split_answer_turn(turn: str) -> Tuple[str, str]:
    """``(reasoning, answer)`` of an answer turn (``agent_turn_text`` with
    an answer): the text before and after its last ``FINAL_ANSWER_HEADER``
    line; ``("", turn)`` without one."""
    text = turn.strip()
    marker = f"{FINAL_ANSWER_HEADER}\n"
    cut = text.rfind(f"\n\n{marker}")
    if cut >= 0:
        return text[:cut].strip(), text[cut + 2 + len(marker):].strip()
    if text.startswith(marker):
        return "", text[len(marker):].strip()
    return "", text


def _after_last_search(steps: List[Dict[str, Any]]) -> List[Optional[str]]:
    """The reasoning of the *steps* after the last search iteration (the
    last step with a ``search_step``, the number the agent gives every step
    of an iteration that searched; trajectory steps or their saved lines)."""
    last = max((i for i, s in enumerate(steps) if s.get("search_step") is not None), default=-1)
    return [s.get("think") for s in steps[last + 1:]]


def answer_reasoning(steps: List[Dict[str, Any]]) -> str:
    """The answer turn's reasoning without the answer, as the final belief
    update reads it (``agent_turn_text`` of the steps after the last
    search); "" when there is none."""
    return agent_turn_text(_after_last_search(steps))


def answer_turn(steps: List[Dict[str, Any]], answer: str) -> Optional[str]:
    """The answer turn: the reasoning of the *steps* after the last search
    iteration and the final *answer*; None without an answer."""
    if not answer or not answer.strip():
        return None
    return agent_turn_text(_after_last_search(steps), answer=answer)
