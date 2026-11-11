"""The agent's turn as the belief update reads it.

The turn of search iteration t is the agent's reasoning of that iteration,
written after it read the passages of iteration t-1; its search queries are
not part of it (they stay in the trajectory).  The answer turn is the
reasoning after the last iteration and the final answer.  The belief update
of iteration t reads turn t, with the passages its queries returned, for
what the turn says about each criterion (think observations, labelled
"t<t>"), and the final call the answer turn.

The text is built the same way at inference (``BaseAgent`` passes the
iteration's steps) and in belief model training (from the saved
trajectory), so the prompts of both match.
"""

from typing import Any, Dict, Iterable, List, Optional

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


def answer_turn(steps: List[Dict[str, Any]], answer: str) -> Optional[str]:
    """The answer turn: the reasoning of the *steps* after the last search
    step (one with a ``search_query``; trajectory steps or their saved
    lines) and the final *answer*; None without an answer."""
    if not answer or not answer.strip():
        return None
    last = max((i for i, s in enumerate(steps) if s.get("search_query")), default=-1)
    return agent_turn_text([s.get("think") for s in steps[last + 1:]], answer=answer)
