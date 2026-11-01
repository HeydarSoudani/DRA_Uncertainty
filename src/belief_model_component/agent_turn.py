"""The agent's turn as the belief updater reads it.

The turn of search iteration t is the agent's reasoning and search queries
of that iteration, written after it read the passages of iteration t-1; the
answer turn is the reasoning and final answer after the last iteration.
The updater call of iteration t reads turn t to judge whether the agent
used the information that raised criteria at t-1 (``utilized``).

The text is built the same way at inference (``BaseAgent`` passes the
iteration's steps) and in belief model training (from the saved
trajectory), so the prompts of both match.
"""

from typing import Any, Dict, Iterable, List, Optional

# Reasoning shown per turn: its last characters (the conclusions on the
# passages just read come last).
MAX_REASONING_CHARS = 6000


def agent_turn_text(thinks: Iterable[Optional[str]], queries: Iterable[Optional[str]],
                    answer: Optional[str] = None) -> str:
    """The turn of one iteration: its reasoning (the non-empty *thinks*,
    in order, the last ``MAX_REASONING_CHARS`` kept), then its search
    *queries* or, for the answer turn, its final *answer*."""
    reasoning = "\n\n".join(t.strip() for t in thinks if t and t.strip())
    if len(reasoning) > MAX_REASONING_CHARS:
        reasoning = "..." + reasoning[-MAX_REASONING_CHARS:]
    parts: List[str] = [f"Reasoning:\n{reasoning or '(none)'}"]
    if answer is not None:
        parts.append(f"Final answer:\n{answer.strip() or '(none)'}")
    else:
        listed = [q.strip() for q in queries if q and q.strip()]
        parts.append("Search queries:\n" + ("\n".join(f"{j}. {q}" for j, q in enumerate(listed, 1)) or "(none)"))
    return "\n\n".join(parts)


def answer_turn(steps: List[Dict[str, Any]], answer: str) -> Optional[str]:
    """The answer turn: the reasoning of the *steps* after the last search
    step (one with a ``search_query``; trajectory steps or their saved
    lines) and the final *answer*; None without an answer."""
    if not answer or not answer.strip():
        return None
    last = max((i for i, s in enumerate(steps) if s.get("search_query")), default=-1)
    return agent_turn_text([s.get("think") for s in steps[last + 1:]], [], answer=answer)
