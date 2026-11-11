"""One JSON-replying LLM call with its parse retries (the belief update's
and the final scorer's)."""

import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


def complete_with_repair(name: str, llm: Any, messages: List[Dict[str, str]], parse: Callable[[str], Any],
                         repair: str, parse_error: str, retries: int,
                         **kwargs: Any) -> Tuple[Any, Optional[str], List[str]]:
    """``(parsed, raw, errors)`` of one JSON-replying call (the belief
    update's, the final scorer's; *name* for the log): *parse* of the reply, else up to
    *retries* more attempts.  A reply that did not parse is shown back with
    the *repair* note; an empty reply is asked again as it was (no empty
    assistant turn).  *parsed* None when no reply parsed (*raw* the last
    reply, None when the first call raised); a call that raises after a
    reply keeps that reply."""
    errors: List[str] = []
    raw = None
    asked = messages
    for attempt in range(1 + retries):
        try:
            reply = llm.complete(asked, **kwargs) or ""
        except Exception as e:
            logger.warning("%s: call failed: %s", name, e)
            return None, raw, errors + [f"{e}"]
        raw = reply if reply or raw is None else raw
        parsed = parse(reply)
        if parsed is not None:
            return parsed, reply, errors
        errors.append(f"parse (attempt {attempt + 1}): {parse_error}" if reply else
                      f"parse (attempt {attempt + 1}): empty reply")
        asked = ([*messages, {"role": "assistant", "content": reply}, {"role": "user", "content": repair}]
                 if reply else messages)
    return None, raw, errors
