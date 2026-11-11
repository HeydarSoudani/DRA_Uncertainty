"""The agent's own (verbalized) confidence in its final answer.

The agents whose answer format asks for it (``OSS_FORMAT``: glm, oss) end
their answer with ``Confidence: <0-100>%``.  It is read from the saved
generation by a regex (the last such line outside the References block),
never by the judge, and is the baseline of the uncertainty evaluation.  A
generation without the line has none (None), never a default value.
"""

import re
from typing import Optional

from ..judge import strip_references

_CONFIDENCE = re.compile(r"Confidence\s*:\s*\**\s*(\d+(?:\.\d+)?)\s*%", re.IGNORECASE)


def verbal_confidence(generation: str) -> Optional[float]:
    """The confidence the generation states, in [0, 1], or None."""
    found = _CONFIDENCE.findall(strip_references(generation or ""))
    if not found:
        return None
    return max(0.0, min(100.0, float(found[-1]))) / 100.0
