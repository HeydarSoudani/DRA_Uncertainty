"""The agent's own (verbalized) confidence in its final answer.

The agents whose answer format asks for it (``OSS_FORMAT``: glm, oss) end
their answer with ``Confidence: <0-100>%``.  It is read from the saved
generation by a regex (the last one outside the References block; markdown
emphasis and an approximation sign such as "~85 %" accepted; the number
must end the value, so "Confidence: 3 sources agree" is no confidence; a
decimal without "%" at most 1 ("0.85") read as a fraction, any other value
as a percent, as the format asks),
never by the judge, and is the baseline of the uncertainty evaluation.  A
generation without the line has none (None), never a default value.
"""

import re
from typing import Optional

from ..judge import strip_references

_CONFIDENCE = re.compile(
    r"\bConfidence[*_]*[^\S\n]*:[^\S\n]*[*_]*[^\S\n]*"
    r"(?:~|≈|about|approximately|approx\.?)?[^\S\n]*(\d+(?:\.\d+)?)[^\S\n]*(%|/[^\S\n]*100)?[*_]*"
    r"[^\S\n]*(?:[.;,(]|$)",
    re.IGNORECASE | re.MULTILINE)


def verbal_confidence(generation: str) -> Optional[float]:
    """The confidence the generation states, in [0, 1], or None."""
    found = _CONFIDENCE.findall(strip_references(generation or ""))
    if not found:
        return None
    value, percent = found[-1]
    if not percent and "." in value and float(value) <= 1.0:
        return float(value)
    return max(0.0, min(100.0, float(value))) / 100.0
