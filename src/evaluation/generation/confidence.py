"""The agent's own (verbalized) confidence in its final answer.

The agents whose answer format asks for it (``OSS_FORMAT``: glm, oss) end
their answer with ``Confidence: <0-100>%``.  It is read from the saved
generation by a regex, never by the judge, and is the baseline of the
uncertainty evaluation.  The last confidence outside the References block
counts; markdown emphasis and an approximation sign such as "~85 %" are
accepted.  A value with "%" (or "/100") is a confidence wherever it stands:
after "Confidence:" on the same line or the next one, or on the line after a
"**Confidence**" heading without a colon, whatever text follows it
("Confidence: 0% - not found", "Confidence: 60%</think>").  A value without
"%" counts only after "Confidence:" on the same line and must end the value,
so "Confidence: 3 sources agree" is no confidence; such a decimal of at most
1 ("0.85") is read as a fraction, any other value as a percent, as the format
asks.  A generation without a confidence has none (None); the uncertainty
evaluation decides how to score it (``uncertainty/verbal.py``).
"""

import re
from typing import Optional

from .report_sentences import strip_references

# The label, then either a colon (the value on the same line or the next) or
# a line break (a heading without a colon), then the value; the rest of the
# line is looked at, not consumed, so the check below can read it.
_CONFIDENCE = re.compile(
    r"\bConfidence[*_]*"
    r"(?P<sep>[^\S\n]*:[^\S\n]*[*_]*[^\S\n]*\n?|[^\S\n]*\n)"
    r"[^\S\n]*[*_]*[^\S\n]*"
    r"(?:~|≈|about|approximately|approx\.?)?[^\S\n]*"
    r"(?P<value>\d+(?:\.\d+)?)[^\S\n]*(?P<percent>%|/[^\S\n]*100)?"
    r"(?=(?P<rest>[^\n]*))",
    re.IGNORECASE)

# What may follow a value without "%": emphasis, then the end of the line or
# a punctuation mark.
_PLAIN_VALUE_END = re.compile(r"[*_]*[^\S\n]*(?:[.;,(]|$)")


def _accepted(m: "re.Match") -> bool:
    """Whether the match *m* is a confidence (module docstring)."""
    if m.group("percent"):
        return True
    sep = m.group("sep")
    return ":" in sep and "\n" not in sep and bool(_PLAIN_VALUE_END.match(m.group("rest")))


def verbal_confidence(generation: str) -> Optional[float]:
    """The confidence the generation states, in [0, 1], or None."""
    found = [m for m in _CONFIDENCE.finditer(strip_references(generation or "")) if _accepted(m)]
    if not found:
        return None
    value, percent = found[-1].group("value"), found[-1].group("percent")
    if not percent and "." in value and float(value) <= 1.0:
        return float(value)
    return max(0.0, min(100.0, float(value))) / 100.0
