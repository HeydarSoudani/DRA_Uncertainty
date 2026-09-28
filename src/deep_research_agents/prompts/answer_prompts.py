"""Answer prompts: instructions, format strings, and extraction utilities.

- FINAL_ANSWER_INSTRUCTION  -- loop-end: agent must produce a definitive answer.
- Per-agent format instructions (TAG, OSS, REACT, BOXED, DRTULU, SELFASK, ...).
- AnswerCandidateOutput dataclass + extraction helpers, shared by the final
  answer evaluation and the uncertainty estimator's intermediate answers.

The intermediate-answer instruction lives in ``uncertainty_estimator.prompts``.
"""

import json
import logging
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

# ======================================================================
# Instructions
# ======================================================================

FINAL_ANSWER_INSTRUCTION = (
    "You have now reached the maximum context length you can handle. "
    "You should stop making tool calls and, based on all the information "
    "above, think again and provide what you consider the most likely answer."
)

TONGYI_FORCE_ANSWER = (
    "You have now reached the maximum context length you can handle. "
    "You should stop making tool calls and, based on all the information "
    "above, think again and provide what you consider the most likely answer "
    "in the following format:"
    "<think>your final thinking</think>\n"
    "<answer>your answer</answer>"
)

# ======================================================================
# Per-agent format instructions
# ======================================================================
TAG_FORMAT = (
    "Provide your answer in the following format:\n"
    "<think>your final thinking</think>\n"
    "<answer>your answer</answer>"
)

OSS_FORMAT = (
    "Your response should be in the following format:\n"
    "Explanation: {your explanation, citing evidence in [docid] brackets}\n"
    "Exact Answer: {your succinct, final answer}\n"
    "Confidence: {your confidence score between 0% and 100%}"
)

REACT_FORMAT = (
    "Provide your answer in the following format:\n"
    "<think>your final thinking</think>\n"
    '<action>finish(answer="your answer")</action>'
)

WEBWEAVER_FORMAT = (
    "Based on the evidence gathered so far, provide a concise answer.\n"
    "IMPORTANT: The <answer> tag must contain ONLY the short factual answer,no explanations, no reasoning, no bullet points. Put all reasoning inside <think>.\n\n"
    "<think>your reasoning over the collected evidence</think>\n"
    "<answer>your short answer</answer>"
)

CPM_EXPLORE_FORMAT = (
    "## How to Respond\n\n"
    "1. **Think**: Provide a SHORT and CONCISE final thinking.\n"
    "2. **Answer**: Provide your SHORT most-likely answer or 'no candidate' inside <answer></answer> tags."
)

BOXED_FORMAT = "Provide your final answer in \\boxed{} format."

DRTULU_FORMAT = "Provide your answer using <answer>...</answer> tags."

SELFASK_FORMAT = "So the final answer is: "

# ======================================================================
# Structured output + extraction
# ======================================================================

@dataclass
class AnswerCandidateOutput:
    """Structured output from the answer-candidate LLM call."""
    candidate: str
    reasoning: str = ""
    confidence: Optional[float] = None

def _parse_candidate_list(text: str) -> List[str]:
    """Split a candidate value into individual candidates.

    Handles three forms:
    - ``"no candidate"`` -> empty list
    - ``"[c1, c2, c3]"`` -> ``["c1", "c2", "c3"]``
    - ``"single answer"`` -> ``["single answer"]``
    """
    stripped = text.strip()
    if stripped.lower() == "no candidate":
        return []
    if stripped.startswith("[") and stripped.endswith("]"):
        try:
            parsed = json.loads(stripped)
            if isinstance(parsed, list):
                return [str(c).strip() for c in parsed if str(c).strip()]
        except (json.JSONDecodeError, ValueError):
            pass
        inner = stripped[1:-1]
        return [c.strip().strip('"').strip("'") for c in inner.split(",") if c.strip()]
    return [stripped]

_AGENT_TO_PATTERNS = {
    "searcho1":    ["boxed"],
    "react":       ["finish_action"],
    "drtulu":      ["answer_tag"],
    "selfask":     ["so_final"],
    "oss":         ["exact_answer"],
    "glm":         ["exact_answer"],
    "cpm_explore": ["answer_tag"],
    "tongyi":      ["answer_tag"],
    "webweaver":   ["answer_tag"],
}

_ALL_PATTERN_NAMES = ["answer_tag", "finish_action", "exact_answer", "boxed", "so_final"]


def extract_answer_candidates(
    raw: str,
    expected_format: Optional[str] = None,
) -> Tuple[List[AnswerCandidateOutput], bool]:
    """Parse answer candidates and optional ``<think>`` tags from raw LLM output.

    Returns ``(candidates, format_matched)`` where *format_matched* is
    ``True`` when at least one known format pattern was found in the text
    (even if the value was ``"no candidate"``).

    When *expected_format* is an agent name (e.g. ``"react"``, ``"oss"``),
    the expected pattern is tried first; remaining patterns run only as
    fallback.  When ``None``, all patterns run (backward-compatible).
    """
    thinking = ""
    think_m = re.search(r"<think(?:ing)?>(.*?)(?:</think(?:ing)?>|$)", raw, re.DOTALL)
    if think_m:
        thinking = think_m.group(1).strip()

    if not thinking:
        # Matches both the OSS/GLM colon form ("Explanation: …") and the
        # markdown-header form ("## Explanation with Citations\n…"),
        # stopping at the next header / Exact Answer / Confidence section.
        expl_m = re.search(
            r"#{0,3}\s*Explanation(?:\s+with\s+Citations)?\s*:?[ \t]*\n*(.+?)"
            r"(?=\n[ \t]*#{1,3}\s*Exact Answer|\n[ \t]*(?:Exact Answer|Confidence):|$)",
            raw, re.DOTALL,
        )
        if expl_m:
            _expl = expl_m.group(1).strip()
            if not re.search(r"\{your\b", _expl, re.IGNORECASE):
                thinking = _expl

    confidence: Optional[float] = None
    conf_m = re.search(r"Confidence:\s*(\d+(?:\.\d+)?)\s*%", raw)
    if conf_m:
        confidence = max(0.0, min(100.0, float(conf_m.group(1))))

    candidates: List[AnswerCandidateOutput] = []
    seen: set = set()
    format_matched = False

    def _is_template_placeholder(text: str) -> bool:
        # "{your succinct, final answer}" is the literal template; a bare "..."
        # is how a model abbreviates it when restating the format in prose.
        return bool(re.search(r"\{your\b", text, re.IGNORECASE)) or \
            text.strip(" .\u2026") == ""

    def _span_after(text: str, start: int) -> str:
        """The old multi-line span, for a format whose answer really does wrap."""
        m = re.match(
            r"#{0,3}\s*Exact Answer\s*:?[ \t]*\n*[ \t]*(.+?)"
            r"(?:\n[ \t]*#{1,3}|\n[ \t]*Confidence:|\n[ \t]*\n|$)",
            text[start:], re.DOTALL,
        )
        return m.group(1) if m else ""

    def _add(text: str) -> None:
        normed = text.lower()
        if normed and normed != "no candidate" and normed not in seen:
            if _is_template_placeholder(text):
                return
            seen.add(normed)
            candidates.append(AnswerCandidateOutput(
                candidate=text, reasoning=thinking, confidence=confidence,
            ))

    def _add_raw(value: str) -> None:
        for c in _parse_candidate_list(value):
            _add(c)

    def _try_answer_tag() -> bool:
        m = re.search(r"<answer>(.*?)</answer>", raw, re.DOTALL)
        if m:
            _add_raw(m.group(1).strip())
            return True
        return False

    def _try_finish_action() -> bool:
        m = re.search(r'finish\(answer="(.*?)"\)', raw, re.DOTALL)
        if m:
            _add_raw(m.group(1).strip())
            return True
        return False

    def _try_exact_answer() -> bool:
        # Handles the OSS/GLM colon form ("Exact Answer: …") and the
        # markdown-header form ("## Exact Answer\n…").
        #
        # Read from the LAST occurrence backwards, one line at a time.  Both
        # halves matter.  A model that restates the output format before using
        # it ("then Explanation: ... Exact Answer: ...") puts a first occurrence
        # on the page that is not the answer, and `re.search` would anchor
        # there.  And the answer is one line by construction -- a bare number, a
        # name, or one sentence -- so a span that runs to the next blank line
        # swallows any reasoning the model wrote after it.  Together those two
        # scored a run that answered 1886 correctly as having predicted twenty
        # lines of its own deliberation.
        #
        # The multi-line span is kept as a fallback for a format that really
        # does wrap, and earlier occurrences are tried when the last one turns
        # out to be the template rather than an answer.
        matches = list(re.finditer(
            r"#{0,3}[ \t]*Exact Answer[ \t]*:?[ \t]*\n*[ \t]*(?P<line>.*)",
            raw,
        ))
        if not matches:
            return False

        for m in reversed(matches):
            for candidate in (m.group("line"),
                              _span_after(raw, m.start())):
                val = re.sub(r"\s*Confidence:\s*\d+(?:\.\d+)?\s*%\s*$", "",
                             (candidate or "").strip())
                if val and not _is_template_placeholder(val):
                    _add_raw(val)
                    return True
        # Every occurrence was the template itself: the format matched even
        # though nothing usable came out of it, which is what the caller's
        # ``format_matched`` is for.
        return True

    def _try_boxed() -> bool:
        m = re.search(r"\\boxed\{(.*?)}", raw, re.DOTALL)
        if m:
            _add_raw(m.group(1).strip())
            return True
        return False

    def _try_so_final() -> bool:
        m = re.search(r"(?:So the candidate answer is|So the final answer is):\s*(.+?)(?:\n\n|$)", raw, re.DOTALL)
        if m:
            _add_raw(m.group(1).strip())
            return True
        return False

    _PATTERN_FNS = {
        "answer_tag": _try_answer_tag,
        "finish_action": _try_finish_action,
        "exact_answer": _try_exact_answer,
        "boxed": _try_boxed,
        "so_final": _try_so_final,
    }

    primary = _AGENT_TO_PATTERNS.get(expected_format, _ALL_PATTERN_NAMES)
    fallback = [p for p in _ALL_PATTERN_NAMES if p not in primary]

    for name in primary:
        if _PATTERN_FNS[name]():
            format_matched = True

    if not candidates:
        for name in fallback:
            if _PATTERN_FNS[name]():
                format_matched = True

    if format_matched and not candidates:
        candidates.append(AnswerCandidateOutput(
            candidate="no candidate",
            reasoning=thinking or "format matched but no usable candidate extracted",
            confidence=confidence,
        ))

    return candidates, format_matched
