"""Check a report request's criteria list against the request itself.

The nugget evaluation scores criteria against information found in the
collection; this check scores them against what the request states.  One
LLM call per query lists the request's constraints and, for each, the
criteria that carry it, then the criteria that ask for what the request
leaves out:

* ``requirement``: information the report must cover; carried by a
  criterion that asks for it.
* ``limit``: a time, place, group or other scope set on a requirement;
  carried by a criterion whose wording includes it.
* ``exclusion``: what the request leaves out; carried by a criterion that
  states it ("..., not ...").
* ``exception``: what the request lets back in ("unless ..."); carried by a
  criterion that asks for it.
* ``background``: a detail about the asker that limits what information
  applies (country, location, situation); carried like a limit.

Run the judge with its reasoning on (as the information matcher).
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

from utils.text_utils import parse_json_object

logger = logging.getLogger(__name__)

CONSTRAINT_TYPES = ("requirement", "limit", "exclusion", "exception", "background")

REQUEST_COVERAGE_SYSTEM = """\
You check a list of criteria against the report request it was extracted from. The criteria are the pieces of information a research agent will look for.

First list the constraints the request states, each as a short phrase in the request's own words, with its type:
- "requirement": information the report must cover.
- "limit": a time, place, group or other scope the request sets on a requirement.
- "exclusion": something the request says to leave out.
- "exception": something the request lets back in from what it leaves out ("unless ...", "except ...").
- "background": a detail about the asker, such as their country, location or situation, that limits what information applies.
Do not list who is asking or why, or instructions on the format or length of the report. List each constraint once.

For each constraint, list the criteria that carry it:
- A requirement or an exception is carried by a criterion that asks for it, in any wording.
- A limit or a background detail is carried by a criterion whose wording includes it.
- An exclusion is carried by a criterion that states it, for example "..., not ...".

Then list the criteria that ask for information the request leaves out. A criterion that states an exclusion ("..., not ...") does not ask for it.

Respond with ONLY a raw JSON object, no markdown, no code blocks:
{"constraints": [{"text": "...", "type": "requirement", "criteria": [criterion numbers]}], "violations": [{"criterion": criterion number, "reason": "..."}]}
"criteria" and "violations" may be empty."""

REQUEST_COVERAGE_USER_TEMPLATE = """\
Request: {query}

Criteria:
{criteria}"""


def _recall(constraints: List[Dict], types: Tuple[str, ...]) -> Optional[float]:
    units = [c for c in constraints if c["type"] in types]
    return round(sum(bool(c["criteria"]) for c in units) / len(units), 4) if units else None


class RequestCoverageJudge:
    """Scores a criteria list against the constraints of its request.

    Args:
        llm_client: Object with ``complete(messages, **kwargs) -> str``.
        max_tokens / temperature: LLM call settings.
    """

    def __init__(self, llm_client: Any, max_tokens: int = 16000, temperature: float = 0.0) -> None:
        self._llm = llm_client
        self._max_tokens = max_tokens
        self._temperature = temperature

    def check(self, query: str, criteria: List[Dict]) -> Tuple[Dict[str, Any], Optional[str]]:
        """``(record, error)``: the record has ``constraint_recall`` (every
        constraint), ``exclusion_recall``, ``num_constraints``,
        ``num_violations``, the ``constraints`` (``{text, type, criteria}``,
        criterion ids) and the ``violations`` (``{criterion, reason}``);
        the error is None on success (the record is empty on failure)."""
        if not criteria:
            return {}, None
        lines = "\n".join(f"{j + 1}. {c['text']}" for j, c in enumerate(criteria))
        messages = [
            {"role": "system", "content": REQUEST_COVERAGE_SYSTEM},
            {"role": "user", "content": REQUEST_COVERAGE_USER_TEMPLATE.format(query=query, criteria=lines)},
        ]
        try:
            raw = self._llm.complete(messages, max_tokens=self._max_tokens, temperature=self._temperature)
        except Exception as e:
            logger.warning("RequestCoverageJudge: LLM call failed", exc_info=True)
            return {}, f"llm_call: {e}"
        data = parse_json_object(raw or "")
        if data is None or not isinstance(data.get("constraints"), list):
            return {}, "parse: no JSON object with a 'constraints' list"

        def index(value) -> Optional[int]:
            try:
                j = int(value) - 1
            except (TypeError, ValueError):
                return None
            return j if 0 <= j < len(criteria) else None

        constraints = []
        for item in data["constraints"]:
            if not isinstance(item, dict) or item.get("type") not in CONSTRAINT_TYPES:
                continue
            carried = [index(v) for v in item.get("criteria") or []]
            constraints.append({
                "text": str(item.get("text", "")),
                "type": item["type"],
                "criteria": sorted({criteria[j]["id"] for j in carried if j is not None}),
            })
        violations = []
        for item in data.get("violations") or []:
            j = index(item.get("criterion")) if isinstance(item, dict) else None
            if j is not None:
                violations.append({"criterion": criteria[j]["id"], "reason": str(item.get("reason", ""))})
        return {
            "constraint_recall": _recall(constraints, CONSTRAINT_TYPES),
            "exclusion_recall": _recall(constraints, ("exclusion",)),
            "num_constraints": len(constraints),
            "num_violations": len({v["criterion"] for v in violations}),
            "constraints": constraints,
            "violations": violations,
        }, None
