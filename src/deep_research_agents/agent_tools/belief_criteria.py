"""Criteria progress for the belief agent: init once per query, update per search.

A copy of the controller's criteria-coverage signal
(``controller_component.signals.CriteriaCoverageSignal`` and the helpers in
``controller_component.prompts.criteria_coverage``), kept here so the belief
agent does not depend on the controller.  The update logic is unchanged.
What differs:

- the LLM is reached through a ``complete(messages) -> str`` callable, so the
  agent decides the model, temperature and reasoning switch;
- ``update`` takes the passages the policy was shown, with the snippet count
  and length set by the caller (the controller uses 10 x 200 chars);
- failed calls and unparsable outputs are recorded in ``errors`` and on the
  returned summary, not only logged;
- JSON with prose around it and no code fence is still parsed (the outermost
  ``{...}`` is taken);
- the ``critical_gaps`` / ``minor_gaps`` the prompts ask for are ignored:
  nothing reads them, and the statuses already carry the same information.

Statuses are ``not_covered``, ``partial`` and ``covered``.
"""

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from utils.text_utils import doc_text, doc_title
from deep_research_agents.prompts.belief.criteria import (
    CRITERIA_INIT_DYNAMIC_SYSTEM,
    CRITERIA_INIT_DYNAMIC_USER_TEMPLATE,
    CRITERIA_INIT_STATIC_SYSTEM,
    CRITERIA_INIT_STATIC_USER_TEMPLATE,
    CRITERIA_UPDATE_DYNAMIC_SYSTEM,
    CRITERIA_UPDATE_DYNAMIC_USER_TEMPLATE,
    CRITERIA_UPDATE_STATIC_SYSTEM,
    CRITERIA_UPDATE_STATIC_USER_TEMPLATE,
    FROZEN_INSTRUCTION,
    UNFROZEN_INSTRUCTION,
)

logger = logging.getLogger(__name__)

MODES = ("static", "dynamic")
VALID_STATUSES = {"not_covered", "partial", "covered"}
_MAX_EVIDENCE_LENGTH = 500


# ── Dataclasses ───────────────────────────────────────────────────────────────

@dataclass
class Criterion:
    """A single information-need criterion of a query."""
    name: str
    status: str = "not_covered"  # "not_covered" | "partial" | "covered"
    evidence: str = ""

    def to_dict(self) -> Dict[str, str]:
        return {"name": self.name, "status": self.status, "evidence": self.evidence}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Criterion":
        status = d.get("status", "not_covered")
        if status not in VALID_STATUSES:
            status = "not_covered"
        evidence = d.get("evidence", "")
        if len(evidence) > _MAX_EVIDENCE_LENGTH:
            evidence = evidence[:_MAX_EVIDENCE_LENGTH] + "..."
        return cls(name=d.get("name", ""), status=status, evidence=evidence)


@dataclass
class CriteriaCoverageSummary:
    """Structured output from one criteria-coverage evaluation step."""
    criteria: List[Criterion] = field(default_factory=list)
    num_covered: int = 0
    num_partial: int = 0
    num_not_covered: int = 0
    total: int = 0
    new_criteria_this_iter: List[str] = field(default_factory=list)
    removed_criteria_this_iter: List[str] = field(default_factory=list)
    changed_criteria_this_iter: List[str] = field(default_factory=list)
    stable_since: Optional[int] = None
    frozen: bool = False
    reasoning: str = ""
    raw: str = ""
    error: Optional[str] = None  # set when this step's LLM call or parse failed

    def to_dict(self) -> Dict[str, Any]:
        return {
            "criteria": [a.to_dict() for a in self.criteria],
            "num_covered": self.num_covered,
            "num_partial": self.num_partial,
            "num_not_covered": self.num_not_covered,
            "total": self.total,
            "new_criteria_this_iter": self.new_criteria_this_iter,
            "removed_criteria_this_iter": self.removed_criteria_this_iter,
            "changed_criteria_this_iter": self.changed_criteria_this_iter,
            "stable_since": self.stable_since,
            "frozen": self.frozen,
            "reasoning": self.reasoning,
            "raw": self.raw,
            "error": self.error,
        }


@dataclass
class CriterionAction:
    """A single action returned by the LLM for the delta-based update."""
    action: str  # "tick" | "add" | "remove"
    name: str
    status: str = ""
    evidence: str = ""

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> Optional["CriterionAction"]:
        action = d.get("action", "").strip().lower()
        name = d.get("name", "").strip()
        if not action or not name:
            return None
        if action not in ("tick", "add", "remove"):
            return None
        return cls(
            action=action,
            name=name,
            status=d.get("status", "not_covered"),
            evidence=d.get("evidence", ""),
        )


@dataclass
class CriterionActionResult:
    """Parsed output from a delta-based criteria coverage update."""
    reasoning: str = ""
    actions: List[CriterionAction] = field(default_factory=list)


# ── Extraction ────────────────────────────────────────────────────────────────

def _json_body(raw: str) -> str:
    """The JSON object in *raw*: a fenced block, else the outermost ``{...}``."""
    m = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", raw, re.DOTALL)
    if m:
        return m.group(1).strip()
    start, end = raw.find("{"), raw.rfind("}")
    if start != -1 and end > start:
        return raw[start:end + 1]
    return raw.strip()


def extract_criteria_coverage(raw: str) -> Optional[CriteriaCoverageSummary]:
    """Parse a full criterion list from LLM output (used for initialization).

    Expects a raw JSON object with keys: reasoning, criteria.  Only the
    criteria and the reasoning are kept.

    Returns None if parsing fails entirely.
    """
    stripped = _json_body(raw)

    try:
        data = json.loads(stripped)
    except json.JSONDecodeError:
        logger.warning("BeliefCriteria: JSON parse failed for init output")
        return None

    if not isinstance(data, dict) or "criteria" not in data:
        logger.warning("BeliefCriteria: no 'criteria' key found in JSON")
        return None

    reasoning = data.get("reasoning", "")
    criteria_list = data.get("criteria", [])
    if not isinstance(criteria_list, list):
        return None

    criteria = []
    for item in criteria_list:
        if isinstance(item, dict) and item.get("name"):
            criteria.append(Criterion.from_dict(item))

    if not criteria:
        logger.warning("BeliefCriteria: parsed JSON but no valid criteria found")
        return None

    return CriteriaCoverageSummary(criteria=criteria, reasoning=reasoning)


def extract_criterion_actions(raw: str) -> Optional[CriterionActionResult]:
    """Parse a delta-based action list from LLM output (used for updates).

    Expects a raw JSON object with keys: reasoning, actions.

    Returns None if parsing fails entirely.
    """
    stripped = _json_body(raw)

    try:
        data = json.loads(stripped)
    except json.JSONDecodeError:
        logger.warning("BeliefCriteria: JSON parse failed for update output")
        return None

    if not isinstance(data, dict) or "actions" not in data:
        logger.warning("BeliefCriteria: no 'actions' key found in JSON")
        return None

    reasoning = data.get("reasoning", "")
    actions_list = data.get("actions", [])
    if not isinstance(actions_list, list):
        return None

    actions = []
    for item in actions_list:
        if isinstance(item, dict):
            action = CriterionAction.from_dict(item)
            if action is not None:
                actions.append(action)

    return CriterionActionResult(reasoning=reasoning, actions=actions)


def _resolve_criterion_name(name: str, criteria_by_name: Dict[str, Criterion]) -> Optional[str]:
    """Resolve *name* to an existing criterion key, case-insensitively."""
    if name in criteria_by_name:
        return name
    name_lower = name.strip().lower()
    for key in criteria_by_name:
        if key.strip().lower() == name_lower:
            return key
    return None


_STATUS_ALIASES = {
    "partially_covered": "partial",
    "partially covered": "partial",
    "mostly covered": "partial",
    "mostly_covered": "partial",
    "partly covered": "partial",
    "partly_covered": "partial",
    "incomplete": "partial",
    "fully_covered": "covered",
    "fully covered": "covered",
    "complete": "covered",
    "missing": "not_covered",
    "uncovered": "not_covered",
    "not covered": "not_covered",
    "none": "not_covered",
}


def _normalise_status(raw_status: str) -> str:
    """Map an LLM-returned status string to a canonical value."""
    normalised = (raw_status or "").strip().lower()
    if normalised in VALID_STATUSES:
        return normalised
    if normalised in _STATUS_ALIASES:
        return _STATUS_ALIASES[normalised]
    logger.warning("BeliefCriteria: unrecognized status '%s', defaulting to 'not_covered'", raw_status)
    return "not_covered"


def apply_criterion_actions(
    current_criteria: List[Criterion],
    action_result: CriterionActionResult,
    max_criteria: int,
    frozen: bool = False,
) -> tuple:
    """Apply parsed actions to the current criterion list.

    Returns (updated_criteria, added_names, removed_names).
    """
    criteria_by_name = {a.name: a for a in current_criteria}
    added = set()
    removed = set()

    for act in action_result.actions:
        status = _normalise_status(act.status)

        if act.action == "tick":
            resolved = _resolve_criterion_name(act.name, criteria_by_name)
            if resolved is not None:
                if status in VALID_STATUSES:
                    criteria_by_name[resolved].status = status
                if act.evidence:
                    criteria_by_name[resolved].evidence = act.evidence[:_MAX_EVIDENCE_LENGTH]
            else:
                logger.warning("BeliefCriteria: tick for unknown criterion '%s', skipping", act.name)

        elif act.action == "add":
            if frozen:
                logger.info("BeliefCriteria: ignoring 'add' action while frozen")
                continue
            resolved = _resolve_criterion_name(act.name, criteria_by_name)
            if resolved is None and len(criteria_by_name) < max_criteria:
                add_status = status if status in VALID_STATUSES else "not_covered"
                criteria_by_name[act.name] = Criterion(
                    name=act.name, status=add_status, evidence=act.evidence[:_MAX_EVIDENCE_LENGTH])
                added.add(act.name)
            elif resolved is not None:
                if status in VALID_STATUSES:
                    criteria_by_name[resolved].status = status
                if act.evidence:
                    criteria_by_name[resolved].evidence = act.evidence[:_MAX_EVIDENCE_LENGTH]

        elif act.action == "remove":
            if frozen:
                logger.info("BeliefCriteria: ignoring 'remove' action while frozen")
                continue
            resolved = _resolve_criterion_name(act.name, criteria_by_name)
            if resolved is not None:
                del criteria_by_name[resolved]
                removed.add(resolved)

    return list(criteria_by_name.values()), added, removed


# ── Formatting ────────────────────────────────────────────────────────────────

def format_criteria_for_prompt(criteria: List[Criterion]) -> str:
    """Format the current criterion list for the updater's user prompt."""
    if not criteria:
        return "(no criteria yet)"
    lines = []
    for i, a in enumerate(criteria, 1):
        ev = f" | evidence: {a.evidence}" if a.evidence else ""
        lines.append(f"  {i}. {a.name} [{a.status}]{ev}")
    return "\n".join(lines)


def format_doc_snippets(docs: List[Dict[str, Any]], top_k: int = 10, max_text_length: int = 200) -> str:
    """Format retrieved docs as compact snippets for the updater's user prompt."""
    if not docs:
        return "(no documents retrieved)"
    parts = []
    for i, doc in enumerate(docs[:top_k], 1):
        title = doc_title(doc, default="")
        text = doc_text(doc, max_length=max_text_length)
        if title and text:
            parts.append(f"  [{i}] {title}: {text}")
        elif text:
            parts.append(f"  [{i}] {text}")
        elif title:
            parts.append(f"  [{i}] {title}")
    return "\n".join(parts) if parts else "(no documents retrieved)"


def format_summary_for_log(summary: CriteriaCoverageSummary) -> str:
    """Compact one-line summary for verbose logging."""
    return (
        f"criteria: {summary.num_covered}/{summary.total} covered, "
        f"{summary.num_partial}/{summary.total} partial, "
        f"{summary.num_not_covered}/{summary.total} not_covered"
        + (f" | new: [{', '.join(summary.new_criteria_this_iter)}]" if summary.new_criteria_this_iter else "")
        + (f" | removed: [{', '.join(summary.removed_criteria_this_iter)}]" if summary.removed_criteria_this_iter else "")
        + (" | FROZEN" if summary.frozen else "")
        + (f" | ERROR: {summary.error}" if summary.error else "")
    )


# ── Tracker ───────────────────────────────────────────────────────────────────

class CriteriaTracker:
    """Criteria list and statuses for one query, updated after every search.

    Args:
        complete: messages -> raw completion; raises on a failed call.
        mode: ``"static"`` (fixed list, tick only) or ``"dynamic"``.
        max_criteria: Soft cap on the number of criteria.
        stabilization_window: dynamic mode: after this many iterations with no
            add/remove, the list is frozen.
        evidence_top_k: passages per update shown to the updater.
        evidence_chars: characters of one passage shown to the updater.
    """

    MIN_CRITERIA = 2

    def __init__(self, complete: Callable[[List[Dict[str, str]]], str], mode: str = "static",
                 max_criteria: int = 8, stabilization_window: int = 15,
                 evidence_top_k: int = 5, evidence_chars: int = 1500) -> None:
        if mode not in MODES:
            raise ValueError(f"criteria mode must be one of {MODES}, got {mode!r}")
        self._complete = complete
        self.mode = mode
        self.max_criteria = max_criteria
        self.stabilization_window = stabilization_window
        self.evidence_top_k = evidence_top_k
        self.evidence_chars = evidence_chars

        self.criteria: List[Criterion] = []
        self.frozen: bool = mode == "static"
        self._last_structure_change_iter: int = 0
        self.errors: List[str] = []

    def reset(self) -> None:
        """Clear all state for a new query."""
        self.criteria = []
        self.frozen = self.mode == "static"
        self._last_structure_change_iter = 0
        self.errors = []

    # ------------------------------------------------------------------

    def initialize(self, query: str) -> CriteriaCoverageSummary:
        """Extract the query's criteria, all ``not_covered``."""
        fmt = dict(min_criteria=self.MIN_CRITERIA, max_criteria=self.max_criteria)
        if self.mode == "static":
            system = CRITERIA_INIT_STATIC_SYSTEM.format(**fmt)
            user = CRITERIA_INIT_STATIC_USER_TEMPLATE.format(query=query, **fmt)
        else:
            system = CRITERIA_INIT_DYNAMIC_SYSTEM.format(**fmt)
            user = CRITERIA_INIT_DYNAMIC_USER_TEMPLATE.format(query=query, **fmt)
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]

        try:
            raw = self._complete(messages) or ""
        except Exception as exc:
            return self._fail(f"init call failed: {exc}", iter_num=0)

        parsed = extract_criteria_coverage(raw)
        if parsed is None:
            return self._fail("init output could not be parsed", iter_num=0, raw=raw)

        self.criteria = parsed.criteria[:self.max_criteria]
        for a in self.criteria:
            a.status = "not_covered"
            a.evidence = ""
        summary = self._build_summary(iter_num=0)
        summary.reasoning = parsed.reasoning
        summary.raw = raw
        return summary

    def update(self, iter_num: int, docs: List[Dict[str, Any]], subqueries: List[str],
               query: str) -> CriteriaCoverageSummary:
        """Apply the updater's delta for one search; on failure the list is unchanged."""
        if not self.criteria:
            return self._build_summary(iter_num=iter_num)

        doc_snippets = format_doc_snippets(docs, top_k=self.evidence_top_k,
                                           max_text_length=self.evidence_chars)
        current = format_criteria_for_prompt(self.criteria)
        if self.mode == "static":
            system = CRITERIA_UPDATE_STATIC_SYSTEM.format(max_criteria=self.max_criteria)
            user = CRITERIA_UPDATE_STATIC_USER_TEMPLATE.format(
                query=query, current_criteria_formatted=current,
                subqueries="; ".join(subqueries), doc_snippets=doc_snippets)
        else:
            system = CRITERIA_UPDATE_DYNAMIC_SYSTEM.format(
                max_criteria=self.max_criteria,
                frozen_instruction=FROZEN_INSTRUCTION if self.frozen else UNFROZEN_INSTRUCTION)
            user = CRITERIA_UPDATE_DYNAMIC_USER_TEMPLATE.format(
                query=query, current_criteria_formatted=current,
                subqueries="; ".join(subqueries), doc_snippets=doc_snippets)
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]

        try:
            raw = self._complete(messages) or ""
        except Exception as exc:
            return self._fail(f"update call failed: {exc}", iter_num=iter_num)

        result = extract_criterion_actions(raw)
        if result is None:
            return self._fail("update output could not be parsed", iter_num=iter_num, raw=raw)

        before = {a.name: a.status for a in self.criteria}
        self.criteria, added, removed = apply_criterion_actions(
            current_criteria=self.criteria,
            action_result=result,
            max_criteria=self.max_criteria,
            frozen=self.frozen,
        )
        changed = [a.name for a in self.criteria
                   if a.name in before and before[a.name] != a.status]

        if added or removed:
            self._last_structure_change_iter = iter_num
        if (not self.frozen and self.mode == "dynamic" and iter_num > 0
                and iter_num - self._last_structure_change_iter >= self.stabilization_window):
            self.frozen = True

        summary = self._build_summary(iter_num=iter_num)
        summary.new_criteria_this_iter = sorted(added)
        summary.removed_criteria_this_iter = sorted(removed)
        summary.changed_criteria_this_iter = changed
        summary.reasoning = result.reasoning
        summary.raw = raw
        return summary

    # ------------------------------------------------------------------

    def _fail(self, message: str, iter_num: int, raw: str = "") -> CriteriaCoverageSummary:
        logger.warning("BeliefCriteria (iter %d): %s", iter_num, message)
        self.errors.append(f"iter {iter_num}: {message}")
        summary = self._build_summary(iter_num=iter_num)
        summary.error = message
        summary.raw = raw
        return summary

    def _build_summary(self, iter_num: int) -> CriteriaCoverageSummary:
        stable_since = None
        if self.frozen and self.mode == "dynamic":
            stable_since = self._last_structure_change_iter
        return CriteriaCoverageSummary(
            criteria=[Criterion(name=a.name, status=a.status, evidence=a.evidence) for a in self.criteria],
            num_covered=sum(1 for a in self.criteria if a.status == "covered"),
            num_partial=sum(1 for a in self.criteria if a.status == "partial"),
            num_not_covered=sum(1 for a in self.criteria if a.status == "not_covered"),
            total=len(self.criteria),
            stable_since=stable_since,
            frozen=self.frozen,
        )
