"""Criteria reference of the uncertainty estimator.

The report (Section "Instantiation") replaces the hidden state with a fixed
per-query criteria list C = {c_1, ..., c_K} and tracks a criteria state
sigma_t(k) in {uncovered, partially_covered, fully_covered}, updated by
documents only (it can move up or down).

- ``CriteriaSource``: produces C once, at the start of each sample.
  ``LLMCriteriaSource`` derives the criteria from the query.
  TODO(criteria-file): a source that reads C from a file, keyed by query id.
- ``CriteriaState``: sigma_t and the evidence attached to each criterion.

Each criterion is closed (one fact) or open (several parts or answers that
documents establish only together); an open criterion is fully covered only
once ``OPEN_MIN_SOURCES`` distinct documents support it.  The kind follows
from the dataset's query shape and the criterion's position (``_kinds``);
the extractor only lists the criteria.

The judges that update the state and score queries against C are in
``judges``.
"""

import json
import logging
import re
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Set, Tuple

from utils.text_utils import parse_json_object
from .prompts import CRITERIA_INIT_SYSTEMS, CRITERIA_INIT_USER_TEMPLATE
from .types import (
    CLOSED, FULLY_COVERED, MULTI_ASPECT, OPEN, PARTIALLY_COVERED, QUERY_SHAPES, SET, STATUS_VALUE,
    STATUSES, UNCOVERED,
    Criterion, CriterionUpdate, Evidence,
)

logger = logging.getLogger(__name__)

# Distinct supporting documents an open criterion needs to be fully covered;
# with fewer it stays partially covered.
OPEN_MIN_SOURCES = 3

# One complete string item of the criteria list, for output that is not
# valid JSON (cut at the token limit, or a stray character between items).
_ITEM_RE = re.compile(r'"((?:[^"\\]|\\.)*)"\s*(?=[,\]])')


def _kinds(shape: str, n: int) -> List[str]:
    """Kind of each of *n* criteria of a *shape* query: every clue of a single
    target is closed; the first criterion of a set query (the complete set)
    is open and its members are closed; every report criterion is open."""
    if shape == MULTI_ASPECT:
        return [OPEN] * n
    if shape == SET:
        return [OPEN] + [CLOSED] * (n - 1) if n else []
    return [CLOSED] * n


# ---------------------------------------------------------------------------
# Criteria source
# ---------------------------------------------------------------------------

class CriteriaSource(ABC):
    """Produces the fixed criteria list of one query."""

    name: str = ""

    @abstractmethod
    def get(self, query_id: Optional[str], query: str) -> Tuple[List[Criterion], Dict[str, Any]]:
        """Return ``(criteria, info)``; *info* is saved in the meta line.

        *info* carries at least ``errors`` (a list of strings); an empty
        criteria list with a non-empty ``errors`` means the source failed.
        """


class LLMCriteriaSource(CriteriaSource):
    """Derives the criteria of the query with one LLM call.

    The dataset's query shape (single target, set or multi-aspect) picks the
    prompt and the kind of each criterion; the model only lists the
    criteria, as strings.  The shape is saved in *info* as ``query_shape``.

    Args:
        llm_client: Object with ``complete(messages, **kwargs) -> str``.
        max_criteria: Cap on the number of criteria kept (the dataset's
            ``max_criteria`` in layout.DATASET_SPECS).
        model_name: Saved in the meta line.
        max_tokens: Max tokens for the LLM call.
        temperature: LLM temperature.
        query_shape: ``"single_target"``, ``"set"`` or ``"multi_aspect"``
            (the dataset's ``query_shape`` in layout.DATASET_SPECS).
        system_prompt: Replaces the criteria-extraction system prompt (for
            offline prompt comparisons, ``python -m evaluation.criteria``).
    """

    name = "llm"

    def __init__(
        self,
        llm_client: Any,
        max_criteria: int,
        model_name: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.0,
        query_shape: str = "single_target",
        system_prompt: Optional[str] = None,
    ) -> None:
        if query_shape not in QUERY_SHAPES:
            raise ValueError(f"unknown query shape {query_shape!r}; expected one of {QUERY_SHAPES}")
        self._shape = query_shape
        self._system = system_prompt or CRITERIA_INIT_SYSTEMS[query_shape]
        self._llm = llm_client
        self.model_name = model_name
        self._max_criteria = max_criteria
        self._max_tokens = max_tokens
        self._temperature = temperature

    def get(self, query_id: Optional[str], query: str) -> Tuple[List[Criterion], Dict[str, Any]]:
        info: Dict[str, Any] = {"model": self.model_name, "query_shape": self._shape, "errors": []}
        messages = [
            {"role": "system", "content": self._system},
            {"role": "user", "content": CRITERIA_INIT_USER_TEMPLATE.format(
                query=query, max_criteria=self._max_criteria,
            )},
        ]
        try:
            raw = self._llm.complete(messages, max_tokens=self._max_tokens, temperature=self._temperature)
        except Exception as e:
            logger.warning("LLMCriteriaSource: LLM call failed", exc_info=True)
            info["errors"].append(f"llm_call: {e}")
            return [], info

        data = parse_json_object(raw or "")
        if data is None or not isinstance(data.get("criteria"), list):
            # Keep the complete items of malformed or truncated output.
            body = (raw or "").partition('"criteria"')[2].partition("[")[2]
            salvaged = [json.loads(f'"{t}"') for t in _ITEM_RE.findall(body)]
            if not salvaged:
                info["errors"].append("parse: no JSON object with a 'criteria' list")
                info["raw"] = raw
                return [], info
            info["warnings"] = [f"parse: invalid JSON, kept {len(salvaged)} complete items"]
            data = {"criteria": salvaged}

        texts: List[str] = []
        for item in data["criteria"]:
            if isinstance(item, dict):
                item = item.get("text") or ""
            item = str(item).strip()
            if item and item not in texts:
                texts.append(item)
        if not texts:
            info["errors"].append("parse: empty criteria list")
            info["raw"] = raw

        texts = texts[:self._max_criteria]
        criteria = [
            Criterion(id=f"c{k + 1}", text=t, kind=kind)
            for k, (t, kind) in enumerate(zip(texts, _kinds(self._shape, len(texts))))
        ]
        return criteria, info


# ---------------------------------------------------------------------------
# Criteria state
# ---------------------------------------------------------------------------

class CriteriaState:
    """Criteria state sigma_t with the evidence attached to each criterion.

    sigma_0 is all uncovered.  Each step, the coverage judge proposes
    updates from the step's novel documents and ``apply`` enforces:

    - a raise needs at least one supporting passage;
    - an open criterion is fully covered only once its supporting passages
      come from ``OPEN_MIN_SOURCES`` distinct documents; before that it is
      capped at partially covered;
    - a lowering needs at least one contradicting passage and moves one
      level at most per step;
    - an update that keeps the status only attaches its evidence.

    The cited passages are attached to the criterion (role ``support`` or
    ``contradict``) and shown to the judge in later steps, with what a
    partially covered criterion still lacks (``missing``).
    """

    def __init__(self, criteria: List[Criterion]) -> None:
        self.criteria = list(criteria)
        self.statuses: List[str] = [UNCOVERED] * len(self.criteria)
        self._evidence: List[List[Evidence]] = [[] for _ in self.criteria]
        self.missing: List[str] = [""] * len(self.criteria)
        self._index = {c.id: k for k, c in enumerate(self.criteria)}

    def snapshot(self) -> List[str]:
        return list(self.statuses)

    def attached(self, k: int, limit: Optional[int] = None) -> List[Evidence]:
        """Evidence of criterion *k*, oldest first; the last *limit* when set."""
        items = self._evidence[k]
        return items[-limit:] if limit else list(items)

    def apply(self, updates: List[CriterionUpdate]) -> List[Dict[str, Any]]:
        """Apply the judge's updates of one step.

        Returns one record per update: ``{id, from, to, proposed, support,
        contradict, reason, missing, applied, note}``; ``applied`` is False for an
        update that was rejected (``note`` says why).  Only the first update
        of a criterion is used.
        """
        records: List[Dict[str, Any]] = []
        done: Set[int] = set()
        for u in updates:
            k = self._index.get(u.id)
            record = {
                "id": u.id,
                "from": self.statuses[k] if k is not None else None,
                "to": None,
                "proposed": u.status,
                "support": [e.to_dict() for e in u.support],
                "contradict": [e.to_dict() for e in u.contradict],
                "reason": u.reason,
                "missing": u.missing,
                "applied": False,
                "note": "",
            }
            records.append(record)
            if k is None:
                record["note"] = "unknown criterion id"
                continue
            if k in done:
                record["note"] = "duplicate update"
                continue
            if u.status not in STATUSES:
                record["note"] = "unknown status"
                continue
            done.add(k)

            current, proposed = STATUS_VALUE[self.statuses[k]], STATUS_VALUE[u.status]
            if proposed > current and not u.support:
                record["note"] = "raise without supporting passage"
                continue
            if proposed < current and not u.contradict:
                record["note"] = "lowering without contradicting passage"
                continue
            if proposed == current and not (u.support or u.contradict):
                record["note"] = "no cited passage"
                continue
            if proposed < current - 1:
                proposed = current - 1
                record["note"] = "lowered by one level at most"
            self._evidence[k].extend(u.support + u.contradict)
            missing = u.missing
            if self.criteria[k].kind == OPEN and STATUSES[proposed] == FULLY_COVERED:
                sources = self.num_sources(k)
                if sources < OPEN_MIN_SOURCES:
                    proposed = STATUS_VALUE[PARTIALLY_COVERED]
                    record["note"] = f"open criterion needs {OPEN_MIN_SOURCES} sources, has {sources}"
                    missing = missing or (
                        f"support from more documents ({sources} of {OPEN_MIN_SOURCES} so far)"
                    )
            self.statuses[k] = STATUSES[proposed]
            self.missing[k] = missing if self.statuses[k] == PARTIALLY_COVERED else ""
            record["to"] = self.statuses[k]
            record["applied"] = True
        return records

    def num_sources(self, k: int) -> int:
        """Distinct documents among the supporting evidence of criterion *k*."""
        return len({e.doc_id for e in self._evidence[k] if e.role == "support"})

    def evidence(self) -> List[Dict[str, Any]]:
        """Per criterion, its kind, status, what it still lacks and its
        attached evidence (without text)."""
        return [
            {"id": c.id, "kind": c.kind, "status": st, "missing": m, "evidence": [e.to_dict() for e in ev]}
            for c, st, m, ev in zip(self.criteria, self.statuses, self.missing, self._evidence)
        ]
