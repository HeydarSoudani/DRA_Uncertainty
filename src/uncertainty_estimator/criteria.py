"""Criteria reference of the uncertainty estimator.

The report (Section "Instantiation") replaces the hidden state with a fixed
per-query criteria list C = {c_1, ..., c_K} and tracks a criteria state
sigma_t(k) in {uncovered, partially_covered, fully_covered}, updated by
documents only.

- ``CriteriaSource``: produces C once, at the start of each sample.
  ``LLMCriteriaSource`` extracts the criteria stated in the query.
  TODO(criteria-file): a source that reads C from a file, keyed by query id.
- ``CriteriaState``: sigma_t, accumulated over all novel documents so far.

The judges that label documents and queries against C are in ``judges``.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Set, Tuple

from ._helpers import parse_json_object
from .prompts import CRITERIA_INIT_SYSTEM, CRITERIA_INIT_USER_TEMPLATE
from .types import FULLY_COVERED, STATUS_VALUE, STATUSES, UNCOVERED, Criterion

logger = logging.getLogger(__name__)


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
    """Extracts the criteria stated in the query with one LLM call.

    Args:
        llm_client: Object with ``complete(messages, **kwargs) -> str``.
        model_name: Saved in the meta line.
        max_criteria: Cap on the number of criteria kept.
        max_tokens: Max tokens for the LLM call.
        temperature: LLM temperature.
    """

    name = "llm"

    def __init__(
        self,
        llm_client: Any,
        model_name: Optional[str] = None,
        max_criteria: int = 8,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> None:
        self._llm = llm_client
        self.model_name = model_name
        self._max_criteria = max_criteria
        self._max_tokens = max_tokens
        self._temperature = temperature

    def get(self, query_id: Optional[str], query: str) -> Tuple[List[Criterion], Dict[str, Any]]:
        info: Dict[str, Any] = {"model": self.model_name, "reasoning": "", "errors": []}
        messages = [
            {"role": "system", "content": CRITERIA_INIT_SYSTEM},
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
            info["errors"].append("parse: no JSON object with a 'criteria' list")
            info["raw"] = raw
            return [], info

        texts: List[str] = []
        for item in data["criteria"]:
            if isinstance(item, dict):
                item = item.get("text") or item.get("name") or ""
            item = str(item).strip()
            if item and item not in texts:
                texts.append(item)
        if not texts:
            info["errors"].append("parse: empty criteria list")
            info["raw"] = raw
        info["reasoning"] = str(data.get("reasoning", ""))

        criteria = [Criterion(id=f"c{k + 1}", text=t) for k, t in enumerate(texts[:self._max_criteria])]
        return criteria, info


# ---------------------------------------------------------------------------
# Criteria state
# ---------------------------------------------------------------------------

class CriteriaState:
    """Criteria state sigma_t, accumulated over all novel documents so far.

    sigma_t(k) is the highest coverage any document gave c_k so far, so the
    state never moves down; several partial documents do not add up to full
    coverage.  Each criterion also keeps the ids of the documents that
    partially and fully cover it.  sigma_0 is all uncovered.
    """

    def __init__(self, criteria: List[Criterion]) -> None:
        self.criteria = list(criteria)
        self.statuses: List[str] = [UNCOVERED] * len(self.criteria)
        self._partial: List[Set[str]] = [set() for _ in self.criteria]
        self._full: List[Set[str]] = [set() for _ in self.criteria]

    def snapshot(self) -> List[str]:
        return list(self.statuses)

    def apply(self, doc_id: str, statuses: List[str]) -> None:
        """Add one document's coverage of each criterion."""
        if len(statuses) != len(self.criteria):
            raise ValueError(f"expected {len(self.criteria)} statuses, got {len(statuses)}")
        for k, status in enumerate(statuses):
            if status not in STATUSES:
                raise ValueError(f"unknown coverage status {status!r}")
            if status == UNCOVERED:
                continue
            (self._full if status == FULLY_COVERED else self._partial)[k].add(doc_id)
            if STATUS_VALUE[status] > STATUS_VALUE[self.statuses[k]]:
                self.statuses[k] = status

    def evidence(self) -> List[Dict[str, Any]]:
        """Per-criterion ids of the partially and fully covering documents."""
        return [
            {"id": c.id, "partially_covered": sorted(p), "fully_covered": sorted(f)}
            for c, p, f in zip(self.criteria, self._partial, self._full)
        ]
