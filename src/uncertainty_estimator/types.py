"""Constants and dataclasses shared by the uncertainty estimator."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


# Criteria state sigma_t(k) of the report (Section "Instantiation"):
# uncovered (0), partially covered (1), fully covered (2).  The same three
# labels are the per-(document, criterion) coverage a DocCriteriaJudge returns.
UNCOVERED = "uncovered"
PARTIALLY_COVERED = "partially_covered"
FULLY_COVERED = "fully_covered"

STATUS_VALUE: Dict[str, int] = {UNCOVERED: 0, PARTIALLY_COVERED: 1, FULLY_COVERED: 2}
STATUSES = tuple(STATUS_VALUE)

# Criteria that a query can still target: uncovered or partially covered.
OPEN_STATUSES = (UNCOVERED, PARTIALLY_COVERED)


@dataclass
class Criterion:
    """One criterion c_k of the per-query criteria list."""
    id: str
    text: str

    def to_dict(self) -> Dict[str, str]:
        return {"id": self.id, "text": self.text}


@dataclass
class DocJudgment:
    """Coverage of every criterion by one document.

    ``statuses[k]`` is in ``STATUSES``.  ``scores[k]`` is the NLI judge's
    entailment probability (kept so the thresholds can be re-tuned offline);
    ``evidence[k]`` is the LLM judge's supporting quote.
    """
    statuses: List[str]
    scores: Optional[List[float]] = None
    evidence: Optional[List[str]] = None

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"statuses": self.statuses}
        if self.scores is not None:
            out["scores"] = [round(s, 4) for s in self.scores]
        if self.evidence is not None:
            out["evidence"] = self.evidence
        return out
