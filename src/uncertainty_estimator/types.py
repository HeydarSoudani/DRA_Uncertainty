"""Constants and dataclasses shared by the uncertainty estimator."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# Criteria state sigma_t(k) of the report (Section "Instantiation"):
# uncovered (0), partially covered (1), fully covered (2).  The same three
# labels are the statuses the coverage judge proposes.
UNCOVERED = "uncovered"
PARTIALLY_COVERED = "partially_covered"
FULLY_COVERED = "fully_covered"

STATUS_VALUE: Dict[str, int] = {UNCOVERED: 0, PARTIALLY_COVERED: 1, FULLY_COVERED: 2}
STATUSES = tuple(STATUS_VALUE)


@dataclass
class Criterion:
    """One criterion c_k of the per-query criteria list."""
    id: str
    text: str

    def to_dict(self) -> Dict[str, str]:
        return {"id": self.id, "text": self.text}


@dataclass
class Evidence:
    """One passage attached to a criterion by the coverage judge.

    ``role`` is ``support`` or ``contradict``.  ``spans`` are the sentences
    the judge cited from the passage; ``span_verified[i]`` says whether
    ``spans[i]`` occurs verbatim in the passage, and then ``spans[i]`` is the
    passage's own words, else the judge's text.  ``text`` is the start of
    the passage.  Later prompts show the verified spans and the text.
    """
    doc_id: str
    step: int
    role: str
    spans: List[str]
    span_verified: List[bool]
    text: str

    @property
    def verified_spans(self) -> List[str]:
        return [s for s, ok in zip(self.spans, self.span_verified) if ok]

    def to_dict(self, with_text: bool = False) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "doc_id": self.doc_id, "step": self.step, "role": self.role,
            "spans": list(self.spans), "span_verified": list(self.span_verified),
        }
        if with_text:
            out["text"] = self.text
        return out


@dataclass
class CriterionUpdate:
    """One criterion update proposed by the coverage judge for one step.

    ``status`` is the proposed sigma_t(k); ``support`` and ``contradict`` are
    the new passages cited for it; ``missing`` is what a partially covered
    criterion still lacks.
    """
    id: str
    status: str
    support: List[Evidence] = field(default_factory=list)
    contradict: List[Evidence] = field(default_factory=list)
    reason: str = ""
    missing: str = ""
