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

# Criterion kinds: a closed criterion is one fact that a single document can
# establish; an open one has several parts or answers that documents
# establish only together (``CriteriaState`` needs several sources for it),
# such as the complete set of a set query with the property of each member,
# listed or not; an aspect criterion is one
# aspect of a report topic, which has no complete list of parts, so it is
# fully covered once specific facts establish every part its wording names.
CLOSED = "closed"
OPEN = "open"
ASPECT = "aspect"
KINDS = (CLOSED, OPEN, ASPECT)
# Kinds that documents establish only together (``OPEN_MIN_SOURCES``).
MULTI_SOURCE_KINDS = (OPEN, ASPECT)

# Query shapes (layout.DATASET_SPECS): one entity or value described by clues,
# a value computed over every member of a set, or a report on several aspects
# of a topic.  The shape picks the criteria-extraction prompt and the kind of
# each criterion; the extractor never chooses either.
SINGLE_TARGET = "single_target"
SET = "set"
MULTI_ASPECT = "multi_aspect"
QUERY_SHAPES = (SINGLE_TARGET, SET, MULTI_ASPECT)


@dataclass
class Criterion:
    """One criterion c_k of the per-query criteria list."""
    id: str
    text: str
    kind: str = CLOSED

    def to_dict(self) -> Dict[str, str]:
        return {"id": self.id, "text": self.text, "kind": self.kind}


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
