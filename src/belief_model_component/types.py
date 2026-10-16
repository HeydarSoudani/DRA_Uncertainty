"""Constants and dataclasses shared by the belief model."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# Belief state b_t(k): one status per criterion from the seven words of
# Agent-BRACE's verbal belief scale (arXiv 2605.11436), lowest level first.
# The paper ranks "unknown" lowest; here it is the neutral level 0 (b_0, no
# passage relates to the criterion).  The positive levels rate how much of the
# information the criterion asks for the passages found so far give; the
# negative levels rate evidence against it: "unlikely" (the passages point
# against it) and "doubtful" (a passage contradicts it).
BELIEF_STATUSES = (
    "doubtful",
    "unlikely",
    "unknown",
    "possible",
    "probable",
    "almost certain",
    "certain",
)
# Signed level of each status (-2 to 4), for Delta^D and analysis.
STATUS_LEVEL: Dict[str, int] = {
    status: k - BELIEF_STATUSES.index("unknown") for k, status in enumerate(BELIEF_STATUSES)
}
# b_0: no passage found yet.
INITIAL_STATUS = "unknown"
# The paper's "confirmed or certain": "confirmed" read as "certain".
_STATUS_SYNONYMS = {"confirmed": "certain"}


def parse_status(text: Any) -> Optional[str]:
    """The belief status *text* names (case, underscores, hyphens and extra
    spaces ignored; "confirmed" accepted for "certain"), or None."""
    key = " ".join(str(text or "").lower().replace("_", " ").replace("-", " ").split())
    return key if key in STATUS_LEVEL else _STATUS_SYNONYMS.get(key)


# Criterion kinds: a closed criterion is one fact that a single document can
# establish; an open one has several parts or answers that documents
# establish only together; the rest criterion of a set query ("any other
# member: <property>") stands for the members the extractor did not list; an
# aspect criterion is one aspect of a report topic, which has no complete
# list of parts.  The kind picks nothing in the belief update; it is saved
# with the criteria.
CLOSED = "closed"
OPEN = "open"
REST = "rest"
ASPECT = "aspect"

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
    """One passage attached to a criterion by the belief updater.

    ``role`` is ``support`` or ``contradict``.  ``spans`` are the sentences
    the updater cited from the passage; ``span_verified[i]`` says whether
    ``spans[i]`` occurs verbatim in the passage, and then ``spans[i]`` is the
    passage's own words, else the updater's text.  ``title`` is the title
    line of the passage.  Later prompts show the verified spans and the title.
    """
    doc_id: str
    step: int
    role: str
    spans: List[str]
    span_verified: List[bool]
    title: str

    @property
    def verified_spans(self) -> List[str]:
        return [s for s, ok in zip(self.spans, self.span_verified) if ok]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "doc_id": self.doc_id, "step": self.step, "role": self.role,
            "spans": list(self.spans), "span_verified": list(self.span_verified),
        }


@dataclass
class CriterionUpdate:
    """One criterion of the belief the belief updater wrote for one step.

    ``status`` is b_t(k) as written (None when it names no belief status);
    ``support`` and ``contradict`` are the step's passages cited for it.
    """
    id: str
    status: Optional[str]
    support: List[Evidence] = field(default_factory=list)
    contradict: List[Evidence] = field(default_factory=list)
    reason: str = ""
