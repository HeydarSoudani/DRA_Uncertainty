"""Constants and dataclasses shared by the belief model."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# Belief state b_t(k): one label per criterion from the seven words of
# estimative probability of ICD 203 (US Intelligence Community Directive 203,
# "Analytic Standards"), lowest first.  A label rates how likely it is that
# the passages found so far give the information the criterion asks for.
WEP_LABELS = (
    "almost no chance",
    "very unlikely",
    "unlikely",
    "roughly even chance",
    "likely",
    "very likely",
    "almost certain",
)
# Ordinal level of each label (0 to 6), for Delta^D.
WEP_LEVEL: Dict[str, int] = {label: k for k, label in enumerate(WEP_LABELS)}
# Midpoint of each label's ICD 203 probability band (1-5%, 5-20%, 20-45%,
# 45-55%, 55-80%, 80-95%, 95-99%), for analysis only; never shown to a model.
WEP_PROBABILITY: Dict[str, float] = dict(zip(WEP_LABELS, (0.03, 0.125, 0.325, 0.5, 0.675, 0.875, 0.97)))
# b_0: no passage found yet.
INITIAL_LABEL = WEP_LABELS[0]
# ICD 203's alternative terms, read as the label of the same band.
_WEP_SYNONYMS = {
    "remote": "almost no chance",
    "highly improbable": "very unlikely",
    "improbable": "unlikely",
    "roughly even odds": "roughly even chance",
    "probable": "likely",
    "highly probable": "very likely",
    "nearly certain": "almost certain",
}


def parse_label(text: Any) -> Optional[str]:
    """The WEP label *text* names (case, underscores, hyphens and extra
    spaces ignored; ICD 203's alternative terms accepted), or None."""
    key = " ".join(str(text or "").lower().replace("_", " ").replace("-", " ").split())
    return key if key in WEP_LEVEL else _WEP_SYNONYMS.get(key)


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

    ``label`` is b_t(k) as written (None when it names no WEP label);
    ``support`` and ``contradict`` are the step's passages cited for it.
    """
    id: str
    label: Optional[str]
    support: List[Evidence] = field(default_factory=list)
    contradict: List[Evidence] = field(default_factory=list)
    reason: str = ""
