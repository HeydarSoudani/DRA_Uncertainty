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
# aggregate criterion of a set query is how its answer follows from the
# members' values (computed, no document states it); an aspect criterion is
# one aspect of a report topic, which has no complete list of parts.  The
# kind sets whether a criterion has a status (``STATUSLESS_KINDS``), its
# candidate cap (``MAX_CANDIDATES``) and whether it counts in the reasoning
# confidence (``REASONING_KINDS_EXCLUDED``).
CLOSED = "closed"
OPEN = "open"
REST = "rest"
AGGREGATE = "aggregate"
ASPECT = "aspect"

# Kinds without a status: no passage can state an aggregate (the agent
# computes it), so it has answer candidates only, and its status is None
# in every belief state; coverage, Delta^D and the analysis skip it.
STATUSLESS_KINDS = (AGGREGATE,)


def initial_status(kind: str) -> Optional[str]:
    """b_0 of a criterion of *kind*: ``INITIAL_STATUS``, None without a status."""
    return None if kind in STATUSLESS_KINDS else INITIAL_STATUS

# Query shapes (layout.DATASET_SPECS): one entity or value described by clues,
# one entity reached through a chain of other entities (hops), a value
# computed over every member of a set, or a report on several aspects of a
# topic.  The shape picks the criteria-extraction prompt and the kind of each
# criterion; the extractor never chooses either.
SINGLE_TARGET = "single_target"
MULTI_HOP = "multi_hop"
SET = "set"
MULTI_ASPECT = "multi_aspect"
QUERY_SHAPES = (SINGLE_TARGET, MULTI_HOP, SET, MULTI_ASPECT)


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

    ``status`` is b_t(k) as written (None when it names no belief status,
    and ignored for a criterion without a status);
    ``support`` and ``contradict`` are the step's passages cited for it.
    """
    id: str
    status: Optional[str]
    support: List[Evidence] = field(default_factory=list)
    contradict: List[Evidence] = field(default_factory=list)
    reason: str = ""


# Answer candidates (reasoner side): per criterion, the answers the agent
# currently considers for it, each with a belief status read as how likely
# the candidate is the correct answer.  At most MAX_CANDIDATES[kind] per
# criterion: a closed criterion has one answer, so its candidates are
# alternatives (an aggregate's are final answers); the candidates of an
# open or rest criterion are members or values that hold together (a rest
# candidate is "<member>: <value>", ``REST_PAIR``).  An aspect criterion (a
# report) has none.
MAX_CANDIDATES: Dict[str, int] = {CLOSED: 3, AGGREGATE: 3, OPEN: 50, REST: 50, ASPECT: 0}

# Kinds left out of the reasoning confidence R: a rest criterion asks
# whether the set is complete (its status covers it), and an aspect
# criterion has no candidates.
REASONING_KINDS_EXCLUDED = (REST, ASPECT)


# The separator of a rest candidate's member and value.
REST_PAIR = ": "


def candidate_probability(status: str) -> float:
    """The status of a candidate as a probability of being correct:
    (level + 2) / 6, doubtful 0, unknown 1/3, certain 1."""
    return (STATUS_LEVEL[status] + 2) / 6


@dataclass
class Candidate:
    """One answer candidate of a criterion.  ``value`` is the agent's own
    text when it occurs in one of the agent's turns read so far
    (``anchored``), else the updater's text; ``status`` is None when the
    reply names no belief status."""
    value: str
    status: Optional[str]
    anchored: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {"value": self.value, "status": self.status}


@dataclass
class CandidateUpdate:
    """The candidate list the belief updater wrote for one criterion."""
    id: str
    candidates: List[Candidate]
    reason: str = ""
