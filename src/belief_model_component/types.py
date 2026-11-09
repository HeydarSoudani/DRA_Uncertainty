"""Constants and dataclasses shared by the belief model."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


# The status of a criterion: how likely it is covered by the selected
# candidate(s) its statement names, as the passages show it (the agent's
# turns propose candidates and mark doubts, but alone never make one more
# than "possible", ``TURN_ONLY_MAX_STATUS``), one of the seven words of
# Agent-BRACE's verbal belief scale (arXiv 2605.11436), lowest level first.
# The paper ranks "unknown" lowest; here it is the neutral level 0 (no
# observation gives a candidate); "unlikely" and "doubtful" are below it
# (the sources speak against the candidates).
BELIEF_STATUSES = (
    "doubtful",
    "unlikely",
    "unknown",
    "possible",
    "probable",
    "almost certain",
    "certain",
)
# The neutral status: no observation gives a candidate (b_0).
UNKNOWN = "unknown"
# Signed level of each status (-2 to 4), for analysis.
STATUS_LEVEL: Dict[str, int] = {
    status: k - BELIEF_STATUSES.index(UNKNOWN) for k, status in enumerate(BELIEF_STATUSES)
}
# The paper's "confirmed or certain": "confirmed" read as "certain".
_STATUS_SYNONYMS = {"confirmed": "certain"}


def parse_status(text: Any) -> Optional[str]:
    """The belief status *text* names (case, underscores, hyphens and extra
    spaces ignored; "confirmed" accepted for "certain"), or None."""
    key = " ".join(str(text or "").lower().replace("_", " ").replace("-", " ").split())
    return key if key in STATUS_LEVEL else _STATUS_SYNONYMS.get(key)


# The highest status a statement keeps when it cites no passage for its
# selected candidate (only the agent's turns support it); a computed
# criterion (``NO_DOC_KINDS``) is exempt.
TURN_ONLY_MAX_STATUS = "possible"

# The probability of "doubtful": above 0, so that one contradicted
# criterion lowers the geometric mean of the certainty score without
# fixing it at 0 (``certainty.geometric_mean``).
DOUBTFUL_PROBABILITY = 1 / 12


def status_probability(status: str) -> float:
    """A status as a probability that the criterion is covered:
    (level + 2) / 6, doubtful ``DOUBTFUL_PROBABILITY`` (1/12, not 0),
    unlikely 1/6, unknown 1/3, certain 1."""
    return max((STATUS_LEVEL[status] + 2) / 6, DOUBTFUL_PROBABILITY)


# Criterion kinds: a closed criterion is one fact with one answer; the rest
# criterion of a set query ("any other member: <property>") stands for the
# members the extractor did not list; an aggregate criterion of a set query is how its
# answer follows from the members' values (computed, no document states
# it); an aspect criterion is one aspect of a report topic, whose answers
# are the facts on it.
CLOSED = "closed"
REST = "rest"
AGGREGATE = "aggregate"
ASPECT = "aspect"

# Exclusive kinds have one correct answer, so the values of such a
# criterion are alternatives; the values of the other (cumulative) kinds
# are answers that hold together (members, values, facts).
EXCLUSIVE_KINDS = (CLOSED, AGGREGATE)
# Kinds no document can state (the agent computes an aggregate): the
# belief update takes no document observation on them, so their values
# rest on the agent's turns only.
NO_DOC_KINDS = (AGGREGATE,)


def exclusive(kind: str) -> bool:
    """Whether a criterion of *kind* has one correct answer."""
    return kind in EXCLUSIVE_KINDS


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


# Text matching: values and spans are compared on their letters and digits
# only (case, whitespace and punctuation ignored).

def alnum_index(text: str) -> Tuple[str, List[int]]:
    """*text* lowercased with everything but letters and digits removed, and
    the position in *text* of each character kept (to map a match back to
    the text's own words)."""
    keys, pos = [], []
    for i, ch in enumerate(text):
        for k in ch.lower():
            if k.isalnum():
                keys.append(k)
                pos.append(i)
    return "".join(keys), pos


def alnum(text: str) -> str:
    """*text* lowercased, letters and digits only."""
    return "".join(ch for ch in text.lower() if ch.isalnum())


@dataclass
class Criterion:
    """One criterion c_k of the per-query criteria list."""
    id: str
    text: str
    kind: str = CLOSED

    def to_dict(self) -> Dict[str, str]:
        return {"id": self.id, "text": self.text, "kind": self.kind}


# Observation sources: a document and the agent's turn (its think).  A
# document's label is "d<n>", numbered per sample in the order the
# documents are shown to the belief update; a turn's label is "t<step>",
# the belief model's step (the answer turn: the last step + 1).
DOC = "doc"
THINK = "think"


@dataclass
class Observation:
    """What one source says about one criterion, as the belief update
    wrote it: the ``spans`` of the source (the passage, or the agent's
    turn) that give information about the criterion.  It has no stance and
    no candidate: the statement judges what the observations show.

    ``span_verified[i]`` says whether ``spans[i]`` occurs in the source as
    the call showed it (then it is the source's own words); ``grounded``:
    one span is verified.  Only grounded observations enter the belief;
    ``to_dict`` is the saved record of one that entered (its verified
    spans only)."""
    criterion: str
    source: str
    label: str
    step: int
    spans: List[str] = field(default_factory=list)
    span_verified: List[bool] = field(default_factory=list)
    doc_id: Optional[str] = None
    grounded: bool = False

    @property
    def verified_spans(self) -> List[str]:
        return [s for s, ok in zip(self.spans, self.span_verified) if ok]

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"id": self.criterion, "label": self.label, "step": self.step}
        if self.source == DOC:
            out["doc_id"] = self.doc_id
        out["spans"] = self.verified_spans
        return out


@dataclass
class Statement:
    """The belief of one criterion: one statement in free text with a fixed
    opening (``statement_text``), as the belief update wrote it (its status
    word lowered by code when the turn-only cap applies).  Parsed from it:
    ``status``, the ``selected`` candidate(s) the opening names, the labels
    it cites (``cited``) and those of the part the status rests on, before
    the first other-candidate sentence (``support_cited``).  The text is
    all that the next update and the agent read of the criterion."""
    text: str
    status: str = UNKNOWN
    selected: List[str] = field(default_factory=list)
    cited: List[str] = field(default_factory=list)
    support_cited: List[str] = field(default_factory=list)

    def to_dict(self, criterion_id: str) -> Dict[str, Any]:
        return {"id": criterion_id, "status": self.status, "selected": list(self.selected), "text": self.text,
                "cited": list(self.cited), "support_cited": list(self.support_cited)}


@dataclass
class WrittenStatement:
    """The statement text the update wrote for one criterion."""
    id: str
    text: str
