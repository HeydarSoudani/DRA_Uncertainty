"""Constants and dataclasses shared by the belief model."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


# The status of a nugget: how likely its value is a correct answer of its
# criterion, one of the seven words of Agent-BRACE's verbal belief scale
# (arXiv 2605.11436), lowest level first.  The paper ranks "unknown" lowest;
# here it is the neutral level 0 (no document relates to the value, only the
# agent's turns give it); "unlikely" (the documents point against the value)
# and "doubtful" (a document contradicts it) are below it.
BELIEF_STATUSES = (
    "doubtful",
    "unlikely",
    "unknown",
    "possible",
    "probable",
    "almost certain",
    "certain",
)
# Signed level of each status (-2 to 4), for analysis.
STATUS_LEVEL: Dict[str, int] = {
    status: k - BELIEF_STATUSES.index("unknown") for k, status in enumerate(BELIEF_STATUSES)
}
# The paper's "confirmed or certain": "confirmed" read as "certain".
_STATUS_SYNONYMS = {"confirmed": "certain"}


def parse_status(text: Any) -> Optional[str]:
    """The belief status *text* names (case, underscores, hyphens and extra
    spaces ignored; "confirmed" accepted for "certain"), or None."""
    key = " ".join(str(text or "").lower().replace("_", " ").replace("-", " ").split())
    return key if key in STATUS_LEVEL else _STATUS_SYNONYMS.get(key)


def status_probability(status: str) -> float:
    """A nugget's status as a probability that its value is correct:
    (level + 2) / 6, doubtful 0, unknown 1/3, certain 1."""
    return (STATUS_LEVEL[status] + 2) / 6


# Criterion kinds: a closed criterion is one fact with one answer; an open
# one has several answers that hold together; the rest criterion of a set
# query ("any other member: <property>") stands for the members the
# extractor did not list; an aggregate criterion of a set query is how its
# answer follows from the members' values (computed, no document states
# it); an aspect criterion is one aspect of a report topic, whose answers
# are the facts on it.
CLOSED = "closed"
OPEN = "open"
REST = "rest"
AGGREGATE = "aggregate"
ASPECT = "aspect"

# Exclusive kinds have one correct answer, so the nuggets of such a
# criterion are alternatives; the nuggets of the other (cumulative) kinds
# are answers that hold together (members, values, facts).
EXCLUSIVE_KINDS = (CLOSED, AGGREGATE)
# Kinds no document can state (the agent computes an aggregate): the
# belief update takes no document observation on them, so their nuggets
# rest on the agent's turns only.
NO_DOC_KINDS = (AGGREGATE,)
# Nuggets kept per criterion, by kind: the most likely first.
MAX_NUGGETS: Dict[str, int] = {CLOSED: 5, AGGREGATE: 5, OPEN: 50, REST: 50, ASPECT: 50}


def exclusive(kind: str) -> bool:
    """Whether a criterion of *kind* has one correct answer."""
    return kind in EXCLUSIVE_KINDS


# The separator of a rest value's member and value ("<member>: <value>").
REST_PAIR = ": "

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


def same_value(a: str, b: str) -> bool:
    """Whether two values are the same text, case, whitespace and
    punctuation ignored."""
    return alnum(a) == alnum(b)


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
SUPPORTS = "supports"
CONTRADICTS = "contradicts"
STANCES = (SUPPORTS, CONTRADICTS)


def parse_stance(text: Any) -> Optional[str]:
    """``supports`` or ``contradicts`` as *text* names it ("support",
    "contradict" accepted), or None."""
    key = str(text or "").strip().lower()
    key = {"support": SUPPORTS, "contradict": CONTRADICTS}.get(key, key)
    return key if key in STANCES else None


@dataclass
class Evidence:
    """One source of a nugget, as the belief keeps it: the source's
    observation on the criterion (its label, stance and value, and a
    document's verified spans).  The value may be another answer of the
    criterion (a source that points against the nugget)."""
    label: str
    stance: str
    value: str
    spans: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"label": self.label, "stance": self.stance, "value": self.value}
        if self.spans:
            out["spans"] = list(self.spans)
        return out


@dataclass
class Observation:
    """One source stating one answer of one criterion, as the belief update
    wrote it.

    A document observation (``source`` ``doc``) cites ``spans`` of the
    document; ``span_verified[i]`` says whether ``spans[i]`` occurs in it
    verbatim (then it is the document's own words).  A think observation
    (``source`` ``think``) has no spans; its ``value`` is the agent's own
    text when it occurs in the turn.  ``grounded``: a document observation
    has a verified span, a think observation's value occurs in its turn.
    Only grounded observations enter the belief, as ``Evidence``.
    """
    criterion: str
    source: str
    label: str
    step: int
    stance: str
    value: str
    spans: List[str] = field(default_factory=list)
    span_verified: List[bool] = field(default_factory=list)
    doc_id: Optional[str] = None
    grounded: bool = False

    @property
    def verified_spans(self) -> List[str]:
        return [s for s, ok in zip(self.spans, self.span_verified) if ok]

    def evidence(self) -> Evidence:
        return Evidence(label=self.label, stance=self.stance, value=self.value, spans=self.verified_spans)

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"id": self.criterion, "label": self.label, "step": self.step,
                               "stance": self.stance, "value": self.value}
        if self.source == DOC:
            out.update(doc_id=self.doc_id, spans=list(self.spans), span_verified=list(self.span_verified))
        return out


@dataclass
class Nugget:
    """One answer of a criterion in the belief: ``value``, its ``status``
    and its ``evidence``, the sources with an observation on it (each
    source's own observation, ``Evidence``).  The belief is self-contained:
    the nuggets and their evidence are all the belief update reads of b_{t-1}."""
    value: str
    status: str
    evidence: List[Evidence] = field(default_factory=list)

    @property
    def labels(self) -> List[str]:
        return list(dict.fromkeys(e.label for e in self.evidence))

    def to_dict(self) -> Dict[str, Any]:
        return {"value": self.value, "status": self.status, "evidence": [e.to_dict() for e in self.evidence]}


@dataclass
class WrittenNugget:
    """One nugget as the belief update wrote it: ``status`` None when the
    reply names no belief status, ``evidence`` the labels it cites."""
    value: str
    status: Optional[str]
    evidence: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {"value": self.value, "status": self.status, "evidence": list(self.evidence)}


@dataclass
class NuggetUpdate:
    """The nugget list the belief update wrote for one criterion."""
    id: str
    nuggets: List[WrittenNugget]
    reason: str = ""
