"""Constants and dataclasses shared by the belief model."""

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# The status of a candidate: how likely it is an answer of its criterion,
# as the passages show it (the agent's reasoning only proposes names: a
# candidate without passage support is never above "possible",
# ``NO_SUPPORT_MAX_STATUS``); the status of a criterion is derived from its
# candidates (the highest one; "unknown" without candidates).  One of the
# seven words of Agent-BRACE's verbal belief scale (arXiv 2605.11436),
# lowest level first, without its "doubtful": a candidate a passage rules
# out is removed (``REMOVED``), not kept at a lowest status.  The paper
# ranks "unknown" lowest; here it is the neutral level 0 (no observation
# gives a candidate); "unlikely" is below it (a passage speaks against the
# candidate, or another candidate of a one-answer criterion has clearly
# stronger support).
BELIEF_STATUSES = (
    "unlikely",
    "unknown",
    "possible",
    "probable",
    "almost certain",
    "certain",
)
# The neutral status: no observation gives a candidate (b_0).
UNKNOWN = "unknown"
# The lowest status: a passage speaks against the candidate.
UNLIKELY = "unlikely"
# Signed level of each status (-1 to 4), for analysis.
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


# What the update writes in place of a status to remove a candidate: a
# passage states that it fails a part of the criterion.  An action, not a
# status: ``parse_status`` does not accept it.
REMOVED = "removed"


def is_removed(text: Any) -> bool:
    """Whether a reply's status *text* is ``REMOVED`` (case and spaces ignored)."""
    return str(text or "").strip().lower() == REMOVED


# The highest status of a candidate no passage supports (a computed
# criterion, ``NO_DOC_KINDS``, is exempt: the reasoning sets its status).
NO_SUPPORT_MAX_STATUS = "possible"


# What the reasoning says of a candidate (a claim): a status of the scale,
# or that the agent rejects it.
REJECTED = "rejected"


def parse_claim(text: Any) -> Optional[str]:
    """The claim *text* names: a status other than "unknown", or
    ``REJECTED``; None otherwise."""
    if str(text or "").strip().lower() == REJECTED:
        return REJECTED
    status = parse_status(text)
    return status if status not in (None, UNKNOWN) else None


def status_probability(status: str) -> float:
    """A status as a probability that the criterion is covered:
    (level + 2) / 6: unlikely 1/6, unknown 1/3, possible 1/2, probable 2/3,
    almost certain 5/6, certain 1.  No status scores 0, so one weak
    criterion lowers the geometric mean of the certainty score without
    fixing it at 0 (``core.certainty.geometric_mean``)."""
    return (STATUS_LEVEL[status] + 2) / 6


# Criterion kinds: a closed criterion is one fact with one answer; a hop is
# an entity on the chain of a multi-hop query before its answer, which may
# have several answers (two entities can both fit the description and lead
# on to the same next hop); the rest criterion of a multi-member query ("any other
# member: <property>") stands for the members the extractor did not list;
# an aggregate criterion of a multi-member query is how its answer follows from the
# members' values (computed, no document states it); an aspect criterion is
# one aspect of a report topic, whose answers are the facts on it; the
# target criterion of a multi-clue query ("the composer that meets c1 to
# c3") is the entity every clue before it describes, its candidates derived
# from theirs (``DERIVED_KINDS``).
CLOSED = "closed"
HOP = "hop"
REST = "rest"
AGGREGATE = "aggregate"
ASPECT = "aspect"
TARGET = "target"

# Exclusive kinds have one correct answer, so the values of such a
# criterion are alternatives; the values of the other (cumulative) kinds
# are answers that hold together (members, values, facts).
EXCLUSIVE_KINDS = (CLOSED, AGGREGATE, TARGET)
# Kinds whose candidates are names or figures a source writes out (checked
# against the spans and titles of the criterion's passages; a rest
# criterion's "<member>: <value>" each part).  An aggregate is computed,
# never checked.
NAMED_KINDS = (CLOSED, HOP, REST)
# Kinds no document can state (the agent computes an aggregate): the
# belief update takes no document observation on them, so their values
# rest on the agent's turns only.
NO_DOC_KINDS = (AGGREGATE,)
# Kinds whose candidates code derives from other criteria, never the belief
# update (``BeliefState._derive_targets``): a target's candidates are the
# names listed on every clue before it, each at the lowest of its statuses
# there, with their support.
DERIVED_KINDS = (TARGET,)


def exclusive(kind: str) -> bool:
    """Whether a criterion of *kind* has one correct answer."""
    return kind in EXCLUSIVE_KINDS


# Query shapes (layout.DATASET_SPECS): one entity (the target) described by a
# list of clues, the answer the target or something asked about it,
# one entity reached through a chain of other entities (hops), a value
# computed over every member of a set, or a report on several aspects of a
# topic.  The shape picks the criteria-extraction prompt and the kind of each
# criterion; the extractor never chooses either.
MULTI_CLUE = "multi_clue"
MULTI_HOP = "multi_hop"
MULTI_MEMBER = "multi_member"
MULTI_ASPECT = "multi_aspect"
QUERY_SHAPES = (MULTI_CLUE, MULTI_HOP, MULTI_MEMBER, MULTI_ASPECT)


# A reference to another criterion in a criterion's text ("the town of c1").
CRITERION_REF_RE = re.compile(r"\bc(\d+)\b")


def criterion_id(value: Any) -> str:
    """A criterion id as a reply writes it ("C1", "c1.", "[c1]") as "c1"."""
    return str(value if value is not None else "").strip().strip("[].:").strip().lower()


@dataclass
class Criterion:
    """One criterion c_k of the per-query criteria list."""
    id: str
    text: str
    kind: str = CLOSED

    def to_dict(self) -> Dict[str, str]:
        return {"id": self.id, "text": self.text, "kind": self.kind}


# Observation sources: a document and the agent's turn (its reasoning).  A
# document's label is "d<n>", numbered per sample in the order the
# documents are shown to the belief update; a turn's label is "t<step>",
# the belief model's step (the answer turn: the last step + 1).
DOC = "doc"
THINK = "think"


@dataclass
class Observation:
    """What one source says about one criterion, as the belief update
    wrote it.

    A passage observation (``DOC``): the ``spans`` of the passage that
    state a part of the criterion.  It has no stance and no candidate: the
    candidate updates say which observations support or speak against
    which candidate.  A reasoning observation (``THINK``, a claim): one
    span of the agent's turn that names, assesses or rejects one
    ``candidate``, and the ``claim`` it makes (a status, or
    ``REJECTED``).

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
    candidate: Optional[str] = None
    claim: Optional[str] = None

    @property
    def verified_spans(self) -> List[str]:
        return [s for s, ok in zip(self.spans, self.span_verified) if ok]

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"id": self.criterion, "label": self.label, "step": self.step}
        if self.source == DOC:
            out["doc_id"] = self.doc_id
        else:
            out.update(candidate=self.candidate, claim=self.claim)
        out["spans"] = self.verified_spans
        return out


@dataclass
class Support:
    """One passage observation on a candidate: the passage (its label,
    document and title), the step it was observed in, its spans on the
    criterion, and whether it supports the candidate or speaks against it
    (``against``)."""
    label: str
    doc_id: Optional[str]
    title: str
    step: int
    spans: List[str]
    against: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {"label": self.label, "doc_id": self.doc_id, "title": self.title, "step": self.step,
                "spans": list(self.spans), "against": self.against}


# The basis of a candidate (derived from its support, never written by
# the update): "passage" when a passage observation supports it or speaks
# against it, "reasoning" when none does (a computed criterion's
# candidates).
PASSAGE = "passage"
REASONING = "reasoning"


@dataclass
class Candidate:
    """One candidate of a criterion in the belief: its name, its status,
    the step it was added in, its support (the passage observations that
    support it or speak against it, ``Support``) and the agent's latest
    claim on it (``claim``, from the turn ``claim_step``; never shown to
    the update).  A candidate the update does not name keeps its status;
    one the update removes (``REMOVED``) leaves the list, and is kept in
    the hidden removal log of its criterion (``RemovedCandidate``)."""
    name: str
    status: str
    added_step: int = 0
    support: List[Support] = field(default_factory=list)
    claim: Optional[str] = None
    claim_step: Optional[int] = None

    @property
    def basis(self) -> str:
        return PASSAGE if self.support else REASONING

    def supports(self, against: bool = False) -> List[Support]:
        return [x for x in self.support if x.against == against]

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "status": self.status, "basis": self.basis, "added_step": self.added_step,
                "support": [x.to_dict() for x in self.support], "claim": self.claim,
                "claim_step": self.claim_step}


@dataclass
class Proposal:
    """A name the agent's reasoning proposed for a criterion that no
    passage supports yet: no status, not a rival of the candidates.  It
    keeps the agent's latest claim on it and the turn it came from
    (``step``), and the step it was proposed in (``added_step``); a
    passage that supports it makes it a candidate, a claim that rejects
    it drops it."""
    name: str
    claim: Optional[str]
    step: int
    added_step: int

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "claim": self.claim, "step": self.step, "added_step": self.added_step}


@dataclass
class RemovedCandidate:
    """One candidate or proposal the belief dropped: its name, its status
    when removed (null for a proposal), the step it was added in and the
    step it was removed in, the reason (the update's, or the claim
    that rejected a proposal) and whether the call ruled it out
    (``against``: a passage spoke against it, or the agent's claim did: the
    removal of a computed criterion's candidate, a rejected proposal; not a
    proposal dropped for room).  The final scorer reads a ruled-out name as
    removed.  Saved, never shown to the update."""
    name: str
    status: Optional[str]
    basis: str
    added_step: int
    removed_step: int
    reason: str = ""
    against: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "status": self.status, "basis": self.basis, "added_step": self.added_step,
                "removed_step": self.removed_step, "reason": self.reason, "against": self.against}


@dataclass
class CandidateUpdate:
    """One item of the reply's ``candidates`` list, as written: the
    criterion id, the candidate, the labels of the call's passages whose
    observations support it (``support``) or speak against it
    (``against``), the reason of the change ("Was <status>: ...", "Was
    proposed: ..." or "New: ...") and the new status (raw text)."""
    id: str
    candidate: str
    reason: str
    status: str
    support: List[str] = field(default_factory=list)
    against: List[str] = field(default_factory=list)
