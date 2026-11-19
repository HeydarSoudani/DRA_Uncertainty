"""Constants and dataclasses shared by the belief model."""

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# The word of estimative probability (WEP) of a belief: how likely its
# most probable candidate addresses it (an aspect: how well its kept facts
# cover it), judged by the belief update from all the belief's passages
# (``prompts/shared/wep_definitions.txt``); "unknown" when no passage
# supports a candidate.  A belief has one WEP: its most probable candidate
# carries it, its other candidates have none.  Ordered by probability
# (``STATUS_PROBABILITY``), lowest first.
BELIEF_STATUSES = (
    "almost no chance",
    "unknown",
    "unlikely",
    "possible",
    "probable",
    "almost certain",
    "certain",
)
# The status of a belief no passage supports a candidate of (b_0).
UNKNOWN = "unknown"
UNLIKELY = "unlikely"
ALMOST_NO_CHANCE = "almost no chance"
# Signed level of each status, "unknown" 0, for ordering and analysis.
STATUS_LEVEL: Dict[str, int] = {
    status: k - BELIEF_STATUSES.index(UNKNOWN) for k, status in enumerate(BELIEF_STATUSES)
}


def parse_status(text: Any) -> Optional[str]:
    """The status *text* names (case, underscores, hyphens and extra
    spaces ignored), or None."""
    key = " ".join(str(text or "").lower().replace("_", " ").replace("-", " ").split())
    return key if key in STATUS_LEVEL else None


# What the reasoning says of a candidate (a claim): the agent keeps it as
# the answer or a possible answer, or drops it (it fails a part).
KEEP = "keep"
DROP = "drop"
CLAIMS = (KEEP, DROP)


def parse_claim(text: Any) -> Optional[str]:
    """The claim *text* names (``KEEP`` or ``DROP``, case and extra spaces
    ignored), or None."""
    key = " ".join(str(text or "").lower().split())
    return key if key in CLAIMS else None


# The lowest status a selected candidate has (``BeliefState.selected``).
NO_SUPPORT_MAX_STATUS = "possible"


# Each status as a probability, the one scale of every number read from a
# status (the certainty formula, the final scorer's s_T and the status
# calibration): the midpoint of each word's range; "unknown" (no
# evidence) 0.25; "certain" stays below 1, so a log or squared error term
# never saturates.
STATUS_PROBABILITY: Dict[str, float] = {
    "almost no chance": 0.10,
    "unknown": 0.25,
    "unlikely": 0.30,
    "possible": 0.50,
    "probable": 0.70,
    "almost certain": 0.875,
    "certain": 0.97,
}


def status_probability(status: str) -> float:
    """A status as a probability (``STATUS_PROBABILITY``)."""
    return STATUS_PROBABILITY[parse_status(status) or status]


# Belief kinds: a closed belief has one answer (a clue or the asked
# thing of a multi-clue query, a hop of a multi-hop query, a member's value
# of a multi-member query); an aggregate belief combines the answers of
# the beliefs before it (its inputs) and takes no observation: the target
# of a multi-clue query ("the composer that meets B1 to B3",
# ``is_target``), whose candidate code derives as the name that is the
# most probable candidate of every clue (``BeliefState._derive_aggregates``),
# and the computation of a multi-member query (``is_computed``), which
# stays without candidates ("unknown"); an aspect
# belief is one aspect of a report topic: its candidates are the facts
# on it, kept or removed, and the aspect has one status.
CLOSED = "closed"
AGGREGATE = "aggregate"
ASPECT = "aspect"
BELIEF_KINDS = (CLOSED, AGGREGATE, ASPECT)

# Exclusive kinds have one correct answer, so the values of such a
# belief are alternatives; an aspect's facts hold together.
EXCLUSIVE_KINDS = (CLOSED, AGGREGATE)
# Kinds whose candidates are names or figures a source writes out (checked
# against the spans and titles of the belief's passages).  An aggregate
# is combined, never checked.
NAMED_KINDS = (CLOSED,)


def exclusive(kind: str) -> bool:
    """Whether a belief of *kind* has one correct answer."""
    return kind in EXCLUSIVE_KINDS


# The text of a multi-clue query's target: "the <type> that meets B1 to B<k>".
TARGET_RE = re.compile(r"\bmeets\s+B1\s+to\s+B(\d+)\b")


def is_target(kind: str, text: str) -> bool:
    """Whether a belief of *kind* and *text* is a target: an aggregate
    whose candidate is derived from the clues before it (the name that is
    the most probable candidate of every one), never written by the belief
    update."""
    return kind == AGGREGATE and TARGET_RE.search(text or "") is not None


def is_computed(kind: str, text: str) -> bool:
    """Whether a belief of *kind* and *text* is a computed aggregate: the
    value a multi-member query computes over its member beliefs (it takes
    no observation and has no candidates)."""
    return kind == AGGREGATE and not is_target(kind, text)


# The entity a belief is about, written before its colon: a multi-member
# query's member belief "<member>: <property>" (``member_subject``).
_PARENTHETICAL_RE = re.compile(r"\s*\([^)]*\)")


def member_subject(beliefs: List["Belief"], k: int) -> Optional[str]:
    """The member belief *k* is about, or None: a closed belief
    "<member>: <property>" of a multi-member query's list (one with a
    computed aggregate).  Its candidates are the member's value, never the
    member."""
    c = beliefs[k]
    if c.kind != CLOSED or ":" not in c.text or not any(is_computed(x.kind, x.text) for x in beliefs):
        return None
    # The last separator: a member's name may hold one ("Star Wars: Episode IV: box office").
    subject = c.text.rsplit(": " if ": " in c.text else ":", 1)[0].strip()
    return subject or None


def subject_names(subject: str) -> List[str]:
    """The names of a belief's *subject* a candidate may repeat: the
    subject, and without its parenthetical ("New York (state)": "New York")."""
    short = _PARENTHETICAL_RE.sub("", subject).strip()
    return list(dict.fromkeys(n for n in (subject, short) if n))


# Query shapes (layout.DATASET_SPECS): one entity (the target) described by a
# list of clues, the answer the target or something asked about it,
# one entity reached through a chain of other entities (hops), a value
# computed over every member of a set, or a report on several aspects of a
# topic.  The shape picks the belief-extraction prompt and the kind of each
# belief; the extractor never chooses either.
MULTI_CLUE = "multi_clue"
MULTI_HOP = "multi_hop"
MULTI_MEMBER = "multi_member"
MULTI_ASPECT = "multi_aspect"
QUERY_SHAPES = (MULTI_CLUE, MULTI_HOP, MULTI_MEMBER, MULTI_ASPECT)


# A reference to another belief in a belief's text ("born in B1").  Belief
# ids are "B<k>" everywhere: prompts, replies, banks, gold annotations,
# run files and evaluation.
BELIEF_REF_RE = re.compile(r"\bB(\d+)\b")
_ID_RE = re.compile(r"^b(\d+)$")


def belief_id(value: Any) -> str:
    """A belief id as a reply writes it ("B1", "b1.", "[B1]") as "B1"."""
    key = str(value if value is not None else "").strip().strip("[].:").strip().lower()
    m = _ID_RE.match(key)
    return f"B{m.group(1)}" if m else key


@dataclass
class Belief:
    """One belief c_k of the per-query belief list."""
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
    """What one source says about one belief, as the belief update
    wrote it.

    A passage observation (``DOC``): the ``spans`` of the passage that
    state a part of the belief, the ``candidate`` they are about and
    their stance (``against``: the candidate fails the part, "-"; else
    "+").  A reasoning observation (``THINK``, a claim): one span of the
    agent's turn that gives or drops one ``candidate``, and the ``claim``
    it makes (``KEEP`` or ``DROP``).

    ``span_verified[i]`` says whether ``spans[i]`` occurs in the source as
    the call showed it (then it is the source's own words); ``grounded``:
    one span is verified.  Only grounded observations enter the belief;
    ``to_dict`` is the saved record of one that entered (its verified
    spans only)."""
    belief: str
    source: str
    label: str
    step: int
    spans: List[str] = field(default_factory=list)
    span_verified: List[bool] = field(default_factory=list)
    doc_id: Optional[str] = None
    grounded: bool = False
    candidate: Optional[str] = None
    against: bool = False
    # The stance as the reply wrote it, when it is neither "+" nor "-"
    # (an issue; not saved).
    bad_stance: Optional[str] = None
    claim: Optional[str] = None
    # The claim as the reply wrote it (for an issue's detail; not saved).
    claim_written: Optional[str] = None

    @property
    def verified_spans(self) -> List[str]:
        return [s for s, ok in zip(self.spans, self.span_verified) if ok]

    @property
    def stance(self) -> str:
        return "-" if self.against else "+"

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"id": self.belief, "label": self.label, "step": self.step}
        if self.source == DOC:
            out.update(doc_id=self.doc_id, candidate=self.candidate, stance=self.stance)
        else:
            out.update(candidate=self.candidate, claim=self.claim)
        out["spans"] = self.verified_spans
        return out


@dataclass
class Support:
    """One passage observation on a candidate: the passage (its label,
    document and title), the step it was observed in, its spans on the
    belief, and whether it supports the candidate or speaks against it
    (``against``)."""
    label: str
    doc_id: Optional[str]
    title: str
    step: int
    spans: List[str]
    against: bool = False

    def source(self) -> str:
        """The source of the passage: the document it comes from, known by
        its title (case and punctuation ignored), so the chunks of one page
        (InfoSeek, TRQA) are one source; its document id, else its label,
        when it has no title."""
        return " ".join(re.sub(r"\W+", " ", self.title.lower()).split()) or self.doc_id or self.label

    def to_dict(self) -> Dict[str, Any]:
        return {"label": self.label, "doc_id": self.doc_id, "title": self.title, "step": self.step,
                "spans": list(self.spans), "against": self.against}


# The basis of a candidate (derived from its support, never written by
# the update): "passage" when a passage observation supports it or speaks
# against it, "reasoning" when none does (a proposal dropped from the
# proposals).
PASSAGE = "passage"
REASONING = "reasoning"


@dataclass
class Candidate:
    """One candidate of a belief in the belief: its name, its status,
    the step it was added in, its support (the passage observations that
    support it or speak against it, ``Support``) and the agent's latest
    claim on it (``claim``, from the turn ``claim_step``; never shown to
    the update).  The status is the belief's WEP for its most probable
    candidate and for an aspect's facts, code's for a target's derived
    candidate, and None for any other candidate (a belief has one WEP).
    Only an aspect's fact a passage speaks against leaves the list (kept
    in the hidden removal log of its belief, ``RemovedCandidate``); the
    agent's drop is a claim and removes no candidate."""
    name: str
    status: Optional[str]
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
    """A name the agent's reasoning proposed for a belief that no
    passage supports yet: no status, not a rival of the candidates.  It
    keeps the agent's latest claim on it and the turn it came from
    (``step``), and the step it was proposed in (``added_step``); a
    passage on it makes it a candidate, a claim that drops it takes it
    off the proposals (into the removal log)."""
    name: str
    claim: Optional[str]
    step: int
    added_step: int

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "claim": self.claim, "step": self.step, "added_step": self.added_step}


@dataclass
class RemovedCandidate:
    """One candidate or proposal the belief dropped: an aspect's fact a
    passage spoke against, a proposal the agent's claim dropped, or a
    proposal dropped for room.  Its name, its status when removed (null
    for a proposal), the step it was added in and the step it was removed
    in, and the reason.  Saved, never shown."""
    name: str
    status: Optional[str]
    basis: str
    added_step: int
    removed_step: int
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "status": self.status, "basis": self.basis, "added_step": self.added_step,
                "removed_step": self.removed_step, "reason": self.reason}


@dataclass
class BeliefUpdate:
    """One item of the reply's ``beliefs`` list, as written: the belief id,
    the reason, its most probable candidate (None: none written; an
    aspect's is null) and its WEP (raw text)."""
    id: str
    reason: str = ""
    most_probable: Optional[str] = None
    wep: str = ""
