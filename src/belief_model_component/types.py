"""Constants and dataclasses shared by the belief model."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


# The status of a candidate: how likely it is an answer of its criterion,
# as the passages show it (the agent's turns propose candidates, but alone
# never make one more than "possible", ``TURN_ONLY_MAX_STATUS``); the
# status of a criterion is derived from its candidates (the highest one;
# "unknown" without candidates).  One of the seven words of
# Agent-BRACE's verbal belief scale (arXiv 2605.11436), lowest level first,
# without its "doubtful": a candidate a passage rules out is removed
# (``REMOVED``), not kept at a lowest status.  The paper ranks "unknown"
# lowest; here it is the neutral level 0 (no observation gives a candidate);
# "unlikely" is below it (a passage speaks against the candidate, or another
# candidate of a one-answer criterion meets more of it).
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
# passage rules it out, or, when only the agent's turns name it, the turns
# reject it.  An action, not a status: ``parse_status`` does not accept it.
REMOVED = "removed"


def is_removed(text: Any) -> bool:
    """Whether a reply's status *text* is ``REMOVED`` (case and spaces ignored)."""
    return str(text or "").strip().lower() == REMOVED


# The range of the status of a candidate whose basis is "reasoning" (only
# the agent's turns name it): "possible" when they name it, "unlikely" when
# another candidate outweighs it (turns that reject it remove it); a
# computed criterion (``NO_DOC_KINDS``) is exempt.
TURN_ONLY_MAX_STATUS = "possible"
TURN_ONLY_MIN_STATUS = "unlikely"

def status_probability(status: str) -> float:
    """A status as a probability that the criterion is covered:
    (level + 2) / 6: unlikely 1/6, unknown 1/3, possible 1/2, probable 2/3,
    almost certain 5/6, certain 1.  No status scores 0, so one weak
    criterion lowers the geometric mean of the certainty score without
    fixing it at 0 (``certainty.geometric_mean``)."""
    return (STATUS_LEVEL[status] + 2) / 6


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


def _unspaced(ch: str) -> bool:
    """Whether *ch* is of a script written without spaces between words
    (CJK ideographs, kana), where any character may start a word."""
    o = ord(ch)
    return 0x3040 <= o <= 0x30FF or 0x3400 <= o <= 0x9FFF or 0xF900 <= o <= 0xFAFF


def _word_edge(text: str, i: int, inner: str) -> bool:
    """Whether a match whose edge character is *inner* ends at a word
    boundary when *text*[i] is the character just outside it."""
    return not 0 <= i < len(text) or not text[i].isalnum() or _unspaced(text[i]) or _unspaced(inner)


def contains_words(text: str, name: str) -> bool:
    """Whether *name* occurs in *text* as whole words: compared on letters
    and digits only (``alnum``), and the match must start and end at a word
    boundary of *text*, so "12" is not found in "1912" nor "Ann" in "Anne"."""
    key = alnum(name)
    if not key or not text:
        return False
    keys, pos = alnum_index(text)
    j = keys.find(key)
    while j >= 0:
        a, b = pos[j], pos[j + len(key) - 1]
        if _word_edge(text, a - 1, text[a]) and _word_edge(text, b + 1, text[b]):
            return True
        j = keys.find(key, j + 1)
    return False


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
    no candidate: the candidate updates judge what the observations show.

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


# The basis of a candidate (code sets it from the observations, the
# update never writes it): "passage" when a passage observation on the
# criterion bears on it (it names the candidate in a span or its title, or
# the reason of an update of the candidate cites it); "reasoning" when only
# the agent's turns name it.  A candidate's basis never goes back from
# passage to reasoning.
PASSAGE = "passage"
REASONING = "reasoning"

@dataclass
class Candidate:
    """One candidate of a criterion in the belief: its name, its status,
    its basis and the step it was added in.  A candidate the update does
    not name keeps its status; one the update removes (``REMOVED``) leaves
    the list, and is kept in the hidden removal log of its criterion
    (``RemovedCandidate``)."""
    name: str
    status: str
    basis: str = REASONING
    added_step: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "status": self.status, "basis": self.basis, "added_step": self.added_step}


@dataclass
class RemovedCandidate:
    """One candidate the update removed: its name, its status and basis
    when removed, the step it was added in and the step it was removed in,
    and the reason.  Saved, never shown to the update."""
    name: str
    status: str
    basis: str
    added_step: int
    removed_step: int
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "status": self.status, "basis": self.basis, "added_step": self.added_step,
                "removed_step": self.removed_step, "reason": self.reason}


@dataclass
class CandidateUpdate:
    """One item of the reply's ``candidates`` list, as written: the
    criterion id, the candidate, the reason of the change ("Was <status>:
    ..." or "New: ...") and the new status (raw text)."""
    id: str
    candidate: str
    reason: str
    status: str
