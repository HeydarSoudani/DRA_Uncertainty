"""The issue codes of one belief update (``BeliefState.apply``), in one
table (``ISSUES``), and the sets of them by role (``GATE_ISSUES``,
``UNCLEAN_REPLY``), derived from it and reported in the issue table of
the evaluation (``evaluation.belief.issues``).

The table is grouped by what a code is about: an observation, a belief
item, the candidate name of a passage observation that entered, the most
probable candidate or WEP.  Each code has:

- ``effect``: what code did: ``dropped`` (the observation or update is
  ignored), ``corrected`` (code set another most probable candidate or
  WEP than written), ``flagged`` (taken as written, with a
  flaw) or ``note`` (not an error: information only);
- ``gate``: a reply with it fails the format gate (it names what the
  call did not show, or writes an update code cannot take);
- ``unclean``: a reply with it is not a clean sample (a training sample
  should not contain it).

An issue record is ``{id, code, detail, [label]}``.  A ``note`` code goes
to the update's ``notes``, the others to its ``issues``.
"""

from dataclasses import dataclass
from typing import Dict, Tuple

DROPPED, CORRECTED, FLAGGED, NOTE = "dropped", "corrected", "flagged", "note"
# The effects in the order a log shows them.
EFFECTS = (DROPPED, CORRECTED, FLAGGED, NOTE)
# The effects of an error (a note is none).
ERROR_EFFECTS = (DROPPED, CORRECTED, FLAGGED)


@dataclass(frozen=True)
class IssueCode:
    code: str
    effect: str
    gate: bool = False
    unclean: bool = False
    meaning: str = ""


# An observation that did not enter the belief.
ISSUE_UNKNOWN_BELIEF = "unknown_belief"
ISSUE_UNKNOWN_PASSAGE = "unknown_passage"
ISSUE_OBSERVATION_ON_AGGREGATE = "observation_on_aggregate"
ISSUE_SPAN_UNVERIFIED = "span_unverified"
ISSUE_TURN_SPAN_UNVERIFIED = "turn_span_unverified"
ISSUE_CLAIM_INCOMPLETE = "claim_incomplete"
ISSUE_CANDIDATE_MISSING = "candidate_missing"
ISSUE_STANCE_INVALID = "stance_invalid"
ISSUE_SUBJECT_CANDIDATE = "subject_candidate"
# A belief item that is not taken, or missing.
ISSUE_BELIEF_REPEATED = "belief_repeated"
ISSUE_UPDATE_WITHOUT_OBSERVATION = "update_without_observation"
ISSUE_UPDATE_ON_AGGREGATE = "update_on_aggregate"
ISSUE_BELIEF_MISSING = "belief_missing"
# The candidate name of a passage observation that entered.
ISSUE_CANDIDATE_NOT_IN_PASSAGES = "candidate_not_in_passages"
ISSUE_CANDIDATE_NEAR_DUPLICATE = "candidate_near_duplicate"
ISSUE_FACT_MERGED = "fact_merged"
ISSUE_SPAN_OFF_BELIEF = "span_off_belief"
# The most probable candidate and WEP of a belief item.
ISSUE_MOST_PROBABLE_WRONG = "most_probable_wrong"
ISSUE_WEP_INVALID = "wep_invalid"
ISSUE_WEP_WITHOUT_SUPPORT = "wep_without_support"
ISSUE_UNKNOWN_WITH_SUPPORT = "unknown_with_support"
# Notes (not errors).
ISSUE_CANDIDATE_READDED = "candidate_readded"
ISSUE_SPAN_OFF_FACT = "span_off_fact"

ISSUES: Dict[str, IssueCode] = {i.code: i for i in (
    IssueCode(ISSUE_UNKNOWN_BELIEF, DROPPED, gate=True,
              meaning="an observation or belief item of a belief id the belief does not have"),
    IssueCode(ISSUE_UNKNOWN_PASSAGE, DROPPED, gate=True,
              meaning="a passage label that is no passage of the call"),
    IssueCode(ISSUE_OBSERVATION_ON_AGGREGATE, DROPPED, unclean=True,
              meaning="an observation on a belief computed from other beliefs (a target, a computed answer)"),
    IssueCode(ISSUE_SPAN_UNVERIFIED, DROPPED, unclean=True,
              meaning="a passage observation without a span of the passage as shown"),
    IssueCode(ISSUE_TURN_SPAN_UNVERIFIED, DROPPED, unclean=True,
              meaning="a reasoning observation without a span of the agent's turn"),
    IssueCode(ISSUE_CLAIM_INCOMPLETE, DROPPED, gate=True, unclean=True,
              meaning="a reasoning observation without a candidate, or whose claim is neither 'keep' nor 'drop'"),
    IssueCode(ISSUE_CANDIDATE_MISSING, DROPPED, gate=True,
              meaning="a passage observation without a candidate"),
    IssueCode(ISSUE_STANCE_INVALID, DROPPED, gate=True,
              meaning="a passage observation whose stance is neither + nor -"),
    IssueCode(ISSUE_SUBJECT_CANDIDATE, DROPPED, unclean=True,
              meaning="the member a member belief is about, as its candidate or proposal"),

    IssueCode(ISSUE_BELIEF_REPEATED, DROPPED, gate=True, meaning="a second item of one belief in the reply"),
    IssueCode(ISSUE_UPDATE_WITHOUT_OBSERVATION, DROPPED, unclean=True,
              meaning="a belief item of a belief with no passage observation in the call"),
    IssueCode(ISSUE_UPDATE_ON_AGGREGATE, DROPPED, unclean=True,
              meaning="a belief item on a belief computed from other beliefs (it takes none)"),
    IssueCode(ISSUE_BELIEF_MISSING, FLAGGED, unclean=True,
              meaning="a belief with a new passage observation but no belief item: its WEP is kept"),

    IssueCode(ISSUE_CANDIDATE_NOT_IN_PASSAGES, FLAGGED,
              meaning="a new candidate in no span or title of the belief's passages"),
    IssueCode(ISSUE_CANDIDATE_NEAR_DUPLICATE, FLAGGED,
              meaning="a new candidate whose name contains a listed one, or the reverse"),
    IssueCode(ISSUE_FACT_MERGED, FLAGGED,
              meaning="a new fact of an aspect whose words contain a listed one or the reverse: merged into it"),
    IssueCode(ISSUE_SPAN_OFF_BELIEF, FLAGGED,
              meaning="a - passage on a one-answer candidate whose spans share no content word with the belief"),

    IssueCode(ISSUE_MOST_PROBABLE_WRONG, CORRECTED, unclean=True,
              meaning="the most probable candidate written is no listed candidate with a + passage: "
                      "code's pick (the most + passages)"),
    IssueCode(ISSUE_WEP_INVALID, CORRECTED, gate=True,
              meaning="the WEP is no word of the scale: the previous WEP kept"),
    IssueCode(ISSUE_WEP_WITHOUT_SUPPORT, CORRECTED, unclean=True,
              meaning="a WEP above 'unknown' while no passage supports a candidate: set to 'unknown'"),
    IssueCode(ISSUE_UNKNOWN_WITH_SUPPORT, FLAGGED, unclean=True,
              meaning="'unknown' written while a passage supports a candidate: taken as written"),

    IssueCode(ISSUE_CANDIDATE_READDED, NOTE, meaning="a name the belief dropped before, written again"),
    IssueCode(ISSUE_SPAN_OFF_FACT, NOTE,
              meaning="a supporting span of an aspect's fact that shares no content word with it"),
)}

# A reply with one of these fails the format gate.
GATE_ISSUES: Tuple[str, ...] = tuple(c for c, i in ISSUES.items() if i.gate)
# A reply with one of these is not a clean sample.
UNCLEAN_REPLY: Tuple[str, ...] = tuple(c for c, i in ISSUES.items() if i.unclean)
NOTE_CODES: Tuple[str, ...] = tuple(c for c, i in ISSUES.items() if i.effect == NOTE)


def issue_effect(code: str) -> str:
    """The effect of *code* (an unknown code is ``flagged``)."""
    info = ISSUES.get(code)
    return info.effect if info is not None else FLAGGED
