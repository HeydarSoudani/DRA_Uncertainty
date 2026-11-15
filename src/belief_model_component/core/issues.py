"""The issue codes of one belief update (``BeliefState.apply``) and of the
final scorer (``update.final_score.check_issues``), in one table
(``ISSUES``), and the sets of them that training reads (``GATE_ISSUES``,
``UNCLEAN_REPLY``, ``REJECTED_OBSERVATION``), derived from it.

Each code has:

- ``concerns``: what it is about: an ``observation``, a candidate
  ``update`` (its status, name or removal), the ``reason`` or labels of an
  update, a ``status`` code set, the ``belief`` after the reply, or the
  final scorer's ``answer``;
- ``effect``: what code did: ``dropped`` (the observation or update is
  ignored), ``corrected`` (code kept or set another status, or took the
  update otherwise than written), ``flagged`` (taken as written, with a
  flaw) or ``note`` (not an error: information only);
- ``gate``: a reply with it fails the format gate of training (it names
  what the call did not show, or writes an update code cannot take);
- ``unclean``: a reply with it is not an SFT sample.

An issue record is ``{id, code, detail, [label]}``.  A ``note`` code goes
to the update's ``notes``, the others to its ``issues``.
"""

from dataclasses import dataclass
from typing import Dict, Tuple

OBSERVATION, UPDATE, REASON, STATUS, BELIEF, ANSWER = "observation", "update", "reason", "status", "belief", "answer"
DROPPED, CORRECTED, FLAGGED, NOTE = "dropped", "corrected", "flagged", "note"
# The effects in the order a log shows them.
EFFECTS = (DROPPED, CORRECTED, FLAGGED, NOTE)
# The effects of an error (a note is none).
ERROR_EFFECTS = (DROPPED, CORRECTED, FLAGGED)


@dataclass(frozen=True)
class IssueCode:
    code: str
    concerns: str
    effect: str
    gate: bool = False
    unclean: bool = False
    meaning: str = ""


# An observation that did not enter the belief.
ISSUE_UNKNOWN_CRITERION = "unknown_criterion"
ISSUE_UNKNOWN_PASSAGE = "unknown_passage"
ISSUE_OBSERVATION_ON_TARGET = "observation_on_target"
ISSUE_SPAN_UNVERIFIED = "span_unverified"
ISSUE_TURN_SPAN_UNVERIFIED = "turn_span_unverified"
ISSUE_CLAIM_INCOMPLETE = "claim_incomplete"
# A candidate update that is not taken, or taken otherwise than written.
ISSUE_CANDIDATE_MISSING = "candidate_missing"
ISSUE_STATUS_INVALID = "status_invalid"
ISSUE_FACT_STATUS_INVALID = "fact_status_invalid"
ISSUE_ANSWER_NOT_COMPUTED = "answer_not_computed"
ISSUE_CANDIDATE_REPEATED = "candidate_repeated"
ISSUE_UPDATE_WITHOUT_OBSERVATION = "update_without_observation"
ISSUE_UPDATE_ON_TARGET = "update_on_target"
ISSUE_REMOVAL_UNLISTED = "removal_unlisted"
ISSUE_REMOVAL_UNSUPPORTED = "removal_unsupported"
ISSUE_AGAINST_ONLY = "against_only"
ISSUE_SUBJECT_CANDIDATE = "subject_candidate"
ISSUE_ASPECT_STATUS_INVALID = "aspect_status_invalid"
# The reason or labels of a taken update, or its name.
ISSUE_REASON_OPENING_MISSING = "reason_opening_missing"
ISSUE_REASON_PREVIOUS_WRONG = "reason_previous_wrong"
ISSUE_REASON_NEW_ON_LISTED = "reason_new_on_listed"
ISSUE_REF_WITHOUT_OBSERVATION = "ref_without_observation"
ISSUE_CANDIDATE_NOT_IN_PASSAGES = "candidate_not_in_passages"
ISSUE_CANDIDATE_NEAR_DUPLICATE = "candidate_near_duplicate"
ISSUE_FACT_MERGED = "fact_merged"
# A status code corrected (the hard limits).
ISSUE_STATUS_WITHOUT_SUPPORT = "status_without_support"
ISSUE_STATUS_WITHOUT_NEW_SUPPORT = "status_without_new_support"
ISSUE_ASPECT_WITHOUT_FACT = "aspect_without_fact"
ISSUE_CONFIRMED_ONE_SOURCE = "confirmed_one_source"
ISSUE_AGGREGATE_STATUS_DERIVED = "aggregate_status_derived"
# The belief after the reply (the statuses are kept).
ISSUE_CONFIRMED_WITH_RIVAL = "confirmed_with_rival"
ISSUE_UNEQUAL_TIE = "unequal_tie"
# Notes (not errors).
ISSUE_CANDIDATE_READDED = "candidate_readded"
ISSUE_UPDATE_UNCHANGED = "update_unchanged"
ISSUE_SPAN_OFF_FACT = "span_off_fact"
# The final scorer's answer candidates (its status is kept).
ISSUE_ANSWER_CANDIDATE_MISSING = "answer_candidate_missing"
ISSUE_ANSWER_UNKNOWN_CRITERION = "answer_unknown_criterion"
ISSUE_ANSWER_CANDIDATE_UNLISTED = "answer_candidate_unlisted"

ISSUES: Dict[str, IssueCode] = {i.code: i for i in (
    IssueCode(ISSUE_UNKNOWN_CRITERION, OBSERVATION, DROPPED, gate=True,
              meaning="an observation or update of a criterion id the belief does not have"),
    IssueCode(ISSUE_UNKNOWN_PASSAGE, OBSERVATION, DROPPED, gate=True,
              meaning="a passage label that is no passage of the call"),
    IssueCode(ISSUE_OBSERVATION_ON_TARGET, OBSERVATION, DROPPED, unclean=True,
              meaning="an observation on a target criterion (code derives it from the clues)"),
    IssueCode(ISSUE_SPAN_UNVERIFIED, OBSERVATION, DROPPED, unclean=True,
              meaning="a passage observation without a span of the passage as shown"),
    IssueCode(ISSUE_TURN_SPAN_UNVERIFIED, OBSERVATION, DROPPED, unclean=True,
              meaning="a reasoning observation without a span of the agent's turn"),
    IssueCode(ISSUE_CLAIM_INCOMPLETE, OBSERVATION, DROPPED, gate=True, unclean=True,
              meaning="a reasoning observation without a candidate, or whose claim is no status nor 'rejected'"),

    IssueCode(ISSUE_CANDIDATE_MISSING, UPDATE, DROPPED, gate=True, meaning="an update without a candidate name"),
    IssueCode(ISSUE_STATUS_INVALID, UPDATE, DROPPED, gate=True,
              meaning="no status of the scale nor 'removed', or 'unknown' on a listed or cited name"),
    IssueCode(ISSUE_FACT_STATUS_INVALID, UPDATE, CORRECTED, unclean=True,
              meaning="an aspect's fact neither 'kept' nor 'removed': taken as kept"),
    IssueCode(ISSUE_ANSWER_NOT_COMPUTED, UPDATE, DROPPED, unclean=True,
              meaning="a new name on a computed aggregate that no claim computes"),
    IssueCode(ISSUE_CANDIDATE_REPEATED, UPDATE, DROPPED, gate=True,
              meaning="a second update of one candidate in the reply"),
    IssueCode(ISSUE_UPDATE_WITHOUT_OBSERVATION, UPDATE, DROPPED, unclean=True,
              meaning="an update of a criterion with no observation in the call"),
    IssueCode(ISSUE_UPDATE_ON_TARGET, UPDATE, DROPPED, unclean=True,
              meaning="an update on a target criterion (code derives it)"),
    IssueCode(ISSUE_REMOVAL_UNLISTED, UPDATE, DROPPED, unclean=True,
              meaning="'removed' on a name that is neither listed nor proposed"),
    IssueCode(ISSUE_REMOVAL_UNSUPPORTED, UPDATE, CORRECTED, unclean=True,
              meaning="'removed' without a passage that speaks against it: kept"),
    IssueCode(ISSUE_AGAINST_ONLY, UPDATE, DROPPED, unclean=True,
              meaning="a status for a name only passages speak against: not added (a proposal stays one)"),
    IssueCode(ISSUE_SUBJECT_CANDIDATE, UPDATE, DROPPED, unclean=True,
              meaning="the member a member criterion is about, as its candidate or proposal"),
    IssueCode(ISSUE_ASPECT_STATUS_INVALID, UPDATE, DROPPED, unclean=True,
              meaning="an aspect's status item without a status of the scale, or a second one"),

    IssueCode(ISSUE_REASON_OPENING_MISSING, REASON, FLAGGED, unclean=True,
              meaning="the reason opens with none of 'Was <status>:', 'Was proposed:', 'New:'"),
    IssueCode(ISSUE_REASON_PREVIOUS_WRONG, REASON, FLAGGED, unclean=True,
              meaning="the opening names another previous status than the candidate's (or it is not listed)"),
    IssueCode(ISSUE_REASON_NEW_ON_LISTED, REASON, FLAGGED, meaning="'New:' on a listed candidate"),
    IssueCode(ISSUE_REF_WITHOUT_OBSERVATION, REASON, FLAGGED, unclean=True,
              meaning="a support label, or a source of the reason, with no observation on the criterion"),
    IssueCode(ISSUE_CANDIDATE_NOT_IN_PASSAGES, REASON, FLAGGED,
              meaning="a new candidate in no span or title of the criterion's passages"),
    IssueCode(ISSUE_CANDIDATE_NEAR_DUPLICATE, REASON, FLAGGED,
              meaning="a new candidate whose name contains a listed one, or the reverse"),
    IssueCode(ISSUE_FACT_MERGED, REASON, FLAGGED,
              meaning="a new fact of an aspect whose words contain a listed one or the reverse: merged into it"),

    IssueCode(ISSUE_STATUS_WITHOUT_SUPPORT, STATUS, CORRECTED, unclean=True,
              meaning="a new name no passage of the call supports, written above 'unknown': kept as proposed"),
    IssueCode(ISSUE_STATUS_WITHOUT_NEW_SUPPORT, STATUS, CORRECTED, unclean=True,
              meaning="a status change (a candidate's or an aspect's) without new support: kept"),
    IssueCode(ISSUE_ASPECT_WITHOUT_FACT, STATUS, CORRECTED, unclean=True,
              meaning="an aspect with no fact kept: 'unlikely' when a passage removed a fact, else 'unknown'"),
    IssueCode(ISSUE_CONFIRMED_ONE_SOURCE, STATUS, CORRECTED, unclean=True,
              meaning="'confirmed' (a candidate, or an aspect's fact) with fewer than 2 sources: 'almost certain'"),
    IssueCode(ISSUE_AGGREGATE_STATUS_DERIVED, STATUS, CORRECTED, unclean=True,
              meaning="an aggregate's status written other than code derives from its inputs: the derived one"),

    IssueCode(ISSUE_CONFIRMED_WITH_RIVAL, BELIEF, FLAGGED, unclean=True,
              meaning="a one-answer candidate 'confirmed' while another candidate has support"),
    IssueCode(ISSUE_UNEQUAL_TIE, BELIEF, FLAGGED, unclean=True,
              meaning="one-answer candidates with the same support but different statuses"),

    IssueCode(ISSUE_CANDIDATE_READDED, UPDATE, NOTE, meaning="a name the belief dropped before, written again"),
    IssueCode(ISSUE_UPDATE_UNCHANGED, UPDATE, NOTE,
              meaning="no new support and the status written is the current one"),
    IssueCode(ISSUE_SPAN_OFF_FACT, REASON, NOTE,
              meaning="a supporting span of an aspect's fact that shares no content word with it"),

    IssueCode(ISSUE_ANSWER_CANDIDATE_MISSING, ANSWER, FLAGGED, meaning="criteria without an answer's candidate"),
    IssueCode(ISSUE_ANSWER_UNKNOWN_CRITERION, ANSWER, FLAGGED,
              meaning="an answer's candidate of a criterion id the belief does not have"),
    IssueCode(ISSUE_ANSWER_CANDIDATE_UNLISTED, ANSWER, FLAGGED,
              meaning="an answer's candidate the criterion does not list (nor proposes, nor removed)"),
)}

# A reply with one of these fails the format gate of training.
GATE_ISSUES: Tuple[str, ...] = tuple(c for c, i in ISSUES.items() if i.gate)
# A reply with one of these is not used as an SFT sample (with
# ``require_verified_spans``).
UNCLEAN_REPLY: Tuple[str, ...] = tuple(c for c, i in ISSUES.items() if i.unclean)
# Observations that did not enter for a reason of their own (the reply
# should not have written them): the observation-side false positives of
# training's observation score.
REJECTED_OBSERVATION: Tuple[str, ...] = tuple(
    c for c, i in ISSUES.items() if i.concerns == OBSERVATION and i.effect == DROPPED and i.unclean)
NOTE_CODES: Tuple[str, ...] = tuple(c for c, i in ISSUES.items() if i.effect == NOTE)

# The codes of belief files before schema 51, by their current code (a log
# of an old run shows them under the current name and effect).
LEGACY_CODES: Dict[str, str] = {
    "not_new_passage": ISSUE_UNKNOWN_PASSAGE,
    "derived_criterion": ISSUE_OBSERVATION_ON_TARGET,
    "no_span": ISSUE_SPAN_UNVERIFIED,
    "not_in_turn": ISSUE_TURN_SPAN_UNVERIFIED,
    "bad_claim": ISSUE_CLAIM_INCOMPLETE,
    "empty_candidate": ISSUE_CANDIDATE_MISSING,
    "bad_status": ISSUE_STATUS_INVALID,
    "bad_fact_status": ISSUE_FACT_STATUS_INVALID,
    "not_computed": ISSUE_ANSWER_NOT_COMPUTED,
    "duplicate_candidate": ISSUE_CANDIDATE_REPEATED,
    "unexpected_candidate": ISSUE_UPDATE_WITHOUT_OBSERVATION,
    "derived_update": ISSUE_UPDATE_ON_TARGET,
    "remove_unlisted": ISSUE_REMOVAL_UNLISTED,
    "bad_removal": ISSUE_REMOVAL_UNSUPPORTED,
    "bad_aspect_status": ISSUE_ASPECT_STATUS_INVALID,
    "aspect_no_passage": ISSUE_STATUS_WITHOUT_NEW_SUPPORT,
    "no_reason_opening": ISSUE_REASON_OPENING_MISSING,
    "wrong_previous": ISSUE_REASON_PREVIOUS_WRONG,
    "not_listed": ISSUE_REASON_PREVIOUS_WRONG,
    "listed_as_new": ISSUE_REASON_NEW_ON_LISTED,
    "bad_ref": ISSUE_REF_WITHOUT_OBSERVATION,
    "unknown_candidate": ISSUE_CANDIDATE_NOT_IN_PASSAGES,
    "near_duplicate": ISSUE_CANDIDATE_NEAR_DUPLICATE,
    "merged_duplicate": ISSUE_FACT_MERGED,
    "no_support": ISSUE_STATUS_WITHOUT_SUPPORT,
    "no_new_support": ISSUE_STATUS_WITHOUT_NEW_SUPPORT,
    "aspect_no_fact": ISSUE_ASPECT_WITHOUT_FACT,
    "one_source": ISSUE_CONFIRMED_ONE_SOURCE,
    "aspect_one_source": ISSUE_CONFIRMED_ONE_SOURCE,
    "aggregate_status": ISSUE_AGGREGATE_STATUS_DERIVED,
    "certain_with_rival": ISSUE_CONFIRMED_WITH_RIVAL,
    "readded": ISSUE_CANDIDATE_READDED,
    "unchanged": ISSUE_UPDATE_UNCHANGED,
    "span_off_candidate": ISSUE_SPAN_OFF_FACT,
}


def current_code(code: str) -> str:
    """*code* under its current name (``LEGACY_CODES``)."""
    return LEGACY_CODES.get(code, code)


def issue_effect(code: str) -> str:
    """The effect of *code* (an old code by its current one; an unknown
    code is ``flagged``)."""
    info = ISSUES.get(current_code(code))
    return info.effect if info is not None else FLAGGED
