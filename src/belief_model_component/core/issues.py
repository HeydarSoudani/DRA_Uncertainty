"""The issue codes of one belief update (``BeliefState.apply``) and the
sets of them that training reads (``UNCLEAN_REPLY``, ``GATE_ISSUES``)."""

# Issues of one belief update (``BeliefState.apply``): one record
# ``{id, code, detail, [label]}`` per thing the reply got wrong.
# An observation that did not enter the belief:
ISSUE_UNKNOWN_CRITERION = "unknown_criterion"      # also a candidate update of an unknown criterion
ISSUE_NOT_NEW_PASSAGE = "not_new_passage"          # its label is no passage of the call
ISSUE_COMPUTED = "computed_criterion"              # a passage observation on a computed criterion
ISSUE_NO_SPAN = "no_span"                          # a passage observation without a verified span
ISSUE_NOT_IN_TURN = "not_in_turn"                  # a reasoning observation without a span of its turn
ISSUE_BAD_CLAIM = "bad_claim"                      # a reasoning observation without a candidate or a valid claim
# A candidate update that is not taken:
ISSUE_EMPTY_CANDIDATE = "empty_candidate"          # no candidate name
ISSUE_BAD_STATUS = "bad_status"                    # no status of the scale nor "removed", or "unknown"
ISSUE_DUPLICATE_CANDIDATE = "duplicate_candidate"  # a second update of one candidate in the reply
ISSUE_UNEXPECTED_CANDIDATE = "unexpected_candidate"  # a criterion without a new observation
ISSUE_REMOVE_UNLISTED = "remove_unlisted"          # "removed" on a name that is neither listed nor proposed
ISSUE_BAD_REMOVAL = "bad_removal"                  # "removed" without a passage that speaks against it: kept
ISSUE_AGAINST_ONLY = "against_only"                # a status for a name only passages against: not added (a proposal stays one)
# A candidate update that is taken, with a flaw in its reason, its labels or its name:
ISSUE_NO_REASON = "no_reason_opening"              # the reason opens with none of "Was <status>:", "Was proposed:", "New:"
ISSUE_WRONG_PREVIOUS = "wrong_previous"            # "Was <X>:" with X not the candidate's current status (or not proposed)
ISSUE_NOT_LISTED = "not_listed"                    # "Was <X>:" on a candidate that is not listed (added)
ISSUE_LISTED_AS_NEW = "listed_as_new"              # "New:" on a listed candidate (its status updated)
ISSUE_BAD_REF = "bad_ref"                          # a support label, or a source of the reason, with no observation on the criterion
ISSUE_UNKNOWN_CANDIDATE = "unknown_candidate"      # a new candidate in no span or title of the criterion's passages
ISSUE_NEAR_DUPLICATE = "near_duplicate"            # a new candidate whose name contains a listed one, or the reverse
ISSUE_UNCHANGED = "unchanged"                      # no new support and the status written is the current one
ISSUE_READDED = "readded"                          # a name the belief dropped before (information only)
# A status code corrected (the hard limits):
ISSUE_NO_SUPPORT = "no_support"                    # no passage supports the candidate: at most "possible", or a new name kept as proposed
ISSUE_NO_NEW_SUPPORT = "no_new_support"            # a status change without new support (nor, one answer, another candidate's support changed): kept
# The belief after the reply (flags; the statuses are kept):
ISSUE_CERTAIN_WITH_RIVAL = "certain_with_rival"    # a one-answer candidate "certain" while another candidate has support
ISSUE_UNEQUAL_TIE = "unequal_tie"                  # one-answer candidates with the same support but different statuses

# Observations that did not enter for a reason of their own (the reply
# should not have written them).
REJECTED_OBSERVATION = (ISSUE_COMPUTED, ISSUE_NO_SPAN, ISSUE_NOT_IN_TURN, ISSUE_BAD_CLAIM)
# A candidate update whose status code corrected (the reply's status is
# not the one kept).
CORRECTED_STATUS = (ISSUE_NO_SUPPORT, ISSUE_NO_NEW_SUPPORT)
# A reply with one of these is not used as an SFT sample (with
# ``require_verified_spans``): an observation left out for a reason of its
# own, a status code corrected, a reason or label that misreads the belief
# or cites a source with no observation on the criterion, a removal code
# did not take, a candidate update code ignored (no observation on its
# criterion in the call, or only passages against a name not listed), or
# a one-answer criterion whose statuses do not follow its support
# (``certain_with_rival``, ``unequal_tie``).  ``unknown_candidate`` is not
# among them: a candidate the reply writes in other words than its spans
# (an initial, a computed value) raises it too; nor are
# ``near_duplicate``, ``listed_as_new``, ``unchanged`` and ``readded``
# (harmless for the belief).
UNCLEAN_REPLY = (*REJECTED_OBSERVATION, *CORRECTED_STATUS, ISSUE_NO_REASON, ISSUE_WRONG_PREVIOUS,
                 ISSUE_NOT_LISTED, ISSUE_BAD_REF, ISSUE_REMOVE_UNLISTED, ISSUE_BAD_REMOVAL,
                 ISSUE_UNEXPECTED_CANDIDATE, ISSUE_AGAINST_ONLY, ISSUE_CERTAIN_WITH_RIVAL, ISSUE_UNEQUAL_TIE)
# A reply with one of these fails the format gate of training: it names
# what the call did not show, or writes an update code cannot take.
GATE_ISSUES = (ISSUE_UNKNOWN_CRITERION, ISSUE_NOT_NEW_PASSAGE, ISSUE_EMPTY_CANDIDATE, ISSUE_BAD_STATUS,
               ISSUE_DUPLICATE_CANDIDATE, ISSUE_BAD_CLAIM)
