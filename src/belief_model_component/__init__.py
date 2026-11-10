"""Belief model: a prompted belief state over each query's criteria, plus
per-step signals, for deep research agents.

Plugged into any agent with ``--belief-mode monitor`` (observe only) or
``inform`` (also inject a ``<belief_state>`` tag into the trajectory).  The
belief state b_t gives each criterion of a fixed per-query criteria list
its candidates (possible answers), each with an Agent-BRACE belief status
(how likely it is an answer) and its support (the passage observations
that support it or speak against it), and the names the agent proposed
that no passage supports yet.  A step with a new passage makes one call
(b_t = f(query, b_{t-1}, o_t)):
it records the claims of the agent's turns since the last call (reasoning
observations: they only propose or reject names) and the spans of the
passages that state a part of each criterion (passage observations), kept
as the criterion's references, then gives, with the passages that support
or speak against it and its reason, the status of every candidate they
change.  Code keeps every other candidate as it was and enforces the hard
limits (no support: at most "possible").  Other steps carry the belief
forward.  The candidates, their support and the proposals are shown to the
next update (the agent sees the candidates and proposals); the claims and
reasons are saved.  The certainty s_t of each step is a fixed function of
the statuses.  On a short-answer query, the final scorer reads b_{T-1}
and the agent's answer turn and judges how likely the answer is correct
(s_T, a word of estimative probability); on a report the answer turn is
one more update and s_T the certainty of the final belief.  For evaluation only,
it also saves the retrieval gain against the qrels and the agent's
intermediate answers.

Module layout:
    types         belief statuses, criterion kinds, Criterion, Observation, Support,
                  Candidate, Proposal, CandidateUpdate
    reason_text   the opening of a candidate update's reason and its source labels
    certainty     the certainty score s_t of a belief (status probabilities, geometric mean)
    final_score   LLMFinalScorer, build_final_scorer: s_T from b_{T-1} and the answer turn
    criteria      CriteriaSource, LLMCriteriaSource, BankedCriteriaSource,
                  GoldCriteriaSource, FallbackCriteriaSource, BeliefState
                  (the candidates, their trace of observations, the references)
    criteria_init the criteria source of a dataset and criteria_source (bank,
                  gold or silver): check_criteria_inputs, build_criteria_source,
                  check_max_criteria, prepare_criteria, belief_llm
    agent_turn    the agent's turn as the belief update reads it
    updater       LLMBeliefUpdater, build_belief_updater
    signals       BeliefUpdateSignal, RetrievalGainSignal,
                  IntermediateAnswerSignal
    belief_state  format_belief (the update prompt's view of
                  b_{t-1}), render_belief_state, strip_belief_state (the inform tag)
    belief_model  BeliefModel
"""

from .types import (
    BELIEF_STATUSES,
    STATUS_LEVEL,
    UNKNOWN,
    REMOVED,
    EXCLUSIVE_KINDS,
    NO_DOC_KINDS,
    exclusive,
    parse_status,
    status_probability,
    Criterion,
    Observation,
    Candidate,
    CandidateUpdate,
    Proposal,
    RemovedCandidate,
    Support,
)
from .criteria import (
    CriteriaSource,
    LLMCriteriaSource,
    BankedCriteriaSource,
    FallbackCriteriaSource,
    GoldCriteriaSource,
    BeliefState,
)
from .criteria_init import (
    belief_llm,
    build_criteria_source,
    check_criteria_inputs,
    check_max_criteria,
    prepare_criteria,
)
from .updater import LLMBeliefUpdater, build_belief_updater
from .final_score import FINAL_STATUSES, FinalScore, LLMFinalScorer, build_final_scorer, parse_final_score
from .signals import (
    BeliefUpdateSignal,
    RetrievalGainSignal,
    IntermediateAnswerSignal,
)
from .belief_state import format_belief, render_belief_state, strip_belief_state
from .belief_model import BeliefModel

__all__ = [
    "BELIEF_STATUSES",
    "STATUS_LEVEL",
    "UNKNOWN",
    "REMOVED",
    "EXCLUSIVE_KINDS",
    "NO_DOC_KINDS",
    "exclusive",
    "parse_status",
    "status_probability",
    "Criterion",
    "Observation",
    "Candidate",
    "CandidateUpdate",
    "Proposal",
    "RemovedCandidate",
    "Support",
    "CriteriaSource",
    "LLMCriteriaSource",
    "BankedCriteriaSource",
    "FallbackCriteriaSource",
    "GoldCriteriaSource",
    "BeliefState",
    "belief_llm",
    "build_criteria_source",
    "check_criteria_inputs",
    "check_max_criteria",
    "prepare_criteria",
    "LLMBeliefUpdater",
    "build_belief_updater",
    "FINAL_STATUSES",
    "FinalScore",
    "LLMFinalScorer",
    "build_final_scorer",
    "parse_final_score",
    "BeliefUpdateSignal",
    "RetrievalGainSignal",
    "IntermediateAnswerSignal",
    "format_belief",
    "render_belief_state",
    "strip_belief_state",
    "BeliefModel",
]
