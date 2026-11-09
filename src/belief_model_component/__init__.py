"""Belief model: a prompted belief state over each query's criteria, plus
per-step signals, for deep research agents.

Plugged into any agent with ``--belief-mode monitor`` (observe only) or
``inform`` (also inject a ``<belief_state>`` tag into the trajectory).  The
belief state b_t gives each criterion of a fixed per-query criteria list
one statement, in text: an Agent-BRACE belief status (how likely it is
covered), its selected candidate(s), the observations the status rests on
with their sources ("d<n>" a document, "t<step>" the agent's turn) and the
other candidates with why they lose or tie.  At the end of each search
iteration one call updates it (b_t = f(query, b_{t-1}, o_t)): it records
the spans of the agent's turn (think observations) and of the new
documents (document observations) that give information about each
criterion, kept as the criterion's references, then rewrites the
statements of the criteria with a new observation.  Only the statements
are shown (to the next update and to the agent); the observations and
references are saved.  The answer turn is one more update, and
the certainty score s_T is a fixed function of the final statuses.  For
evaluation only, it also saves the retrieval gain against the qrels and the
agent's intermediate answers.

Module layout:
    types         belief statuses, criterion kinds, Criterion, Observation, Statement,
                  WrittenStatement
    statement_text the fixed opening of a statement text and its source labels
    certainty     the certainty score s_t of a belief (status probabilities, geometric mean)
    criteria      CriteriaSource, LLMCriteriaSource, BankedCriteriaSource,
                  GoldCriteriaSource, FallbackCriteriaSource, BeliefState
                  (the statements, their trace of observations, the references)
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
    EXCLUSIVE_KINDS,
    NO_DOC_KINDS,
    exclusive,
    parse_status,
    status_probability,
    Criterion,
    Observation,
    Statement,
    WrittenStatement,
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
    "EXCLUSIVE_KINDS",
    "NO_DOC_KINDS",
    "exclusive",
    "parse_status",
    "status_probability",
    "Criterion",
    "Observation",
    "Statement",
    "WrittenStatement",
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
    "BeliefUpdateSignal",
    "RetrievalGainSignal",
    "IntermediateAnswerSignal",
    "format_belief",
    "render_belief_state",
    "strip_belief_state",
    "BeliefModel",
]
