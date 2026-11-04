"""Belief model: a prompted belief state over each query's criteria, plus
per-step signals, for deep research agents.

Plugged into any agent with ``--belief-mode monitor`` (observe only) or
``inform`` (also inject a ``<belief_state>`` tag into the trajectory).  The
belief state b_t gives each criterion of a fixed per-query criteria list
its nuggets: the answers the documents and the agent's turns give for it,
each with an Agent-BRACE belief status (how likely it is correct) and the
sources it rests on ("d<n>" a document, "t<step>" the agent's turn).  At
the end of each search iteration one call updates it (b_t = f(query,
b_{t-1}, o_t)): it records what the agent's turn states (think
observations) and what the new documents state (document observations),
then rewrites the nuggets of the criteria with new observations.  For
evaluation only, it also saves the retrieval gain against the qrels and the
agent's intermediate answers.

Module layout:
    types         belief statuses, criterion kinds, Criterion, Observation, Evidence, Nugget,
                  WrittenNugget, NuggetUpdate
    certainty     the certainty score s_t of a belief (criterion scores, geometric mean)
    criteria      CriteriaSource, LLMCriteriaSource, BankedCriteriaSource,
                  GoldCriteriaSource, FallbackCriteriaSource, BeliefState
                  (nuggets and their evidence)
    criteria_init the criteria source of a dataset and criteria_source (bank,
                  gold or silver): check_criteria_inputs, build_criteria_source,
                  check_max_criteria, prepare_criteria, belief_llm
    agent_turn    the agent's turn as the belief update reads it
    updater       LLMBeliefUpdater, build_belief_updater
    signals       BeliefUpdateSignal, RetrievalGainSignal,
                  IntermediateAnswerSignal
    belief_state  nugget sentences, format_belief (the update prompt's view of
                  b_{t-1}), render_belief_state, strip_belief_state (the inform tag)
    belief_model  BeliefModel
"""

from .types import (
    BELIEF_STATUSES,
    STATUS_LEVEL,
    MAX_NUGGETS,
    EXCLUSIVE_KINDS,
    NO_DOC_KINDS,
    exclusive,
    parse_status,
    status_probability,
    Criterion,
    Observation,
    Evidence,
    Nugget,
    WrittenNugget,
    NuggetUpdate,
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
from .belief_state import format_belief, nugget_sentence, render_belief_state, strip_belief_state
from .belief_model import BeliefModel

__all__ = [
    "BELIEF_STATUSES",
    "STATUS_LEVEL",
    "MAX_NUGGETS",
    "EXCLUSIVE_KINDS",
    "NO_DOC_KINDS",
    "exclusive",
    "parse_status",
    "status_probability",
    "Criterion",
    "Observation",
    "Evidence",
    "Nugget",
    "WrittenNugget",
    "NuggetUpdate",
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
    "nugget_sentence",
    "render_belief_state",
    "strip_belief_state",
    "BeliefModel",
]
