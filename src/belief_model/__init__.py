"""Belief model: a prompted belief state over each query's criteria, plus
per-step signals, for deep research agents.

Plugged into any agent with ``--belief-mode monitor`` (observe only) or
``inform`` (also inject a ``<belief_state>`` tag into the trajectory).  At
the end of each search iteration the belief updater rewrites the belief
state b_t, one ICD 203 word of estimative probability per criterion of a
fixed per-query criteria list (Agent-BRACE style: b_t = f(query, b_{t-1},
o_t)), and the model computes the report's per-step signals
x_t = (nu^D_t, Delta^D_t, nu^q_t, a_t), plus extra saved information
(retrieval gain against qrels, intermediate answers).

Module layout:
    types         WEP labels, Criterion, Evidence, CriterionUpdate
    criteria      CriteriaSource, LLMCriteriaSource, BankedCriteriaSource, BeliefState
    judges        LLMBeliefUpdater, LLMQueryScorer, build_belief_judges
    signals       DocNoveltySignal, QueryNoveltySignal, BeliefUpdateSignal,
                  CriteriaTargetingSignal, RetrievalGainSignal,
                  IntermediateAnswerSignal, encode_fn_from_retriever
    belief_state  render_belief_state, strip_belief_state (the inform tag)
    belief_model  BeliefModel
"""

from .types import (
    WEP_LABELS,
    WEP_LEVEL,
    WEP_PROBABILITY,
    INITIAL_LABEL,
    parse_label,
    Criterion,
    CriterionUpdate,
    Evidence,
)
from .criteria import (
    CriteriaSource,
    LLMCriteriaSource,
    BankedCriteriaSource,
    BeliefState,
)
from .judges import (
    LLMBeliefUpdater,
    LLMQueryScorer,
    build_belief_judges,
)
from .signals import (
    DocNoveltySignal,
    QueryNoveltySignal,
    BeliefUpdateSignal,
    CriteriaTargetingSignal,
    RetrievalGainSignal,
    IntermediateAnswerSignal,
    encode_fn_from_retriever,
)
from .belief_state import render_belief_state, strip_belief_state
from .belief_model import BeliefModel

__all__ = [
    "WEP_LABELS",
    "WEP_LEVEL",
    "WEP_PROBABILITY",
    "INITIAL_LABEL",
    "parse_label",
    "Criterion",
    "CriterionUpdate",
    "Evidence",
    "CriteriaSource",
    "LLMCriteriaSource",
    "BankedCriteriaSource",
    "BeliefState",
    "LLMBeliefUpdater",
    "LLMQueryScorer",
    "build_belief_judges",
    "DocNoveltySignal",
    "QueryNoveltySignal",
    "BeliefUpdateSignal",
    "CriteriaTargetingSignal",
    "RetrievalGainSignal",
    "IntermediateAnswerSignal",
    "encode_fn_from_retriever",
    "render_belief_state",
    "strip_belief_state",
    "BeliefModel",
]
