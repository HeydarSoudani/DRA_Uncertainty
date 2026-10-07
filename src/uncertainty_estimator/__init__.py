"""Uncertainty estimator: passive per-step uncertainty signals for deep research agents.

Plugged into any agent with ``--uncertainty-estimator-mode monitor`` (observe
only) or ``inform`` (also inject a ``<certainty>`` tag into the trajectory).  At the end of
each search iteration it computes the report's per-step signals
x_t = (nu^D_t, Delta^D_t, nu^q_t, a_t) against a fixed per-query criteria
list, plus extra saved information (retrieval gain against qrels, intermediate answers).

Module layout:
    types      criteria statuses, Criterion, Evidence, CriterionUpdate
    criteria   CriteriaSource, LLMCriteriaSource, BankedCriteriaSource, CriteriaState
    judges     LLMCoverageJudge, LLMQueryScorer, build_criteria_judges
    signals    DocNoveltySignal, QueryNoveltySignal, CriteriaCoverageSignal,
               CriteriaTargetingSignal, RetrievalGainSignal,
               IntermediateAnswerSignal, encode_fn_from_retriever
    certainty  render_certainty, strip_certainty (the inform tag)
    estimator  UncertaintyEstimator
"""

from .types import (
    UNCOVERED,
    PARTIALLY_COVERED,
    FULLY_COVERED,
    STATUS_VALUE,
    STATUSES,
    Criterion,
    CriterionUpdate,
    Evidence,
)
from .criteria import (
    CriteriaSource,
    LLMCriteriaSource,
    BankedCriteriaSource,
    CriteriaState,
)
from .judges import (
    LLMCoverageJudge,
    LLMQueryScorer,
    build_criteria_judges,
)
from .signals import (
    DocNoveltySignal,
    QueryNoveltySignal,
    CriteriaCoverageSignal,
    CriteriaTargetingSignal,
    RetrievalGainSignal,
    IntermediateAnswerSignal,
    encode_fn_from_retriever,
)
from .certainty import render_certainty, strip_certainty
from .estimator import UncertaintyEstimator

__all__ = [
    "UNCOVERED",
    "PARTIALLY_COVERED",
    "FULLY_COVERED",
    "STATUS_VALUE",
    "STATUSES",
    "Criterion",
    "CriterionUpdate",
    "Evidence",
    "CriteriaSource",
    "LLMCriteriaSource",
    "BankedCriteriaSource",
    "CriteriaState",
    "LLMCoverageJudge",
    "LLMQueryScorer",
    "build_criteria_judges",
    "DocNoveltySignal",
    "QueryNoveltySignal",
    "CriteriaCoverageSignal",
    "CriteriaTargetingSignal",
    "RetrievalGainSignal",
    "IntermediateAnswerSignal",
    "encode_fn_from_retriever",
    "render_certainty",
    "strip_certainty",
    "UncertaintyEstimator",
]
