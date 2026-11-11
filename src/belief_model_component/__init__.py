"""Belief model: a prompted belief state over each query's criteria, plus
per-step signals, for deep research agents.

Plugged into any agent with ``--belief-mode monitor`` (observe only) or
``inform`` (also inject a ``<belief_state>`` tag into the trajectory).  The
design is described in ``README.md``.

Layout:
    belief_model    BeliefModel, the per-sample entry point
    core/           types, issues, reason_text, state (BeliefState), certainty
    update/         updater, signal (BeliefUpdateSignal), final_score, repair
    criteria/       sources, parsing, factory (the criteria C of a query)
    text/           matching, passages, agent_turn (plain string helpers)
    views/          tag (prompt view and <belief_state> tag), log_view
    eval_signals/   retrieval_gain, intermediate_answer (evaluation only)
    prompts/        system prompts and user templates

Code outside the package imports from here, never from a module path.
"""

from .core.certainty import AGGREGATORS, CERTAINTY_AGGREGATOR, answer_certainty, certainty, criterion_scores
from .core.issues import GATE_ISSUES, REJECTED_OBSERVATION, UNCLEAN_REPLY
from .core.state import BeliefState
from .core.types import (
    AGGREGATE,
    BELIEF_STATUSES,
    CLOSED,
    HOP,
    MULTI_ASPECT,
    NO_DOC_KINDS,
    REMOVED,
    REST,
    STATUS_LEVEL,
    UNKNOWN,
    Candidate,
    Criterion,
    Observation,
    Support,
    exclusive,
    status_probability,
)
from .criteria.factory import (
    belief_llm,
    build_criteria_source,
    check_criteria_inputs,
    check_max_criteria,
    prepare_criteria,
)
from .criteria.parsing import shape_kinds
from .criteria.sources import (
    BANK_ONLY,
    FINAL,
    GOLD_CRITERIA_LEVEL,
    GOLD_CRITERIA_MODEL,
    BankedCriteriaSource,
    CriteriaSource,
    FallbackCriteriaSource,
    LLMCriteriaSource,
)
from .eval_signals.intermediate_answer import IntermediateAnswerSignal
from .text.agent_turn import agent_turn_text, answer_turn
from .text.passages import Passage, span_in_passage, split_span
from .update.final_score import (
    build_final_scorer,
    check_issues,
    parse_final_score,
)
from .update.signal import BeliefUpdateSignal, CallPlan
from .update.updater import LLMBeliefUpdater, build_belief_updater
from .views.log_view import belief_lines, belief_md, criteria_lines, criteria_md
from .views.tag import strip_belief_state
from .belief_model import BeliefModel

__all__ = [
    # core
    "AGGREGATE", "BELIEF_STATUSES", "CLOSED", "HOP", "MULTI_ASPECT", "NO_DOC_KINDS",
    "REMOVED", "REST", "STATUS_LEVEL", "UNKNOWN",
    "Candidate", "Criterion", "Observation", "Support",
    "exclusive", "status_probability",
    "GATE_ISSUES", "REJECTED_OBSERVATION", "UNCLEAN_REPLY",
    "BeliefState",
    "AGGREGATORS", "CERTAINTY_AGGREGATOR", "answer_certainty", "certainty", "criterion_scores",
    # criteria
    "BANK_ONLY", "FINAL", "GOLD_CRITERIA_LEVEL", "GOLD_CRITERIA_MODEL",
    "CriteriaSource", "LLMCriteriaSource", "BankedCriteriaSource", "FallbackCriteriaSource",
    "shape_kinds",
    "belief_llm", "build_criteria_source", "check_criteria_inputs", "check_max_criteria", "prepare_criteria",
    # update
    "LLMBeliefUpdater", "build_belief_updater", "BeliefUpdateSignal", "CallPlan",
    "build_final_scorer", "check_issues", "parse_final_score",
    # text
    "Passage", "agent_turn_text", "answer_turn", "span_in_passage", "split_span",
    # views
    "strip_belief_state",
    "belief_lines", "belief_md", "criteria_lines", "criteria_md",
    # evaluation only
    "IntermediateAnswerSignal",
    # entry point
    "BeliefModel",
]
