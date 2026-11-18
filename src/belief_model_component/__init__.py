"""Belief model: a prompted belief state over each query's criteria, plus
per-step signals, for deep research agents.

Run over saved agent trajectories by the belief stage of
``experiments/inference/dra_inference.py`` (``--stage belief``,
``orchestration.belief_stage``); it never changes what the agent does.  The
design is described in ``README.md``.

Layout:
    belief_model    BeliefModel, the per-sample entry point
    core/           types, issues, state (BeliefState), certainty
    update/         updater, signal (BeliefUpdateSignal), final_score, repair
    criteria/       sources, parsing, factory (the criteria C of a query)
    text/           matching, passages, agent_turn (plain string helpers)
    views/          belief_text (the belief as prompt text), log_view
    eval_signals/   retrieval_gain (evaluation only)
    prompts/        system prompts and user templates

Code outside the package imports from here, never from a module path.
"""

from .core.certainty import certainty, counted
from .core.issues import EFFECTS, ERROR_EFFECTS, GATE_ISSUES, ISSUES, NOTE_CODES, UNCLEAN_REPLY, issue_effect
from .core.state import BeliefState
from .core.types import (
    AGGREGATE,
    ASPECT,
    BELIEF_STATUSES,
    CLOSED,
    MULTI_ASPECT,
    STATUS_LEVEL,
    UNKNOWN,
    Candidate,
    Criterion,
    Observation,
    Support,
    exclusive,
    is_computed,
    is_target,
    parse_status,
    status_probability,
)
from .criteria.factory import (
    belief_llm,
    belief_vllm_api_base,
    build_criteria_source,
    check_criteria_inputs,
    check_max_criteria,
    prepare_criteria,
)
from .criteria.parsing import criterion_kinds
from .criteria.sources import (
    BANK_ONLY,
    FINAL,
    GOLD_CRITERIA_MODEL,
    BankedCriteriaSource,
    CriteriaSource,
    FallbackCriteriaSource,
    LLMCriteriaSource,
)
from .text.agent_turn import agent_turn_text, answer_turn
from .text.passages import Passage, passage_title, span_in_passage, split_span
from .update.final_score import (
    build_final_scorer,
    parse_final_score,
)
from .update.signal import BeliefUpdateSignal, CallPlan
from .update.updater import LLMBeliefUpdater, build_belief_updater
from .views.log_view import belief_lines, belief_md, criteria_lines, criteria_md
from .belief_model import BeliefModel

__all__ = [
    # core
    "AGGREGATE", "ASPECT", "BELIEF_STATUSES", "CLOSED", "MULTI_ASPECT",
    "STATUS_LEVEL", "UNKNOWN",
    "Candidate", "Criterion", "Observation", "Support",
    "exclusive", "is_computed", "is_target", "parse_status", "status_probability",
    "EFFECTS", "ERROR_EFFECTS", "GATE_ISSUES", "ISSUES", "NOTE_CODES", "UNCLEAN_REPLY",
    "issue_effect",
    "BeliefState",
    "certainty", "counted",
    # criteria
    "BANK_ONLY", "FINAL", "GOLD_CRITERIA_MODEL",
    "CriteriaSource", "LLMCriteriaSource", "BankedCriteriaSource", "FallbackCriteriaSource",
    "criterion_kinds",
    "belief_llm", "belief_vllm_api_base", "build_criteria_source", "check_criteria_inputs", "check_max_criteria", "prepare_criteria",
    # update
    "LLMBeliefUpdater", "build_belief_updater", "BeliefUpdateSignal", "CallPlan",
    "build_final_scorer", "parse_final_score",
    # text
    "Passage", "passage_title", "agent_turn_text", "answer_turn", "span_in_passage", "split_span",
    # views
    "belief_lines", "belief_md", "criteria_lines", "criteria_md",
    # entry point
    "BeliefModel",
]
