"""Belief model: a prompted belief state over each query's beliefs, plus
per-step signals, for deep research agents.

Run over saved agent trajectories by the belief stage of
``experiments/inference/dra_inference.py`` (``--stage belief``,
``orchestration.belief_stage``); it never changes what the agent does.  The
design is described in ``README.md``.

Layout:
    belief_model    BeliefModel, the per-sample entry point
    core/           types, issues, state (BeliefState), certainty
    update/         updater, signal (BeliefUpdateSignal), final_score, repair
    init/           sources, parsing, factory (the beliefs of a query)
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
    Belief,
    Observation,
    Support,
    exclusive,
    is_computed,
    is_target,
    parse_status,
    status_probability,
)
from .init.factory import (
    bank_model_id,
    belief_llm,
    belief_provider_pin,
    belief_vllm_api_base,
    build_belief_source,
    check_belief_inputs,
    check_max_beliefs,
    prepare_beliefs,
)
from .init.parsing import belief_kinds
from .init.sources import (
    BANK_ONLY,
    FINAL,
    GOLD_BELIEF_MODEL,
    BankedBeliefSource,
    BeliefSource,
    FallbackBeliefSource,
    LLMBeliefSource,
)
from .text.agent_turn import agent_turn_text, answer_reasoning, answer_turn, split_answer_turn
from .text.passages import Passage, passage_title, span_in_passage, split_span
from .update.final_score import (
    build_final_scorer,
    parse_final_score,
)
from .update.signal import BeliefUpdateSignal, CallPlan
from .update.updater import LLMBeliefUpdater, build_belief_updater
from .views.log_view import belief_lines, belief_md, init_lines, init_md
from .belief_model import BeliefModel

__all__ = [
    # core
    "AGGREGATE", "ASPECT", "BELIEF_STATUSES", "CLOSED", "MULTI_ASPECT",
    "STATUS_LEVEL", "UNKNOWN",
    "Candidate", "Belief", "Observation", "Support",
    "exclusive", "is_computed", "is_target", "parse_status", "status_probability",
    "EFFECTS", "ERROR_EFFECTS", "GATE_ISSUES", "ISSUES", "NOTE_CODES", "UNCLEAN_REPLY",
    "issue_effect",
    "BeliefState",
    "certainty", "counted",
    # beliefs
    "BANK_ONLY", "FINAL", "GOLD_BELIEF_MODEL",
    "BeliefSource", "LLMBeliefSource", "BankedBeliefSource", "FallbackBeliefSource",
    "belief_kinds",
    "bank_model_id", "belief_llm", "belief_provider_pin", "belief_vllm_api_base", "build_belief_source", "check_belief_inputs", "check_max_beliefs", "prepare_beliefs",
    # update
    "LLMBeliefUpdater", "build_belief_updater", "BeliefUpdateSignal", "CallPlan",
    "build_final_scorer", "parse_final_score",
    # text
    "Passage", "passage_title", "agent_turn_text", "answer_reasoning", "answer_turn", "split_answer_turn", "span_in_passage", "split_span",
    # views
    "belief_lines", "belief_md", "init_lines", "init_md",
    # entry point
    "BeliefModel",
]
