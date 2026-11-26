"""The agent's trajectory: searches, tokens and how the query ended.

Reads the results loaded back from the saved files: ``trajectory`` (the
lines of ``trajectory/{qid}.jsonl``), ``num_searches`` and the agent's
``token_usage``.  A line's action type is the one it saved, else inferred
from its keys (``trajectory_logger.infer_action_type``), so every agent's
lines are counted the same way.

Per query:
    searches        search steps (from the steps, else the saved count)
    forced_answer   the answer was forced: a force-answer action, or no
                    terminal action
    max_iter        the iteration cap was reached: a ``max_iter_force``
                    action, or neither a terminal nor a force-answer action
                    (a ``context_limit`` ending is not one)
    agent_tokens    the agent's total tokens (None when it records none)

Rows (``trajectory`` group): ``searches`` and ``agent tokens`` (means),
``forced answers`` and ``max iterations reached`` (counts).  No LLM call.
"""

from typing import Any, Dict, Optional

from utils.run_files.trajectory_logger import infer_action_type as _infer_action_type

from ..shared.context import GENERATION, EvalContext, EvalSetup
from ..shared.measures import mean
from ..shared.result import GroupResult
from .metrics import AGENT_TOKENS, FORCED, GROUP, MAX_ITER, SEARCHES

#: Every action type an agent here ends a query on.  A type missing from this
#: set reads as "never terminated", which inflates both counts for an agent
#: that in fact finished cleanly, so it has to track the agents.  cpm_report
#: ends on its section writing (``write``), with no final action.
TERMINAL_ACTIONS = {"finish", "predict", "done", "answer", "terminate", "write"}
FORCE_ANSWER_ACTIONS = {"context_limit", "max_iter_force"}


def _is_search_step(action_type: str, step: Dict[str, Any]) -> bool:
    """A step with a retrieval call (an agent may set no action type but have docs)."""
    if action_type == "search":
        return True
    return bool(step.get("docs") or step.get("all_docs"))


def trajectory_stats(results: Dict[str, Dict[str, Any]]) -> GroupResult:
    """The rows and per-query values of the module docstring."""
    out: Dict[str, Dict[str, Any]] = {}
    for qid, result in results.items():
        trajectory = result.get("trajectory") or []
        if trajectory:
            actions = [_infer_action_type(step) for step in trajectory]
            searches = sum(_is_search_step(a, step) for a, step in zip(actions, trajectory))
            terminal = any(a in TERMINAL_ACTIONS for a in actions)
            force_actions = {a for a in actions if a in FORCE_ANSWER_ACTIONS}
            forced = bool(force_actions) or not terminal
            max_iter = "max_iter_force" in force_actions or not (terminal or force_actions)
        else:
            searches = int(result.get("num_searches") or 0)
            forced, max_iter = False, False
        usage = result.get("token_usage") or {}
        out[qid] = {
            "searches": searches,
            "forced_answer": forced,
            "max_iter": max_iter,
            "agent_tokens": usage.get("total_tokens") if usage else None,
        }
    res = GroupResult(per_query=out)
    res.add(GROUP, SEARCHES, mean([r["searches"] for r in out.values()]), n=len(out))
    agent_tokens = [r["agent_tokens"] for r in out.values() if r["agent_tokens"] is not None]
    res.add(GROUP, AGENT_TOKENS, mean(agent_tokens), n=len(agent_tokens))
    res.add(GROUP, FORCED, sum(r["forced_answer"] for r in out.values()), n=len(out))
    res.add(GROUP, MAX_ITER, sum(r["max_iter"] for r in out.values()), n=len(out))
    return res


class TrajectoryEvaluator:
    """:func:`trajectory_stats` of a run; every dataset."""

    name = "trajectory"
    scope = GENERATION
    provides = None

    @classmethod
    def from_setup(cls, setup: EvalSetup) -> Optional["TrajectoryEvaluator"]:
        return cls()

    def evaluate(self, ctx: EvalContext) -> GroupResult:
        return trajectory_stats(ctx.results)
