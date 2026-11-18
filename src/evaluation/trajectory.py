"""Per-query trajectory statistics (the ``trajectory`` group of the run's
``metrics.md``).

Reads the results the runner loads back from the saved files
(``utils.run_files.load_result_from_saved_files``): ``trajectory`` (the
lines of ``trajectory/{qid}.jsonl``), ``num_searches`` and the agent's
``token_usage``.  A line's action type is the one it saved, else inferred
from its keys (``trajectory_logger.infer_action_type``: a ``search_query``
is a search, a ``prediction`` an answer, ...), so every agent's lines are
counted the same way.  The belief model's tokens are one belief model's
(:func:`belief_tokens`, once per belief model of the run).
"""

from typing import Any, Dict

from utils.run_files.trajectory_logger import infer_action_type as _infer_action_type

from .measures import mean
from .metrics_file import GroupResult

# Every action type an agent here ends a query on.  A type missing from
# this set is read as "never terminated", which inflates both counts below
# for an agent that in fact finished cleanly, so it has to track the agents.
# cpm_report ends on its section writing (``write``), with no final action.
_TERMINAL_ACTIONS = {"finish", "predict", "done", "answer", "terminate", "write"}
_FORCE_ANSWER_ACTIONS = {"context_limit", "max_iter_force"}


def _is_search_step(action_type: str, step: Dict[str, Any]) -> bool:
    """Return True if this step involved a retrieval call."""
    if action_type == "search":
        return True
    # Some agents may not set action_type but do have docs
    return bool(step.get("docs") or step.get("all_docs"))


class TrajectoryEvaluator:
    """Per-query trajectory statistics from the unified result format.

    Usage::

        evaluator = TrajectoryEvaluator()
        group = evaluator.evaluate(results)
    """

    def evaluate(self, results: Dict[str, Dict[str, Any]]) -> GroupResult:
        """The ``trajectory`` rows (means and counts) of, per query:

        - ``searches``: search steps (from the steps, else the saved count);
        - ``forced_answer``: the answer was forced (a force-answer action, or
          no terminal action);
        - ``max_iter``: the iteration cap was reached (a ``max_iter_force``
          action, or no terminal and no force-answer action; a
          ``context_limit`` ending is not one);
        - ``agent_tokens``: the agent's total tokens (``token_usage``; None
          when the agent records none).
        """
        out: Dict[str, Dict[str, Any]] = {}
        for qid, result in results.items():
            trajectory = result.get("trajectory") or []
            if trajectory:
                actions = [_infer_action_type(step) for step in trajectory]
                searches = sum(_is_search_step(a, step) for a, step in zip(actions, trajectory))
                terminal = any(a in _TERMINAL_ACTIONS for a in actions)
                force_actions = {a for a in actions if a in _FORCE_ANSWER_ACTIONS}
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
        res.add("trajectory", "searches", mean([r["searches"] for r in out.values()]), n=len(out))
        agent_tokens = [r["agent_tokens"] for r in out.values() if r["agent_tokens"] is not None]
        res.add("trajectory", "agent tokens", mean(agent_tokens), n=len(agent_tokens))
        res.add("trajectory", "forced answers", sum(r["forced_answer"] for r in out.values()), n=len(out))
        res.add("trajectory", "max iterations reached", sum(r["max_iter"] for r in out.values()), n=len(out))
        return res


def belief_tokens(files: Dict[str, Any]) -> GroupResult:
    """The ``belief tokens`` row of one belief model and its per-query
    ``belief_tokens``: the model's total tokens of each query of its belief
    *files* (``load_belief_files``)."""
    tokens = {q: (f[0].get("tokens") or {}).get("total_tokens") for q, f in files.items()}
    defined = [v for v in tokens.values() if v is not None]
    res = GroupResult(per_query={q: {"belief_tokens": v} for q, v in tokens.items()})
    if defined:
        res.add("trajectory", "belief tokens", mean(defined), n=len(defined))
    return res
