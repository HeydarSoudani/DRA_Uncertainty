"""Trajectory statistics for deep-research agents.

Accepts the **unified** result format produced by all agents:

    {
        query_id: {
            "trajectory": List[Dict]  – per-step trace
            "num_steps":  int
            "num_searches": int
        }
    }

Each trajectory step is a dict whose schema varies by agent type:

    ReAct / ReAct-WoPlan (action_type field):
        {"action_type": "search", "search_query": str, "docs": List[Dict], "think": str}
        {"action_type": "finish", "conclusion": str, "think": str}

    SearchR1 / ReSearch / StepSearch / SelfAsk (inferred from keys):
        {"think": str, "search_query": str, "docs": List[Dict]}
        {"think": str, "prediction": str}

    AgentCPM (action field):
        {"step": int, "state": str, "action": str,
         "input": {...}, "output": {...}}

The evaluator normalises all these formats into a common ``action_type``
string and then computes aggregate statistics.
"""

from collections import defaultdict
from typing import Any, Dict, List, Optional

from utils.trajectory_logger import (
    infer_action_type as _infer_action_type,
    seen_doc_ids as _seen_doc_ids,
)

from ..common import summary_stats


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_num_docs(step: Dict[str, Any]) -> Optional[int]:
    """How many documents the retriever returned for a step.

    ``None`` when the step carries no surfaced ranking to count, the normal
    case for a trajectory read back from disk: the logger keeps only the seen
    ids and writes the full ranking to ``retrieval/surfaced/{qid}.trec``, whose
    per-step lists the evaluator counts instead (``surfaced_docs_iterations``).
    Returning 0 would report "the retriever found nothing"; the count is
    unknown, not zero.
    """
    # Flat doc list (most agents)
    docs = step.get("docs")
    if docs:
        return len(docs)

    # all_docs: can be flat list of dicts (OSS/GLM/Tongyi) or list-of-lists
    # (CPM/ReactWithPlan).  Detect format by checking the first element.
    all_docs = step.get("all_docs")
    if all_docs:
        if isinstance(all_docs[0], dict):
            # Flat list of doc dicts
            return len(all_docs)
        else:
            # List of lists — sum inner lengths
            return sum(len(d) for d in all_docs if d)

    # AgentCPM: output.num_results
    output = step.get("output", {})
    if isinstance(output, dict) and "num_results" in output:
        return int(output["num_results"])

    return None


def _get_num_seen_docs(step: Dict[str, Any]) -> int:
    """How many of a search step's documents actually reached the model.

    ``_get_num_docs`` counts what the retriever *returned*; this counts the
    top-k slice injected into the prompt (the agent's ``seen_top_k``).  Reading
    the first as "docs the agent read" overstates it by the ranking depth --
    100 surfaced against 5 seen is typical here.
    """
    return len(_seen_doc_ids(step))


def _is_search_step(action_type: str, step: Dict[str, Any]) -> bool:
    """Return True if this step involved a retrieval call."""
    if action_type == "search":
        return True
    # Some agents may not set action_type but do have docs
    if step.get("docs") or step.get("all_docs"):
        return True
    return False


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

class TrajectoryEvaluator:
    """Evaluate agent trajectories from the unified result format.

    Usage::

        evaluator = TrajectoryEvaluator()
        metrics = evaluator.evaluate(results)
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(self, results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Compute trajectory statistics aggregated over all queries.

        Always returns:
        - ``num_queries``
        - ``steps`` – mean/std/min/max for total steps per query
        - ``search_steps`` – mean/std/min/max for search steps per query
        - ``docs_surfaced_per_search`` – mean/std/min/max docs the retriever
          *returned* per search step (the full ranked list): from the steps,
          else from the per-step lists of ``retrieval/surfaced/{qid}.trec``
          (``surfaced_docs_iterations``, a run read back from disk); ``None``
          when neither is there
        - ``docs_seen_per_search`` – how many of those actually reached the
          model, i.e. the agent's ``seen_top_k``.  The two differ by an order
          of magnitude for most agents here, so reporting only the first
          overstates what the model read.
        - ``action_type_counts`` – avg count of each action type per query
        - ``queries_no_search`` – number of queries with zero search steps
        - ``queries_max_iter`` – number of queries whose trajectory ended without
          a finish/predict action (possible truncation)

        Args:
            results: Unified agent results keyed by *query_id*.

        Returns:
            Nested metrics dict.
        """
        if not results:
            return {}

        # Every action type an agent here ends a query on.  A type missing from
        # this set is read as "never terminated", which inflates both
        # ``queries_max_iter_reached`` and ``queries_with_force_answer`` for an
        # agent that in fact finished cleanly -- so it has to track the agents.
        _TERMINAL_ACTIONS = {"finish", "predict", "done", "answer", "terminate"}
        _FORCE_ANSWER_ACTIONS = {"context_limit", "max_iter_force"}

        total_steps_per_query: List[int] = []
        search_steps_per_query: List[int] = []
        docs_per_search_all: List[int] = []
        docs_seen_per_search_all: List[int] = []
        action_type_counts_per_query: List[Dict[str, int]] = []
        queries_no_search = 0
        queries_with_force_answer = 0
        queries_max_iter = 0

        for query_id, result in results.items():
            trajectory = result.get("trajectory", [])

            if not trajectory:
                n_steps = result.get("num_steps", 0)
                n_searches = result.get("num_searches", 0)
                total_steps_per_query.append(n_steps)
                search_steps_per_query.append(n_searches)
                if n_searches == 0:
                    queries_no_search += 1
                action_type_counts_per_query.append({})
                continue

            step_action_counts: Dict[str, int] = defaultdict(int)
            n_search = 0
            has_terminal = False
            has_force_answer = False
            surfaced_counts: List[int] = []

            for step in trajectory:
                atype = _infer_action_type(step)
                step_action_counts[atype] += 1

                if _is_search_step(atype, step):
                    n_search += 1
                    n_docs = _get_num_docs(step)
                    if n_docs is not None:
                        surfaced_counts.append(n_docs)
                    docs_seen_per_search_all.append(_get_num_seen_docs(step))

                if atype in _TERMINAL_ACTIONS:
                    has_terminal = True
                if atype in _FORCE_ANSWER_ACTIONS:
                    has_force_answer = True

            if not surfaced_counts:
                surfaced_counts = [len(docs) for docs in result.get("surfaced_docs_iterations") or []]
            docs_per_search_all.extend(surfaced_counts)

            total_steps_per_query.append(len(trajectory))
            search_steps_per_query.append(n_search)
            action_type_counts_per_query.append(dict(step_action_counts))

            if n_search == 0:
                queries_no_search += 1
            if has_force_answer or not has_terminal:
                queries_with_force_answer += 1
            if not has_terminal:
                queries_max_iter += 1

        # Aggregate action type counts (mean per query)
        all_action_types = set(
            atype
            for counts in action_type_counts_per_query
            for atype in counts
        )
        avg_action_type_counts: Dict[str, float] = {}
        n_queries = len(results)
        for atype in sorted(all_action_types):
            total = sum(c.get(atype, 0) for c in action_type_counts_per_query)
            avg_action_type_counts[atype] = round(total / n_queries, 3)

        metrics: Dict[str, Any] = {
            "num_queries": n_queries,
            "steps": summary_stats(total_steps_per_query),
            "search_steps": summary_stats(search_steps_per_query),
            # None, not zeros: "no ranking recorded" and "the retriever
            # returned nothing" are different claims.
            "docs_surfaced_per_search": (
                summary_stats(docs_per_search_all) if docs_per_search_all else None),
            "docs_seen_per_search": summary_stats(docs_seen_per_search_all),
            "avg_action_type_counts": avg_action_type_counts,
            "queries_no_search": queries_no_search,
            "queries_with_force_answer": queries_with_force_answer,
            "queries_max_iter_reached": queries_max_iter,
        }

        # Token usage (per-query totals attached by the agents as result["token_usage"])
        token_usages = [
            result["token_usage"]
            for result in results.values()
            if result.get("token_usage")
        ]
        if token_usages:
            def _tok(key: str) -> List[float]:
                return [float(tu.get(key, 0) or 0) for tu in token_usages]
            metrics["tokens"] = {
                "num_queries_with_tokens": len(token_usages),
                "input_tokens": summary_stats(_tok("input_tokens")),
                "output_tokens": summary_stats(_tok("output_tokens")),
                "total_tokens": summary_stats(_tok("total_tokens")),
                "llm_calls": summary_stats(_tok("num_calls")),
            }

        return metrics

