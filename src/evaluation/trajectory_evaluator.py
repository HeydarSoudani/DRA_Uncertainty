"""Trajectory evaluator for deep-research agents.

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

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from utils.trajectory_logger import (
    build_meta_line as _build_meta_line,
    dump_line as _dump_line,
    infer_action_type as _infer_action_type,
    iter_label as _iter_label,
    seen_doc_ids as _seen_doc_ids,
    step_to_line as _step_to_line,
)


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_num_docs(step: Dict[str, Any]) -> Optional[int]:
    """How many documents the retriever returned for a step.

    ``None`` when the step carries no surfaced ranking to count -- which is the
    normal case for a trajectory reloaded from disk, since the logger keeps only
    the seen ids and writes the full ranking to ``retrieval/surfaced/{qid}.trec``
    instead.  Returning 0 there would report "the retriever found nothing" for
    every ``--eval-only`` pass; the count is unknown, not zero.
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
        evaluator.print_results(metrics)
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
          *returned* per search step (the full ranked list), or ``None`` when the
          trajectory carried no ranking to count.  That is the normal case for a
          trajectory reloaded from disk, which keeps only the seen ids — the
          surfaced ranking lives in ``retrieval/surfaced/{qid}.trec``
        - ``docs_seen_per_search`` – how many of those actually reached the
          model, i.e. the agent's ``seen_top_k``.  The two differ by an order
          of magnitude for most agents here, so reporting only the first
          overstates what the model read.  ``docs_per_search`` is kept as an
          alias of the surfaced count for existing readers.
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

            for step in trajectory:
                atype = _infer_action_type(step)
                step_action_counts[atype] += 1

                if _is_search_step(atype, step):
                    n_search += 1
                    n_docs = _get_num_docs(step)
                    if n_docs is not None:
                        docs_per_search_all.append(n_docs)
                    docs_seen_per_search_all.append(_get_num_seen_docs(step))

                if atype in _TERMINAL_ACTIONS:
                    has_terminal = True
                if atype in _FORCE_ANSWER_ACTIONS:
                    has_force_answer = True

            total_steps_per_query.append(len(trajectory))
            search_steps_per_query.append(n_search)
            action_type_counts_per_query.append(dict(step_action_counts))

            if n_search == 0:
                queries_no_search += 1
            if has_force_answer or not has_terminal:
                queries_with_force_answer += 1
            if not has_terminal:
                queries_max_iter += 1

        def _stats(lst: List) -> Dict[str, float]:
            if not lst:
                return {"mean": 0.0, "std": 0.0, "min": 0, "max": 0}
            arr = np.array(lst, dtype=float)
            return {
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr)),
                "min": int(np.min(arr)),
                "max": int(np.max(arr)),
            }

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
            "steps": _stats(total_steps_per_query),
            "search_steps": _stats(search_steps_per_query),
            # None, not zeros: "no ranking recorded" and "the retriever
            # returned nothing" are different claims.
            "docs_per_search": _stats(docs_per_search_all) if docs_per_search_all else None,
            "docs_surfaced_per_search": (
                _stats(docs_per_search_all) if docs_per_search_all else None),
            "docs_seen_per_search": _stats(docs_seen_per_search_all),
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
                "input_tokens": _stats(_tok("input_tokens")),
                "output_tokens": _stats(_tok("output_tokens")),
                "total_tokens": _stats(_tok("total_tokens")),
                "llm_calls": _stats(_tok("num_calls")),
            }

        return metrics

    def print_results(self, metrics: Dict[str, Any], header: str = "TRAJECTORY STATISTICS") -> None:
        """Pretty-print trajectory statistics.

        Args:
            metrics: Output of :meth:`evaluate`.
            header:  Section header string.
        """
        if not metrics:
            print("  ⚠ No trajectory metrics available")
            return

        print("\n" + "=" * 80)
        print(header)
        print("=" * 80)

        n = metrics.get("num_queries", 0)
        print(f"  Queries evaluated:        {n}")

        def _fmt_stats(d: Optional[Dict]) -> str:
            if not d:
                return "n/a"
            return (
                f"{d['mean']:.1f} ± {d['std']:.1f}"
                f"  (min: {d['min']}, max: {d['max']})"
            )

        print(f"  Steps per query:          {_fmt_stats(metrics.get('steps'))}")
        print(f"  Search steps per query:   {_fmt_stats(metrics.get('search_steps'))}")
        print(f"  Docs surfaced per search: {_fmt_stats(metrics.get('docs_surfaced_per_search'))}")
        print(f"  Docs seen per search:     {_fmt_stats(metrics.get('docs_seen_per_search'))}")

        no_search = metrics.get("queries_no_search", 0)
        pct = (no_search / n * 100) if n else 0
        print(f"  Queries with no search:   {no_search} ({pct:.1f}%)")

        force_ans = metrics.get("queries_with_force_answer", 0)
        pct_force = (force_ans / n * 100) if n else 0
        print(f"  Queries w/ force answer:  {force_ans} ({pct_force:.1f}%)")

        max_iter = metrics.get("queries_max_iter_reached", 0)
        pct_max = (max_iter / n * 100) if n else 0
        print(f"  Queries hitting max iter: {max_iter} ({pct_max:.1f}%)")

        tokens = metrics.get("tokens")
        if tokens:
            tot = tokens.get("total_tokens", {})
            inp = tokens.get("input_tokens", {})
            out = tokens.get("output_tokens", {})
            calls = tokens.get("llm_calls", {})
            print(f"  Tokens/query (total):     {tot.get('mean', 0):.0f} ± {tot.get('std', 0):.0f}"
                  f"  (in: {inp.get('mean', 0):.0f}, out: {out.get('mean', 0):.0f})")
            print(f"  LLM calls per query:      {calls.get('mean', 0):.1f}")

        action_counts = metrics.get("avg_action_type_counts", {})
        if action_counts:
            print(f"  Action type distribution (avg per query):")
            for atype, avg in sorted(action_counts.items(), key=lambda x: -x[1]):
                print(f"    {atype:<20s}: {avg:.2f}")

        print("=" * 80)

    def save_item(self, query_id: str, question: str, result: Dict[str, Any], output_dir) -> None:
        """Save per-query trajectory as a JSONL file (one line per step).

        Each line is one trajectory step, followed by a final
        ``{"record": "meta", ...}`` line carrying the metadata needed to
        reconstruct the result on resume (generation, step counts,
        agent-specific resume data).  The meta record is written last so that
        :class:`utils.trajectory_logger.TrajectoryLogger` can stream the same
        file step by step during the run; readers dispatch on ``record`` per
        line, so its position does not matter.  Steps are shaped as: search steps carry ``iter`` / ``action_type`` / ``think`` /
        ``search_query`` / ``seen_docs`` (``action_type`` and ``think`` only on
        the first subquery of an iteration); terminal steps carry ``iter`` /
        ``action_type`` /
        ``generation``.

        Deliberately *not* stored here (to avoid duplicating data that lives
        elsewhere):
        - the full surfaced doc ranking → ``retrieval/surfaced/{qid}.trec``;
        - uncertainty signals → ``uncertainty/{qid}.jsonl``;
        - cited docs → ``retrieval/cited/{qid}.trec``.

        Args:
            query_id:   Query identifier.
            question:   Original query text.
            result:     Unified agent result dict for this query.
            output_dir: Directory where ``{query_id}.jsonl`` will be written.
        """
        output_dir_str = str(output_dir)
        Path(output_dir_str).mkdir(parents=True, exist_ok=True)

        json_path = f"{output_dir_str.rstrip('/')}/{query_id}.jsonl"
        with open(json_path, "w") as f:
            last_iter = 0
            for step in result.get("trajectory", []):
                it = step.get("iteration")
                if it is not None:
                    last_iter = int(it)
                    label = _iter_label(step, last_iter)
                else:
                    last_iter += 1
                    label = _iter_label(step, last_iter)
                f.write(_dump_line(_step_to_line(step, label)) + "\n")
            f.write(_dump_line(_build_meta_line(query_id, question, result)) + "\n")

    def save_results(self, metrics: Dict[str, Any], output_path, summary: Optional[Dict[str, Any]] = None, summary_path=None) -> None:
        """Save trajectory statistics and optionally the run summary to JSON files.

        Args:
            metrics:      Output of :meth:`evaluate`.
            output_path:  Destination file for trajectory metrics (parents created if needed).
            summary:      Optional full-run summary dict to persist alongside metrics.
            summary_path: Destination file for the summary (required when *summary* is given).
        """
        if not metrics:
            return
        output_path_str = str(output_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, default=str)
        print(f"  ✓ Saved trajectory metrics: {output_path_str}")

        if summary and summary_path:
            summary_path_str = str(summary_path)
            summary_path = Path(summary_path)
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, default=str)
            print(f"  ✓ Saved summary: {summary_path_str}")
