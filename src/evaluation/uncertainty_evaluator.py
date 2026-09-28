"""Uncertainty evaluator: persist and aggregate the uncertainty estimator's signals.

Saves what ``UncertaintyEstimator`` records into one JSONL file per query
under ``uncertainty/``: a ``meta`` line, then one ``step`` line per search
iteration.  The files are the input of the offline analysis and of the
scoring model g_phi.  Every line carries ``query_id``; correctness labels
stay in the accuracy output (``accuracy.jsonl``, same ``query_id`` key).
The per-iteration seen doc ids live in the trajectory file
(``trajectory/{query_id}.jsonl``).

Per-query JSONL schema (``uncertainty/{query_id}.jsonl``), schema_version 2::

    line 1  {"record": "meta", "schema_version": 2, "query_id": "q1", "question": "...",
             "agent": "react", "llm_model": "...", "dataset": "browsecomp_plus",
             "llm_criteria": "...", "max_criteria": 8,
             "criteria_source": "llm", "criteria_judge": "nli:...",
             "query_scorer": "embedding:...", "encoder": "...",
             "num_criteria": 4, "num_iterations": 8, "num_unique_docs": 34, "num_relevant": 6,
             "criteria": [{"id": "c1", "text": "..."}, ...],
             "criteria_info": {"model": ..., "reasoning": ..., "errors": []},
             "final_criteria_state": ["fully_covered", "uncovered", ...],
             "criteria_evidence": [{"id": "c1", "partially_covered": [...], "fully_covered": ["d1"]}, ...]}

    line 2  {"record": "step", "query_id": "q1", "iteration": 1, "agent_iteration": 0,
             "num_subqueries": 1, "num_docs": 5, "num_new_docs": 5,
             "doc_novelty": 0.93, "criteria_delta": 3, "query_novelty": 1.0, "criteria_targeting": 0.71,
             "marginal_recall": 0.3333, "new_relevant_frac": 0.4,
             "num_new_relevant": 2, "num_repeated_relevant": 0, "num_irrelevant": 3,
             "intermediate_answers": ["..."], "intermediate_answer_confidence": 0.75,
             "subqueries": ["..."],
             "queries": [{"text": "...", "max_sim_to_earlier": null, "novelty": 1.0,
                          "target_scores": [0.71, 0.42, ...], "criteria_targeting": 0.71}],
             "docs": [{"doc_id": "d1", "seen_before": false, "max_sim_to_seen": null, "novelty": 1.0}, ...],
             "criteria_state_before": ["uncovered", ...], "criteria_state_after": ["fully_covered", ...],
             "criteria_judgments": [{"doc_id": "d1", "statuses": [...], "scores": [...]}, ...],
             "intermediate_answer_reasoning": "...", "errors": []}
    ...

``iteration`` counts from 1 for every agent; ``agent_iteration`` is the
agent's own counter.  A signal that could not be computed is null, never 0,
and every float is finite and rounded to 4 decimals.  A query without any
search still gets its meta line.  The file is written atomically.
``utils.io_utils.load_result_from_saved_files`` reads it back.
"""

import json
import logging
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


logger = logging.getLogger(__name__)

SCHEMA_VERSION = 2

# Per-step scalar signals: x_t of the report, then the extra ones.
SIGNAL_KEYS = (
    "doc_novelty", "criteria_delta", "query_novelty", "criteria_targeting",
    "marginal_recall", "new_relevant_frac", "intermediate_answer_confidence",
)


def _clean(obj: Any) -> Any:
    """JSON-safe copy: numpy scalars to Python, non-finite floats to None,
    floats rounded to 4 decimals, tuples and sets to lists."""
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_clean(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _clean(obj.tolist())
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        value = float(obj)
        return round(value, 4) if math.isfinite(value) else None
    return obj


def _dump(obj: Dict[str, Any]) -> str:
    return json.dumps(_clean(obj), separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def _correct_flag(row: Dict[str, Any]) -> Optional[bool]:
    """Per-query correctness from an accuracy row (judge ``correct`` or TRQA
    ``exact_match``)."""
    for key in ("correct", "exact_match"):
        if row.get(key) is not None:
            return bool(row[key])
    return None


class UncertaintyEvaluator:
    """Persist and aggregate uncertainty signal data.

    Usage::

        evaluator = UncertaintyEvaluator()

        # Per-query save (inside the query loop)
        evaluator.save_item(query_id, query_text, result, uncertainty_dir)

        # Aggregate evaluation (after all queries)
        metrics = evaluator.evaluate(results, accuracy_metrics)
        evaluator.print_results(metrics)
    """

    # ------------------------------------------------------------------
    # Per-query persistence
    # ------------------------------------------------------------------

    def save_item(
        self,
        query_id: str,
        question: str,
        result: Dict[str, Any],
        output_dir,
    ) -> None:
        """Write ``{output_dir}/{query_id}.jsonl`` from the result dict.

        Reads ``uncertainty_meta`` and ``uncertainty_steps`` (attached by
        ``_attach_uncertainty_stats`` in the agent mixin); does nothing when
        the estimator was off.  Writes to a temporary file first, so an
        interrupted write never leaves a truncated file.
        """
        meta = result.get("uncertainty_meta")
        if meta is None:
            return

        out_dir = Path(str(output_dir))
        out_dir.mkdir(parents=True, exist_ok=True)

        header = {
            "record": "meta", "schema_version": SCHEMA_VERSION,
            "query_id": query_id, "question": question, **meta,
        }
        path = out_dir / f"{query_id}.jsonl"
        tmp_path = out_dir / f".{query_id}.jsonl.tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            f.write(_dump(header) + "\n")
            for step in result.get("uncertainty_steps") or []:
                f.write(_dump({"record": "step", "query_id": query_id, **step}) + "\n")
        os.replace(tmp_path, path)

    # ------------------------------------------------------------------
    # Aggregate evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self,
        results: Dict[str, Dict[str, Any]],
        accuracy_metrics: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Aggregate statistics over all queries with uncertainty data.

        ``signals`` pools every step; ``signals_per_query`` first averages
        each signal within a query, so long trajectories do not dominate.
        ``null_rate`` is the share of steps where a signal is null.  With
        *accuracy_metrics* (its ``per_query`` rows), the per-query means are
        also split into correct and incorrect queries.
        """
        correct: Dict[str, bool] = {}
        for row in (accuracy_metrics or {}).get("per_query") or []:
            flag = _correct_flag(row)
            if flag is not None and row.get("query_id") is not None:
                correct[str(row["query_id"])] = flag

        num_queries = 0
        num_steps = 0
        steps_per_query: List[float] = []
        unique_docs_per_query: List[float] = []
        criteria_per_query: List[float] = []
        queries_without_criteria = 0
        signal_values: Dict[str, List[float]] = {k: [] for k in SIGNAL_KEYS}
        null_counts: Dict[str, int] = {k: 0 for k in SIGNAL_KEYS}
        query_means: Dict[str, List[float]] = {k: [] for k in SIGNAL_KEYS}
        by_outcome: Dict[str, Dict[str, List[float]]] = {
            "correct": {k: [] for k in SIGNAL_KEYS},
            "incorrect": {k: [] for k in SIGNAL_KEYS},
        }
        steps_with_errors = 0
        queries_with_answer = 0

        for qid, result in results.items():
            meta = result.get("uncertainty_meta")
            if meta is None:
                continue
            steps = result.get("uncertainty_steps") or []
            num_queries += 1
            num_steps += len(steps)
            steps_per_query.append(float(len(steps)))
            unique_docs_per_query.append(float(meta.get("num_unique_docs") or 0))
            n_criteria = len(meta.get("criteria") or [])
            criteria_per_query.append(float(n_criteria))
            if n_criteria == 0:
                queries_without_criteria += 1

            outcome = None
            if str(qid) in correct:
                outcome = "correct" if correct[str(qid)] else "incorrect"

            has_answer = False
            for key in SIGNAL_KEYS:
                values = [float(step[key]) for step in steps if step.get(key) is not None]
                null_counts[key] += len(steps) - len(values)
                signal_values[key].extend(values)
                if values:
                    query_means[key].append(float(np.mean(values)))
                    if outcome is not None:
                        by_outcome[outcome][key].append(float(np.mean(values)))
            for step in steps:
                if step.get("errors"):
                    steps_with_errors += 1
                if step.get("intermediate_answers"):
                    has_answer = True
            if has_answer:
                queries_with_answer += 1

        if num_queries == 0:
            return {}

        def _stats(values: List[float]) -> Optional[Dict[str, float]]:
            if not values:
                return None
            arr = np.array(values, dtype=float)
            return {
                "mean": round(float(np.mean(arr)), 4),
                "std": round(float(np.std(arr)), 4),
                "min": round(float(np.min(arr)), 4),
                "max": round(float(np.max(arr)), 4),
                "median": round(float(np.median(arr)), 4),
                "count": int(arr.size),
            }

        metrics = {
            "num_queries": num_queries,
            "num_steps": num_steps,
            "steps_per_query": _stats(steps_per_query),
            "unique_docs_per_query": _stats(unique_docs_per_query),
            "criteria_per_query": _stats(criteria_per_query),
            "queries_without_criteria": queries_without_criteria,
            "signals": {key: _stats(values) for key, values in signal_values.items()},
            "signals_per_query": {key: _stats(values) for key, values in query_means.items()},
            "null_rate": {
                key: round(null_counts[key] / num_steps, 4) if num_steps else None
                for key in SIGNAL_KEYS
            },
            "steps_with_errors": steps_with_errors,
            "queries_with_intermediate_answer": queries_with_answer,
        }
        if correct:
            metrics["signals_per_query_by_outcome"] = {
                outcome: {key: _stats(values) for key, values in per_key.items()}
                for outcome, per_key in by_outcome.items()
            }
        return metrics

    # ------------------------------------------------------------------
    # Pretty-print
    # ------------------------------------------------------------------

    def print_results(
        self,
        metrics: Dict[str, Any],
        header: str = "UNCERTAINTY SIGNALS",
    ) -> None:
        """Pretty-print aggregate uncertainty statistics."""
        if not metrics:
            print("  (no uncertainty data available)")
            return

        print("\n" + "=" * 80)
        print(header)
        print("=" * 80)

        def _fmt(d: Optional[Dict]) -> str:
            if not d:
                return "n/a"
            return (
                f"{d['mean']:.3f} +/- {d['std']:.3f}"
                f"  (min: {d['min']:.3f}, max: {d['max']:.3f}, med: {d['median']:.3f}, n: {d['count']})"
            )

        print(f"  Queries:                  {metrics.get('num_queries', 0)}")
        print(f"  Steps per query:          {_fmt(metrics.get('steps_per_query'))}")
        print(f"  Unique docs per query:    {_fmt(metrics.get('unique_docs_per_query'))}")
        print(f"  Criteria per query:       {_fmt(metrics.get('criteria_per_query'))}")
        print(f"  Queries without criteria: {metrics.get('queries_without_criteria', 0)}")
        print("  Signals (across all steps; null values excluded):")
        null_rate = metrics.get("null_rate") or {}
        for name, stats in (metrics.get("signals") or {}).items():
            rate = null_rate.get(name)
            rate_str = f"  null: {rate:.0%}" if rate is not None else ""
            print(f"    {name:<31s}: {_fmt(stats)}{rate_str}")
        by_outcome = metrics.get("signals_per_query_by_outcome")
        if by_outcome:
            print("  Per-query means, correct vs incorrect:")
            for name in (metrics.get("signals") or {}):
                c = (by_outcome.get("correct") or {}).get(name)
                w = (by_outcome.get("incorrect") or {}).get(name)
                c_str = f"{c['mean']:.3f} (n={c['count']})" if c else "n/a"
                w_str = f"{w['mean']:.3f} (n={w['count']})" if w else "n/a"
                print(f"    {name:<31s}: correct {c_str}  |  incorrect {w_str}")
        print(f"  Steps with errors:        {metrics.get('steps_with_errors', 0)}")
        print(f"  Queries with an intermediate answer: {metrics.get('queries_with_intermediate_answer', 0)}")
        print("=" * 80)
