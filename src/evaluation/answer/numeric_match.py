"""Answer accuracy by numeric match, for datasets with numeric answers (TRQA).

Correctness is judged by numeric matching rather than an LLM judge
(``DATASET_SPECS[...].answer_eval == "numeric_match"``).  The matching logic
(number extraction and exact / tolerant comparison) is ported verbatim from the
official Total-Recall-QA repository:

    https://github.com/mahta-r/total-recall-qa/blob/main/c5_task_evaluation/metrics/generation_eval_metrics.py

Metrics (as in ``run_evalution.py`` of that repository):
    * ``exact_match``        fraction with an exact numeric match.
    * ``soft_exact_match``   fraction within each tolerance % in
                             ``[1, 5, 10, 20, 50, 90]``.

The agent's final short answer is extracted from the generation by
:func:`extract_prediction`.  The interface matches
:class:`evaluation.answer.llm_judge.AccuracyEvaluator`, so the runner uses
either one in the accuracy slot.
"""

import logging
import math
import re
from typing import Any, Dict, List

from ..common import print_header, write_jsonl
from ..judge import strip_references

logger = logging.getLogger(__name__)

# Tolerance percentages reported by the TRQA reference (run_evalution.py).
DEFAULT_TOLERANCE_PCTS = (1.0, 5.0, 10.0, 20.0, 50.0, 90.0)


# ---------------------------------------------------------------------------
# Numeric matching, ported verbatim from total-recall-qa generation_eval_metrics
# ---------------------------------------------------------------------------

def normalize_number(value, decimals=2):
    """Rounds numeric value to fixed decimals for comparison."""
    if value is None or math.isnan(value):
        return None
    return round(float(value), decimals)


def safe_float_convert(value):
    """Safely convert a value to float, return None if not possible."""
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


# Regex to find first number in text (int or decimal, optional minus, optional comma thousands)
_NUMBER_RE = re.compile(r"[-+]?(?:\d+(?:,\d{3})*(?:\.\d+)?|\d*\.\d+)")


def extract_number_from_string(s):
    """Extract the first number from a string, stripping text/units.

    E.g. "8269 km" -> 8269.0, "1,234.5 units" -> 1234.5.  Returns float or None.
    """
    if s is None or (isinstance(s, str) and s.strip() == ""):
        return None
    s = str(s).strip()
    m = _NUMBER_RE.search(s)
    if not m:
        return None
    num_str = m.group(0).replace(",", "")
    try:
        return float(num_str)
    except ValueError:
        return None


def soft_exact_match(prediction, gold, decimals=3, tolerance_pct=None):
    """Compute soft exact match with optional tolerance.

    Returns a dict with ``exact_match`` (bool) and, when *tolerance_pct* is
    given, ``soft_match`` (bool within ``tolerance_pct`` % of gold).
    """
    pred_float = extract_number_from_string(prediction)
    gold_float = extract_number_from_string(gold)

    # Fallback: pure numeric string without extra text
    if pred_float is None:
        pred_float = safe_float_convert(prediction)
    if gold_float is None:
        gold_float = safe_float_convert(gold)

    if pred_float is None or gold_float is None:
        result = {"exact_match": False}
        if tolerance_pct is not None:
            result["soft_match"] = False
        return result

    npred = normalize_number(pred_float, decimals)
    ngold = normalize_number(gold_float, decimals)

    if npred is None or ngold is None:
        result = {"exact_match": False}
        if tolerance_pct is not None:
            result["soft_match"] = False
        return result

    exact = npred == ngold
    result = {"exact_match": exact}

    if tolerance_pct is not None:
        abs_err = abs(npred - ngold)
        soft = abs_err <= (tolerance_pct / 100.0) * abs(ngold) if ngold != 0 else (abs_err == 0)
        result["soft_match"] = soft

    return result


# ---------------------------------------------------------------------------
# Final-answer extraction from the agent generation
# ---------------------------------------------------------------------------

def extract_prediction(generation: str) -> str:
    """The agent's final short answer from its (possibly long) generation.

    Uses the answer-candidate parsers (``\\boxed{}``, ``<answer>``,
    ``Exact Answer:``, ...).  Falls back to the generation without its
    ``## References`` block when nothing structured matches, so
    :func:`extract_number_from_string` can still find the number.
    """
    if not generation:
        return ""
    try:
        from deep_research_agents.prompts.answer_prompts import extract_answer_candidates

        candidates, _matched = extract_answer_candidates(generation)
        if candidates:
            return candidates[0].candidate
    except Exception as e:  # pragma: no cover - extraction is best-effort
        logger.debug(f"Answer extraction failed, using raw generation: {e}")
    return strip_references(generation)


# ---------------------------------------------------------------------------
# NumericMatchEvaluator
# ---------------------------------------------------------------------------

class NumericMatchEvaluator:
    """Numeric exact / soft-exact match of the final answer.

    Usage::

        evaluator = NumericMatchEvaluator(answers={"q1": "8269", ...})
        metrics = evaluator.evaluate(results)   # {query_id: {"generation": str, ...}}
        evaluator.print_results(metrics)

    Args:
        answers: ``query_id -> ground-truth answer``.
        tolerance_pcts: Tolerance percentages of the soft match.
        decimals: Decimals kept before comparison.
    """

    def __init__(
        self,
        answers: Dict[str, str],
        tolerance_pcts=DEFAULT_TOLERANCE_PCTS,
        decimals: int = 3,
    ) -> None:
        self.answers = answers
        self.tolerance_pcts = tuple(tolerance_pcts)
        self.decimals = decimals

    def evaluate(self, results: Dict[str, Dict[str, Any]], run_dir=None) -> Dict[str, Any]:
        """Exact and soft-exact match over the queries with a ground-truth answer
        (*run_dir* is unused: the match needs no judge to cache).

        Returns ``exact_match``, ``accuracy`` (the same value, for the
        accuracy slot of the summary), ``soft_exact_match`` (``{pct:
        fraction}``), ``num_correct``, ``num_evaluated`` and ``per_query``;
        ``{}`` when no query has an answer.
        """
        evaluable_ids = [qid for qid in results if self.answers.get(qid)]
        if not evaluable_ids:
            logger.warning("No queries with ground-truth answers to evaluate")
            return {}

        per_query: List[Dict[str, Any]] = []
        exact_count = 0
        soft_counts = {pct: 0 for pct in self.tolerance_pcts}

        for qid in evaluable_ids:
            gold = self.answers[qid]
            prediction = extract_prediction(results[qid].get("generation", ""))

            exact = soft_exact_match(prediction, gold, decimals=self.decimals)["exact_match"]
            exact_count += exact

            soft_flags: Dict[str, bool] = {}
            for pct in self.tolerance_pcts:
                soft = soft_exact_match(
                    prediction, gold, decimals=self.decimals, tolerance_pct=pct,
                )["soft_match"]
                soft_flags[str(pct)] = soft
                soft_counts[pct] += soft

            per_query.append({
                "query_id": qid,
                "prediction": prediction,
                "gold": gold,
                "exact_match": exact,
                "soft_match": soft_flags,
            })

        num_evaluated = len(per_query)
        return {
            "exact_match": round(exact_count / num_evaluated, 5),
            "accuracy": round(exact_count / num_evaluated, 5),
            "soft_exact_match": {
                str(pct): round(soft_counts[pct] / num_evaluated, 5)
                for pct in self.tolerance_pcts
            },
            "num_correct": exact_count,
            "num_evaluated": num_evaluated,
            "per_query": per_query,
        }

    def print_results(
        self,
        metrics: Dict[str, Any],
        header: str = "NUMERIC MATCH EVALUATION (exact / soft match)",
    ) -> None:
        """Pretty-print the numeric-match metrics."""
        if not metrics:
            print("  No numeric-match metrics available (no ground-truth answers)")
            return
        print_header(header)
        print(f"  Queries evaluated:  {metrics.get('num_evaluated', 0)}")
        print(f"  Exact Match:        {metrics.get('exact_match', 0):.4f}")
        soft = metrics.get("soft_exact_match", {})
        if soft:
            print("  Soft Exact Match (within tolerance):")
            for pct in sorted(float(p) for p in soft):
                print(f"    {pct:5.1f}% tolerance: {soft[str(pct)]:.4f}")
        print("=" * 80)

    def save_results(self, metrics: Dict[str, Any], output_path) -> None:
        """Write ``accuracy.jsonl``: a ``{"record": "meta", ...}`` line with the
        run-level aggregates, then one line per query."""
        if not metrics:
            return
        meta = {
            "record": "meta",
            "exact_match": metrics.get("exact_match"),
            "accuracy": metrics.get("accuracy"),
            "soft_exact_match": metrics.get("soft_exact_match"),
            "num_correct": metrics.get("num_correct"),
            "num_evaluated": metrics.get("num_evaluated"),
        }
        write_jsonl(output_path, [meta] + metrics.get("per_query", []))
        print(f"  Saved numeric-match results: {output_path}")
