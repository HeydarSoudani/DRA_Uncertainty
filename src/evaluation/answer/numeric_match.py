"""Answer accuracy by numeric match, for datasets with numeric answers (TRQA).

Correctness is judged by numeric matching rather than an LLM judge
(``DATASET_SPECS[...].answer_eval == "numeric_match"``).  The matching logic
(number extraction and exact / tolerant comparison) is ported verbatim from the
official Total-Recall-QA repository:

    https://github.com/mahta-r/total-recall-qa/blob/main/c5_task_evaluation/metrics/generation_eval_metrics.py

Metric: ``exact_match``, the fraction with an exact numeric match (as in
``run_evalution.py`` of that repository; its tolerant matches are not
reported).

The agent's final short answer is extracted from the generation by
:func:`extract_prediction`.  The interface matches
:class:`evaluation.answer.llm_judge.AccuracyEvaluator`, so the runner uses
either one in the accuracy slot.
"""

import logging
import math
import re
from typing import Any, Dict, List

from utils.io_utils import write_jsonl
from ..judge import strip_references
from .confidence import verbal_confidence

logger = logging.getLogger(__name__)

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


def numeric_exact_match(prediction, gold, decimals=3) -> bool:
    """Whether the number of *prediction* equals the number of *gold*, both
    rounded to *decimals* (False when either has none)."""
    pred_float = extract_number_from_string(prediction)
    gold_float = extract_number_from_string(gold)

    # Fallback: pure numeric string without extra text
    if pred_float is None:
        pred_float = safe_float_convert(prediction)
    if gold_float is None:
        gold_float = safe_float_convert(gold)
    if pred_float is None or gold_float is None:
        return False

    npred = normalize_number(pred_float, decimals)
    ngold = normalize_number(gold_float, decimals)
    return npred is not None and ngold is not None and npred == ngold


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
    """Numeric exact match of the final answer.

    Usage::

        evaluator = NumericMatchEvaluator(answers={"q1": "8269", ...})
        metrics = evaluator.evaluate(results)   # {query_id: {"generation": str, ...}}

    Args:
        answers: ``query_id -> ground-truth answer``.
        decimals: Decimals kept before comparison.
    """

    def __init__(self, answers: Dict[str, str], decimals: int = 3) -> None:
        self.answers = answers
        self.decimals = decimals

    def evaluate(self, results: Dict[str, Dict[str, Any]], run_dir=None) -> Dict[str, Any]:
        """Exact match over the queries with a ground-truth answer (*run_dir*
        is unused: the match needs no judge to cache).

        Returns ``accuracy`` (the exact match), ``num_correct``,
        ``num_evaluated``, ``num_judge_errors`` (0) and ``per_query``
        (``{query_id, prediction, gold, exact_match, correct,
        verbal_confidence}``); ``{}`` when no query has an answer.
        """
        evaluable_ids = [qid for qid in results if self.answers.get(qid)]
        if not evaluable_ids:
            logger.warning("No queries with ground-truth answers to evaluate")
            return {}

        per_query: List[Dict[str, Any]] = []
        for qid in sorted(evaluable_ids):
            gold = self.answers[qid]
            generation = results[qid].get("generation", "")
            prediction = extract_prediction(generation)
            exact = numeric_exact_match(prediction, gold, decimals=self.decimals)
            per_query.append({
                "query_id": qid,
                "prediction": prediction,
                "gold": gold,
                "exact_match": exact,
                "correct": exact,
                "verbal_confidence": verbal_confidence(generation),
            })

        num_correct = sum(r["correct"] for r in per_query)
        return {
            "accuracy": round(num_correct / len(per_query), 5),
            "num_correct": num_correct,
            "num_evaluated": len(per_query),
            "num_judge_errors": 0,
            "per_query": per_query,
        }

    def save_results(self, metrics: Dict[str, Any], output_path) -> None:
        """Write ``accuracy.jsonl``: one line per query."""
        if not metrics:
            return
        write_jsonl(output_path, metrics.get("per_query", []))
