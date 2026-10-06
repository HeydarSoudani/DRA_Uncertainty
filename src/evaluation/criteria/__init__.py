"""Criteria evaluation: how well a per-query criteria list covers the gold.

* :class:`CriteriaEvaluator`: scores criteria lists against gold units
  (``evaluation.gold``); in a run, per query as the criteria are extracted
  and over the run at evaluation (:meth:`CriteriaEvaluator.evaluate_run`).
* ``python -m evaluation.criteria``: derives the criteria of a dataset split
  with the uncertainty estimator's ``LLMCriteriaSource`` (no agent run), or
  reads them from a run, and scores them.
"""

from .evaluator import (
    CRITERIA_JUDGMENTS_FILE, CriteriaEvaluator, build_criteria_evaluator, compare_with_report,
)

__all__ = ["CRITERIA_JUDGMENTS_FILE", "CriteriaEvaluator", "build_criteria_evaluator",
           "compare_with_report"]
