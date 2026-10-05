"""Criteria evaluation: how well a per-query criteria list covers the gold.

* :class:`CriteriaEvaluator`: scores criteria lists against gold units
  (``evaluation.gold``).
* ``python -m evaluation.criteria``: derives the criteria of a dataset split
  with the uncertainty estimator's ``LLMCriteriaSource`` (no agent run), or
  reads them from a run, and scores them.
"""

from .evaluator import CriteriaEvaluator

__all__ = ["CriteriaEvaluator"]
