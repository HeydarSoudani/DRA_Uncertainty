"""Uncertainty evaluation (the ``uncertainty`` group of the run's ``metrics.md``).

* :class:`UncertaintyEvaluator`: s_T (the belief model's final scorer), its
  formula ablations and the agent's verbal confidence against the
  correctness of the final answer (AUROC, Brier, ECE).
"""

from .evaluator import UncertaintyEvaluator

__all__ = ["UncertaintyEvaluator"]
