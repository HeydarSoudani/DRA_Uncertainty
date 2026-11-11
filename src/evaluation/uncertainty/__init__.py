"""Uncertainty evaluation (the ``uncertainty`` group of the run's ``metrics.md``).

* :class:`UncertaintyEvaluator`: s_T (the belief model's final scorer), its
  formula ablations and the agent's verbal confidence against the
  correctness of the final answer (AUROC, Brier, ECE).
* :class:`ReportUncertaintyEvaluator`: a report's s_T (the certainty of the
  final coverage) against its nugget coverage (Kendall's tau-b).
"""

from .evaluator import UncertaintyEvaluator
from .report import ReportUncertaintyEvaluator

__all__ = ["ReportUncertaintyEvaluator", "UncertaintyEvaluator"]
