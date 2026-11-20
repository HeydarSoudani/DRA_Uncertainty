"""Uncertainty evaluation (the ``uncertainty`` group of the run's ``metrics.md``).

* :class:`VerbalConfidenceEvaluator` (:mod:`.verbal`): the agent's verbal
  confidence against the correctness of the final answer (AUROC, Brier), a
  measure of the generation (once per run).
* :class:`UncertaintyEvaluator` (:mod:`.qa`): the belief LLM score (s_T,
  the belief model's final scorer) and the belief rule-based score (the
  mean status probability of the final beliefs) against the correctness of
  the final answer (AUROC, Brier), once per belief model.
* :class:`ReportUncertaintyEvaluator` (:mod:`.report`): a report's s_T (the
  certainty of the final coverage) against its nugget coverage (Kendall's
  tau-b).
"""

from .qa import UncertaintyEvaluator
from .report import ReportUncertaintyEvaluator
from .verbal import VerbalConfidenceEvaluator

__all__ = ["ReportUncertaintyEvaluator", "UncertaintyEvaluator", "VerbalConfidenceEvaluator"]
