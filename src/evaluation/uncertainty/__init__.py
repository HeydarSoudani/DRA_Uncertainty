"""Uncertainty evaluation: the ``uncertainty`` group of ``metrics.md``.

    verbal.py    VerbalConfidenceEvaluator: the agent's verbal confidence vs
                 correctness (AUROC, Brier); QA, once per run
    qa.py        UncertaintyEvaluator: the belief rule-based score and the
                 belief LLM score (s_T) vs correctness (AUROC, Brier); QA,
                 once per belief model
    report.py    ReportUncertaintyEvaluator: a report's s_T vs its nugget
                 coverage (Kendall's tau-b, Brier); reports, once per belief model
    metrics.py   metric names and the log lines

Every score here is read against the generation group's output (the answer
verdicts or the Auto-ARGUE records), never judged again.
"""

from .qa import UncertaintyEvaluator, belief_scores
from .report import ReportUncertaintyEvaluator, report_scores
from .verbal import VerbalConfidenceEvaluator, verbal_confidence_scores

__all__ = ["ReportUncertaintyEvaluator", "UncertaintyEvaluator", "VerbalConfidenceEvaluator",
           "belief_scores", "report_scores", "verbal_confidence_scores"]
