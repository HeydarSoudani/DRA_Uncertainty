"""Answer evaluation (``generation.correctness`` and ``generation.nuggets`` in ``summary.json``).

* :class:`AccuracyEvaluator`: answer correctness by LLM judge
  (``answer_eval == "llm_judge"``, BrowseComp-Plus).
* :class:`NumericMatchEvaluator`: answer correctness by numeric exact / soft
  match (``answer_eval == "numeric_match"``, TRQA).
* :class:`ArgueReportEvaluator`: a report against the request's nuggets with
  Auto-ARGUE (``report_eval == "argue"``, NeuCLIR and RAGTIME).
"""

from .argue import ArgueReportEvaluator
from .llm_judge import AccuracyEvaluator
from .numeric_match import NumericMatchEvaluator

__all__ = ["AccuracyEvaluator", "NumericMatchEvaluator", "ArgueReportEvaluator"]
