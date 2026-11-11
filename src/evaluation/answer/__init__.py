"""Answer evaluation (the ``generation`` group of the run's ``metrics.md``).

* :class:`AccuracyEvaluator`: answer correctness by LLM judge
  (``answer_eval == "llm_judge"``: BrowseComp-Plus, deep_qa).
* :class:`NumericMatchEvaluator`: answer correctness by numeric exact
  match (``answer_eval == "numeric_match"``, TRQA).
* :class:`ArgueReportEvaluator`: a report against the request's nuggets with
  Auto-ARGUE (``report_eval == "argue"``, NeuCLIR and RAGTIME).
* :func:`verbal_confidence`: the agent's own confidence in its answer, read
  from the generation (glm, oss).
"""

from .argue import ArgueReportEvaluator
from .confidence import verbal_confidence
from .llm_judge import AccuracyEvaluator
from .numeric_match import NumericMatchEvaluator

__all__ = ["AccuracyEvaluator", "NumericMatchEvaluator", "ArgueReportEvaluator", "verbal_confidence"]
