"""Generation evaluation: the ``generation`` group of ``metrics.md``.

    llm_judge.py        AccuracyEvaluator: answer correctness by LLM judge
                        (``answer_eval == "llm_judge"``: BrowseComp-Plus, infoseek_hard, ...)
    numeric_match.py    NumericMatchEvaluator: answer correctness by numeric
                        exact match (``answer_eval == "numeric_match"``: TRQA)
    accuracy.py         the accuracy rows both of them produce
    argue.py            ArgueReportEvaluator: a report against the request's
                        nuggets with Auto-ARGUE (``report_eval == "argue"``: NeuCLIR, RAGTIME)
    confidence.py       verbal_confidence: the agent's confidence, read from the generation
    references.py       the References block of a generation
    report_sentences.py a report split into cited sentences (Auto-ARGUE's input)
    metrics.py          metric names and the log line

The answer verdicts and the Auto-ARGUE records are this group's raw output,
read by the uncertainty group.
"""

from .argue import ArgueReportEvaluator
from .confidence import verbal_confidence
from .llm_judge import AccuracyEvaluator
from .numeric_match import NumericMatchEvaluator

__all__ = ["AccuracyEvaluator", "ArgueReportEvaluator", "NumericMatchEvaluator", "verbal_confidence"]
