"""Retrieval evaluation: the ``retrieval`` group of ``metrics.md``.

    seen_docs.py   SeenDocEvaluator: Recall@N, Precision@N of the seen documents
    metrics.py     metric names and the log line
"""

from .seen_docs import SeenDocEvaluator, seen_doc_scores

__all__ = ["SeenDocEvaluator", "seen_doc_scores"]
