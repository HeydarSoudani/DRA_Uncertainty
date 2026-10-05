"""Surfaced-doc retrieval evaluator.

"Surfaced" documents are **every** doc the retriever returned in each step
(top-k surfaced, before the agent down-selects what it actually reads).  They
are stored in each trajectory step as ``step["docs"]`` (flat) or
``step["all_docs"]`` (flat list of dicts, or list-of-lists).  A trajectory
reloaded from disk keeps only the seen doc ids; its surfaced lists are rebuilt
from ``retrieval/surfaced/{qid}.trec`` into ``surfaced_docs_iterations``
(see ``utils.io_utils``).

The run summary scores the surfaced docs with the fusion evaluation
(:mod:`evaluation.retrieval.fusion`), which reuses :meth:`doc_iterations`.
"""

from typing import Any, Dict, List

from .base import BaseDocRetrievalEvaluator


def _rank_score(doc: Dict[str, Any], fallback_rank: int) -> float:
    """The doc's retrieval score (``rank_score``, ``rerank_score`` or ``score``),
    else ``1 / rank``."""
    for key in ("rank_score", "rerank_score", "score"):
        val = doc.get(key)
        if val is not None:
            return val
    return 1.0 / fallback_rank


def _trajectory_iterations(trajectory: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
    """One surfaced doc list per retrieval step of a trajectory."""
    iterations = []
    for step in trajectory:
        all_docs = step.get("all_docs")
        if all_docs:
            if isinstance(all_docs[0], dict):
                turn_docs = all_docs
            else:
                turn_docs = [doc for retrieve_docs in all_docs for doc in retrieve_docs]
            if turn_docs:
                iterations.append(turn_docs)
        elif step.get("docs"):
            iterations.append(step["docs"])
    return iterations


class SurfacedDocEvaluator(BaseDocRetrievalEvaluator):
    """Retrieval quality over all surfaced (retriever-returned) docs.

    Usage::

        evaluator = SurfacedDocEvaluator(qrels=qrels, k_values=[1, 5, 10, 100])
        metrics = evaluator.evaluate(results)
    """

    emit_metrics_at_n = False
    default_header = "RETRIEVAL EVALUATION RESULTS"

    @classmethod
    def doc_iterations(cls, result: Dict[str, Any]) -> List[List[Dict[str, Any]]]:
        iterations = _trajectory_iterations(result.get("trajectory", []))
        if not iterations:
            iterations = result.get("surfaced_docs_iterations", [])
        return iterations

    @staticmethod
    def _score(doc: Dict[str, Any], rank: int) -> float:
        return _rank_score(doc, rank)
