"""Retrieval evaluation (the ``retrieval`` group of the run's ``metrics.md``).

Three document levels share one base class:

* :class:`SurfacedDocEvaluator`: all docs the retriever returned per step
  (saved only).
* :class:`SeenDocEvaluator`: docs actually passed to the agent/LLM, scored
  per query (Recall@N, Precision@N, N = the docs seen).
* :class:`CitedDocEvaluator`: docs cited by the final report
  (:mod:`.citations`; saved only).

Each class also writes its per-query ``retrieval/{level}/{query_id}.trec``.
"""

from .base import BaseDocRetrievalEvaluator
from .cited import CitedDocEvaluator
from .seen import SeenDocEvaluator
from .surfaced import SurfacedDocEvaluator

__all__ = [
    "BaseDocRetrievalEvaluator",
    "SurfacedDocEvaluator",
    "SeenDocEvaluator",
    "CitedDocEvaluator",
]
