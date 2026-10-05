"""Retrieval evaluation (the ``retrieval`` group of ``summary.json``).

Three document levels share one base class and metric set:

* :class:`SurfacedDocEvaluator`: all docs the retriever returned per step,
  scored per fusion method by :func:`.fusion.run_fusion_eval`.
* :class:`SeenDocEvaluator`: docs actually passed to the agent/LLM.
* :class:`CitedDocEvaluator`: docs cited by the final report
  (:mod:`.citations`).

Each class also writes its per-query ``retrieval/{level}/{query_id}.trec``.
"""

from .base import BaseDocRetrievalEvaluator
from .cited import CitedDocEvaluator
from .metrics import compute_trec_metrics, evaluate_results, metrics_at_n
from .seen import SeenDocEvaluator
from .surfaced import SurfacedDocEvaluator

__all__ = [
    "BaseDocRetrievalEvaluator",
    "SurfacedDocEvaluator",
    "SeenDocEvaluator",
    "CitedDocEvaluator",
    "compute_trec_metrics",
    "evaluate_results",
    "metrics_at_n",
]
