"""Evaluation of deep-research agent runs.

The subpackages follow the groups of a run's ``metrics.md`` (``report``):

retrieval
    The surfaced / seen / cited document levels and their per-query TREC
    files; the seen docs are scored (Recall@N, Precision@N).
answer
    Answer correctness (``AccuracyEvaluator`` by LLM judge,
    ``NumericMatchEvaluator`` by numeric match), the report evaluation
    against nuggets (``ArgueReportEvaluator``, Auto-ARGUE) and the agent's
    verbal confidence: the ``generation`` group.
uncertainty
    s_T and its baselines against correctness (``UncertaintyEvaluator``).
belief
    The per-query belief file, the belief update evaluation
    (``BeliefUpdateEvaluator``) and the gold match shared with training.
trajectory
    Per-query trajectory statistics and the per-query trajectory file.
generation
    The per-query generation file.
gold
    Per-criterion gold (TRQA's gold bank, the training pools) and TRQA's
    gold entities with their matcher (``analysis/criteria_reachability.py``).

Beside them: ``runner`` (evaluation of an inference run), ``report`` (its
files and log), ``metrics`` (the shared measures), ``judge`` (the
LLM-judge client) and ``common`` (file and statistics helpers).
"""

from .answer import AccuracyEvaluator, ArgueReportEvaluator, NumericMatchEvaluator
from .generation import GenerationEvaluator
from .retrieval import CitedDocEvaluator, SeenDocEvaluator, SurfacedDocEvaluator
from .trajectory import TrajectoryEvaluator

__all__ = [
    "AccuracyEvaluator",
    "NumericMatchEvaluator",
    "ArgueReportEvaluator",
    "GenerationEvaluator",
    "SurfacedDocEvaluator",
    "SeenDocEvaluator",
    "CitedDocEvaluator",
    "TrajectoryEvaluator",
]
