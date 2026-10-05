"""Evaluation of deep-research agent runs.

The subpackages follow the groups of a run's ``summary.json``:

answer
    Answer correctness (``AccuracyEvaluator`` by LLM judge,
    ``NumericMatchEvaluator`` by numeric match) and the report evaluation
    against nuggets (``ArgueReportEvaluator``, Auto-ARGUE).
retrieval
    Surfaced / seen / cited document evaluators on one base class and metric
    set, the citation helpers, and the surfaced-doc fusion evaluation
    (``retrieval.fusion``).
trajectory
    Trajectory statistics (incl. token usage) and the per-query trajectory
    file.
generation
    Generation statistics and the per-query generation file.
uncertainty
    The per-query uncertainty-signal file.

Beside them: ``runner`` (evaluation of an inference run), ``judge`` (the
LLM-judge client), ``common`` (file and statistics helpers), and the offline
criteria evaluation (``criteria``, ``python -m evaluation.criteria``) with its
gold units and matchers (``gold``).
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
