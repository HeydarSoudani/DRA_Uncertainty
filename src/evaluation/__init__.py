"""Evaluation of deep-research agent runs.

The subpackages follow the groups of a run's ``summary.json``:

answer
    Answer correctness (``AccuracyEvaluator`` by LLM judge,
    ``NumericMatchEvaluator`` by numeric match) and the report evaluation
    against nuggets (``ArgueReportEvaluator``, Auto-ARGUE); in
    ``summary.json`` they are ``generation.correctness`` and
    ``generation.nuggets``.
retrieval
    Surfaced / seen / cited document evaluators on one base class and metric
    set, the citation helpers, and the surfaced-doc fusion evaluation
    (``retrieval.fusion``).
trajectory
    Trajectory statistics (incl. token usage) and the per-query trajectory
    file.
generation
    Generation statistics (``generation.stats``) and the per-query
    generation file.
gold
    The gold entities of TRQA and the entity matcher, for the entity
    reachability of the criteria (``analysis/criteria_reachability.py``).
uncertainty
    The per-query uncertainty-signal file.

Beside them: ``runner`` (evaluation of an inference run), ``judge`` (the
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
