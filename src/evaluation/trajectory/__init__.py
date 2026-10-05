"""Trajectory evaluation (the ``trajectory`` group of ``summary.json``).

* :class:`TrajectoryEvaluator`: step, search, document and token statistics.
* :func:`save_trajectory`: the per-query ``trajectory/{query_id}.jsonl``.
"""

from .evaluator import TrajectoryEvaluator
from .writer import save_trajectory

__all__ = ["TrajectoryEvaluator", "save_trajectory"]
