"""Trajectory evaluation (the ``trajectory`` group of the run's ``metrics.md``).

* :class:`TrajectoryEvaluator`: per query searches, forced answer and agent tokens.
* :func:`save_trajectory`: the per-query ``trajectory/{query_id}.jsonl``.
"""

from .evaluator import TrajectoryEvaluator
from .writer import save_trajectory

__all__ = ["TrajectoryEvaluator", "save_trajectory"]
