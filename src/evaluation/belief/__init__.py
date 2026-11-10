"""Belief-model output of a run.

* :func:`save_belief`: the per-query ``belief/{query_id}.jsonl``
  (schema in :mod:`.writer`), the input of belief model training and of
  the offline analyses (``analysis/final_score_ablation.py``).
"""

from .writer import SCHEMA_VERSION, save_belief

__all__ = ["SCHEMA_VERSION", "save_belief"]
