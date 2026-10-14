"""Belief-model output of a run.

* :func:`save_belief`: the per-query ``belief/{query_id}.jsonl``
  (schema in :mod:`.writer`), the input of the offline signal analysis
  (``analysis/signal_correlation.py``).
"""

from .writer import SCHEMA_VERSION, save_belief

__all__ = ["SCHEMA_VERSION", "save_belief"]
