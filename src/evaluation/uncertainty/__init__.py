"""Uncertainty-signal output of a run.

* :func:`save_uncertainty`: the per-query ``uncertainty/{query_id}.jsonl``
  (schema in :mod:`.writer`), the input of the offline signal analysis
  (``analysis/signal_correlation.py``).
"""

from .writer import SCHEMA_VERSION, save_uncertainty

__all__ = ["SCHEMA_VERSION", "save_uncertainty"]
