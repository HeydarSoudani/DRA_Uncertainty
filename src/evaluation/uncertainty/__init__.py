"""Uncertainty-signal output of a run.

* :func:`save_uncertainty`: the per-query ``uncertainty/{query_id}.jsonl``
  (schema in :mod:`.writer`), the input of the offline signal analysis
  (``analysis/signal_correlation.py``).
* :func:`load_uncertainty_meta`: the meta lines of a run's files;
  :func:`update_uncertainty_meta` sets fields of one.
"""

from .writer import SCHEMA_VERSION, load_uncertainty_meta, save_uncertainty, update_uncertainty_meta

__all__ = ["SCHEMA_VERSION", "load_uncertainty_meta", "save_uncertainty", "update_uncertainty_meta"]
