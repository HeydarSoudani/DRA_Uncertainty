"""Belief-model output of a run and its evaluation.

* :func:`save_belief`: the per-query ``belief/{query_id}.jsonl``
  (schema in :mod:`.writer`), the input of belief model training, of the
  evaluation and of the offline analyses.
* :class:`BeliefUpdateEvaluator`: what the belief update does with a gold
  document (the ``belief`` group of the run's ``metrics.md``).
* :mod:`.gold`: whether a text has a criterion's gold answer (match and
  judge), shared with belief model training.
"""

from .evaluator import BeliefUpdateEvaluator, load_belief_files, passage_chars, read_belief_file, read_run_config
from .writer import SCHEMA_VERSION, clean_record, save_belief

__all__ = ["SCHEMA_VERSION", "clean_record", "save_belief", "BeliefUpdateEvaluator", "load_belief_files",
           "passage_chars", "read_belief_file", "read_run_config"]
