"""Belief evaluation (the ``belief`` group of the run's ``metrics.md``).

The belief files are written and read by ``utils.run_files.belief``.

* :class:`BeliefUpdateEvaluator`: what the belief update does with a gold
  document, and whether the final belief is correct and calibrated
  (datasets with gold per belief, :data:`GOLD_DATASETS`).
* :func:`issue_summary`: how often the update's replies have issues
  (rows of the ``belief`` group and the issue table of ``metrics.md``).
* :mod:`.gold_rows`: the gold answers and documents per belief of a
  run's queries; :mod:`.gold_match`: whether a text has a belief's gold
  answer (match, then judge).
"""

from .evaluator import CALIBRATION_COLUMNS, BeliefUpdateEvaluator
from .gold_rows import GOLD_DATASETS
from .issues import ISSUE_COLUMNS, issue_summary

__all__ = ["BeliefUpdateEvaluator", "CALIBRATION_COLUMNS", "GOLD_DATASETS", "ISSUE_COLUMNS", "issue_summary"]
