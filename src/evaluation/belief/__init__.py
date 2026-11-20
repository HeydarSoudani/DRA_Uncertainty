"""Belief evaluation (the ``belief`` group of the run's ``metrics.md``).

The belief files are written and read by ``utils.run_files.belief``.

* :func:`evidence_uptake` (:mod:`.uptake`): whether the belief's certainty
  s_t rises at the steps that bring a gold document (evidence uptake) and
  at the steps that bring only non-gold documents (false uptake), from the
  run's qrels (every dataset with qrels).
* :func:`issue_summary` (:mod:`.issues`): how often the update's replies
  fail the format gate (rows of the ``belief`` group) and every issue code
  that occurs (the issue table of ``metrics.md``).
"""

from .issues import ISSUE_COLUMNS, issue_summary
from .uptake import evidence_uptake

__all__ = ["ISSUE_COLUMNS", "evidence_uptake", "issue_summary"]
