"""The SFT filter: which of the teacher's gated samples are trained on.

TODO(user): decide the filter.  Every gated sample is kept for now
(Agent-BRACE does not filter its teacher's transitions).  A sample is
``{set, task, dataset, split, policy_model, qid, step, messages,
completion, y}`` (``sft.samples``): e.g. an outcome filter on the teacher's
final score would keep the episodes whose s_T is on the side of y.
"""

from typing import Any, Dict


def keep(sample: Dict[str, Any]) -> bool:
    """Whether *sample* is trained on."""
    return True
