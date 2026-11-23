"""The filter rules of stage filter: which SFT samples and RL episodes are trained on.

TODO(user): decide the SFT sample rule.  Every gated sample is kept for now
(Agent-BRACE does not filter its teacher's transitions).  A sample is
``{set, task, dataset, split, policy_model, qid, step, messages,
completion, y}`` (``filtering.samples``): e.g. an outcome filter on the
teacher's final score would keep the episodes whose s_T is on the side of y.

An RL episode (the rl split; the val episodes are all kept) is left out when it has more search
iterations with a document than ``filter.rl_max_chain_steps`` (not cut:
the final score always reads a complete belief).
"""

from typing import Any, Dict, Optional

from ..runs import Episode


def keep_sample(sample: Dict[str, Any]) -> bool:
    """Whether the SFT *sample* is trained on."""
    return True


def episode_drop_reason(episode: Episode, max_chain_steps: Optional[int]) -> Optional[str]:
    """Why the RL *episode* is left out (None: it is kept)."""
    if max_chain_steps is not None and episode.doc_steps > max_chain_steps:
        return "longer_than_max_chain_steps"
    return None
