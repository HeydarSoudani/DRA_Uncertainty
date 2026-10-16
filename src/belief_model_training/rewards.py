"""Phase 3 rewards of the student belief updater.  TODO: to be designed.

A reward function scores every sample of a rollout batch (one updater call
each, ``env.BeliefSample``) with a scalar; the trainer whitens the scalars
over the batch.  Agent-BRACE combines state tracking, state correctness,
diversity and task terms behind a format gate; which of these carry over
to retrieval, and against what gold, is still open.
"""

from typing import List, Protocol

from .config import TrainingConfig
from .env import BeliefSample
from .episodes import Episode


class RewardFn(Protocol):
    def __call__(self, samples: List[BeliefSample], episodes: List[Episode]) -> List[float]:
        """One reward per sample, in order."""
        ...


def build_reward(cfg: TrainingConfig) -> RewardFn:
    """The reward of ``cfg.rl``.  TODO."""
    raise NotImplementedError("Phase 3 rewards are TODO (to be designed); see belief_model_training/rewards.py")
