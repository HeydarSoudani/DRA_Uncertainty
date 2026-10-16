"""Real GRPO backend — TODO (veRL adapter).

The weight-update math (FSDP actor, KL to ref, importance-sampling / staleness
correction, optimizer step) is delegated to veRL, the RL library most search
agents are trained with (Search-R1 among them), rather than hand-rolled.

Samples are per turn: each ``Turn`` of each trajectory is one
``(prompt_token_ids, gen_token_ids)`` record whose response mask is
``gen_token_ids`` and whose advantage is the trajectory's.  To check against
the installed veRL: whether it can train on samples produced by this
package's rollout driver, or whether the environment must be wrapped as a
veRL agent loop instead (the environment itself is unchanged either way).

TODO(grpo-backend):
  * construct the veRL trainer from cfg.trainer.extra,
  * in update(): feed each sample's token_ids + loss_mask + per-token advantages
    (``advantage.per_token_advantages``) into the engine's optimization step,
  * apply KL(cfg.kl_coef) to the SFT ref and the decoupled IS/staleness term,
  * bump the weight version and implement push_weights() as a GPU-GPU broadcast
    to the policy servers (see weight_sync.py).
"""

from __future__ import annotations

from typing import Any, Dict, List

from .base import TrainerBackend
from ..config import TrainerConfig
from ..data.schema import Trajectory


class GRPOBackend(TrainerBackend):
    def __init__(self, cfg: TrainerConfig) -> None:
        self.cfg = cfg
        self._version = 0
        # TODO(grpo-backend): initialize the veRL engine, ref model, optimizer.

    @property
    def version(self) -> int:
        return self._version

    def update(self, groups: List[List[Trajectory]]) -> Dict[str, Any]:
        raise NotImplementedError(
            "GRPOBackend is a TODO. Delegate the GRPO/FSDP update to veRL "
            "(advantages come from trainer.advantage). Use MockTrainer to validate "
            "the loop plumbing first."
        )
