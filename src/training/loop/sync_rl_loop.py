"""Synchronous GRPO driver — build/validate this BEFORE async.

Single weight version at a time: sample groups -> score -> update -> (implicitly
new version).  No buffer, no staleness — the simplest thing that exercises
masking, group formation, advantage shaping, and the trainer contract.  Prove
correctness here, then switch to the async driver.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import List

from ..config import TrainingConfig
from ..data.schema import PromptRecord
from ..rollout.sampler import Sampler
from ..reward.base import RewardModel
from ..trainer.base import TrainerBackend
from ..data.trajectory_io import write_group
from ..utils.logging import log_stats, rollout_stats

logger = logging.getLogger(__name__)


async def run_sync_rl(
    cfg: TrainingConfig,
    prompts: List[PromptRecord],
    sampler: Sampler,
    reward: RewardModel,
    trainer: TrainerBackend,
) -> None:
    rl = cfg.rl
    idx = 0
    for step in range(rl.total_steps):
        # take the next slice of prompts (wrap around)
        batch: List[PromptRecord] = []
        for _ in range(rl.prompts_per_step):
            batch.append(prompts[idx % len(prompts)])
            idx += 1

        results = await asyncio.gather(*[_sample_and_score(sampler, reward, p) for p in batch],
                                       return_exceptions=True)
        groups = []
        for p, res in zip(batch, results):
            if isinstance(res, BaseException):
                # A crashed rollout says nothing about the policy: drop the group.
                logger.error("rollout group for %s failed: %r", p.id, res, exc_info=res)
                continue
            groups.append(res)
        if not groups:
            logger.warning("sync step %d: every group failed; no update", step + 1)
            continue

        stats = trainer.update(groups)
        # keep the policy version in lockstep with the trainer (on-policy)
        trainer.push_weights([sampler.policy])
        stats.update(rollout_stats(groups))
        log_stats(f"[sync step {step + 1}/{rl.total_steps}]", stats)
        save_groups(cfg, f"step{step + 1:05d}", groups)

        if cfg.trainer.save_every and (step + 1) % cfg.trainer.save_every == 0:
            trainer.save(step + 1)


async def _sample_and_score(sampler: Sampler, reward: RewardModel, prompt: PromptRecord):
    group = await sampler.sample_group(prompt)
    await reward.score_group(group, prompt)
    return group


def save_groups(cfg: TrainingConfig, tag: str, groups) -> None:
    if not (cfg.output_dir and cfg.save_rollouts):
        return
    root = Path(cfg.output_dir) / "rollouts"
    for g in groups:
        write_group(root, tag, g)
