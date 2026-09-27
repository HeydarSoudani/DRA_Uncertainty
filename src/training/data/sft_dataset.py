"""SFT cold-start data: teacher trajectories, filtered, as per-turn chat samples.

The teacher (``sft.teacher_model``, a hosted model) runs through the SAME
environment as RL, so the samples show the policy exactly the context it will
see in RL and at inference.  Trajectories are kept when their outcome reward
reaches ``sft.min_outcome`` (reject sampling) and, with
``sft.drop_format_retries``, only when no turn had to be re-asked.

Each kept policy turn becomes one sample, matching the per-turn layout of RL:

    {"messages": [<context the teacher saw> ..., {"role": "assistant", "content": <turn>}],
     "prompt_id": ..., "turn": i, "outcome": 1.0}

Every mainstream SFT trainer (TRL, LLaMA-Factory, veRL multi-turn SFT) trains on
the final assistant message only, which is the masking this layout needs: the
context, with every observation in it, is never trained on.  The assistant
content is the environment's normalised turn (``Turn.info["target_text"]``,
e.g. the belief agent's ``<think>…</think>\\n<search>…</search>``) when it
provides one, else the raw generation.
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

from .schema import PromptRecord, Trajectory
from ..config import SFTConfig

logger = logging.getLogger(__name__)


def keep_trajectory(traj: Trajectory, cfg: SFTConfig) -> bool:
    if traj.reward is None or traj.reward.outcome < cfg.min_outcome:
        return False
    if cfg.drop_format_retries and any(not t.valid for t in traj.turns):
        return False
    return traj.answered


def trajectory_to_sft_samples(traj: Trajectory) -> List[Dict[str, Any]]:
    samples = []
    for t in traj.turns:
        if not t.valid or not t.messages:
            continue
        target = t.info.get("target_text") or t.text
        samples.append({
            "messages": list(t.messages) + [{"role": "assistant", "content": target}],
            "prompt_id": traj.prompt_id,
            "turn": t.index,
            "action": t.action_kind,
            "outcome": traj.reward.outcome if traj.reward else None,
        })
    return samples


async def generate_sft_data(cfg: SFTConfig, prompts: List[PromptRecord], sampler, reward,
                            out_path: Path, rollouts_dir: Path = None) -> Tuple[Path, Dict[str, Any]]:
    """Roll the teacher out on *prompts*, keep the good trajectories, write samples."""
    from .trajectory_io import write_group

    out_path.parent.mkdir(parents=True, exist_ok=True)
    stats = {"prompts": 0, "failed_prompts": 0, "trajectories": 0, "kept": 0, "samples": 0}
    lock = asyncio.Lock()
    fh = out_path.open("w")

    async def one(p: PromptRecord) -> None:
        try:
            group = await sampler.sample_group(p, group_size=cfg.samples_per_prompt)
            await reward.score_group(group, p)
        except Exception as exc:
            logger.error("SFT rollout for %s failed: %r", p.id, exc)
            stats["failed_prompts"] += 1
            return
        kept = [t for t in group if keep_trajectory(t, cfg)]
        async with lock:
            stats["prompts"] += 1
            stats["trajectories"] += len(group)
            stats["kept"] += len(kept)
            for t in kept:
                for s in trajectory_to_sft_samples(t):
                    fh.write(json.dumps(s, ensure_ascii=False) + "\n")
                    stats["samples"] += 1
            fh.flush()
            if rollouts_dir is not None:
                write_group(rollouts_dir, "teacher", group)
            if stats["prompts"] % 10 == 0:
                logger.info("SFT data: %s", stats)

    try:
        await asyncio.gather(*[one(p) for p in prompts])
    finally:
        fh.close()
    return out_path, stats
