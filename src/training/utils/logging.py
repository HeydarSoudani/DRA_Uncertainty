"""Lightweight stats logging for the training loops.

Console by default; W&B is optional and guarded so nothing here pulls a heavy
dependency into the CPU smoke path.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

logger = logging.getLogger("training")

_WANDB = None


def init_wandb(project: str, name: str, config: Dict[str, Any]) -> None:  # pragma: no cover
    global _WANDB
    try:
        import wandb  # noqa
        _WANDB = wandb
        wandb.init(project=project, name=name, config=config)
    except Exception:
        logger.warning("wandb unavailable — console logging only")
        _WANDB = None


def log_stats(prefix: str, stats: Dict[str, Any]) -> None:
    kv = "  ".join(
        f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
        for k, v in stats.items()
    )
    logger.info("%s %s", prefix, kv)
    print(f"{prefix} {kv}")
    if _WANDB is not None:  # pragma: no cover
        _WANDB.log(stats)


def rollout_stats(groups) -> Dict[str, Any]:
    """Behaviour of the scored rollouts of one update: answers, length, format."""
    trajs = [t for g in groups for t in g]
    n = len(trajs) or 1
    stats: Dict[str, Any] = {
        "outcome": sum(t.reward.outcome for t in trajs if t.reward) / n,
        "answered": sum(1 for t in trajs if t.answered) / n,
        "gens_per_traj": sum(len(t.turns) for t in trajs) / n,
        "searches_per_traj": sum(t.num_search_turns for t in trajs) / n,
        "invalid_turns": sum(1 for t in trajs for x in t.turns if not x.valid) / n,
        "gen_tokens": sum(len(x.gen_token_ids) for t in trajs for x in t.turns) / n,
    }
    ends: Dict[str, int] = {}
    for t in trajs:
        ends[t.stop_reason] = ends.get(t.stop_reason, 0) + 1
    for reason, count in sorted(ends.items()):
        stats[f"end.{reason}"] = count
    # Groups whose rewards are all equal carry no GRPO signal.
    stats["flat_groups"] = sum(
        1 for g in groups if len({(t.reward.total if t.reward else 0.0) for t in g}) == 1)
    return stats
