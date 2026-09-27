"""GRPO advantage computation — INFRASTRUCTURE (not designable).

Group Relative Policy Optimization: the advantage of each trajectory in a group
is its return normalized by the group's mean and std.  This is standard and
shared by the mock and real backends; only the weight-update step differs.

    A_i = (R_i - mean(R)) / (std(R) + eps)

Every sample (policy turn) of trajectory i gets A_i: the per-token advantage
broadcasts A_i onto each sample's generated tokens (loss_mask == 1); prompt
tokens, which hold every observation, get 0.
"""

from __future__ import annotations

from typing import List

from ..data.schema import Trajectory


def group_advantages(group: List[Trajectory], eps: float = 1e-6) -> List[float]:
    """Compute + assign trajectory-level GRPO advantages for one group.

    Requires each trajectory to have ``reward.total`` set.  Mutates
    ``traj.advantage`` in place and returns the advantages.
    """
    returns = [t.reward.total if t.reward else 0.0 for t in group]
    n = len(returns)
    mean = sum(returns) / n if n else 0.0
    var = sum((r - mean) ** 2 for r in returns) / n if n else 0.0
    std = var ** 0.5
    advs = [(r - mean) / (std + eps) for r in returns]
    for t, a in zip(group, advs):
        t.advantage = a
    return advs


def per_token_advantages(traj: Trajectory) -> List[List[float]]:
    """Per sample: A_i on generated tokens, 0 on prompt tokens."""
    a = traj.advantage or 0.0
    return [[a if m == 1 else 0.0 for m in turn.loss_mask] for turn in traj.samples()]
