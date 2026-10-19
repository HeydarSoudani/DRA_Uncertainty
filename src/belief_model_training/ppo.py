# The loss functions below are adapted from veRL (verl/trainer/ppo/core_algos.py,
# Copyright 2024 Bytedance Ltd. and/or its affiliates), as vendored by Agent-BRACE
# (meow_tea_train/verl), licensed under the Apache License, Version 2.0
# (http://www.apache.org/licenses/LICENSE-2.0).
"""PPO losses of the belief updater update: the ``vanilla`` dual-clip
policy loss, its aggregation and the KL penalty that Agent-BRACE's
BeliefStateLMTrainer takes from veRL.  The ratio is clipped to
``[1 - clip_ratio_low, 1 + clip_ratio_high]`` (veRL's
``cliprange_low`` / ``cliprange_high``; DAPO's clip-higher when high > low)."""

from typing import Tuple

import torch


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (values * mask).sum() / (mask.sum() + 1e-8)


def agg_loss(loss_mat: torch.Tensor, loss_mask: torch.Tensor, loss_agg_mode: str) -> torch.Tensor:
    """Aggregate a ``(batch, response_length)`` loss matrix into a scalar."""
    if loss_agg_mode == "token-mean":
        return masked_mean(loss_mat, loss_mask)
    if loss_agg_mode == "seq-mean-token-sum":
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1)
        return masked_mean(seq_losses, (torch.sum(loss_mask, dim=-1) > 0).float())
    if loss_agg_mode == "seq-mean-token-mean":
        seq_len = torch.sum(loss_mask, dim=-1)
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1) / (seq_len + 1e-8)
        return masked_mean(seq_losses, (seq_len > 0).float())
    raise ValueError(f"invalid loss_agg_mode: {loss_agg_mode}")


def policy_loss_vanilla(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    clip_ratio_low: float,
    clip_ratio_high: float,
    clip_ratio_c: float = 3.0,
    loss_agg_mode: str = "token-mean",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Dual-clip PPO policy loss; returns ``(pg_loss, pg_clipfrac, ppo_kl)``."""
    if clip_ratio_c <= 1.0:
        raise ValueError(f"clip_ratio_c must be > 1.0, got {clip_ratio_c}")
    negative_approx_kl = torch.clamp(log_prob - old_log_prob, min=-20.0, max=20.0)
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = masked_mean(-negative_approx_kl, response_mask)

    pg_losses1 = -advantages * ratio
    pg_losses2 = -advantages * torch.clamp(ratio, 1 - clip_ratio_low, 1 + clip_ratio_high)
    clip_pg_losses1 = torch.maximum(pg_losses1, pg_losses2)
    pg_clipfrac = masked_mean(torch.gt(pg_losses2, pg_losses1).float(), response_mask)

    pg_losses3 = -advantages * clip_ratio_c
    clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)
    pg_losses = torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1)
    return agg_loss(pg_losses, response_mask, loss_agg_mode), pg_clipfrac, ppo_kl


def kl_penalty(log_prob: torch.Tensor, ref_log_prob: torch.Tensor, kind: str) -> torch.Tensor:
    """Per-token KL estimate to the reference policy (http://joschu.net/blog/kl-approx.html)."""
    if kind in ("kl", "k1"):
        return log_prob - ref_log_prob
    if kind == "abs":
        return (log_prob - ref_log_prob).abs()
    if kind in ("mse", "k2"):
        return 0.5 * (log_prob - ref_log_prob).square()
    if kind in ("low_var_kl", "k3"):
        kl = torch.clamp(ref_log_prob - log_prob, min=-20, max=20)
        return torch.clamp(torch.exp(kl) - kl - 1, min=-10, max=10)
    raise ValueError(f"unknown kl penalty {kind!r}")
