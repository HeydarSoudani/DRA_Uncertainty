"""PPO update of the student belief model, ported from Agent-BRACE's
``BeliefStateLMTrainer.update`` (meow_tea_train/agentic_menu/sync_textworld/
belief_state_lm_trainer.py), in the setting of its recipes: no value
function, no KL to a reference by default.  The advantages are GRPO's
with each reward term normalized apart (computed by the caller,
``group_advantages``).

- Full-parameter HF model in bf16 with ``StochasticRoundingAdamW``: fp32
  moments and an fp32 update, written into the bf16 weights with
  stochastic rounding, so that updates below bf16's resolution are kept on
  average (an fp32 master copy does not fit next to vLLM on one GPU).
- Advantage (GRPO with GDPO's per-term normalization, arXiv 2601.05242):
  each reward term normalized within its group, the replies to the same
  prompt (``(r - mean) / max(std, std_floor)``; 0 for a term with zero
  spread), the weighted sum divided by its standard deviation over the
  batch, broadcast to every reply token.
- The mean token entropy of the replies (at the sampling temperature) is
  reported (``entropy``), to watch for entropy collapse.
- ``log pi_old`` is recomputed by the trainer before the first optimizer
  step (as veRL), so the first ratio is exactly 1; the gap to vLLM's
  log-probs is reported (``rollout_logprob_gap``).  Log-probs are those of
  the logits divided by the sampling temperature.
- ``ppo_epochs`` passes over the batch in shuffled mini-batches; each
  sample is a separate forward pass, and its summed token loss is scaled so
  that a mini-batch's loss is ``loss_agg_mode`` over all its reply tokens
  (``token-mean``: divided by the mini-batch's reply tokens); gradients
  clipped to ``grad_clip``.
- Log-probs are computed only for the reply tokens (``logits_to_keep``),
  as the updater prompts are long.

The model and its optimizer state sit on the GPU only during ``update``
(``onload`` / ``offload``); the rollout engine uses the GPU otherwise.
"""

import gc
import logging
import random
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

from .config import RLConfig
from .ppo import kl_penalty, policy_loss_vanilla

logger = logging.getLogger(__name__)


def group_advantages(terms: List[Dict[str, float]], groups: List[int], weights: Dict[str, float],
                     std_floor: float) -> List[float]:
    """GDPO advantages: per group (the replies to one prompt, which share
    their available terms) and term, the term minus its group mean divided
    by ``max(std, std_floor)`` (0 when the term has zero spread); the
    weighted sum over the terms, divided by its standard deviation over the
    replies of groups with spread (its batch mean is 0).  A group with no
    spread in any term gets 0."""
    by_group: Dict[int, List[int]] = {}
    for i, g in enumerate(groups):
        by_group.setdefault(g, []).append(i)
    adv = [0.0] * len(terms)
    spread: List[int] = []
    for idxs in by_group.values():
        for k in terms[idxs[0]]:
            t = torch.tensor([terms[i][k] for i in idxs], dtype=torch.float32)
            std = t.std(unbiased=False).item()
            if std < 1e-8:
                continue
            z = (t - t.mean()) / max(std, std_floor)
            for i, v in zip(idxs, z.tolist()):
                adv[i] += weights[k] * v
        if any(adv[i] != 0.0 for i in idxs):
            spread += idxs
    if len(spread) > 1:
        scale = torch.tensor([adv[i] for i in spread]).std(unbiased=False).item()
        if scale > 1e-8:
            adv = [a / scale for a in adv]
    return adv


def _stochastic_round_bf16(x: torch.Tensor) -> torch.Tensor:
    """*x* (fp32) rounded to bf16 up or down with probability given by its
    distance to each: random low 16 bits added, then cut off."""
    bits = x.view(torch.int32)
    noise = torch.randint_like(bits, 0, 1 << 16)
    return ((bits + noise) & -65536).view(torch.float32).to(torch.bfloat16)


class StochasticRoundingAdamW(torch.optim.Optimizer):
    """AdamW (``torch.optim.AdamW``'s defaults) for bf16 parameters: the
    moments are fp32, the update is computed in fp32 and the new weights are
    written to bf16 with stochastic rounding."""

    def __init__(self, params, lr: float, betas=(0.9, 0.999), eps: float = 1e-8,
                 weight_decay: float = 0.01) -> None:
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            lr, (b1, b2), eps = group["lr"], group["betas"], group["eps"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if not state:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p, dtype=torch.float32)
                    state["exp_avg_sq"] = torch.zeros_like(p, dtype=torch.float32)
                state["step"] += 1
                t = state["step"]
                g = p.grad.float()
                m, v = state["exp_avg"], state["exp_avg_sq"]
                m.lerp_(g, 1 - b1)
                v.mul_(b2).addcmul_(g, g, value=1 - b2)
                denom = (v / (1 - b2 ** t)).sqrt_().add_(eps)
                x = p.float().mul_(1 - lr * group["weight_decay"])
                x.addcdiv_(m, denom, value=-lr / (1 - b1 ** t))
                p.copy_(_stochastic_round_bf16(x) if p.dtype == torch.bfloat16 else x)


def _sample_weight(mode: str, n_tokens: int, mb_tokens: int, mb_size: int) -> float:
    """Scale of one sample's summed token loss such that the mini-batch's
    loss is ``ppo.agg_loss`` with *mode* over all its samples."""
    if mode == "token-mean":
        return 1.0 / mb_tokens
    if mode == "seq-mean-token-mean":
        return 1.0 / (mb_size * n_tokens)
    if mode == "seq-mean-token-sum":
        return 1.0 / mb_size
    raise ValueError(f"invalid loss_agg_mode: {mode}")


class BeliefPPOTrainer:
    def __init__(self, model_path: str, cfg: RLConfig, device: str = "cuda", seed: int = 0) -> None:
        from transformers import AutoModelForCausalLM

        self.cfg = cfg
        self.device = torch.device(device)
        self._rng = random.Random(seed)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16)
        self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        self.model.config.use_cache = False
        self.optimizer = StochasticRoundingAdamW(self.model.parameters(), lr=cfg.learning_rate)
        self.ref_model = None
        if cfg.use_kl_loss:
            self.ref_model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16)
            self.ref_model.requires_grad_(False)
            self.ref_model.eval()

    # -- placement --

    def _move_optimizer(self, device: torch.device) -> None:
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)

    def onload(self) -> None:
        self.model.to(self.device)
        self._move_optimizer(self.device)

    def offload(self) -> None:
        self.model.to("cpu")
        self._move_optimizer(torch.device("cpu"))
        gc.collect()
        torch.cuda.empty_cache()

    def save(self, path: str, tokenizer=None) -> None:
        self.model.save_pretrained(path)
        if tokenizer is not None:
            tokenizer.save_pretrained(path)

    # -- update --

    def _response_log_probs(self, model, prompt_ids: List[int], response_ids: List[int],
                            entropy: bool = False):
        """log pi(response token | prefix) for every reply token, at the
        sampling temperature; with *entropy*, also the entropy of each
        token's distribution: ``(log_probs, entropies)``."""
        ids = torch.tensor([prompt_ids + response_ids], dtype=torch.long, device=self.device)
        L = len(response_ids)
        logits = model(ids, logits_to_keep=L + 1).logits[0, :-1, :].float() / self.cfg.temperature
        labels = ids[0, -L:]
        log_p = F.log_softmax(logits, dim=-1)
        lp = log_p.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
        if not entropy:
            return lp
        # In chunks of tokens: a reply of thousands of tokens over the whole vocabulary.
        ent = torch.cat([-(c.exp() * c).sum(-1) for c in log_p.split(1024)])
        return lp, ent

    @torch.no_grad()
    def _log_probs(self, model, prompt_ids: List[List[int]], response_ids: List[List[int]],
                   batch: List[int]) -> Dict[int, torch.Tensor]:
        return {i: self._response_log_probs(model, prompt_ids[i], response_ids[i]) for i in batch}

    @torch.no_grad()
    def _log_probs_entropy(self, prompt_ids: List[List[int]], response_ids: List[List[int]],
                           batch: List[int]) -> Tuple[Dict[int, torch.Tensor], torch.Tensor]:
        """The policy's log-probs of *batch* and the mean token entropy over their replies."""
        lps, total, n = {}, 0.0, 0
        for i in batch:
            lps[i], ent = self._response_log_probs(self.model, prompt_ids[i], response_ids[i], entropy=True)
            total += float(ent.sum())
            n += ent.numel()
        return lps, torch.tensor(total / max(n, 1))

    def update(self, prompt_ids: List[List[int]], response_ids: List[List[int]],
               rollout_log_probs: List[List[float]], advantages: List[float],
               rewards: List[float]) -> Dict[str, float]:
        """One PPO update on the rollout batch; returns its metrics.  Samples
        without a reply or with a zero advantage carry no gradient and are
        skipped.  *rollout_log_probs* (vLLM's) are only compared with the
        recomputed ``log pi_old``."""
        cfg = self.cfg
        r = torch.tensor(rewards, dtype=torch.float32)
        batch = [i for i, ids in enumerate(response_ids) if ids and advantages[i] != 0.0]
        if not batch:
            return {"n_samples": 0, "reward_mean": float(r.mean()), "reward_std": float(r.std(unbiased=False))}

        self.model.eval()
        old_lps, entropy = self._log_probs_entropy(prompt_ids, response_ids, batch)
        gap = torch.cat([(torch.tensor(rollout_log_probs[i], device=self.device) - old_lps[i]).abs() for i in batch])
        ref_lps: Dict[int, torch.Tensor] = {}
        if self.ref_model is not None:
            self.ref_model.to(self.device)
            ref_lps = self._log_probs(self.ref_model, prompt_ids, response_ids, batch)
            self.ref_model.to("cpu")
            torch.cuda.empty_cache()

        self.model.train()
        stats = {"pg_loss": [], "pg_clipfrac": [], "ppo_kl": [], "kl_loss": [], "grad_norm": []}
        for _ in range(cfg.ppo_epochs):
            order = list(batch)
            self._rng.shuffle(order)
            for start in range(0, len(order), cfg.ppo_mini_batch_size):
                mb = order[start:start + cfg.ppo_mini_batch_size]
                mb_tokens = sum(len(response_ids[i]) for i in mb)
                self.optimizer.zero_grad()
                for i in mb:
                    n = len(response_ids[i])
                    w = _sample_weight(cfg.loss_agg_mode, n, mb_tokens, len(mb))
                    new_lp = self._response_log_probs(self.model, prompt_ids[i], response_ids[i]).unsqueeze(0)
                    old_lp = old_lps[i].unsqueeze(0)
                    mask = torch.ones_like(old_lp)
                    a = torch.full_like(old_lp, float(advantages[i]))
                    # The summed token loss of the sample (one sequence).
                    pg_sum, clipfrac, ppo_kl = policy_loss_vanilla(
                        old_lp, new_lp, a, mask, cfg.clip_ratio_low, cfg.clip_ratio_high, cfg.clip_ratio_c,
                        "seq-mean-token-sum",
                    )
                    loss = pg_sum
                    if self.ref_model is not None:
                        kl_sum = kl_penalty(new_lp, ref_lps[i].unsqueeze(0), cfg.kl_loss_type).sum()
                        loss = loss + cfg.kl_loss_coef * kl_sum
                        stats["kl_loss"].append(kl_sum.item() / n)
                    (loss * w).backward()
                    stats["pg_loss"].append(pg_sum.item() / n)
                    stats["pg_clipfrac"].append(clipfrac.item())
                    stats["ppo_kl"].append(ppo_kl.item())
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip)
                stats["grad_norm"].append(float(grad_norm))
                self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)

        out = {k: sum(v) / len(v) for k, v in stats.items() if v}
        out.update(n_samples=len(batch), reward_mean=float(r.mean()), reward_std=float(r.std(unbiased=False)),
                   rollout_logprob_gap=float(gap.mean()), entropy=float(entropy))
        return out
