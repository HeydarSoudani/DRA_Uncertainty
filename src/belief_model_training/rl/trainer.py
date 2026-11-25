"""PPO update of the student belief model, ported from Agent-BRACE's
``BeliefStateLMTrainer.update`` (no value function, no KL to a reference by
default; the advantages are computed by the caller, ``rl.loop``).

- Full-parameter HF model in bf16 with ``StochasticRoundingAdamW``: fp32
  moments and an fp32 update, written into the bf16 weights with
  stochastic rounding (an fp32 master copy does not fit next to vLLM on one
  GPU).  A vision-language student (Qwen3.5) keeps its vision encoder
  frozen, outside the optimizer.
- ``log pi_old`` is recomputed by the trainer before the first optimizer
  step (as veRL), so the first ratio is exactly 1; the gap to vLLM's
  log-probs is reported (``rollout_logprob_gap``).  Log-probs are those of
  the logits divided by the sampling temperature.
- ``ppo_epochs`` passes over the batch in shuffled mini-batches of
  ``ppo_mini_batch_chains`` chains (all the calls of a chain in one
  mini-batch); each sample is a separate forward pass, and its summed
  token loss is scaled so a mini-batch's loss is ``loss_agg_mode`` over its
  reply tokens; gradients clipped to ``grad_clip``.
- The mean token entropy of the replies is reported (entropy collapse).

The model and its optimizer state sit on the GPU only during ``update``
(``onload`` / ``offload``); the rollout engine uses the GPU otherwise.  On
the CPU they are held in pinned buffers, allocated at the first offload
and reused, so each move is one asynchronous copy per tensor.
"""

import gc
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch

from ..shared.config import RLConfig
from ..shared.student import load_student
from .ppo import kl_penalty, policy_loss_vanilla

# Reply tokens per chunk of the log-prob and entropy computation.
LOGIT_CHUNK = 1024


@dataclass
class Sample:
    """One trained call: its tokens, vLLM's log-probs and its advantage."""
    prompt_ids: List[int]
    response_ids: List[int]
    rollout_log_probs: List[float]
    advantage: float


def _stochastic_round_bf16(x: torch.Tensor) -> torch.Tensor:
    """*x* (fp32) rounded to bf16 up or down with probability given by its
    distance to each: random low 16 bits added, then cut off."""
    bits = x.view(torch.int32)
    noise = torch.randint_like(bits, 0, 1 << 16)
    return ((bits + noise) & -65536).view(torch.float32).to(torch.bfloat16)


class StochasticRoundingAdamW(torch.optim.Optimizer):
    """AdamW (``torch.optim.AdamW``'s defaults) for bf16 parameters: fp32
    moments and update, new weights written to bf16 with stochastic rounding."""

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
    loss aggregates its samples' token losses with *mode* (veRL's ``loss_agg_mode``)."""
    if mode == "token-mean":
        return 1.0 / mb_tokens
    if mode == "seq-mean-token-mean":
        return 1.0 / (mb_size * n_tokens)
    if mode == "seq-mean-token-sum":
        return 1.0 / mb_size
    raise ValueError(f"invalid loss_agg_mode: {mode}")


def _chunk_log_probs(logits: torch.Tensor, labels: torch.Tensor, temperature: float):
    z = logits.float() / temperature
    lse = torch.logsumexp(z, dim=-1)
    lp = z.gather(-1, labels.unsqueeze(-1)).squeeze(-1) - lse
    p = torch.softmax(z, dim=-1)
    return lp, lse - (p * z).sum(-1)


def token_log_probs(logits: torch.Tensor, labels: torch.Tensor, temperature: float, entropy: bool = False):
    """log softmax(logits / temperature) at *labels*, per token, in fp32
    chunks of ``LOGIT_CHUNK`` tokens; with *entropy*, also each token's
    entropy: ``(log_probs, entropies)``."""
    if not entropy:
        lps = []
        for c, y in zip(logits.split(LOGIT_CHUNK), labels.split(LOGIT_CHUNK)):
            z = c.float() / temperature
            lps.append(z.gather(-1, y.unsqueeze(-1)).squeeze(-1) - torch.logsumexp(z, dim=-1))
        return torch.cat(lps)
    parts = [_chunk_log_probs(c, y, temperature) for c, y in zip(logits.split(LOGIT_CHUNK), labels.split(LOGIT_CHUNK))]
    return torch.cat([lp for lp, _ in parts]), torch.cat([e for _, e in parts])


class BeliefPPOTrainer:
    def __init__(self, model_path: str, cfg: RLConfig, device: str = "cuda", seed: int = 0) -> None:
        self.cfg = cfg
        self.device = torch.device(device)
        self._rng = random.Random(seed)
        self.model = load_student(model_path)
        self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        self.model.config.use_cache = False
        self.optimizer = StochasticRoundingAdamW([p for p in self.model.parameters() if p.requires_grad],
                                                 lr=cfg.learning_rate)
        # Pinned CPU copies of the parameters and optimizer state (``offload``), by tensor.
        self._pinned: Dict[Tuple[int, str], torch.Tensor] = {}
        self.ref_model = None
        if cfg.use_kl_loss:
            self.ref_model = load_student(model_path)
            self.ref_model.requires_grad_(False)
            self.ref_model.eval()

    # -- placement --

    def _to_pinned(self, key: Tuple[int, str], tensor: torch.Tensor) -> torch.Tensor:
        """*tensor* copied (asynchronously) into its pinned CPU buffer."""
        buf = self._pinned.get(key)
        if buf is None or buf.shape != tensor.shape or buf.dtype != tensor.dtype:
            buf = torch.empty(tensor.shape, dtype=tensor.dtype, device="cpu", pin_memory=True)
            self._pinned[key] = buf
        buf.copy_(tensor, non_blocking=True)
        return buf

    def onload(self) -> None:
        for p in self.model.parameters():
            p.data = p.data.to(self.device, non_blocking=True)
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(self.device, non_blocking=True)
        self.model.to(self.device)  # the buffers
        torch.cuda.synchronize()

    def offload(self) -> None:
        if next(self.model.parameters()).device.type != "cpu":
            for p in self.model.parameters():
                p.data = self._to_pinned((id(p), "param"), p.data)
            for p, state in self.optimizer.state.items():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor) and v.device.type != "cpu":
                        state[k] = self._to_pinned((id(p), k), v)
            torch.cuda.synchronize()
        self.model.to("cpu")  # the buffers
        gc.collect()
        torch.cuda.empty_cache()

    def save(self, path: str, processor=None) -> None:
        self.model.save_pretrained(path)
        if processor is not None:
            processor.save_pretrained(path)

    # -- update --

    def _response_log_probs(self, model, s: Sample, entropy: bool = False):
        ids = torch.tensor([s.prompt_ids + s.response_ids], dtype=torch.long, device=self.device)
        L = len(s.response_ids)
        logits = model(ids, logits_to_keep=L + 1).logits[0, :-1, :]
        return token_log_probs(logits, ids[0, -L:], self.cfg.temperature, entropy)

    @torch.no_grad()
    def _old_log_probs(self, samples: List[Sample], batch: List[int]) -> Tuple[Dict[int, torch.Tensor], float]:
        lps, total, n = {}, 0.0, 0
        for i in batch:
            lps[i], ent = self._response_log_probs(self.model, samples[i], entropy=True)
            total += float(ent.sum())
            n += ent.numel()
        return lps, total / max(n, 1)

    def update(self, samples: List[Sample], units: List[List[int]]) -> Dict[str, float]:
        """One PPO update; *units* are the samples of each chain (a
        mini-batch takes whole chains).  Samples without a reply or with a
        zero advantage carry no gradient and are skipped."""
        cfg = self.cfg
        trained = {i for i, s in enumerate(samples) if s.response_ids and s.advantage != 0.0}
        units = [[i for i in u if i in trained] for u in units]
        units = [u for u in units if u]
        batch = [i for u in units for i in u]
        if not batch:
            return {"n_samples": 0}

        self.model.eval()
        old_lps, entropy = self._old_log_probs(samples, batch)
        gap = torch.cat([(torch.tensor(samples[i].rollout_log_probs, device=self.device) - old_lps[i]).abs()
                         for i in batch])
        ref_lps: Dict[int, torch.Tensor] = {}
        if self.ref_model is not None:
            self.ref_model.to(self.device)
            with torch.no_grad():
                ref_lps = {i: self._response_log_probs(self.ref_model, samples[i]) for i in batch}
            self.ref_model.to("cpu")
            torch.cuda.empty_cache()

        self.model.train()
        stats = {"pg_loss": [], "pg_clipfrac": [], "ppo_kl": [], "kl_loss": [], "grad_norm": []}
        for _ in range(cfg.ppo_epochs):
            order = list(units)
            self._rng.shuffle(order)
            for start in range(0, len(order), cfg.ppo_mini_batch_chains):
                mb = [i for u in order[start:start + cfg.ppo_mini_batch_chains] for i in u]
                mb_tokens = sum(len(samples[i].response_ids) for i in mb)
                self.optimizer.zero_grad()
                for i in mb:
                    s = samples[i]
                    n = len(s.response_ids)
                    w = _sample_weight(cfg.loss_agg_mode, n, mb_tokens, len(mb))
                    new_lp = self._response_log_probs(self.model, s).unsqueeze(0)
                    old_lp = old_lps[i].unsqueeze(0)
                    mask = torch.ones_like(old_lp)
                    a = torch.full_like(old_lp, float(s.advantage))
                    pg_sum, clipfrac, ppo_kl = policy_loss_vanilla(
                        old_lp, new_lp, a, mask, cfg.clip_ratio_low, cfg.clip_ratio_high, cfg.clip_ratio_c)
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
        out.update(n_samples=len(batch), rollout_logprob_gap=float(gap.mean()), entropy=entropy)
        return out
