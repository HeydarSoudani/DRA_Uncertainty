"""PPO update of the student belief updater, ported from Agent-BRACE's
``BeliefStateLMTrainer.update`` (meow_tea_train/agentic_menu/sync_textworld/
belief_state_lm_trainer.py), in the setting of its recipes: no value
function, no KL to a reference by default.

- Full-parameter HF model in bf16 with a plain AdamW optimizer.
- Advantage: the samples' scalar rewards whitened over the batch
  (``(r - mean) / std``; 0 for a batch of one or with zero spread),
  broadcast to every reply token.
- ``ppo_epochs`` passes over the batch in shuffled mini-batches; each
  sample is a separate forward pass and its loss (``loss_agg_mode`` over its
  own tokens) is divided by the mini-batch size; gradients clipped to
  ``grad_clip``.
- Log-probs are computed only for the reply tokens (``logits_to_keep``),
  as the updater prompts are long.

The model and its optimizer state sit on the GPU only during ``update``
(``onload`` / ``offload``); the rollout engine uses the GPU otherwise.
"""

import gc
import logging
import random
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F

from .config import RLConfig
from .ppo import agg_loss, kl_penalty, policy_loss_vanilla

logger = logging.getLogger(__name__)


def whiten(rewards: torch.Tensor) -> torch.Tensor:
    if len(rewards) < 2:
        return torch.zeros_like(rewards)
    std = rewards.std(unbiased=False).item()
    if std < 1e-8:
        return torch.zeros_like(rewards)
    return (rewards - rewards.mean()) / (std + 1e-8)


class BeliefPPOTrainer:
    def __init__(self, model_path: str, cfg: RLConfig, device: str = "cuda", seed: int = 0) -> None:
        from transformers import AutoModelForCausalLM

        self.cfg = cfg
        self.device = torch.device(device)
        self._rng = random.Random(seed)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16)
        self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        self.model.config.use_cache = False
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=cfg.learning_rate)
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

    def _response_log_probs(self, model, prompt_ids: List[int], response_ids: List[int]) -> torch.Tensor:
        """log pi(response token | prefix) for every reply token."""
        ids = torch.tensor([prompt_ids + response_ids], dtype=torch.long, device=self.device)
        L = len(response_ids)
        logits = model(ids, logits_to_keep=L + 1).logits[0, :-1, :].float()
        labels = ids[0, -L:]
        return F.log_softmax(logits, dim=-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)

    def update(self, prompt_ids: List[List[int]], response_ids: List[List[int]],
               old_log_probs: List[List[float]], rewards: List[float]) -> Dict[str, float]:
        """One PPO update on the rollout batch; returns its metrics."""
        cfg = self.cfg
        batch = [i for i, r in enumerate(response_ids) if r and len(old_log_probs[i]) == len(r)]
        if not batch:
            return {"n_samples": 0}
        r = torch.tensor([rewards[i] for i in batch], dtype=torch.float32)
        adv = whiten(r)

        ref_lps: Dict[int, torch.Tensor] = {}
        if self.ref_model is not None:
            self.ref_model.to(self.device)
            with torch.no_grad():
                for i in batch:
                    ref_lps[i] = self._response_log_probs(self.ref_model, prompt_ids[i], response_ids[i])
            self.ref_model.to("cpu")
            torch.cuda.empty_cache()

        self.model.train()
        stats = {"pg_loss": [], "pg_clipfrac": [], "ppo_kl": [], "kl_loss": [], "grad_norm": []}
        for _ in range(cfg.ppo_epochs):
            order = list(range(len(batch)))
            self._rng.shuffle(order)
            for start in range(0, len(order), cfg.ppo_mini_batch_size):
                mb = order[start:start + cfg.ppo_mini_batch_size]
                self.optimizer.zero_grad()
                for j in mb:
                    i = batch[j]
                    new_lp = self._response_log_probs(self.model, prompt_ids[i], response_ids[i]).unsqueeze(0)
                    old_lp = torch.tensor([old_log_probs[i]], dtype=torch.float32, device=self.device)
                    mask = torch.ones_like(old_lp)
                    a = torch.full_like(old_lp, float(adv[j]))
                    pg_loss, clipfrac, ppo_kl = policy_loss_vanilla(
                        old_lp, new_lp, a, mask, cfg.clip_ratio, cfg.clip_ratio_c, cfg.loss_agg_mode,
                    )
                    loss = pg_loss
                    if self.ref_model is not None:
                        kl = agg_loss(kl_penalty(new_lp, ref_lps[i].unsqueeze(0), cfg.kl_loss_type),
                                      mask, cfg.loss_agg_mode)
                        loss = loss + cfg.kl_loss_coef * kl
                        stats["kl_loss"].append(kl.item())
                    (loss / len(mb)).backward()
                    stats["pg_loss"].append(pg_loss.item())
                    stats["pg_clipfrac"].append(clipfrac.item())
                    stats["ppo_kl"].append(ppo_kl.item())
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip)
                stats["grad_norm"].append(float(grad_norm))
                self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)

        out = {k: sum(v) / len(v) for k, v in stats.items() if v}
        out.update(n_samples=len(batch), reward_mean=float(r.mean()), reward_std=float(r.std(unbiased=False)))
        return out
