"""Phase 3: RL of the student belief model (GRPO on teacher-forced belief
updates, PPO update from Agent-BRACE's BeliefStateLMTrainer, agent frozen).

Each iteration, on one GPU shared by the vLLM engine and the trainer:

    sample    prompts_per_batch updater steps of the ``rl`` episodes
              (env.StepSampler), each prompt built from the teacher's
              belief b_{t-1}
    rollout   group_size student replies per prompt (temperature 1)
    reward    one scalar per reply (rewards.Reward)
    update    GRPO advantages within each group -> engine.sleep ->
              trainer.onload -> PPO -> save to sync_dir -> trainer.offload
              -> engine.load_weights(sync_dir)

Validation, before training and every ``val_every`` iterations: one greedy
reply per step of the ``val`` episodes (at most ``val_max_prompts``, a fixed
sample), its mean reward and terms, and ``scoring.measures`` over its steps.
The teacher's replies to the same steps are scored once, as the bar to beat.
``final/`` holds the checkpoint with the best validation reward (the start
model when none beats it).

Outputs (``phase3_rl/``): ``metrics.jsonl`` (one line per iteration),
``validation.jsonl`` (teacher, then one line per validation),
``step_{n}/`` every ``save_every`` iterations (the last ``save_total_limit``
kept), ``final/``.  A run starts over: its earlier logs are removed.
"""

import json
import logging
import math
import random
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import RunPaths, TrainingConfig
from .env import StepPrompt, StepSampler, StudentSample, apply_reply, build_prompts, rollout_groups, step_items
from .episodes import DocumentStore, read_episodes
from .rewards import Reward
from .rollout import VLLMRolloutEngine
from .scoring import measures, step_row
from .trainer import BeliefPPOTrainer, group_advantages

logger = logging.getLogger(__name__)

TERMS = ("format", "st", "sc", "ts")


def _mean(values: List[Optional[float]]) -> Optional[float]:
    vals = [v for v in values if v is not None]
    return sum(vals) / len(vals) if vals else None


def score_stats(scores: List[Dict[str, Any]], prefix: str = "") -> Dict[str, Any]:
    """Mean reward and mean of each term (over the replies where it applies)."""
    out = {f"{prefix}reward": _mean([s["reward"] for s in scores])}
    out.update({f"{prefix}{k}": _mean([s[k] for s in scores]) for k in TERMS})
    return out


def step_rows(prompts: List[StepPrompt], scores: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """``scoring.measures`` rows of scored replies; a reply that fails the
    format gate is left out (the format rate is reported apart)."""
    return [step_row(p.item.episode, p.item.t, s["cited"], s["coverage"], p.item.last)
            for p, s in zip(prompts, scores) if s["coverage"] is not None]


def evaluate_replies(prompts: List[StepPrompt], texts: List[Optional[str]], finish: List[Optional[str]],
                     reward_fn: Reward) -> Dict[str, Any]:
    scores = [reward_fn(p, apply_reply(p, t), f) for p, t, f in zip(prompts, texts, finish)]
    return {**score_stats(scores), **measures(step_rows(prompts, scores))}


def rollout_stats(samples: List[StudentSample]) -> Dict[str, Any]:
    n = max(len(samples), 1)
    return {
        "num_replies": len(samples),
        "parsed_rate": sum(s.outcome.get("parsed", False) for s in samples) / n,
        "truncated_rate": sum(s.finish_reason == "length" for s in samples) / n,
        "prompt_too_long": sum(s.finish_reason == "prompt_too_long" for s in samples),
        "mean_response_tokens": sum(len(s.response_ids) for s in samples) / n,
        "mean_prompt_tokens": sum(len(s.prompt_ids) for s in samples) / n,
    }


def _round(record: Dict[str, Any]) -> Dict[str, Any]:
    return {k: (round(v, 4) if isinstance(v, float) and not math.isnan(v) else v) for k, v in record.items()}


def _log(path: Path, record: Dict[str, Any]) -> None:
    with open(path, "a") as f:
        f.write(json.dumps(_round(record)) + "\n")


def _save_checkpoint(sync_dir: Path, rl_dir: Path, it: int, keep: int) -> None:
    """Copy the current weights to ``step_{it}/``; only the last *keep* stay."""
    target = rl_dir / f"step_{it}"
    shutil.rmtree(target, ignore_errors=True)
    shutil.copytree(sync_dir, target)
    steps = sorted(rl_dir.glob("step_*"), key=lambda p: int(p.name.split("_", 1)[1]))
    for old in steps[:-keep] if keep > 0 else steps:
        shutil.rmtree(old, ignore_errors=True)


def run_rl(cfg: TrainingConfig, paths: RunPaths) -> None:
    rc = cfg.rl
    reward_fn = Reward(rc)
    if not paths.episodes_file.exists():
        raise FileNotFoundError(f"{paths.episodes_file} not found; run phase 1 first")
    episodes = [ep for ep in read_episodes(paths.episodes_file) if ep.replay_steps]
    items = step_items([ep for ep in episodes if ep.use == "rl"])
    if not items:
        raise SystemExit("[phase 3] no updater steps in the rl episodes")
    rng = random.Random(cfg.shared.seed)
    val_items = step_items([ep for ep in episodes if ep.use == "val"])
    if len(val_items) > rc.val_max_prompts:
        val_items = random.Random(cfg.shared.seed).sample(val_items, rc.val_max_prompts)
    init = rc.init_belief_model or str(paths.sft_merged)
    print(f"[phase 3] {len(items)} rl steps ({sum(i.has_gold for i in items)} with new gold), "
          f"{len(val_items)} validation steps; student {init}; {rc.num_iterations} iterations of "
          f"{rc.prompts_per_batch} prompts x {rc.group_size} replies")

    store = DocumentStore()
    sampler = StepSampler(items, rc.gold_step_share, rc.no_gold_share, rng)
    val_prompts = build_prompts(val_items, store)
    engine = VLLMRolloutEngine(
        init, gpu_memory_utilization=rc.vllm_gpu_memory_utilization,
        max_model_len=cfg.shared.max_seq_len, max_num_seqs=rc.vllm_max_num_seqs, seed=cfg.shared.seed,
    )
    trainer = BeliefPPOTrainer(init, rc, seed=cfg.shared.seed)
    sync_dir = Path(rc.sync_dir) / cfg.shared.run_name
    paths.rl_dir.mkdir(parents=True, exist_ok=True)
    for path in (paths.rl_metrics, paths.rl_validation):
        path.unlink(missing_ok=True)

    def validate(it: int) -> Optional[float]:
        if not val_prompts:
            return None
        gens = [g[0] for g in engine.generate([p.call.messages for p in val_prompts], temperature=0.0,
                                              max_new_tokens=rc.max_new_tokens)]
        record = {"model": "student", "iteration": it,
                  **evaluate_replies(val_prompts, [g.text for g in gens], [g.finish_reason for g in gens], reward_fn)}
        _log(paths.rl_validation, record)
        print(f"[phase 3] validation {json.dumps(_round(record))}", flush=True)
        return record["reward"]

    if val_prompts:
        teacher = [p.item.episode.steps[p.item.t].teacher_reply for p in val_prompts]
        record = {"model": "teacher", "iteration": None,
                  **evaluate_replies(val_prompts, teacher, ["stop"] * len(teacher), reward_fn)}
        _log(paths.rl_validation, record)
        print(f"[phase 3] teacher on validation {json.dumps(_round(record))}", flush=True)
    best = validate(0)
    # The start model is final/ until a checkpoint beats it.
    shutil.rmtree(paths.rl_final, ignore_errors=True)
    trainer.save(str(paths.rl_final), engine.tokenizer)

    for it in range(1, rc.num_iterations + 1):
        started = time.monotonic()
        prompts = build_prompts(sampler.sample(rc.prompts_per_batch), store)
        samples = rollout_groups(prompts, engine, group_size=rc.group_size, temperature=rc.temperature,
                                 max_new_tokens=rc.max_new_tokens)
        scores = [reward_fn(s.prompt, s.outcome, s.finish_reason) for s in samples]
        rewards = [s["reward"] for s in scores]
        advantages = group_advantages(rewards, [s.group for s in samples])
        rollout_seconds = time.monotonic() - started

        engine.sleep()
        trainer.onload()
        update_started = time.monotonic()
        metrics = trainer.update([s.prompt_ids for s in samples], [s.response_ids for s in samples],
                                 [s.rollout_log_probs for s in samples], advantages, rewards)
        update_seconds = time.monotonic() - update_started
        sync_started = time.monotonic()
        trainer.save(str(sync_dir), engine.tokenizer)
        trainer.offload()
        engine.load_weights(str(sync_dir))
        sync_seconds = time.monotonic() - sync_started

        record = {"iteration": it, **rollout_stats(samples), **score_stats(scores), **metrics,
                  "zero_spread_groups": sum(all(a == 0.0 for a in advantages[g * rc.group_size:(g + 1) * rc.group_size])
                                            for g in range(len(prompts))),
                  "rollout_seconds": round(rollout_seconds, 1), "update_seconds": round(update_seconds, 1),
                  "sync_seconds": round(sync_seconds, 1), "seconds": round(time.monotonic() - started, 1)}
        _log(paths.rl_metrics, record)
        print(f"[phase 3] {json.dumps(_round(record))}", flush=True)

        if it % rc.save_every == 0:
            _save_checkpoint(sync_dir, paths.rl_dir, it, rc.save_total_limit)
        if it % rc.val_every == 0 or it == rc.num_iterations:
            reward = validate(it)
            if reward is not None and (best is None or reward > best):
                best = reward
                shutil.rmtree(paths.rl_final, ignore_errors=True)
                shutil.copytree(sync_dir, paths.rl_final)
                print(f"[phase 3] best validation reward {best:.4f} at iteration {it}", flush=True)
    if not val_prompts:
        shutil.rmtree(paths.rl_final, ignore_errors=True)
        shutil.copytree(sync_dir, paths.rl_final)
    print(f"[phase 3] final model (best validation reward {best}) saved to {paths.rl_final}")
