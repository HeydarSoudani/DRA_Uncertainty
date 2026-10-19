"""Phase 3: RL of the student belief model (GRPO on belief updates from the
student's own belief chain, PPO update from Agent-BRACE's
BeliefStateLMTrainer, agent frozen).

The prompts come from the off-mode run of phase 1 on the ``rl`` split, the
validation from the monitor run on the ``val`` split (its trajectories
only; ``runs.load_episodes``).  Each iteration, on one GPU shared by the
vLLM engine and the trainer:

    sample    prompts_per_batch updater steps of the rl episodes
              (env.StepSampler), each from the student's own belief
              b_{t-1}: its criteria and greedy chain (env.student_prompts,
              DAgger-style: the states the student meets when it runs
              alone); a step without a prompt (no criteria parsed, or no
              call there in the student's chain) is left out
    rollout   group_size student replies per prompt (temperature 1); a
              group whose replies all score alike gives no gradient and is
              replaced, up to REFILL_ROUNDS more rollouts (DAPO's dynamic
              sampling)
    reward    per reply (rewards.Reward); r_ts on the episode's last step
    update    GDPO advantages (each term normalized within its group) ->
              engine.sleep -> trainer.onload -> PPO -> save to sync_dir ->
              trainer.offload -> engine.load_weights(sync_dir)

Validation, before training and every ``val_every`` iterations, greedy, on
a fixed sample of val episodes: the student's own chain over each whole
episode (its criteria, then every call), scored against the gold; r_ts on
its last call.  ``final/`` holds the checkpoint with the best validation
reward (the start model when none beats it).  Each record holds the mean
reward, its terms and ``scoring.measures`` over its calls.

Outputs (``phase3_rl/``): ``metrics.jsonl`` (one line per iteration),
``validation.jsonl`` (one line per validation), ``step_{n}/`` every
``save_every`` iterations (the last ``save_total_limit`` kept), ``final/``.
A run starts over: its earlier logs are removed.
"""

import json
import logging
import math
import random
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import RLConfig, RunPaths, TrainingConfig
from .env import (StepPrompt, StepSampler, StudentSample, apply_reply, rollout_groups, run_student_chains,
                  step_items, student_criteria, student_prompts)
from .episodes import DocumentStore, Episode
from .rewards import Reward
from .rollout import VLLMRolloutEngine
from .runs import check_same_settings, load_episodes
from .scoring import measures, step_row
from .trainer import BeliefPPOTrainer, group_advantages

logger = logging.getLogger(__name__)

TERMS = ("format", "st", "sc", "ts")
# Extra rollouts per iteration to replace the groups without spread.
REFILL_ROUNDS = 2


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
    return [step_row(p.item.episode, p.item.t, s["cited"], s["coverage"], p.final, p.gold)
            for p, s in zip(prompts, scores) if s["coverage"] is not None]


def evaluate_replies(prompts: List[StepPrompt], texts: List[Optional[str]], finish: List[Optional[str]],
                     reward_fn: Reward) -> Dict[str, Any]:
    scores = [reward_fn(p, apply_reply(p, t), f) for p, t, f in zip(prompts, texts, finish)]
    return {"calls": len(prompts), **score_stats(scores), **measures(step_rows(prompts, scores))}


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


def _has_spread(scores: List[Dict[str, Any]]) -> bool:
    """Some term differs between the replies of a group."""
    return any(len({round(s["terms"][k], 12) for s in scores}) > 1 for k in scores[0]["terms"])


def free_running(episodes: List[Episode], store: DocumentStore, engine: VLLMRolloutEngine, rc: RLConfig,
                 reward_fn: Reward) -> Dict[str, Any]:
    """The student's own chain over each of *episodes*, greedy: its
    criteria, then every updater call; r_ts on the chain's last call."""
    criteria = student_criteria(episodes, engine)
    keep = [e for e, c in enumerate(criteria) if c]
    chains = run_student_chains([episodes[e] for e in keep], [criteria[e] for e in keep],
                                [len(episodes[e].steps) - 1 for e in keep], store, engine,
                                max_new_tokens=rc.max_new_tokens, reply_at_target=True)
    calls = [cc for chain in chains for cc in chain]
    for chain in chains:
        if chain:
            chain[-1].prompt.final = True
    record = evaluate_replies([cc.prompt for cc in calls], [cc.text for cc in calls],
                              [cc.finish_reason for cc in calls], reward_fn)
    return {"criteria_failed": len(episodes) - len(keep),
            "mean_criteria": _mean([len(criteria[e]) for e in keep]), **record}


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
    check_same_settings(cfg, ["rl", "val"])
    items = step_items(load_episodes(cfg, "rl", teacher=False, tag="[phase 3]"))
    if not items:
        raise SystemExit("[phase 3] no updater steps in the rl episodes")
    rng = random.Random(cfg.shared.seed)
    val_episodes = [ep for ep in load_episodes(cfg, "val", teacher=False, tag="[phase 3]") if ep.steps]
    if len(val_episodes) > rc.val_max_episodes:
        val_episodes = random.Random(cfg.shared.seed).sample(val_episodes, rc.val_max_episodes)
    init = rc.init_belief_model or str(paths.sft_merged)
    print(f"[phase 3] {len(items)} rl steps ({sum(i.has_gold for i in items)} with new gold), "
          f"{len(val_episodes)} validation episodes; student {init}; {rc.num_iterations} iterations of "
          f"{rc.prompts_per_batch} prompts x {rc.group_size} replies")

    store = DocumentStore()
    sampler = StepSampler(items, rc.gold_step_share, rc.no_gold_share, rng)
    engine = VLLMRolloutEngine(
        init, gpu_memory_utilization=rc.vllm_gpu_memory_utilization,
        max_model_len=cfg.shared.max_seq_len, max_num_seqs=rc.vllm_max_num_seqs, seed=cfg.shared.seed,
    )
    trainer = BeliefPPOTrainer(init, rc, seed=cfg.shared.seed)
    sync_dir = Path(rc.sync_dir) / cfg.shared.training_data / cfg.shared.policy_model
    paths.rl_dir.mkdir(parents=True, exist_ok=True)
    for path in (paths.rl_metrics, paths.rl_validation):
        path.unlink(missing_ok=True)

    def validate(it: int) -> Optional[float]:
        if not val_episodes:
            return None
        record = {"iteration": it, **free_running(val_episodes, store, engine, rc, reward_fn)}
        _log(paths.rl_validation, record)
        print(f"[phase 3] validation {json.dumps(_round(record))}", flush=True)
        return record["reward"]

    best = validate(0)
    # The start model is final/ until a checkpoint beats it.
    shutil.rmtree(paths.rl_final, ignore_errors=True)
    trainer.save(str(paths.rl_final), engine.tokenizer)

    for it in range(1, rc.num_iterations + 1):
        started = time.monotonic()
        generated: List[StudentSample] = []
        all_scores: List[Dict[str, Any]] = []
        samples: List[StudentSample] = []
        scores: List[Dict[str, Any]] = []
        n_groups = skipped = no_prompt = rounds = 0
        for rounds in range(1, REFILL_ROUNDS + 2):
            prompts, rest = student_prompts(sampler.sample(rc.prompts_per_batch - n_groups), store, engine,
                                            max_new_tokens=rc.max_new_tokens)
            no_prompt += len(rest)
            new = rollout_groups(prompts, engine, group_size=rc.group_size, temperature=rc.temperature,
                                 max_new_tokens=rc.max_new_tokens)
            new_scores = [reward_fn(s.prompt, s.outcome, s.finish_reason) for s in new]
            generated += new
            all_scores += new_scores
            for g in range(len(prompts)):
                group = slice(g * rc.group_size, (g + 1) * rc.group_size)
                if not _has_spread(new_scores[group]):
                    skipped += 1
                    continue
                for s in new[group]:
                    s.group = n_groups
                samples += new[group]
                scores += new_scores[group]
                n_groups += 1
            if n_groups >= rc.prompts_per_batch:
                break
        advantages = group_advantages([s["terms"] for s in scores], [s.group for s in samples],
                                      reward_fn.weights, rc.adv_std_floor)
        rewards = [s["reward"] for s in scores]
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

        record = {"iteration": it, "groups": n_groups, "skipped_groups": skipped,
                  "rollout_rounds": rounds, "no_prompt": no_prompt,
                  **rollout_stats(generated), **score_stats(all_scores), **metrics,
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
    if not val_episodes:
        shutil.rmtree(paths.rl_final, ignore_errors=True)
        shutil.copytree(sync_dir, paths.rl_final)
    print(f"[phase 3] final model (best validation reward {best}) saved to {paths.rl_final}")
