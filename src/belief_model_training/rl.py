"""Phase 3: RL of the student belief model (GRPO on whole belief chains,
PPO update from Agent-BRACE's BeliefStateLMTrainer, agent frozen).

The episodes come from the off-mode run of phase 1 on the ``rl`` split, the
validation from the monitor run on the ``val`` split (its trajectories
only; ``runs.load_episodes``); only queries with gold criteria are used.
Each iteration, on one GPU shared by the vLLM engine and the trainer:

    sample    episodes_per_batch rl episodes (env.EpisodeSampler, balanced
              on the correctness of the final answer)
    rollout   group_size chains per episode (env.run_chains, temperature
              1): the student's updater over the whole episode from its
              gold criteria; a group whose chains all score alike gives no
              gradient and is replaced, up to REFILL_ROUNDS more rollouts
              (DAPO's dynamic sampling)
    reward    per chain (rewards.Reward): R = w_format format + w_assign assign
              + w_cal cal + w_final r_final (assign and cal per criterion,
              then averaged over the criteria); on a short-answer query
              r_final is the final scorer's, and each of the group_size
              replies to a chain's scorer call has its own
    update    GDPO advantages in two streams (rewards.py): per chain (each
              term normalized within its group of chains), given to every
              belief update call of the chain; per scorer reply (r_final
              normalized within the replies to one scorer call) -> engine.sleep ->
              trainer.onload -> PPO on all the calls -> save to sync_dir ->
              trainer.offload -> engine.load_weights(sync_dir)

Validation, before training and every ``val_every`` iterations: one greedy
chain per val episode (a fixed sample), scored as in training.  ``final/``
holds the checkpoint with the best mean validation reward (the start model
when none beats it).  Each record holds the mean reward, its terms and
``scoring.measures`` over the chains (per criterion, and the certainty
score against correctness).

Outputs (``phase3_rl/``): ``metrics.jsonl`` (one line per iteration),
``validation.jsonl`` (one line per validation), ``step_{n}/`` every
``save_every`` iterations (the last ``save_total_limit`` kept), ``final/``.
A run starts over: its earlier logs and ``step_{n}/`` checkpoints are
removed.  An iteration where no group has spread makes no update.
"""

import json
import math
import random
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import RunPaths, TrainingConfig
from .env import Chain, EpisodeSampler, run_chains
from .episodes import DocumentStore, Episode
from .rewards import TERMS, Reward
from .rollout import VLLMRolloutEngine
from .runs import check_same_settings, load_episodes, nugget_gold
from .scoring import chain_values, mean_or_none, measures
from .sft import load_processor
from .trainer import BeliefPPOTrainer, group_advantages

# Extra rollouts per iteration to replace the groups without spread.
REFILL_ROUNDS = 2


def score_chains(chains: List[Chain], reward_fn: Reward) -> List[Dict[str, Any]]:
    """The reward of each chain, after the gold of all their values is
    resolved in one batch, with ``scorer_terms`` the terms of each reply
    to its scorer call (empty without one)."""
    reward_fn.gold.resolve([v for c in chains
                            for v in chain_values(c.episode, [cc.outcome for cc in c.calls], c.beliefs)])
    return [{**reward_fn(c.episode, [cc.call for cc in c.calls], [cc.outcome for cc in c.calls],
                         [cc.finish_reason for cc in c.calls], c.beliefs),
             "scorer_terms": reward_fn.scorer_terms(c.episode, [cc.outcome for cc in c.scorer],
                                                    [cc.finish_reason for cc in c.scorer])}
            for c in chains]


def score_stats(scores: List[Dict[str, Any]], prefix: str = "") -> Dict[str, Any]:
    """Mean reward and mean of each term (over the chains where it applies)."""
    out = {f"{prefix}reward": mean_or_none(s["reward"] for s in scores)}
    out.update({f"{prefix}{k}": mean_or_none(s["terms"][k] for s in scores) for k in TERMS})
    return out


def rollout_stats(chains: List[Chain]) -> Dict[str, Any]:
    calls = [cc for c in chains for cc in c.calls]
    n = max(len(calls), 1)
    return {
        "num_chains": len(chains), "num_calls": len(calls),
        "calls_per_chain": len(calls) / max(len(chains), 1),
        "parsed_rate": sum(cc.outcome.get("parsed", False) for cc in calls) / n,
        "truncated_rate": sum(cc.finish_reason == "length" for cc in calls) / n,
        "prompt_too_long": sum(cc.finish_reason == "prompt_too_long" for cc in calls),
        "mean_response_tokens": sum(len(cc.response_ids) for cc in calls) / n,
        "mean_prompt_tokens": sum(len(cc.prompt_ids) for cc in calls) / n,
        "scorer_replies": len(replies := [cc for c in chains for cc in c.scorer]),
        "scorer_parsed_rate": (sum(cc.outcome.get("parsed", False) for cc in replies) / len(replies)
                               if replies else None),
    }


def _spread(terms: List[Dict[str, Optional[float]]], keys) -> bool:
    return any(len({round(t[k], 12) for t in terms if t.get(k) is not None}) > 1 for k in keys)


def _has_spread(scores: List[Dict[str, Any]]) -> bool:
    """Some term of the update calls differs between the chains of a group,
    or r_final between the replies to one of their scorer calls."""
    if _spread([s["chain_terms"] for s in scores], TERMS):
        return True
    return any(_spread(s["scorer_terms"], ("final",)) for s in scores if s["scorer_terms"])


def training_calls(chains: List[Chain], scores: List[Dict[str, Any]], reward_fn: Reward, std_floor: float):
    """``(call, advantage, reward)`` of every call trained: each belief update
    call with its chain's advantage (``chain_terms`` within the chains'
    groups), each scorer reply with its own (r_final within the replies to
    its call)."""
    chain_adv = group_advantages([s["chain_terms"] for s in scores], [c.group for c in chains],
                                 reward_fn.weights, std_floor)
    out = [(cc, a, s["reward"]) for c, a, s in zip(chains, chain_adv, scores) for cc in c.update_calls]
    replies, terms, groups = [], [], []
    for i, (c, s) in enumerate(zip(chains, scores)):
        own = s["scorer_terms"]
        if not own:
            continue
        replies += c.scorer
        terms += own
        groups += [i] * len(own)
    scorer_adv = group_advantages(terms, groups, {"final": reward_fn.weights["final"]}, std_floor)
    out += [(cc, a, t["final"] if t["final"] is not None else 0.0) for cc, a, t in zip(replies, scorer_adv, terms)]
    return out


def validate_chains(episodes: List[Episode], store: DocumentStore, engine: VLLMRolloutEngine,
                    reward_fn: Reward) -> Dict[str, Any]:
    """One greedy chain per episode, scored."""
    chains = run_chains(episodes, store, engine, group_size=1, temperature=0.0)
    scores = score_chains(chains, reward_fn)
    return {"episodes": len(episodes), **rollout_stats(chains), **score_stats(scores),
            **measures([s["score"] for s in scores])}


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


def start_over(paths: RunPaths, sync_dir: Path) -> None:
    """Remove the logs, ``step_{n}/`` checkpoints and synced weights of an
    earlier run (the checkpoint rotation keeps the highest step numbers)."""
    paths.rl_dir.mkdir(parents=True, exist_ok=True)
    for path in (paths.rl_metrics, paths.rl_validation):
        path.unlink(missing_ok=True)
    for old in paths.rl_dir.glob("step_*"):
        shutil.rmtree(old, ignore_errors=True)
    # The weights of the last update; written by the first update.
    shutil.rmtree(sync_dir, ignore_errors=True)


def run_rl(cfg: TrainingConfig, paths: RunPaths) -> None:
    rc = cfg.rl
    reward_fn = Reward(rc, nugget_gold(cfg))
    check_same_settings(cfg, ["rl", "val"])
    episodes = [ep for ep in load_episodes(cfg, "rl", teacher=False, tag="[phase 3]") if ep.steps]
    if not episodes:
        raise SystemExit("[phase 3] no rl episodes with gold criteria and a search step")
    rng = random.Random(cfg.shared.seed)
    val_episodes = [ep for ep in load_episodes(cfg, "val", teacher=False, tag="[phase 3]") if ep.steps]
    if len(val_episodes) > rc.val_max_episodes:
        val_episodes = random.Random(cfg.shared.seed).sample(val_episodes, rc.val_max_episodes)
    init = rc.init_belief_model or str(paths.sft_merged)
    print(f"[phase 3] {len(episodes)} rl episodes, {len(val_episodes)} validation episodes; student {init}; "
          f"{rc.num_iterations} iterations of {rc.episodes_per_batch} episodes x {rc.group_size} chains; "
          f"reward weights {reward_fn.weights}")

    store = DocumentStore()
    sampler = EpisodeSampler(episodes, rng)
    engine = VLLMRolloutEngine(
        init, gpu_memory_utilization=rc.vllm_gpu_memory_utilization,
        max_model_len=cfg.shared.max_seq_len, max_num_seqs=rc.vllm_max_num_seqs, seed=cfg.shared.seed,
    )
    trainer = BeliefPPOTrainer(init, rc, seed=cfg.shared.seed)
    processor = load_processor(init)
    sync_dir = Path(rc.sync_dir) / cfg.shared.training_data / cfg.shared.policy_model
    start_over(paths, sync_dir)

    def validate(it: int) -> Optional[float]:
        if not val_episodes:
            return None
        record = {"iteration": it, **validate_chains(val_episodes, store, engine, reward_fn)}
        _log(paths.rl_validation, record)
        print(f"[phase 3] validation {json.dumps(_round(record))}", flush=True)
        return record["reward"]

    best = validate(0)
    # The start model is final/ until a checkpoint beats it.
    shutil.rmtree(paths.rl_final, ignore_errors=True)
    trainer.save(str(paths.rl_final), processor)

    for it in range(1, rc.num_iterations + 1):
        started = time.monotonic()
        generated: List[Chain] = []
        all_scores: List[Dict[str, Any]] = []
        kept: List[Chain] = []
        kept_scores: List[Dict[str, Any]] = []
        n_groups = skipped = rounds = 0
        # Episodes drawn in this iteration: a refill draws new ones.
        drawn: set = set()
        for rounds in range(1, REFILL_ROUNDS + 2):
            batch = sampler.sample(rc.episodes_per_batch - n_groups, exclude=drawn)
            if not batch:
                break
            chains = run_chains(batch, store, engine, group_size=rc.group_size, temperature=rc.temperature)
            scores = score_chains(chains, reward_fn)
            generated += chains
            all_scores += scores
            for g in range(len(batch)):
                group = slice(g * rc.group_size, (g + 1) * rc.group_size)
                if not _has_spread(scores[group]):
                    skipped += 1
                    continue
                for c in chains[group]:
                    c.group = n_groups
                kept += chains[group]
                kept_scores += scores[group]
                n_groups += 1
            if n_groups >= rc.episodes_per_batch:
                break
        calls = training_calls(kept, kept_scores, reward_fn, rc.adv_std_floor)
        rollout_seconds = time.monotonic() - started

        update_seconds = sync_seconds = 0.0
        if calls:
            engine.sleep()
            trainer.onload()
            update_started = time.monotonic()
            metrics = trainer.update([cc.prompt_ids for cc, _, _ in calls], [cc.response_ids for cc, _, _ in calls],
                                     [cc.rollout_log_probs for cc, _, _ in calls], [a for _, a, _ in calls],
                                     [r for _, _, r in calls])
            # The two streams' rewards are on other scales (R of the chain,
            # r_final of a scorer reply): logged apart, not as one mean.
            for key in ("reward_mean", "reward_std"):
                metrics.pop(key, None)
            update_seconds = time.monotonic() - update_started
            sync_started = time.monotonic()
            trainer.save(str(sync_dir), processor)
            trainer.offload()
            engine.load_weights(str(sync_dir))
            sync_seconds = time.monotonic() - sync_started
        else:
            # No group with spread: no gradient, the weights stay.
            metrics = {"n_samples": 0}

        record = {"iteration": it, "groups": n_groups, "skipped_groups": skipped, "rollout_rounds": rounds,
                  **rollout_stats(generated), **score_stats(all_scores), **metrics,
                  "update_reward_mean": mean_or_none(r for cc, _, r in calls if not cc.call.is_scorer),
                  "scorer_reward_mean": mean_or_none(r for cc, _, r in calls if cc.call.is_scorer),
                  "rollout_seconds": round(rollout_seconds, 1), "update_seconds": round(update_seconds, 1),
                  "sync_seconds": round(sync_seconds, 1), "seconds": round(time.monotonic() - started, 1)}
        _log(paths.rl_metrics, record)
        print(f"[phase 3] {json.dumps(_round(record))}", flush=True)

        if not sync_dir.exists():
            # No update yet: the start model, saved as final/, is the current one.
            continue
        if it % rc.save_every == 0:
            _save_checkpoint(sync_dir, paths.rl_dir, it, rc.save_total_limit)
        if it % rc.val_every == 0 or it == rc.num_iterations:
            reward = validate(it)
            if reward is not None and (best is None or reward > best):
                best = reward
                shutil.rmtree(paths.rl_final, ignore_errors=True)
                shutil.copytree(sync_dir, paths.rl_final)
                print(f"[phase 3] best validation reward {best:.4f} at iteration {it}", flush=True)
    if not val_episodes and sync_dir.exists():
        shutil.rmtree(paths.rl_final, ignore_errors=True)
        shutil.copytree(sync_dir, paths.rl_final)
    print(f"[phase 3] final model (best validation reward {best}) saved to {paths.rl_final}")
