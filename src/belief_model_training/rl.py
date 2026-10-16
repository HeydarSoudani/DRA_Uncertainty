"""Phase 3: RL of the student belief updater (Agent-BRACE's belief PPO with
the agent frozen).

Each iteration, on one GPU shared by the vLLM engine and the trainer:

    rollout   episodes_per_batch phase 1 episodes (replayed in full, or the
              first max_episode_steps steps), one sample per updater call
    reward    one scalar per sample (rewards.build_reward; TODO)
    update    engine.sleep -> trainer.onload -> PPO -> save to sync_dir
              -> trainer.offload -> engine.load_weights(sync_dir)

Only episodes whose teacher replay matched (up to the first mismatching
step) are used.  Outputs (``phase3_rl/``): ``metrics.jsonl`` (one line per
iteration), ``step_{n}/`` every ``save_every`` iterations, ``final/``.
"""

import json
import logging
import random
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List

from .config import RunPaths, TrainingConfig
from .env import BeliefSample, replay_rollout
from .episodes import DocumentStore, read_episodes
from .rewards import build_reward
from .rollout import VLLMRolloutEngine
from .trainer import BeliefPPOTrainer

logger = logging.getLogger(__name__)


def rollout_stats(samples: List[BeliefSample]) -> Dict[str, Any]:
    n = max(len(samples), 1)
    return {
        "num_calls": len(samples),
        "parsed_rate": sum(s.outcome.get("parsed", False) for s in samples) / n,
        "truncated_rate": sum(s.finish_reason == "length" for s in samples) / n,
        "prompt_too_long": sum(s.finish_reason == "prompt_too_long" for s in samples),
        "mean_response_tokens": sum(len(s.response_ids) for s in samples) / n,
        "mean_prompt_tokens": sum(len(s.prompt_ids) for s in samples) / n,
    }


def run_rl(cfg: TrainingConfig, paths: RunPaths) -> None:
    rc = cfg.rl
    if rc.env != "replay":
        raise NotImplementedError(f"rl.env={rc.env!r}: only 'replay' is implemented (online is TODO)")
    reward_fn = build_reward(cfg)

    if not paths.episodes_file.exists():
        raise FileNotFoundError(f"{paths.episodes_file} not found; run phase 1 first")
    episodes = [ep for ep in read_episodes(paths.episodes_file) if ep.replay_steps]
    init = rc.student_init or str(paths.sft_merged)
    print(f"[phase 3] {len(episodes)} episodes; student {init}; {rc.num_iterations} iterations "
          f"of {rc.episodes_per_batch} episodes")

    engine = VLLMRolloutEngine(
        init, gpu_memory_utilization=rc.vllm_gpu_memory_utilization,
        max_model_len=rc.vllm_max_model_len, max_num_seqs=rc.vllm_max_num_seqs, seed=cfg.shared.seed,
    )
    trainer = BeliefPPOTrainer(init, rc, seed=cfg.shared.seed)
    store = DocumentStore()
    rng = random.Random(cfg.shared.seed)
    sync_dir = Path(rc.sync_dir) / cfg.shared.run_name
    paths.rl_dir.mkdir(parents=True, exist_ok=True)

    for it in range(1, rc.num_iterations + 1):
        started = time.monotonic()
        batch = rng.sample(episodes, min(rc.episodes_per_batch, len(episodes)))
        samples = replay_rollout(batch, store, engine, temperature=rc.temperature,
                                 max_new_tokens=rc.max_new_tokens, max_steps=rc.max_episode_steps)
        rewards = reward_fn(samples, batch)
        rollout_seconds = time.monotonic() - started

        engine.sleep()
        trainer.onload()
        metrics = trainer.update([s.prompt_ids for s in samples], [s.response_ids for s in samples],
                                 [s.old_log_probs for s in samples], rewards)
        trainer.save(str(sync_dir), engine.tokenizer)
        trainer.offload()
        engine.load_weights(str(sync_dir))

        record = {"iteration": it, **rollout_stats(samples), **metrics,
                  "rollout_seconds": round(rollout_seconds, 1),
                  "seconds": round(time.monotonic() - started, 1)}
        with open(paths.rl_metrics, "a") as f:
            f.write(json.dumps(record) + "\n")
        print(f"[phase 3] {json.dumps(record)}", flush=True)
        if it % rc.save_every == 0 or it == rc.num_iterations:
            target = paths.rl_final if it == rc.num_iterations else paths.rl_dir / f"step_{it}"
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(sync_dir, target)
    print(f"[phase 3] final model saved to {paths.rl_final}")
