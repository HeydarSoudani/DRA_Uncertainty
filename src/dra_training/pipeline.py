"""Top-level orchestrators: build components from config and run SFT / RL.

Called by the entry scripts (``experiments/dra_*_train.py``).  All the wiring
(which policy / tool / reward / trainer to instantiate) lives here so the
entry scripts stay thin.  An entry script plugs its agent in by passing
``make_env``: given the config and the shared search tool, it returns the
factory that builds one environment per rollout.  Without one, the reference
``BasicSearchEnv`` is used.

Everything defaults to ``mock`` so ``run(TrainingConfig())`` is a full CPU
smoke test with no GPUs, server, or dataset.
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Callable, Optional

from .config import TrainingConfig, to_dict
from .data.prompt_dataset import load_prompts
from .rollout.env import EnvFactory
from .rollout.policy_client import MockPolicyClient, PolicyClient
from .rollout.tool_env import MockToolEnv, ToolEnv
from .rollout.sampler import Sampler
from .reward.base import CompositeReward, RewardModel
from .reward.outcome_reward import OutcomeReward, MockOutcomeReward
from .reward.process_reward import ProcessReward, MockProcessReward
from .trainer.base import TrainerBackend
from .trainer.mock_trainer import MockTrainer
from .trainer.weight_sync import LocalVersionSync
from .buffer.trajectory_buffer import TrajectoryBuffer

logger = logging.getLogger(__name__)

# (config, shared search tool) -> one-environment-per-rollout factory
MakeEnv = Callable[[TrainingConfig, ToolEnv], EnvFactory]


# ---------------------------------------------------------------------------
# Component builders (each honors the "mock" default)
# ---------------------------------------------------------------------------

def _build_policy(cfg: TrainingConfig) -> PolicyClient:
    ro = cfg.rollout
    if ro.policy == "mock":
        return MockPolicyClient(search_turns=min(2, ro.max_turns - 1),
                                malformed_rate=ro.mock_malformed_rate, seed=cfg.rl.seed)
    if ro.policy == "server":
        if not ro.server_url or not ro.model:
            raise ValueError("rollout.server_url and rollout.model required for policy='server'")
        from .rollout.policy_client import ServerPolicyClient
        return ServerPolicyClient(ro.server_url, ro.model, tokenizer=ro.tokenizer,
                                  chat_template_kwargs=ro.chat_template_kwargs,
                                  request_timeout=ro.request_timeout)
    raise ValueError(f"unknown rollout.policy '{ro.policy}'")


def _build_teacher(cfg: TrainingConfig) -> PolicyClient:
    if cfg.rollout.policy == "mock":
        return _build_policy(cfg)
    from .rollout.policy_client import ApiPolicyClient
    return ApiPolicyClient(cfg.sft.teacher_model)


def _build_tool(cfg: TrainingConfig) -> ToolEnv:
    if cfg.rollout.tool == "mock":
        return MockToolEnv(top_k=cfg.rollout.top_k_docs)
    if cfg.rollout.tool == "retrieval":
        # READ-ONLY reuse of the finalized searcher, built exactly as inference does.
        from .rollout.tool_env import build_retrieval_tool
        return build_retrieval_tool(cfg.data, cfg.retrieval, seen_top_k=cfg.rollout.top_k_docs)
    raise ValueError(f"unknown rollout.tool '{cfg.rollout.tool}'")


def _build_reward(cfg: TrainingConfig) -> RewardModel:
    if cfg.reward.outcome_metric == "mock":
        outcome = MockOutcomeReward(cfg.reward)
        process = MockProcessReward(cfg.reward)
    else:
        outcome = OutcomeReward(cfg.reward)
        process = ProcessReward(cfg.reward)      # TODO(reward): real signals (lambda_process = 0 for now)
    return CompositeReward(outcome, process, cfg.reward)


def _build_trainer(cfg: TrainingConfig) -> TrainerBackend:
    if cfg.trainer.backend == "mock":
        return MockTrainer(save_dir=cfg.trainer.save_dir)
    if cfg.trainer.backend == "grpo":
        from .trainer.grpo_backend import GRPOBackend
        return GRPOBackend(cfg.trainer)
    raise ValueError(f"unknown trainer.backend '{cfg.trainer.backend}'")


def _default_make_env(cfg: TrainingConfig, tool: ToolEnv) -> EnvFactory:
    from .rollout.basic_env import BasicSearchEnv
    return lambda: BasicSearchEnv(tool, cfg.rollout)


def _save_run_config(cfg: TrainingConfig) -> None:
    if not cfg.output_dir:
        return
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "run_config.json").write_text(json.dumps(to_dict(cfg), indent=2, default=str))


# ---------------------------------------------------------------------------
# Stage entry points
# ---------------------------------------------------------------------------

def run_rl(cfg: TrainingConfig, make_env: Optional[MakeEnv] = None) -> None:
    """Run the RL stage (sync or async) with the configured components."""
    prompts = load_prompts(cfg.data, seed=cfg.rl.seed)
    if not prompts:
        raise RuntimeError("no prompts loaded")

    policy = _build_policy(cfg)
    tool = _build_tool(cfg)
    reward = _build_reward(cfg)
    trainer = _build_trainer(cfg)
    sampler = Sampler(policy, (make_env or _default_make_env)(cfg, tool), cfg.rollout)
    _save_run_config(cfg)

    print(f"[dra_train] RL stage | mode={cfg.rl.mode} | prompts={len(prompts)} "
          f"| policy={cfg.rollout.policy} tool={cfg.rollout.tool} trainer={cfg.trainer.backend} "
          f"| reward={cfg.reward.outcome_metric}")

    async def main() -> None:
        try:
            if cfg.rl.mode == "sync":
                from .loop.sync_rl_loop import run_sync_rl
                await run_sync_rl(cfg, prompts, sampler, reward, trainer)
            elif cfg.rl.mode == "async":
                from .loop.async_rl_loop import run_async_rl
                buffer = TrajectoryBuffer(
                    group_size=cfg.rollout.group_size,
                    staleness_k=cfg.buffer.staleness_k,
                    capacity=cfg.buffer.capacity,
                )
                weight_sync = LocalVersionSync()  # TODO(weight-sync): BroadcastWeightSync for real backend
                await run_async_rl(cfg, prompts, sampler, reward, trainer, buffer, weight_sync)
            else:
                raise ValueError(f"unknown rl.mode '{cfg.rl.mode}'")
        finally:
            await policy.close()

    asyncio.run(main())
    print("[dra_train] RL stage complete")


def run_sft(cfg: TrainingConfig, make_env: Optional[MakeEnv] = None) -> None:
    """Build the SFT cold-start data with the teacher, then run the SFT backend."""
    from .data.sft_dataset import generate_sft_data

    prompts = load_prompts(cfg.data, seed=cfg.rl.seed)
    if not prompts:
        raise RuntimeError("no prompts loaded")
    out_dir = Path(cfg.output_dir or "run_outputs/training") / "sft"
    out_path = Path(cfg.sft.data_out) if cfg.sft.data_out else out_dir / "sft_data.jsonl"

    teacher = _build_teacher(cfg)
    tool = _build_tool(cfg)
    reward = _build_reward(cfg)
    rollout_cfg = cfg.rollout
    rollout_cfg.temperature = cfg.sft.teacher_temperature
    rollout_cfg.concurrency = cfg.sft.concurrency
    sampler = Sampler(teacher, (make_env or _default_make_env)(cfg, tool), rollout_cfg)
    _save_run_config(cfg)

    teacher_name = "mock" if cfg.rollout.policy == "mock" else cfg.sft.teacher_model
    print(f"[dra_train] SFT data | teacher={teacher_name} | prompts={len(prompts)} "
          f"x {cfg.sft.samples_per_prompt} | reward={cfg.reward.outcome_metric} | out={out_path}")
    rollouts_dir = out_dir / "rollouts" if cfg.save_rollouts else None
    path, stats = asyncio.run(generate_sft_data(cfg.sft, prompts, sampler, reward, out_path, rollouts_dir))
    print(f"[dra_train] SFT data written: {path} | {stats}")

    if cfg.sft.backend == "none":
        return
    from .trainer.sft_backend import run_sft_backend
    run_sft_backend(cfg.sft, str(path))


def run(cfg: TrainingConfig, make_env: Optional[MakeEnv] = None) -> None:
    if cfg.stage == "rl":
        run_rl(cfg, make_env)
    elif cfg.stage == "sft":
        run_sft(cfg, make_env)
    else:
        raise ValueError(f"unknown stage '{cfg.stage}'")
