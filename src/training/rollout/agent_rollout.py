"""Async multi-turn agent rollout — agent-agnostic driver.

The environment (``rollout.env.AgentEnv``) owns the agent's protocol; this
driver only alternates policy calls and environment steps, and records every
policy generation as one training sample (``prompt_token_ids`` masked,
``gen_token_ids`` trained on).  It deliberately does NOT reuse the inference
agents' loops (those are text-only), only whatever pieces an environment
imports from them.
"""

from __future__ import annotations

import logging

from ..config import RolloutConfig
from ..data.schema import PromptRecord, Trajectory, Turn
from .env import AgentEnv
from .masking import validate_trajectory
from .policy_client import ContextLengthExceeded, PolicyClient

logger = logging.getLogger(__name__)


async def rollout_once(
    prompt: PromptRecord,
    policy: PolicyClient,
    env: AgentEnv,
    cfg: RolloutConfig,
    *,
    group_id: str,
) -> Trajectory:
    """Run a single trajectory for ``prompt`` and return it (reward unset).

    A policy or tool failure propagates (the caller drops the group: the
    failure says nothing about the policy); a context overflow ends the run as
    ``context_overflow``, which the policy caused by its transcript.
    """
    traj = Trajectory(prompt_id=prompt.id, group_id=group_id, policy_version=policy.version)
    await env.reset(prompt)

    for i in range(cfg.max_generations):
        messages = env.messages()
        params = env.gen_params()
        try:
            gen = await policy.generate(messages, max_tokens=params.max_tokens,
                                        temperature=cfg.temperature, stop=params.stop)
        except ContextLengthExceeded as exc:
            logger.info("rollout %s: context overflow at generation %d (%s)", prompt.id, i, exc)
            traj.stop_reason = "context_overflow"
            break
        res = await env.step(gen.text)
        traj.turns.append(Turn(
            index=i,
            text=gen.text,
            action_kind=res.action_kind,
            query=res.query,
            valid=res.valid,
            messages=messages,
            prompt_token_ids=gen.prompt_token_ids,
            gen_token_ids=gen.token_ids,
            gen_logprobs=gen.logprobs,
            finish_reason=gen.finish_reason,
            obs_text=res.obs_text,
            obs_doc_ids=res.obs_doc_ids,
            info=res.info,
        ))
        if res.done:
            traj.final_answer = res.final_answer or ""
            traj.stop_reason = res.end_reason or "done"
            break
    else:
        traj.stop_reason = "max_generations"

    traj.meta = env.summary()
    validate_trajectory(traj)
    return traj
