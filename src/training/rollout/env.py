"""Agent environment interface: everything about a rollout except the policy.

The rollout driver (``agent_rollout.rollout_once``) is agent-agnostic: it asks
the environment for the chat messages of the next policy call, generates, and
hands the text back.  The environment owns the agent's protocol: the prompts,
parsing a turn, format re-asks, calling the search tool, rendering what the
policy reads next, and deciding when the run ends.

One environment instance serves ONE rollout (``reset`` is called once), so it
may keep per-run state freely.  Shared resources (search tool, auxiliary LLMs)
are handed in by the factory that builds it.

An agent is plugged into training by passing an ``EnvFactory`` to
``training.pipeline.run``; the agent-specific environment lives with its entry
script (e.g. ``experiments/dra_uncertainty_aware_train.py``).
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from ..data.schema import PromptRecord


@dataclass
class GenParams:
    """Generation settings the environment asks for on its next policy call."""
    max_tokens: int = 512
    stop: Optional[List[str]] = None


@dataclass
class StepResult:
    """The environment's reading of one policy generation.

    ``action_kind``: "search" | "answer" | "retry" (malformed, re-asked) |
    "stop" (malformed, run over).  ``end_reason`` is set when ``done``.
    """
    action_kind: str
    done: bool = False
    valid: bool = True
    query: Optional[str] = None
    final_answer: Optional[str] = None
    end_reason: Optional[str] = None
    obs_text: str = ""
    obs_doc_ids: List[str] = field(default_factory=list)
    info: Dict[str, Any] = field(default_factory=dict)


class AgentEnv(abc.ABC):
    """One rollout's environment."""

    @abc.abstractmethod
    async def reset(self, prompt: PromptRecord) -> None:
        """Start a run for *prompt*."""

    @abc.abstractmethod
    def messages(self) -> List[Dict[str, str]]:
        """Chat messages for the next policy call."""

    def gen_params(self) -> GenParams:
        return GenParams()

    @abc.abstractmethod
    async def step(self, text: str) -> StepResult:
        """Consume one policy generation."""

    def summary(self) -> Dict[str, Any]:
        """Run-level data saved on the trajectory (``Trajectory.meta``)."""
        return {}


# Called once per rollout; returns a fresh environment.
EnvFactory = Callable[[], AgentEnv]
