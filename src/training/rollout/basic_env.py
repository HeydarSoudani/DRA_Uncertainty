"""Reference environment: a minimal SearchR1-style search loop.

The default when an entry script plugs in no agent of its own, and the one the
``--smoke`` path of the common pipeline exercises.  It is deliberately small:
no system prompt, no format re-asks; a malformed turn ends the run.  Real
agents bring their own environment (see ``rollout.env``).
"""

from __future__ import annotations

import asyncio
from typing import Dict, List

from ..config import RolloutConfig
from ..data.schema import PromptRecord
from .env import AgentEnv, GenParams, StepResult
from .masking import make_observation_text
from .protocol import parse_action
from .tool_env import ToolEnv


class BasicSearchEnv(AgentEnv):
    def __init__(self, tool: ToolEnv, cfg: RolloutConfig) -> None:
        self.tool = tool
        self.cfg = cfg
        self._prompt: PromptRecord = None
        self._transcript = ""
        self._turns = 0

    async def reset(self, prompt: PromptRecord) -> None:
        self._prompt = prompt
        self._transcript = f"Question: {prompt.question}"
        self._turns = 0

    def messages(self) -> List[Dict[str, str]]:
        return [{"role": "user", "content": self._transcript}]

    def gen_params(self) -> GenParams:
        return GenParams(max_tokens=self.cfg.max_gen_tokens, stop=["</search>", "</answer>"])

    async def step(self, text: str) -> StepResult:
        self._turns += 1
        action = parse_action(text)
        if action.kind == "answer":
            return StepResult("answer", done=True, final_answer=action.answer or "", end_reason="answered")
        if action.kind == "stop":
            return StepResult("stop", done=True, valid=False, end_reason="format_failure")

        docs = await asyncio.to_thread(self.tool.search, action.query,
                                       original_query=self._prompt.question)
        shown = docs[:self.cfg.top_k_docs]
        obs = make_observation_text(shown, max_docs=self.cfg.top_k_docs)
        self._transcript += f"\n\n{text.strip()}\n{obs}\n"
        done = self._turns >= self.cfg.max_turns
        return StepResult(
            "search", done=done, query=action.query,
            end_reason="max_turns" if done else None,
            obs_text=obs, obs_doc_ids=[str(d.get("doc_id") or d.get("id") or "") for d in shown],
        )
