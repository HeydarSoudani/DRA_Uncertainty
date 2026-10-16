"""Policy clients — the generation endpoint a rollout samples from.

The inference ``reasoner_component`` generators return TEXT only (no token ids /
logprobs), so they cannot drive RL as-is.  This module defines training-side
clients over chat messages:

    PolicyClient        — interface: generate(messages) -> Generation; .version
    MockPolicyClient    — pure-Python, CPU-only; emits a valid think/search/answer
                          protocol so the whole loop is testable with no server.
    ServerPolicyClient  — self-hosted vLLM server: the chat template is applied
                          here with the policy's own tokenizer, the prompt goes
                          out as token ids, and the generated token ids +
                          logprobs come back (what RL trains on).
    ApiPolicyClient     — text-only hosted model through ``reasoner_component``
                          (a teacher generating SFT trajectories; no tokens).
"""

from __future__ import annotations

import abc
import asyncio
import logging
import random
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class Generation:
    """One generation step's output."""
    text: str
    token_ids: List[int] = field(default_factory=list)
    prompt_token_ids: List[int] = field(default_factory=list)
    logprobs: Optional[List[float]] = None
    finish_reason: str = "stop"


class ContextLengthExceeded(RuntimeError):
    """The context no longer fits the policy's window: the run cannot continue."""


class PolicyClient(abc.ABC):
    """Generation endpoint whose weights change across training steps.

    ``version`` is the weight version currently served; rollouts stamp it onto
    their trajectories so the buffer can form same-version GRPO groups and bound
    staleness.
    """

    def __init__(self) -> None:
        self._version = 0

    @property
    def version(self) -> int:
        return self._version

    def set_version(self, v: int) -> None:
        self._version = v

    @abc.abstractmethod
    async def generate(self, messages: List[Dict[str, str]], *, max_tokens: int,
                       temperature: float, stop: Optional[List[str]] = None) -> Generation:
        ...

    async def close(self) -> None:  # optional cleanup hook
        return None


# ---------------------------------------------------------------------------
# Mock (CPU, no server)
# ---------------------------------------------------------------------------

def _toy_tokenize(text: str) -> List[int]:
    """Whitespace toy tokenizer -> stable int ids (no real tokenizer needed)."""
    return [(abs(hash(tok)) % 50000) + 1 for tok in text.split()]


def _render_messages(messages: List[Dict[str, str]]) -> str:
    return "\n".join(f"<|{m['role']}|>\n{m['content']}" for m in messages)


class MockPolicyClient(PolicyClient):
    """Emits a valid protocol for smoke-testing the loop on CPU.

    It searches until the context holds ``search_turns`` ``<information>``
    blocks, then answers.  For a synthetic question ``i`` the answer is the gold
    ``answer-i`` with probability ``answer_accuracy``, so GRPO groups show reward
    variance.  With ``malformed_rate`` > 0 some turns carry no action, which
    exercises the environments' format re-asks.
    """

    _SYN_RE = re.compile(r"synthetic question (\d+)")

    def __init__(self, search_turns: int = 2, answer_accuracy: float = 0.5,
                 malformed_rate: float = 0.0, seed: int = 0) -> None:
        super().__init__()
        self.search_turns = search_turns
        self.answer_accuracy = answer_accuracy
        self.malformed_rate = malformed_rate
        self._rng = random.Random(seed)

    async def generate(self, messages, *, max_tokens, temperature, stop=None) -> Generation:
        context = [m["content"] for m in messages if m["role"] != "system"]
        n = sum(c.count("</information>") for c in context)
        question = context[0].splitlines()[0] if context and context[0] else ""
        if self._rng.random() < self.malformed_rate:
            text = "<think>let me think about it</think> I am not sure what to do."
        elif n < self.search_turns:
            text = (f"<think>need more evidence for step {n}</think>\n"
                    f"<search>query {n} about {question[-40:].strip()} v{self._version}</search>")
        else:
            m = self._SYN_RE.search(question)
            if m and self._rng.random() < self.answer_accuracy:
                answer = f"answer-{int(m.group(1))}"
            else:
                answer = "answer-wrong"
            text = f"<think>enough evidence</think>\n<answer>{answer}</answer>"
        return Generation(text=text, token_ids=_toy_tokenize(text),
                          prompt_token_ids=_toy_tokenize(_render_messages(messages)))


# ---------------------------------------------------------------------------
# vLLM server client (token-level)
# ---------------------------------------------------------------------------

class ServerPolicyClient(PolicyClient):
    """vLLM ``/v1/completions`` client returning token ids + logprobs.

    The chat template is applied HERE with the policy's tokenizer and the prompt
    is sent as token ids, so the ids the trainer sees are exactly the ids the
    server generated from; the server returns the generated ids
    (``return_token_ids``, vLLM >= 0.10.2).  ``chat_template_kwargs`` reaches the
    template (e.g. ``{"enable_thinking": false}`` for Qwen3.x, matching the
    inference agents' reasoning switch).  Stop strings are kept in the output
    so text and token ids stay aligned.

    The served weight version is set by weight sync after each update.
    """

    def __init__(self, server_url: str, model: str, *, tokenizer: Optional[str] = None,
                 chat_template_kwargs: Optional[Dict[str, Any]] = None,
                 request_timeout: float = 600.0, max_retries: int = 3) -> None:
        super().__init__()
        base = server_url.rstrip("/")
        self.base_url = base[:-3] if base.endswith("/v1") else base
        self.model = model
        self.tokenizer_name = tokenizer or model
        self.chat_template_kwargs = dict(chat_template_kwargs or {})
        self.request_timeout = request_timeout
        self.max_retries = max_retries
        self._tokenizer = None
        self._http = None
        self._logged_template = False

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            from transformers import AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
        return self._tokenizer

    def encode_messages(self, messages: List[Dict[str, str]]) -> List[int]:
        """Token ids of *messages* under the policy's chat template, ready to generate."""
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, **self.chat_template_kwargs)
        if not self._logged_template:
            # Templates differ in what the kwargs do (a thinking-only template
            # opens <think> whatever enable_thinking says): show the tail once.
            self._logged_template = True
            logger.info("policy prompt ends with %r (chat_template_kwargs=%s)",
                        text[-120:], self.chat_template_kwargs)
        return list(self.tokenizer.encode(text, add_special_tokens=False))

    async def _post(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        import httpx

        if self._http is None:
            self._http = httpx.AsyncClient(timeout=self.request_timeout)
        url = f"{self.base_url}/v1/completions"
        for attempt in range(self.max_retries + 1):
            try:
                resp = await self._http.post(url, json=payload)
            except (httpx.TransportError, httpx.TimeoutException) as exc:
                if attempt == self.max_retries:
                    raise
                logger.warning("policy server call failed (%s); retry %d", exc, attempt + 1)
                await asyncio.sleep(2 ** attempt)
                continue
            if resp.status_code == 400 and ("context length" in resp.text or "max_model_len" in resp.text
                                            or "maximum context" in resp.text):
                raise ContextLengthExceeded(resp.text[:500])
            if resp.status_code >= 500 and attempt < self.max_retries:
                logger.warning("policy server %d; retry %d", resp.status_code, attempt + 1)
                await asyncio.sleep(2 ** attempt)
                continue
            resp.raise_for_status()
            return resp.json()
        raise RuntimeError("unreachable")

    async def generate(self, messages, *, max_tokens, temperature, stop=None) -> Generation:
        prompt_ids = self.encode_messages(messages)
        payload: Dict[str, Any] = {
            "model": self.model,
            "prompt": prompt_ids,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "logprobs": 1,
            "return_token_ids": True,
            "include_stop_str_in_output": True,
            "skip_special_tokens": True,
        }
        if stop:
            payload["stop"] = stop
        data = await self._post(payload)
        choice = data["choices"][0]
        token_ids = choice.get("token_ids")
        if token_ids is None:
            raise RuntimeError("policy server returned no token_ids: needs vLLM >= 0.10.2 "
                               "(completions request field return_token_ids)")
        served_prompt = choice.get("prompt_token_ids")
        if served_prompt is not None and list(served_prompt) != prompt_ids:
            logger.warning("server prompt ids differ from the local encoding (%d vs %d tokens)",
                           len(served_prompt), len(prompt_ids))
        logprobs = None
        lp = choice.get("logprobs")
        if lp and lp.get("token_logprobs") is not None:
            logprobs = [float(x) if x is not None else 0.0 for x in lp["token_logprobs"]]
            if len(logprobs) != len(token_ids):
                logger.warning("logprob/token length mismatch (%d vs %d); dropping logprobs",
                               len(logprobs), len(token_ids))
                logprobs = None
        return Generation(text=choice.get("text") or "", token_ids=list(token_ids),
                          prompt_token_ids=prompt_ids, logprobs=logprobs,
                          finish_reason=choice.get("finish_reason") or "stop")

    async def close(self) -> None:
        if self._http is not None:
            await self._http.aclose()
            self._http = None


# ---------------------------------------------------------------------------
# Hosted text-only client (SFT teacher)
# ---------------------------------------------------------------------------

class ApiPolicyClient(PolicyClient):
    """A hosted model driven like the inference agents drive it; no token ids.

    For SFT data generation only: RL needs token ids and logprobs, which a
    hosted API does not return.
    """

    def __init__(self, model: str, *, disable_native_thinking: bool = True,
                 request_timeout: Optional[float] = None) -> None:
        super().__init__()
        from .env_llm import TextLLM
        self.model = model
        self.llm = TextLLM(model, disable_native_thinking=disable_native_thinking,
                           request_timeout=request_timeout)

    async def generate(self, messages, *, max_tokens, temperature, stop=None) -> Generation:
        text = await asyncio.to_thread(self.llm.complete, messages, temperature=temperature,
                                       max_tokens=max_tokens, stop=stop)
        return Generation(text=text)
