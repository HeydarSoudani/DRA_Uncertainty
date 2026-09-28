"""Text-only LLM calls through ``reasoner_component`` (READ-ONLY reuse).

Used where training needs a model that is NOT the policy being trained: an
environment's auxiliary model (e.g. a frozen judge), or a
teacher generating SFT trajectories.  Both must behave as in inference, so the
call mirrors ``UncertaintyAwareAgent._call``: the same generator factory, the same
``complete`` kwargs, and the same switch that turns the model's own reasoning
mode off.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from reasoner_component import no_thinking_extra_body


class TextLLM:
    """``complete(messages) -> str`` over a ``reasoner_component`` generator.

    Blocking; async callers run it with ``asyncio.to_thread``.
    """

    def __init__(self, model: str, *, temperature: float = 0.0, max_tokens: int = 1024,
                 disable_native_thinking: bool = True, strip_think: bool = False,
                 api_base: Optional[str] = None, request_timeout: Optional[float] = None) -> None:
        from reasoner_component import create_generator

        kwargs: Dict[str, Any] = {"temperature": temperature, "metadata": {"model": model}}
        if request_timeout is not None:
            kwargs["request_timeout"] = request_timeout
        if model.startswith("vllm/"):
            kwargs.update(api_base=api_base or "http://127.0.0.1:6008/v1", api_key="EMPTY")
        self.model = model
        self.generator = create_generator(model, **kwargs)
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.strip_think = strip_think
        self._extra_body = no_thinking_extra_body(self.generator) if disable_native_thinking else None

    def complete(self, messages: List[Dict[str, str]], *, temperature: Optional[float] = None,
                 max_tokens: Optional[int] = None, stop: Optional[List[str]] = None) -> str:
        kwargs: Dict[str, Any] = {
            "temperature": self.temperature if temperature is None else temperature,
            "max_completion_tokens": max_tokens or self.max_tokens,
            "strip_think": self.strip_think,
        }
        if stop:
            kwargs["stop"] = stop
        if self._extra_body is not None:
            kwargs["extra_body"] = self._extra_body
        return self.generator.complete(messages, **kwargs) or ""

    def __call__(self, messages: List[Dict[str, str]]) -> str:
        return self.complete(messages)
