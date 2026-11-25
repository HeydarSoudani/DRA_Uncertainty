"""``VLLMGenerator``: a self-hosted vLLM server behind ``BaseGenerator``.

Talks to the server's OpenAI-compatible API with the OpenAI SDK.  The servers
themselves (their settings, starting and stopping them) are in
``reasoner_component.vllm_server``.
"""

from __future__ import annotations

import asyncio
import weakref
from typing import Any, Dict, List, Optional, Tuple, Type

from openai import AsyncOpenAI, OpenAI

from utils.token_meter import TokenMeter

from .base import (BaseGenerator, PatternScan, T, json_schema_format, parse_structured, parse_tool_calls,
                   reply_text, resolve_token_cap)
from .vllm_server.registry import bare_model_name, require_spec

# Parameters the OpenAI SDK takes; any other one is a vLLM extension (top_k,
# min_p, repetition_penalty, chat_template_kwargs, ...) and goes in extra_body.
_OPENAI_PARAMS = frozenset({
    "temperature", "top_p", "max_tokens", "max_completion_tokens", "stop", "seed", "n",
    "presence_penalty", "frequency_penalty", "logit_bias", "logprobs", "top_logprobs",
    "response_format", "tools", "tool_choice", "parallel_tool_calls", "user",
    "stream", "stream_options", "timeout",
})


def _merge_extra_body(defaults: Optional[Dict[str, Any]], call: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """The call's ``extra_body`` over the defaults'; ``chat_template_kwargs``
    merged key by key (a per-call one keeps ``disable_thinking``'s switch)."""
    body = {**(defaults or {}), **(call or {})}
    template = {**((defaults or {}).get("chat_template_kwargs") or {}),
                **((call or {}).get("chat_template_kwargs") or {})}
    if template:
        body["chat_template_kwargs"] = template
    return body


class VLLMGenerator(BaseGenerator):
    """The full ``BaseGenerator`` over a vLLM server's OpenAI-compatible API.

    Args:
        model_name:  The served model (a ``vllm/`` prefix is dropped).
        api_base:    The server; by default the registered server of the model.
        temperature, max_tokens, **params:  Default request parameters, each
                     overridable per call.  OpenAI ones (top_p, stop,
                     presence_penalty, ...) go in the request, vLLM's own
                     (top_k, min_p, repetition_penalty, ...) in ``extra_body``.
        timeout:     Seconds per request (``request_timeout``, the API
                     backend's name for it, is taken too, also per call).
        max_retries: Retries, with the SDK's exponential backoff, of a failed
                     connection, a timeout, a 429 or a 5xx reply.  Any other
                     error, a prompt past the context window among them,
                     raises at once.

    Thread-safe: the sync methods share one connection pool, the async ones
    get a client per event loop.
    """

    def __init__(
        self,
        model_name: str,
        api_base: Optional[str] = None,
        api_key: str = "EMPTY",
        *,
        temperature: float = 0.0,
        max_tokens: int = 4000,
        timeout: float = 600.0,
        max_retries: int = 5,
        **params: Any,
    ):
        self.model_name = bare_model_name(model_name)
        if api_base is None:
            api_base = require_spec(self.model_name).api_base
        self.api_base = api_base.rstrip("/")
        self.token_meter = TokenMeter()
        if "request_timeout" in params:
            timeout = params.pop("request_timeout")
        self._params: Dict[str, Any] = {"temperature": temperature, "max_tokens": max_tokens, **params}
        self._client_kwargs = {"base_url": self.api_base, "api_key": api_key,
                               "timeout": timeout, "max_retries": max_retries}
        self._sync = OpenAI(**self._client_kwargs)
        self._async: "weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, AsyncOpenAI]" = weakref.WeakKeyDictionary()

    def disable_thinking(self) -> "VLLMGenerator":
        """Switch the model's own reasoning off for every call (the chat
        template's ``enable_thinking``); returns the generator."""
        body = dict(self._params.get("extra_body") or {})
        body["chat_template_kwargs"] = {**(body.get("chat_template_kwargs") or {}), "enable_thinking": False}
        self._params["extra_body"] = body
        return self

    # ── Plain completion ─────────────────────────────────────────────────────

    def complete(self, messages: List[Dict[str, Any]], *, strip_think: bool = True,
                 return_reasoning_fallback: bool = False, **params: Any) -> str:
        response = self._sync.chat.completions.create(**self._request(messages, params))
        return self._reply(response, strip_think, return_reasoning_fallback)

    async def acomplete(self, messages: List[Dict[str, Any]], *, strip_think: bool = True,
                        return_reasoning_fallback: bool = False, **params: Any) -> str:
        response = await self._async_client().chat.completions.create(**self._request(messages, params))
        return self._reply(response, strip_think, return_reasoning_fallback)

    # ── Structured output (vLLM guided decoding on the JSON schema) ──────────

    def complete_with_structured_output(self, messages: List[Dict[str, Any]], response_format: Type[T],
                                        **params: Any) -> T:
        request = self._request(messages, {**params, "response_format": json_schema_format(response_format)})
        return self._structured(self._sync.chat.completions.create(**request), response_format)

    async def acomplete_with_structured_output(self, messages: List[Dict[str, Any]], response_format: Type[T],
                                               **params: Any) -> T:
        request = self._request(messages, {**params, "response_format": json_schema_format(response_format)})
        return self._structured(await self._async_client().chat.completions.create(**request), response_format)

    # ── Tool calls (the server needs a tool parser: ServerSpec.tool_parser) ──

    def complete_with_tools(self, messages: List[Dict[str, Any]], tools: List[Dict[str, Any]],
                            tool_choice: Any = "auto", *, strip_think: bool = True,
                            **params: Any) -> Tuple[str, List[Dict[str, Any]]]:
        request = self._request(messages, {**params, "tools": tools, "tool_choice": tool_choice})
        return self._tool_reply(self._sync.chat.completions.create(**request), strip_think)

    async def acomplete_with_tools(self, messages: List[Dict[str, Any]], tools: List[Dict[str, Any]],
                                   tool_choice: Any = "auto", *, strip_think: bool = True,
                                   **params: Any) -> Tuple[str, List[Dict[str, Any]]]:
        request = self._request(messages, {**params, "tools": tools, "tool_choice": tool_choice})
        return self._tool_reply(await self._async_client().chat.completions.create(**request), strip_think)

    # ── Stop on a regex ──────────────────────────────────────────────────────
    # For plain strings pass ``stop=[...]`` to ``complete``: the server stops
    # itself.  Here the reply streams and closing the stream on a match makes
    # vLLM abort the request.  An early stop records no token usage (the
    # server sends it with the last chunk only).

    def complete_with_stop_on_pattern(self, messages: List[Dict[str, Any]], stop_patterns: List[str],
                                      **params: Any) -> Tuple[str, Optional[str]]:
        scan = PatternScan(stop_patterns)
        with self._sync.chat.completions.create(**self._stream_request(messages, params)) as stream:
            for chunk in stream:
                if scan.feed(chunk, self.token_meter):
                    break
        return scan.result()

    async def acomplete_with_stop_on_pattern(self, messages: List[Dict[str, Any]], stop_patterns: List[str],
                                             **params: Any) -> Tuple[str, Optional[str]]:
        scan = PatternScan(stop_patterns)
        stream = await self._async_client().chat.completions.create(**self._stream_request(messages, params))
        async with stream:
            async for chunk in stream:
                if scan.feed(chunk, self.token_meter):
                    break
        return scan.result()

    def cleanup(self) -> None:
        """Close the connection pools.  An async client is closed on its own
        loop when that loop is idle (a closed loop took its sockets along)."""
        self._sync.close()
        clients, self._async = list(self._async.items()), weakref.WeakKeyDictionary()
        for loop, client in clients:
            if not loop.is_closed() and not loop.is_running():
                try:
                    loop.run_until_complete(client.close())
                except Exception:  # noqa: BLE001 - best effort
                    pass

    # ── Internals ────────────────────────────────────────────────────────────

    def _request(self, messages: List[Dict[str, Any]], params: Dict[str, Any]) -> Dict[str, Any]:
        """The ``chat.completions.create`` keyword arguments of one call: the
        defaults, the call's *params* on top (its ``chat_template_kwargs``
        merged into the defaults', not replacing them)."""
        overrides = dict(params)
        if "request_timeout" in overrides:
            overrides["timeout"] = overrides.pop("request_timeout")
        merged = {**self._params, **overrides}
        merged.pop("metadata", None)  # LiteLLM cost-tracking tag: not a request field
        resolve_token_cap(merged, overrides)
        extra_body = _merge_extra_body(self._params.get("extra_body"), overrides.get("extra_body"))
        merged.pop("extra_body", None)
        request: Dict[str, Any] = {"model": self.model_name, "messages": messages}
        for key, value in merged.items():
            if value is None:
                continue
            (request if key in _OPENAI_PARAMS else extra_body)[key] = value
        if extra_body:
            request["extra_body"] = extra_body
        return request

    def _stream_request(self, messages: List[Dict[str, Any]], params: Dict[str, Any]) -> Dict[str, Any]:
        return self._request(messages, {**params, "stream": True, "stream_options": {"include_usage": True}})

    def _async_client(self) -> AsyncOpenAI:
        # An async HTTP pool belongs to the event loop it was opened on.
        loop = asyncio.get_running_loop()
        client = self._async.get(loop)
        if client is None:
            client = self._async[loop] = AsyncOpenAI(**self._client_kwargs)
        return client

    def _reply(self, response: Any, strip_think: bool, return_reasoning_fallback: bool) -> str:
        self.token_meter.record_usage(response.usage)
        choice = response.choices[0]
        return reply_text(choice.message, strip_think=strip_think, return_reasoning_fallback=return_reasoning_fallback,
                          finish_reason=choice.finish_reason, model=self.model_name)

    def _structured(self, response: Any, response_format: Type[T]) -> T:
        self.token_meter.record_usage(response.usage)
        return parse_structured(response.choices[0].message.content, response_format)

    def _tool_reply(self, response: Any, strip_think: bool) -> Tuple[str, List[Dict[str, Any]]]:
        self.token_meter.record_usage(response.usage)
        message = response.choices[0].message
        return reply_text(message, strip_think=strip_think), parse_tool_calls(message)

