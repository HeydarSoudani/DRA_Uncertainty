"""``APIGenerator``: hosted APIs (Claude / OpenAI / OpenRouter) behind ``BaseGenerator``.

The model name is resolved to its provider from environment variables
(``providers.routing.resolve_provider_config``); every call goes through a
``LiteLLMClient`` (``providers.litellm_client``).
"""

from typing import Any, Dict, List, Optional, Tuple, Type

from .base import BaseGenerator, T
from .providers import get_litellm_client


class APIGenerator(BaseGenerator):
    """A hosted-API model.

    Args:
        model_name:  A known slug (``"gpt-4.1"``, ``"claude-sonnet-4-6"``, ...) or
                     ``openrouter/<vendor>/<model>``.
        temperature, max_tokens, **kwargs:  Default request parameters, passed to
                     LiteLLM (top_p, request_timeout, api_key, ...); each call can
                     override them.
        metadata:    Added to every call's metadata (cost tracking).
    """

    def __init__(
        self,
        model_name: str,
        temperature: float = 0.0,
        max_tokens: int = 4000,
        metadata: Optional[Dict[str, str]] = None,
        **kwargs,
    ):
        self._client = get_litellm_client(
            model_name=model_name, temperature=temperature, max_tokens=max_tokens, metadata=metadata, **kwargs,
        )

    @property
    def token_meter(self):
        """Cumulative token usage of every call."""
        return self._client.token_meter

    def set_extra_body(self, **fields: Any) -> "APIGenerator":
        """Merge *fields* into the ``extra_body`` of every call (e.g. an
        OpenRouter ``provider`` pin); returns the generator."""
        self._client.set_extra_body(**fields)
        return self

    def disable_thinking(self) -> "APIGenerator":
        self._client.disable_thinking()
        return self

    def complete(self, messages: List[Dict[str, Any]], **kwargs) -> str:
        return self._client.complete(messages, **kwargs)

    async def acomplete(self, messages: List[Dict[str, Any]], **kwargs) -> str:
        return await self._client.acomplete(messages, **kwargs)

    def complete_with_structured_output(self, messages: List[Dict[str, Any]], response_format: Type[T],
                                        **kwargs) -> T:
        return self._client.complete_with_structured_output(messages, response_format, **kwargs)

    async def acomplete_with_structured_output(self, messages: List[Dict[str, Any]], response_format: Type[T],
                                               **kwargs) -> T:
        return await self._client.acomplete_with_structured_output(messages, response_format, **kwargs)

    def complete_with_stop_on_pattern(self, messages: List[Dict[str, Any]], stop_patterns: List[str],
                                      **kwargs) -> Tuple[str, Optional[str]]:
        return self._client.complete_with_stop_on_pattern(messages, stop_patterns, **kwargs)

    async def acomplete_with_stop_on_pattern(self, messages: List[Dict[str, Any]], stop_patterns: List[str],
                                             **kwargs) -> Tuple[str, Optional[str]]:
        return await self._client.acomplete_with_stop_on_pattern(messages, stop_patterns, **kwargs)

    def complete_with_tools(self, messages: List[Dict[str, Any]], tools: List[Dict[str, Any]],
                            **kwargs) -> Tuple[str, List[Dict[str, Any]]]:
        return self._client.complete_with_tools(messages, tools, **kwargs)

    async def acomplete_with_tools(self, messages: List[Dict[str, Any]], tools: List[Dict[str, Any]],
                                   **kwargs) -> Tuple[str, List[Dict[str, Any]]]:
        return await self._client.acomplete_with_tools(messages, tools, **kwargs)

    def cleanup(self) -> None:
        self._client.cleanup()
