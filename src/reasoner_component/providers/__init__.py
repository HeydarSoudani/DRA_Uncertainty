"""How a hosted-API call reaches its provider.

``routing``         a model name → the LiteLLM config / client that serves it
``litellm_client``  ``LiteLLMClient``: the transport (retries, sync over async)
"""

from .litellm_client import LiteLLMClient, is_context_window_error
from .routing import get_litellm_client, get_openai_client, resolve_provider_config

__all__ = [
    "LiteLLMClient",
    "is_context_window_error",
    "get_litellm_client",
    "get_openai_client",
    "resolve_provider_config",
]
