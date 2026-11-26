"""The generators, and how a hosted-API call reaches its provider.

``api``             ``APIGenerator``: hosted APIs (Claude / OpenAI / OpenRouter)
``vllm``            ``VLLMGenerator``: a self-hosted vLLM server
``routing``         a model name → the LiteLLM config / client that serves it
``litellm_client``  ``LiteLLMClient``: the transport (retries, sync over async)
"""

from .api import APIGenerator
from .litellm_client import LiteLLMClient, is_context_window_error
from .routing import get_litellm_client, get_openai_client, resolve_provider_config
from .vllm import VLLMGenerator

__all__ = [
    "APIGenerator",
    "VLLMGenerator",
    "LiteLLMClient",
    "is_context_window_error",
    "get_litellm_client",
    "get_openai_client",
    "resolve_provider_config",
]
