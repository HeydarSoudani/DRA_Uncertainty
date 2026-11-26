"""LLM reasoner component: every LLM call of the project goes through here.

Every generator implements ``BaseGenerator`` (base.py); there are two kinds,
both in ``providers/``:

  providers/api.py   ``APIGenerator``   hosted APIs: Claude / OpenAI / OpenRouter
  providers/vllm.py  ``VLLMGenerator``  a self-hosted vLLM server (every local model)

``create_generator(model_name)`` (factory.py) picks the right one; ``setup_llm``
builds the pipeline's from its CLI args.

  providers/    the generators and how a call reaches its provider: model-name
                routing and the ``LiteLLMClient`` transport
  vllm_server/  the vLLM servers: their registry (``SERVERS``), starting and
                stopping them, and ``python -m reasoner_component.vllm_server``
"""

from .base import REASONING_FALLBACK_PREFIX, BaseGenerator
from .factory import create_generator, disable_native_thinking, run_servers, setup_llm
from .providers import APIGenerator, LiteLLMClient, VLLMGenerator, get_litellm_client, get_openai_client

__all__ = [
    # Generators
    "BaseGenerator",
    "APIGenerator",
    "VLLMGenerator",
    # Building one
    "create_generator",
    "disable_native_thinking",
    "setup_llm",
    "run_servers",
    # Raw clients
    "LiteLLMClient",
    "get_litellm_client",
    "get_openai_client",
    "REASONING_FALLBACK_PREFIX",
]
