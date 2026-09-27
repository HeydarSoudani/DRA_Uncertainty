"""Hosted-API LLM generator and provider routing.

A single ``APIGenerator`` covers all hosted commercial providers:
  * Anthropic / Claude
  * OpenAI (Azure-hosted and direct)
  * OpenRouter  (``openrouter/<vendor>/<model>``)

Provider routing (``resolve_provider_config`` / ``get_litellm_client``) resolves a
friendly ``model_name`` into the LiteLLM config dict (model string, api_base,
api_key, api_version) using environment variables. This is the single place that
knows about the supported hosted providers.
"""

import os
from typing import Any, Dict, Optional

from .base import LiteLLMBackedGenerator
from .litellm_client import LiteLLMClient

# OpenRouter OpenAI-compatible endpoint.
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Azure OpenAI models: shared OPENAI_ENDPOINT + OPENAI_API_KEY,
# per-model {PREFIX}_DEPLOYMENT and {PREFIX}_API_VERSION.
AZURE_CONFIG_MAP = {
    "gpt-4o":       "OPENAI_GPT4O",
    "gpt-4o-mini":  "OPENAI_GPT4OMINI",
    "gpt-4.1":      "OPENAI_GPT41",
    "gpt-4.1-mini": "OPENAI_GPT41MINI",
    "gpt-4.1-nano": "OPENAI_GPT41NANO",
}

# Anthropic models: shared ANTHROPIC_ENDPOINT + ANTHROPIC_API_KEY,
# per-model {PREFIX}_DEPLOYMENT and {PREFIX}_API_VERSION (optional).
ANTHROPIC_CONFIG_MAP = {
    "claude-sonnet-4-5": "ANTHROPIC_CLAUDE_SONNET45",
    "claude-sonnet-4-6": "ANTHROPIC_CLAUDE_SONNET46",
    "claude-opus-4-7": "ANTHROPIC_CLAUDE_OPUS47",
}

# OpenAI direct API models (not Azure-hosted).
# Uses OPENAI_DIRECT_API_KEY env var (separate from Azure OPENAI_API_KEY).
OPENAI_DIRECT_CONFIG_MAP = {
    "gpt-5.2": "openai/gpt-5.2",
}


def _openrouter_provider_pin() -> Optional[Dict[str, Any]]:
    """Build OpenRouter's ``provider`` routing block from the environment.

    OpenRouter serves one slug from several hosts, and they do not agree on
    everything an agent parses: reasoning-tag handling, tokenizer quirks and
    tail latency all differ.  Auto-routing can therefore move a long run onto a
    host whose output the caller's parser does not accept, halfway through.
    Pinning makes a run reproducible.

    ``OPENROUTER_PROVIDER_ORDER`` is a comma-separated preference list (e.g.
    ``"Parasail,CoreWeave"``).  Fallbacks are disabled whenever an order is
    given, so an unavailable host fails loudly instead of silently rerouting;
    set ``OPENROUTER_ALLOW_FALLBACKS=1`` to keep them.  Unset, this returns None
    and OpenRouter routes as it likes.
    """
    order = [h.strip() for h in os.getenv("OPENROUTER_PROVIDER_ORDER", "").split(",") if h.strip()]
    if not order:
        return None
    allow_fallbacks = os.getenv("OPENROUTER_ALLOW_FALLBACKS", "").lower() in ("1", "true", "yes")
    return {"order": order, "allow_fallbacks": allow_fallbacks}


def _assert_routable(
    config: Dict[str, Any], model_name: Optional[str], env_prefixed: Dict[str, str]
) -> None:
    """Fail fast when LiteLLM cannot infer a provider for the resolved model.

    A model string with no recognised provider prefix (``qwen/qwen3-32b``
    rather than ``openrouter/qwen/qwen3-32b``) is accepted here but dies deep
    inside LiteLLM, as a ``BadRequestError`` the retry wrapper mistakes for a
    transient failure and repeats seven times.  Checking at resolution time --
    an offline string lookup, no network -- names the offending model and the
    env var that set it, instead.
    """
    import litellm

    model = config.get("model")
    if not isinstance(model, str) or not model:
        return
    try:
        litellm.get_llm_provider(
            model,
            custom_llm_provider=config.get("custom_llm_provider"),
            api_base=config.get("api_base"),
        )
    except Exception as exc:
        source = (
            f"env var {env_prefixed[model_name]}_DEPLOYMENT"
            if model_name in env_prefixed
            else f"model name '{model_name}'"
        )
        raise ValueError(
            f"LiteLLM cannot route model string '{model}' (from {source}): no "
            f"provider could be inferred.  Prefix it with its provider, e.g. "
            f"'anthropic/{model}' or 'openrouter/{model}'.  "
            f"Providers: https://docs.litellm.ai/docs/providers"
        ) from exc


def resolve_provider_config(model_name: Optional[str] = None, **kwargs) -> Dict[str, Any]:
    """Resolve *model_name* into a LiteLLM config dict.

    Routing (in priority order):
      1. ``openrouter/<name>``           → OpenRouter endpoint + OPENROUTER_API_KEY
                                           (+ provider pin, see _openrouter_provider_pin)
      2. known Azure OpenAI slug         → Azure deployment/version/endpoint/key
      3. known Anthropic/Claude slug     → Anthropic deployment/version/endpoint/key
      4. known OpenAI-direct slug        → openai/<model> + OPENAI_DIRECT_API_KEY
      5. otherwise                       → caller must supply ``model=`` via kwargs

    All extra *kwargs* (temperature, max_tokens, api_key, ...) override/supplement
    the resolved config. ``model_name == "baseline"`` skips the missing-model check.
    """
    config: Dict[str, Any] = {}

    if model_name and model_name.startswith("openrouter/"):
        bare = model_name.removeprefix("openrouter/")
        config["model"] = f"openrouter/{bare}"
        config["api_base"] = OPENROUTER_BASE_URL
        if api_key := os.getenv("OPENROUTER_API_KEY"):
            config["api_key"] = api_key
        # Ask for the chain of thought back.  OpenRouter withholds it by
        # default on reasoning models, which would leave every <think> tag
        # empty for callers that parse one (see acomplete's re-inlining).  The
        # tokens are generated and billed either way, so this costs nothing.
        extra_body: Dict[str, Any] = {"reasoning": {"enabled": True}}
        if provider := _openrouter_provider_pin():
            extra_body["provider"] = provider
        config["extra_body"] = extra_body

    elif model_name and (env_base := AZURE_CONFIG_MAP.get(model_name)):
        # Per-model deployment and API version (versions differ across Azure models)
        if model_deployment := os.environ.get(f"{env_base}_DEPLOYMENT"):
            config["model"] = model_deployment
        if api_version := os.environ.get(f"{env_base}_API_VERSION"):
            config["api_version"] = api_version
        # Endpoint: per-model override → shared fallback
        api_base = os.environ.get(f"{env_base}_ENDPOINT") or os.environ.get("OPENAI_ENDPOINT")
        if api_base:
            config["api_base"] = api_base
        # API key: per-model override → shared fallback
        api_key = os.environ.get(f"{env_base}_API_KEY") or os.environ.get("OPENAI_API_KEY")
        if api_key:
            config["api_key"] = api_key

    elif model_name and (env_base := ANTHROPIC_CONFIG_MAP.get(model_name)):
        # Deployment (model string, e.g. "anthropic/claude-sonnet-4-6")
        if model_deployment := os.environ.get(f"{env_base}_DEPLOYMENT"):
            config["model"] = model_deployment
        # API version: per-model (optional, skip if empty)
        if api_version := os.environ.get(f"{env_base}_API_VERSION"):
            config["api_version"] = api_version
        # Endpoint: per-model override → shared ANTHROPIC_ENDPOINT fallback
        api_base = os.environ.get(f"{env_base}_ENDPOINT") or os.environ.get("ANTHROPIC_ENDPOINT")
        if api_base:
            config["api_base"] = api_base
        # API key: per-model override → shared ANTHROPIC_API_KEY fallback
        api_key = os.environ.get(f"{env_base}_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")
        if api_key:
            config["api_key"] = api_key

    elif model_name and (litellm_model := OPENAI_DIRECT_CONFIG_MAP.get(model_name)):
        # OpenAI direct API (not Azure). API key from OPENAI_DIRECT_API_KEY env var.
        config["model"] = litellm_model
        if api_key := os.environ.get("OPENAI_DIRECT_API_KEY"):
            config["api_key"] = api_key

    # kwargs override / supplement env-var config (or provide full config for unknown models)
    config.update(kwargs)

    # Only the env-prefixed (Azure/Anthropic) maps carry a {PREFIX}_DEPLOYMENT
    # variable; use them for the hint. OpenAI-direct/unknown models fall back.
    env_prefixed = {**AZURE_CONFIG_MAP, **ANTHROPIC_CONFIG_MAP}
    if model_name != "baseline" and "model" not in config:
        raise ValueError(
            f"Model configuration is missing for '{model_name}'. "
            f"Set {env_prefixed.get(model_name, '<MODEL_PREFIX>')}_DEPLOYMENT env variable "
            f"or pass model= directly."
        )

    _assert_routable(config, model_name, env_prefixed)

    return config


def get_litellm_client(model_name=None, metadata=None, **kwargs) -> LiteLLMClient:
    """Create a ``LiteLLMClient`` for *model_name* (Azure/Anthropic/OpenAI/OpenRouter).

    Config is resolved from environment variables (see ``resolve_provider_config``).
    Config can also be passed directly via kwargs (model, api_key, api_base,
    api_version, temperature, top_p, max_tokens, max_completion_tokens, ...).

    Examples:
        client = get_litellm_client(model_name="gpt-4o")
        client = get_litellm_client(model="azure/gpt-4o", api_key="...", api_base="...")
    """
    config = resolve_provider_config(model_name, **kwargs)
    return LiteLLMClient(config, metadata=metadata)


class APIGenerator(LiteLLMBackedGenerator):
    """LLM generator for hosted APIs (Claude / OpenAI / OpenRouter).

    The model name is resolved to the correct provider config via environment
    variables (see ``resolve_provider_config``). OpenRouter models are addressed
    with the ``openrouter/`` prefix, e.g.
    ``"openrouter/anthropic/claude-3.5-sonnet"``. Extra kwargs are forwarded to
    LiteLLM (e.g. top_p, request_timeout, api_key).
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
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            metadata=metadata,
            **kwargs,
        )
