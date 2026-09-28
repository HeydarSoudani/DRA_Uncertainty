"""Factory for creating the right generator from a model name.

Three backends, each behind ``BaseGenerator``:
  * ``api``   → ``APIGenerator``   (Claude / OpenAI / OpenRouter, hosted)
  * ``vllm``  → ``VLLMGenerator``  (self-hosted OpenAI-compatible vLLM server)
  * ``hf``    → ``HFGenerator``    (weights loaded in-process via transformers)
"""

from typing import Any, Dict, Optional

from .base import BaseGenerator

# HuggingFace model repo slugs that must be loaded in-process (HF backend).
# Currently empty: every finetuned model in the flow is served via an
# OpenAI-compatible vLLM server (see is_local_finetuned / setup_llm). To route a
# future model to in-process loading, add its slug here or address it with an
# ``hf/`` prefix; create_generator picks the HF backend automatically.
_HF_MODELS: set[str] = set()


def _infer_backend(model_name: str) -> str:
    """Infer the backend from *model_name* prefixes / known slugs."""
    if model_name.startswith("openrouter/"):
        return "api"
    if model_name.startswith("vllm/"):
        return "vllm"
    if model_name in _HF_MODELS or model_name.startswith("hf/"):
        return "hf"
    return "api"


def create_generator(
    model_name: str,
    backend: Optional[str] = None,
    **kwargs,
) -> BaseGenerator:
    """Instantiate the correct generator for *model_name*.

    Args:
        model_name: Friendly model name. Recognised prefixes:
            ``openrouter/`` → api, ``vllm/`` → vllm, ``hf/`` or a known HF slug → hf.
        backend: Optional explicit backend (``"api"`` / ``"vllm"`` / ``"hf"``) that
            overrides prefix inference.
        **kwargs: Forwarded to the generator constructor.
    """
    backend = (backend or _infer_backend(model_name)).lower()

    if backend == "hf":
        # Import lazily so callers that only need API/vLLM don't pull in torch.
        from .hf import HFGenerator, ensure_transformers_version

        bare = model_name.removeprefix("hf/")
        ensure_transformers_version(bare)
        return HFGenerator(bare, **kwargs)

    if backend == "vllm":
        from .vllm import VLLMGenerator

        return VLLMGenerator(model_name, **kwargs)

    if backend == "api":
        from .api import APIGenerator

        return APIGenerator(model_name, **kwargs)

    raise ValueError(f"Unknown backend '{backend}' for model '{model_name}'.")


def no_thinking_extra_body(generator) -> Optional[Dict[str, Any]]:
    """``extra_body`` that switches a model's own reasoning off.

    Copies the configured body (OpenRouter provider pin) and extends it, so
    it can be passed per call in place of the configured one: OpenRouter takes
    ``reasoning.enabled``, a self-hosted vLLM model takes the chat-template
    switch.  Other APIs get nothing (they may reject the vLLM kwarg).
    """
    client = getattr(generator, "_client", generator)
    config = getattr(client, "config", None) or {}
    body = dict(config.get("extra_body") or {})
    model = str(config.get("model", "") or getattr(generator, "model_name", ""))
    if model.startswith("openrouter/"):
        body["reasoning"] = {"enabled": False}
    elif model.startswith(("vllm/", "hosted_vllm/", "openai/")):
        body["chat_template_kwargs"] = {"enable_thinking": False}
    else:
        return None
    return body


def disable_native_thinking(generator) -> BaseGenerator:
    """Switch *generator*'s own reasoning off for every call; returns it.

    For auxiliary models that must answer in plain text within a small token
    budget (criteria extraction, LLM criteria judge): with reasoning on, a
    thinking model can spend the whole budget before writing the answer.
    """
    body = no_thinking_extra_body(generator)
    client = getattr(generator, "_client", None)
    if body is not None and isinstance(getattr(client, "config", None), dict):
        client.config["extra_body"] = body
    return generator
