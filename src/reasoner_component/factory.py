"""Picking and building a generator.

``create_generator``         a model name → the right ``BaseGenerator``
``disable_native_thinking``  switch a generator's own reasoning off
``setup_llm``                the pipeline's generator, from the parsed CLI args
``run_servers``              the vLLM servers a pipeline run needs

Two backends, each behind ``BaseGenerator``:
  * ``api``   → ``APIGenerator``   hosted: Claude / OpenAI / OpenRouter  (providers/api.py)
  * ``vllm``  → ``VLLMGenerator``  a self-hosted vLLM server             (providers/vllm.py)

``$VLLM_API_BASE`` points the pipeline's vLLM generator at a server of your
own (another port or node) instead of its registered one; no server is then
started for it.
"""

import logging
import os
from typing import List, Optional

from utils.models import SELF_MANAGED_LLM_AGENTS, resolve_agent_backend

from .base import BaseGenerator
from .vllm_server.registry import ServerSpec, server_spec

logger = logging.getLogger(__name__)

# CLI reranker name → the model of the vLLM server it calls.  Only these
# rerankers get a server started for them; the others are served by hand
# (vllm_server_scripts/rerankers/).
RERANKER_SERVER_MODELS = {
    "rank1": "jhu-clsp/rank1-7b",
    "qwen3_reranker": "Qwen/Qwen3-Reranker-4B",
}


# ── Building a generator ────────────────────────────────────────────────────

def create_generator(model_name: str, backend: Optional[str] = None, **kwargs) -> BaseGenerator:
    """The generator for *model_name*.

    Args:
        model_name: ``vllm/<model>`` → vllm; anything else (``openrouter/...``,
            a known API slug) → api.
        backend: ``"api"`` / ``"vllm"``, overriding the inference.
        **kwargs: Passed to the generator's constructor.
    """
    backend = (backend or ("vllm" if model_name.startswith("vllm/") else "api")).lower()

    if backend == "api":
        from .providers.api import APIGenerator

        return APIGenerator(model_name, **kwargs)

    if backend == "vllm":
        from .providers.vllm import VLLMGenerator

        return VLLMGenerator(model_name, **kwargs)

    raise ValueError(f"Unknown backend '{backend}' for model '{model_name}'.")


def disable_native_thinking(generator):
    """Switch *generator*'s own reasoning off for every call; returns it.

    For models that must answer in plain text within a token budget (belief
    extraction, belief updater, the belief-aware policy, the judge): with
    reasoning on, a thinking model can spend the whole budget before writing
    the answer.  Takes any generator or a ``LiteLLMClient``.
    """
    return generator.disable_thinking()


# ── Pipeline wiring (CLI args) ──────────────────────────────────────────────

def setup_llm(args) -> Optional[BaseGenerator]:
    """The generator the pipeline runs on, from parsed CLI args.

      * self-managed agents        → ``None`` (they create their own connection)
      * a model with a vLLM server → ``VLLMGenerator`` at its registered
        server (vllm_server/registry.py), or at ``$VLLM_API_BASE`` when set
      * everything else            → ``APIGenerator``
        (Azure/Anthropic/OpenAI/OpenRouter)
    """
    if args.policy_model in SELF_MANAGED_LLM_AGENTS:
        return None

    spec = server_spec(args.llm_model)
    if spec is not None:
        api_base = os.getenv("VLLM_API_BASE") or spec.api_base
        print(f"Using vLLM-served model: {spec.name} at {api_base}")
        return create_generator(
            spec.name,
            backend="vllm",
            api_base=api_base,
            temperature=args.llm_temperature,
            top_p=args.llm_top_p,
            max_tokens=args.llm_max_tokens_per_call,
            timeout=args.request_timeout,
        )

    return create_generator(
        args.llm_model,
        temperature=args.llm_temperature,
        top_p=args.llm_top_p,
        max_completion_tokens=args.llm_max_tokens_per_call,
        metadata={"model": args.llm_model},
        request_timeout=args.request_timeout,
    )


def run_servers(args) -> List[ServerSpec]:
    """The vLLM servers the run of *args* needs: its agent's (unless the agent
    runs on OpenRouter, or ``$VLLM_API_BASE`` points ``setup_llm`` at a server
    of your own) and its vLLM rerankers', one per port."""
    specs: List[ServerSpec] = []
    cli_name = getattr(args, "policy_model_cli", args.policy_model)
    if resolve_agent_backend(cli_name)[0] == "api":
        print(f"[vLLM] '{cli_name}' routed to OpenRouter: no vLLM server needed.")
    elif args.policy_model not in SELF_MANAGED_LLM_AGENTS and os.getenv("VLLM_API_BASE"):
        print(f"[vLLM] VLLM_API_BASE is set: using the server at {os.getenv('VLLM_API_BASE')}, starting none.")
    else:
        # The oss agent asks its server for "openai/<llm_model>" (oss_agent.py).
        model = f"openai/{args.llm_model}" if args.policy_model == "oss" else args.llm_model
        agent_spec = server_spec(model)
        if agent_spec is not None:
            specs.append(agent_spec)

    for flag in ("post_retrieval_reranker", "post_fusion_reranker"):
        model = RERANKER_SERVER_MODELS.get(getattr(args, flag, "null"))
        spec = server_spec(model) if model else None
        if spec is None or spec in specs:
            continue
        if any(s.port == spec.port for s in specs):
            logger.warning("%s: port %d already has a server in this run; not starting %s",
                           flag, spec.port, spec.label)
            continue
        specs.append(spec)
    return specs
