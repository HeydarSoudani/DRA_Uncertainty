"""The inference configuration an agent passes to its LLM calls."""

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class InferenceConfig:
    """Immutable bundle of parameters shared by the main loop, force answer,
    and intermediate answer components of an agent."""

    # API dispatch
    api_type: str = "chat_completion"  # "responses_api" | "chat_completion"

    # Model identity
    model_name: str = ""
    api_base: Optional[str] = None
    api_key: str = "EMPTY"

    # Generation limits
    max_output_tokens: int = 20000

    # Reasoning (Responses API only; None disables the reasoning block)
    reasoning_effort: Optional[str] = "high"

    # Extra request fields (Responses API only), e.g. OpenRouter's provider pin
    extra_body: Optional[Dict[str, Any]] = None

    # Prompt scaffolding
    system_prompt: Optional[str] = None
    format_instructions: str = ""
