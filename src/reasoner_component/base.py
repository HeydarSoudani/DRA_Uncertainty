"""The generator interface, and the reply handling every backend shares.

``BaseGenerator`` is the abstract interface every generator implements
(``APIGenerator``, ``VLLMGenerator``).

The helpers below are shared by ``LiteLLMClient`` and ``VLLMGenerator``, so
every backend takes the same parameters and answers the same way:
``resolve_token_cap`` (the output-token limit of one call), ``reply_text`` (the
text), ``parse_structured`` (a Pydantic object), ``parse_tool_calls`` (the tool
calls) and ``PatternScan`` (a streamed reply cut at a stop regex).
"""

import json
import logging
import re
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar

from pydantic import BaseModel

from utils.token_meter import TokenMeter

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

# Prefix of a reply that had no content, only reasoning, returned when the
# caller asked for ``return_reasoning_fallback``.
REASONING_FALLBACK_PREFIX = "[reasoning_fallback]"


# ── Generator interface ─────────────────────────────────────────────────────

class BaseGenerator(ABC):
    """One LLM behind a uniform chat interface.

    ``complete`` / ``acomplete`` are required; the structured-output, tool and
    stop-pattern calls are optional (``NotImplementedError`` when a backend
    lacks them).  Every sync method has an async twin prefixed with ``a``.

    ``token_meter`` holds the cumulative token usage of every call.
    """

    token_meter: TokenMeter

    @abstractmethod
    def complete(self, messages: List[Dict[str, Any]], **kwargs) -> str: ...

    @abstractmethod
    async def acomplete(self, messages: List[Dict[str, Any]], **kwargs) -> str: ...

    def complete_with_structured_output(
        self, messages: List[Dict[str, Any]], response_format: Type[T], **kwargs
    ) -> T:
        raise NotImplementedError

    async def acomplete_with_structured_output(
        self, messages: List[Dict[str, Any]], response_format: Type[T], **kwargs
    ) -> T:
        raise NotImplementedError

    def complete_with_stop_on_pattern(
        self, messages: List[Dict[str, Any]], stop_patterns: List[str], **kwargs
    ) -> Tuple[str, Optional[str]]:
        raise NotImplementedError

    async def acomplete_with_stop_on_pattern(
        self, messages: List[Dict[str, Any]], stop_patterns: List[str], **kwargs
    ) -> Tuple[str, Optional[str]]:
        raise NotImplementedError

    def complete_with_tools(
        self, messages: List[Dict[str, Any]], tools: List[Dict[str, Any]], **kwargs
    ) -> Tuple[str, List[Dict[str, Any]]]:
        raise NotImplementedError

    async def acomplete_with_tools(
        self, messages: List[Dict[str, Any]], tools: List[Dict[str, Any]], **kwargs
    ) -> Tuple[str, List[Dict[str, Any]]]:
        raise NotImplementedError

    def disable_thinking(self) -> "BaseGenerator":
        """Switch the model's own reasoning off for every call; returns the
        generator.  A no-op for a backend without such a switch."""
        return self

    def cleanup(self) -> None:
        pass


# ── Request parameters ──────────────────────────────────────────────────────

_TOKEN_CAP_KEYS = ("max_completion_tokens", "max_tokens")


def resolve_token_cap(request: Dict[str, Any], overrides: Dict[str, Any]) -> None:
    """Leave *request* (defaults merged with the call's *overrides*) with one
    output-token limit, in place.

    A limit the call passes wins over the defaults under either name: a client
    built with ``max_completion_tokens=4000`` and called with ``max_tokens=100``
    sends 100.  The value goes under the name the defaults used
    (``max_completion_tokens`` when they set both), else the call's.
    """
    defaults = {k: v for k, v in request.items() if k in _TOKEN_CAP_KEYS and overrides.get(k) is None}
    per_call_key = next((k for k in _TOKEN_CAP_KEYS if overrides.get(k) is not None), None)
    default_key = next((k for k in _TOKEN_CAP_KEYS if defaults.get(k) is not None), None)
    if per_call_key is not None:
        value = overrides[per_call_key]
    elif default_key is not None:
        value = defaults[default_key]
    else:
        value = None
    for k in _TOKEN_CAP_KEYS:
        request.pop(k, None)
    if value is not None:
        request[default_key or per_call_key] = value


# ── Reply handling ──────────────────────────────────────────────────────────

def message_reasoning(message: Any) -> str:
    """The out-of-band chain of thought of a chat-completion *message*.

    Backends disagree on the field name: vLLM's reasoning parsers and
    DeepSeek-style APIs use ``reasoning_content``, OpenRouter and newer vLLM
    use ``reasoning``.
    """
    return getattr(message, "reasoning_content", None) or getattr(message, "reasoning", None) or ""


def _split_inline_thinking(content: str) -> Tuple[str, str]:
    """``(thinking, answer)`` of a reply whose thinking is inline.

    The answer is what follows the last ``</think>``.  A reply that opens a
    ``<think>`` block and never closes it (cut off by the token limit while
    thinking) is all thinking, with no answer.
    """
    if "</think>" in content:
        thinking, answer = content.rsplit("</think>", 1)
        return thinking.replace("<think>", "", 1).strip(), answer.strip()
    stripped = content.lstrip()
    if stripped.startswith("<think>"):
        return stripped[len("<think>"):].strip(), ""
    return "", content.strip()


def reply_text(
    message: Any,
    *,
    strip_think: bool = True,
    return_reasoning_fallback: bool = False,
    finish_reason: Optional[str] = None,
    model: Optional[str] = None,
) -> str:
    """The text a caller gets for a chat-completion *message*.

    Reasoning comes either out of band (OpenRouter, a vLLM server with a
    reasoning parser) or inline in ``<think>...</think>`` (our vLLM servers);
    both are handled alike:

    * ``strip_think``: the answer only, thinking dropped.  A reply cut off
      while thinking (an unclosed ``<think>``) has no answer: ``""``.
      Otherwise reasoning returned out of band is put back inline as
      ``<think>...</think>``, so callers that parse the tag still see it.
    * ``return_reasoning_fallback``: an empty answer with reasoning returns
      that reasoning, prefixed with :data:`REASONING_FALLBACK_PREFIX`.
    * ``finish_reason == "length"`` is logged: a reasoning model can spend
      the whole output budget thinking and return no answer.
    """
    content = getattr(message, "content", None) or ""
    reasoning = message_reasoning(message)

    if finish_reason == "length":
        logger.warning(
            "Completion hit the output-token limit (model=%s, content_chars=%d, "
            "reasoning_chars=%d); the response is truncated.",
            model, len(content), len(reasoning),
        )

    if strip_think:
        inline_thinking, content = _split_inline_thinking(content)
        reasoning = reasoning or inline_thinking
    elif reasoning.strip() and "</think>" not in content:
        content = f"<think>{reasoning.strip()}</think>\n{content}"

    if not content.strip() and return_reasoning_fallback and reasoning.strip():
        return REASONING_FALLBACK_PREFIX + reasoning
    return content


def parse_structured(content: Optional[str], response_format: Type[T]) -> T:
    """Parse a JSON reply *content* into *response_format*.

    Thinking before ``</think>`` and a surrounding Markdown code fence are
    dropped first.  Raises ``ValueError`` when the reply is not valid JSON
    for the model.
    """
    text = (content or "").split("</think>")[-1].strip()
    if text.startswith("```"):
        lines = text.split("\n")[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    try:
        return response_format.model_validate(json.loads(text))
    except (json.JSONDecodeError, ValueError) as e:
        logger.error("Failed to parse structured output: %s\nRaw content: %s", e, text)
        raise ValueError(f"Failed to parse structured output: {e}") from e


def json_schema_format(model: Type[BaseModel]) -> Dict[str, Any]:
    """The ``response_format`` that constrains a reply to *model*'s JSON schema."""
    return {"type": "json_schema", "json_schema": {"name": model.__name__, "schema": model.model_json_schema()}}


def parse_tool_calls(message: Any) -> List[Dict[str, Any]]:
    """The tool calls of a chat-completion *message*, as
    ``[{"name": str, "arguments": dict}]``; unparseable arguments become ``{}``."""
    calls = []
    for call in getattr(message, "tool_calls", None) or []:
        try:
            arguments = json.loads(call.function.arguments or "{}")
        except (json.JSONDecodeError, TypeError):
            arguments = {}
        calls.append({"name": call.function.name, "arguments": arguments if isinstance(arguments, dict) else {}})
    return calls


class PatternScan:
    """Accumulates a streamed reply until one of the stop regexes matches.

    Feed it every chunk; stop reading the stream once ``feed`` returns True.
    When several patterns match, the one that starts first in the text wins.
    """

    def __init__(self, patterns: List[str]):
        self._patterns = [re.compile(p) for p in patterns]
        self._text = ""
        self._match: Optional[re.Match] = None

    def feed(self, chunk: Any, meter: Any = None) -> bool:
        """Take one stream chunk (recording its usage on *meter*, when it
        carries any); True once a pattern matched."""
        if meter is not None and getattr(chunk, "usage", None):
            meter.record_usage(chunk.usage)
        if not chunk.choices or not chunk.choices[0].delta.content:
            return False
        self._text += chunk.choices[0].delta.content
        matches = [m for m in (p.search(self._text) for p in self._patterns) if m]
        if matches:
            self._match = min(matches, key=lambda m: m.start())
            return True
        return False

    def result(self) -> Tuple[str, Optional[str]]:
        """The text before the match and the matched text (None: no match)."""
        if self._match is None:
            return self._text.strip(), None
        return self._text[:self._match.start()].strip(), self._match.group(0)
