"""The LLM calls of the training dataset build and of TRQA's gold beliefs:
a client that answers with a JSON object (``JsonWriter``: the fact LLM,
``gold_passages/facts.py``; the writer of TRQA's gold belief texts,
``belief_model_component/init/member_texts.py``), the cap on replies
without a usable answer and the API key check.

The judge of the gold passages (``gold_passages/judge.py``) answers with a
label, not a JSON object; it shares ``MAX_REPLY_FAILURES`` and
``require_api_key``.
"""

import os
from typing import Any, Dict

from utils.text_utils import parse_json_object

# Replies without a usable answer (not a JSON object, no label; one per run)
# after which a call's failure is final: the writers and the judge answer at
# temperature 0, the next run would get the same.
MAX_REPLY_FAILURES = 2


class ReplyError(ValueError):
    """A reply that is not a JSON object: counted against
    ``MAX_REPLY_FAILURES``, unlike an API error."""


class JsonWriter:
    """An LLM whose reply is a JSON object (native reasoning off)."""

    def __init__(self, model: str, max_tokens: int, llm: Any = None) -> None:
        """*llm*: a client with ``complete(messages, **kwargs) -> str`` for
        *model* (None: an API client, reasoning off)."""
        self.model = model
        if llm is None:
            from reasoner_component.factory import create_generator, disable_native_thinking
            llm = disable_native_thinking(create_generator(model, backend="api"))
        self._llm = llm
        self._max_tokens = max_tokens

    @property
    def token_meter(self):
        """The client's token usage (None when it keeps none)."""
        return getattr(self._llm, "token_meter", None)

    def ask(self, system: str, user: str) -> Dict[str, Any]:
        """The reply as a JSON object; a reply that is none raises
        ``ReplyError``."""
        raw = self._llm.complete([{"role": "system", "content": system}, {"role": "user", "content": user}],
                                 max_tokens=self._max_tokens, temperature=0.0)
        data = parse_json_object(raw or "")
        if not isinstance(data, dict):
            raise ReplyError(f"reply is not a JSON object: {(raw or '')[:200]!r}")
        return data


def require_api_key(model: str, what: str, flag: str) -> None:
    """Refuse to go on without the API key of *model* (an OpenRouter
    model) when *what* needs its calls; *flag* names the setting that
    chooses another model."""
    if model.startswith("openrouter/") and not os.environ.get("OPENROUTER_API_KEY"):
        raise SystemExit(f"{what} need {model}: set OPENROUTER_API_KEY (.env) or pass another {flag}")
