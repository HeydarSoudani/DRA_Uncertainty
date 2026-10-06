"""LLM-as-judge plumbing shared by the answer evaluators and the criteria matchers.

The judge is Qwen3-32B served via OpenRouter (``openrouter/qwen/qwen3-32b``),
the official BrowseComp-Plus / AgentIR leaderboard judge, hosted instead of run
on local GPUs.  Requires ``OPENROUTER_API_KEY`` in the environment.

* :func:`make_judge_client` / :func:`complete_within` / :func:`judge_all`:
  the free-text grader of the answer evaluation.
* :class:`YesNoJudge`: the YES/NO judge of Auto-ARGUE (``answer.argue``) and
  of the criteria evaluation that mirrors it (``gold.nugget_ask``).
"""

import asyncio
import concurrent.futures
import logging
from typing import Any, Callable, Dict, List, Sequence

from tqdm import tqdm

from reasoner_component.api import get_litellm_client
from reasoner_component.factory import disable_native_thinking
from utils.llm_client import LiteLLMClient

logger = logging.getLogger(__name__)

DEFAULT_JUDGE_MODEL = "openrouter/qwen/qwen3-32b"

#: Wall-clock seconds one judge request may take before it is abandoned and
#: asked again.  The HTTP read timeout is not enough: OpenRouter keeps a slow
#: request open with keep-alive bytes, which reset it, so a stalled request
#: would hold the evaluation indefinitely.
JUDGE_TIMEOUT = 180
#: Requests made before a judge call that keeps timing out fails.
JUDGE_ATTEMPTS = 3

#: Header of the references block appended to every generation.
REFERENCES_MARKER = "\n\n## References\n"


def strip_references(text: str) -> str:
    """*text* without the appended ``## References`` block."""
    idx = text.find(REFERENCES_MARKER)
    if idx != -1:
        text = text[:idx].rstrip()
    return text.strip()


def run_sync(coro):
    """Run *coro* to completion from synchronous code, also when called from
    inside a running event loop (then in a thread of its own)."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()


def make_judge_client(judge_model: str) -> LiteLLMClient:
    """The judge client with the BrowseComp-Plus sampling settings."""
    return get_litellm_client(
        model_name=judge_model,
        temperature=0.7,
        top_p=0.8,
        top_k=20,
        max_tokens=4096,
        timeout=JUDGE_TIMEOUT,
    )


def complete_within(client: LiteLLMClient, messages: List[Dict[str, Any]],
                    timeout: float = JUDGE_TIMEOUT, attempts: int = JUDGE_ATTEMPTS) -> str:
    """``client.complete(messages)`` with a wall-clock limit per request; a
    request past *timeout* seconds is abandoned and made again, up to
    *attempts* requests, then :class:`TimeoutError`."""
    for attempt in range(1, attempts + 1):
        try:
            return run_sync(asyncio.wait_for(client.acomplete(messages), timeout))
        except asyncio.TimeoutError:
            logger.warning(f"Judge request timed out after {timeout:.0f}s (attempt {attempt}/{attempts})")
    raise TimeoutError(f"judge request timed out {attempts} times ({timeout:.0f}s each)")


def judge_all(
    items: Sequence[Any],
    judge_one: Callable[[Any], Dict[str, Any]],
    on_error: Callable[[Any, Exception], Dict[str, Any]],
    max_workers: int,
    desc: str,
) -> List[Dict[str, Any]]:
    """Run ``judge_one`` over *items* in a thread pool, in completion order.

    An exception raised by ``judge_one`` is logged and replaced by
    ``on_error(item, exc)``, so every item yields one record.
    """
    records: List[Dict[str, Any]] = []
    bar = tqdm(
        total=len(items), desc=desc,
        bar_format="{desc} {percentage:3.0f}%|{bar}| {n}/{total} [{elapsed}<{remaining}]",
        dynamic_ncols=True,
    )
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(judge_one, item): item for item in items}
        for future in concurrent.futures.as_completed(futures):
            item = futures[future]
            try:
                records.append(future.result())
            except Exception as e:
                logger.error(f"{desc} failed for {item!r}: {e}")
                records.append(on_error(item, e))
            bar.update(1)
    bar.close()
    return records


# ---------------------------------------------------------------------------
# YES/NO judge (Auto-ARGUE settings)
# ---------------------------------------------------------------------------

#: Reply tokens: the judge answers YES or NO; a few tokens of slack over
#: Auto-ARGUE's 10.
YES_NO_TOKENS = 16
#: Asks again of a reply without YES/NO.
YES_NO_RETRIES = 2
#: Wall-clock seconds before a YES/NO request is abandoned and asked again
#: (see :data:`JUDGE_TIMEOUT`: the HTTP read timeout alone never trips).
YES_NO_TIMEOUT = 60


def has_yes_no(text: str) -> bool:
    """Whether a reply says YES or NO."""
    text = (text or "").upper()
    return "YES" in text or "NO" in text


def is_yes(text: str, default: bool = False) -> bool:
    """Auto-ARGUE's reading of a reply: YES when it says YES, NO when it says
    NO, else *default* (the check's default answer)."""
    text = (text or "").strip().upper()
    if "YES" in text:
        return True
    if "NO" in text:
        return False
    return default


class YesNoJudge:
    """A YES/NO judge at Auto-ARGUE's settings: temperature 0, reasoning off
    (``/no_think`` appended for a Qwen3 judge, since OpenRouter honors
    ``reasoning.enabled=false`` for some Qwen3 providers only), a reply
    without YES/NO asked again :data:`YES_NO_RETRIES` times, a request past
    :data:`YES_NO_TIMEOUT` seconds counted as a reply without YES/NO.

    ``calls`` counts the requests made and ``malformed`` the questions left
    without YES/NO.
    """

    def __init__(self, judge_model: str) -> None:
        self._client = disable_native_thinking(
            get_litellm_client(model_name=judge_model, temperature=0.0, max_tokens=YES_NO_TOKENS,
                               timeout=YES_NO_TIMEOUT))
        self._no_think = " /no_think" if "qwen3" in judge_model.lower() else ""
        self.calls = 0
        self.malformed = 0

    async def ask(self, chat: List[Dict[str, str]]) -> str:
        """The judge's reply to *chat* (``[{role, content}]``), asked again
        while it has no YES/NO; the last reply when none has."""
        chat = [dict(m) for m in chat]
        chat[-1]["content"] += self._no_think
        for _ in range(YES_NO_RETRIES + 1):
            try:
                text = (await asyncio.wait_for(self._client.acomplete(chat), YES_NO_TIMEOUT) or "").strip()
            except asyncio.TimeoutError:
                text = ""
            self.calls += 1
            if has_yes_no(text):
                break
        else:
            self.malformed += 1
        return text
