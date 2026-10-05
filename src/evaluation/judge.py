"""LLM-as-judge plumbing shared by the answer evaluators and the criteria matchers.

The judge is Qwen3-32B served via OpenRouter (``openrouter/qwen/qwen3-32b``),
the official BrowseComp-Plus / AgentIR leaderboard judge, hosted instead of run
on local GPUs.  Requires ``OPENROUTER_API_KEY`` in the environment.
"""

import asyncio
import concurrent.futures
import logging
from typing import Any, Callable, Dict, List, Sequence

from tqdm import tqdm

from reasoner_component.api import get_litellm_client
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
    *attempts* requests, then :class:`TimeoutError`.  Called from inside a
    running event loop, each request runs in a thread of its own."""
    def request() -> str:
        return asyncio.run(asyncio.wait_for(client.acomplete(messages), timeout))

    for attempt in range(1, attempts + 1):
        try:
            try:
                asyncio.get_running_loop()
            except RuntimeError:
                return request()
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                return pool.submit(request).result()
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
