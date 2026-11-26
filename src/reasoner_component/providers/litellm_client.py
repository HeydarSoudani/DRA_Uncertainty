"""``LiteLLMClient``: the transport to every hosted API (and ``hosted_vllm/``).

One client wraps one LiteLLM config (model string, endpoint, key, default
sampling parameters; built by ``routing.get_litellm_client``).  Every call
retries transient failures with exponential backoff, records its token usage
on ``token_meter``, and turns the reply into text through the helpers of
``reasoner_component.base``, so it answers like ``VLLMGenerator``.

The async methods do the work; each sync method runs its async twin, on this
thread's persistent event loop or, when a loop is already running here, on
the persistent loop of one shared background thread.  Loops are never opened
and closed per call: LiteLLM caches its async HTTP clients, bound to the loop
they were opened on, for every client of the process.
"""

import asyncio
import inspect
import logging
import threading
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple, Type

import litellm

from utils.token_meter import TokenMeter

from ..base import PatternScan, T, parse_structured, parse_tool_calls, reply_text, resolve_token_cap

logger = logging.getLogger(__name__)


# ── LiteLLM patches ─────────────────────────────────────────────────────────

# LiteLLM prints a bare, un-attributed banner to stdout whenever it cannot infer
# a provider from a model string:
#
#     Provider List: https://docs.litellm.ai/docs/providers
#
# It comes from a raw print() in get_llm_provider (no logger, no traceback, no
# model name), and several of LiteLLM's own metadata helpers trigger it on model
# slugs we never passed -- then swallow the exception, so the banner is noise
# with no matching failure.  Route every provider problem through logging
# instead: resolve_provider_config validates our own model strings up front
# (see providers.routing), and real call failures surface through the retry
# wrapper below.
litellm.suppress_debug_info = True


# After every async call LiteLLM queues its success callbacks on one global
# LoggingWorker, whose background task lives on a single event loop.  We call
# LiteLLM from many threads, each with its own loop (see _thread_loop), so
# every call from another thread rebinds the worker and drops its task on the
# previous loop, which asyncio then reports as "Task was destroyed but it is
# pending!".  We register no LiteLLM callbacks, so the queued handlers have
# nothing to run: close them instead of queuing them, and only fall back to the
# worker when a callback is set.
def _patch_logging_worker() -> None:
    try:
        from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER
    except ImportError:
        return
    enqueue = GLOBAL_LOGGING_WORKER.ensure_initialized_and_enqueue

    def enqueue_if_callbacks(async_coroutine):
        if litellm.callbacks or litellm.success_callback or litellm._async_success_callback:
            return enqueue(async_coroutine)
        async_coroutine.close()

    GLOBAL_LOGGING_WORKER.ensure_initialized_and_enqueue = enqueue_if_callbacks


_patch_logging_worker()


# ── Errors and retries ──────────────────────────────────────────────────────

# Attempts of a structured-output call whose reply is not valid JSON for the
# model: the request is sent again (a sampled reply can come out right), then
# the ``ValueError`` raises.
STRUCTURED_PARSE_ATTEMPTS = 3

# HTTP statuses worth retrying: timeout, conflict, rate limit (5xx too).
_TRANSIENT_STATUSES = frozenset({408, 409, 429})


def is_context_window_error(exc: BaseException) -> bool:
    """True when *exc* says the prompt does not fit the model's context window.

    Covers LiteLLM's ``ContextWindowExceededError`` and the 400 replies of an
    OpenAI-compatible server (vLLM says "maximum context length" or "longer
    than the maximum model length", depending on the version).
    """
    text = str(exc).lower()
    return ("ContextWindowExceededError" in type(exc).__name__
            or "context length" in text or "maximum model length" in text)


def is_transient_error(exc: BaseException) -> bool:
    """True when retrying *exc* can help: a timeout, a dropped connection, a
    rate limit or a server error.  A bad request, a bad key, an unknown model
    or a prompt past the context window fails the same way every time.

    LiteLLM's exceptions carry the HTTP ``status_code`` of the reply (its
    timeouts 408, its connection errors 500); an exception without one is
    transient when it is a timeout, a connection / OS error, or a reply with
    no choices (``IndexError``: a provider hiccup).
    """
    if is_context_window_error(exc):
        return False
    status = getattr(exc, "status_code", None)
    if isinstance(status, int):
        return status in _TRANSIENT_STATUSES or status >= 500
    if isinstance(exc, (asyncio.TimeoutError, TimeoutError, ConnectionError, OSError, IndexError)):
        return True
    name = type(exc).__name__
    return "Timeout" in name or "Connection" in name


async def _with_retries(call: Callable[[], Awaitable[Any]], *, max_total_wait: float = 60.0,
                        initial_wait: float = 1.0, max_wait: float = 32.0, backoff: float = 2.0) -> Any:
    """Await ``call()`` until it succeeds, backing off exponentially between
    attempts.  An error retrying cannot help (``is_transient_error``) raises
    at once; after *max_total_wait* seconds of waiting the last transient
    error raises as a ``RuntimeError`` ("query skipped")."""
    waited, wait, attempt = 0.0, initial_wait, 0
    while True:
        try:
            result = await call()
            if attempt:
                logger.info("Query succeeded after %d retry attempt(s) and %.1fs of waiting", attempt, waited)
            return result
        except Exception as e:
            attempt += 1
            if not is_transient_error(e):
                logger.error("%s on attempt %d, not retrying: %s", type(e).__name__, attempt, e)
                raise
            if waited >= max_total_wait:
                msg = f"Query skipped after {attempt} attempts and {waited:.1f}s total wait time. Last error: {e}"
                logger.error(msg)
                raise RuntimeError(msg) from e
            pause = min(wait, max_wait, max_total_wait - waited)
            logger.warning("Attempt %d failed (%s); retrying in %.1fs (waited %.1fs so far)",
                           attempt, e, pause, waited)
            await asyncio.sleep(pause)
            waited += pause
            wait = min(wait * backoff, max_wait)


# ── Running async calls from sync code ──────────────────────────────────────

_thread_local = threading.local()

# The loop sync calls made from inside a running loop run on (see _run_sync).
_bridge_lock = threading.Lock()
_bridge_loop: Optional[asyncio.AbstractEventLoop] = None


def _thread_loop() -> asyncio.AbstractEventLoop:
    """This thread's persistent event loop (created on first use)."""
    loop = getattr(_thread_local, "loop", None)
    if loop is None or loop.is_closed():
        loop = _thread_local.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop


def _bridge() -> asyncio.AbstractEventLoop:
    """The persistent loop of the background thread that runs the sync calls
    made from inside a running loop (started on first use, a daemon)."""
    global _bridge_loop
    with _bridge_lock:
        if _bridge_loop is None or _bridge_loop.is_closed():
            loop = asyncio.new_event_loop()
            threading.Thread(target=loop.run_forever, name="litellm-sync-bridge", daemon=True).start()
            _bridge_loop = loop
        return _bridge_loop


def _close_loop(loop: asyncio.AbstractEventLoop) -> None:
    """Close LiteLLM's async HTTP clients, cancel pending tasks, close *loop*.

    Must run on the thread that owns the loop, before the loop is abandoned:
    aiohttp connectors left bound to a stale loop raise "Future attached to a
    different loop" on the next call.  LiteLLM's client cache is shared by the
    whole process, so this also closes the clients other threads use (they
    reopen on their next call).
    """
    try:
        loop.run_until_complete(litellm.close_litellm_async_clients())
    except Exception:
        pass
    pending = asyncio.all_tasks(loop)
    for task in pending:
        task.cancel()
    if pending:
        try:
            loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
        except Exception:
            pass
    try:
        loop.close()
    except Exception:
        pass
    asyncio.set_event_loop(None)


def _run_sync(make_coro: Callable[[], Awaitable[Any]]) -> Any:
    """Run the coroutine ``make_coro()`` to completion from sync code.

    On this thread's persistent loop; or, when a loop is already running on
    this thread (sync code called from async code), on the shared background
    loop (``_bridge``), blocking this thread until it is done.
    """
    try:
        running = asyncio.get_running_loop()
    except RuntimeError:
        return _thread_loop().run_until_complete(make_coro())
    bridge = _bridge()
    if running is bridge:
        raise RuntimeError("a sync LiteLLMClient method was called from a coroutine on its own background "
                           "loop: await the async method instead")
    return asyncio.run_coroutine_threadsafe(make_coro(), bridge).result()


async def _close_stream(stream: Any) -> None:
    """Close a LiteLLM stream read only in part, so the provider stops
    generating (and billing) the rest.  Best effort: the wrapper and the
    provider stream under it are both closed where they can be (their close
    methods differ across LiteLLM versions and providers)."""
    for target in (stream, getattr(stream, "completion_stream", None), getattr(stream, "response", None)):
        close = getattr(target, "aclose", None) or getattr(target, "close", None)
        if not callable(close):
            continue
        try:
            result = close()
            if inspect.isawaitable(result):
                await result
        except Exception:  # noqa: BLE001 - already closed, or not closable this way
            pass


# ── Client ──────────────────────────────────────────────────────────────────

class LiteLLMClient:
    """A LiteLLM model with its default request parameters.

    Args:
        config:   LiteLLM parameters: ``model`` (required), ``api_key``,
                  ``api_base``, ``api_version``, ``temperature``, ``top_p``,
                  ``max_tokens``, ``timeout`` / ``request_timeout``,
                  ``extra_body``, ...  Keyword arguments are merged on top.
        metadata: Added to the ``metadata`` of every call (cost tracking).

    Every call takes the same per-call overrides as keyword arguments, plus
    ``metadata``.  Text replies take ``strip_think`` (drop the thinking, default)
    and ``return_reasoning_fallback`` (see ``base.reply_text``).
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None, metadata: Optional[Dict[str, str]] = None, **kwargs):
        self.config: Dict[str, Any] = {**(config or {}), **kwargs}
        self.metadata = metadata or {}
        self.token_meter = TokenMeter()  # cumulative usage of every call

    # ── Configuration ───────────────────────────────────────────────────────

    def set_extra_body(self, **fields: Any) -> "LiteLLMClient":
        """Merge *fields* into the ``extra_body`` sent with every call; returns the client."""
        self.config["extra_body"] = {**(self.config.get("extra_body") or {}), **fields}
        return self

    def disable_thinking(self) -> "LiteLLMClient":
        """Switch the model's own reasoning off for every call; returns the client.

        OpenRouter takes ``reasoning.enabled``; other APIs are left alone
        (they may reject the field).  A vLLM server is ``VLLMGenerator``'s.
        """
        if str(self.config.get("model", "")).startswith("openrouter/"):
            self.set_extra_body(reasoning={"enabled": False})
        return self

    # ── Plain completion ────────────────────────────────────────────────────

    def complete(self, messages: List[Dict[str, Any]], metadata: Optional[Dict[str, Any]] = None,
                 strip_think: bool = True, return_reasoning_fallback: bool = False, **kwargs) -> str:
        """Sync version of :meth:`acomplete`."""
        return _run_sync(lambda: self.acomplete(messages, metadata, strip_think=strip_think,
                                                return_reasoning_fallback=return_reasoning_fallback, **kwargs))

    async def acomplete(self, messages: List[Dict[str, Any]], metadata: Optional[Dict[str, Any]] = None,
                        strip_think: bool = True, return_reasoning_fallback: bool = False, **kwargs) -> str:
        """The reply text to *messages*."""
        request = self._request(messages, metadata, kwargs)

        async def call() -> str:
            response = await litellm.acompletion(**request)
            self.token_meter.record_usage(getattr(response, "usage", None))
            choice = response.choices[0]
            return reply_text(choice.message, strip_think=strip_think,
                              return_reasoning_fallback=return_reasoning_fallback,
                              finish_reason=getattr(choice, "finish_reason", None), model=request["model"])

        return await _with_retries(call)

    # ── Structured output ───────────────────────────────────────────────────

    def complete_with_structured_output(self, messages: List[Dict[str, Any]], response_format: Type[T],
                                        metadata: Optional[Dict[str, Any]] = None, **kwargs) -> T:
        """Sync version of :meth:`acomplete_with_structured_output`."""
        return _run_sync(lambda: self.acomplete_with_structured_output(messages, response_format, metadata, **kwargs))

    async def acomplete_with_structured_output(self, messages: List[Dict[str, Any]], response_format: Type[T],
                                               metadata: Optional[Dict[str, Any]] = None, **kwargs) -> T:
        """The reply to *messages* parsed into *response_format*.  A reply that
        is not valid JSON for it is asked again, up to
        :data:`STRUCTURED_PARSE_ATTEMPTS` requests, then raises ``ValueError``."""
        request = self._request(messages, metadata, kwargs, response_format=response_format)

        async def call() -> T:
            response = await litellm.acompletion(**request)
            self.token_meter.record_usage(getattr(response, "usage", None))
            message = response.choices[0].message
            return getattr(message, "parsed", None) or parse_structured(message.content, response_format)

        for attempt in range(1, STRUCTURED_PARSE_ATTEMPTS + 1):
            try:
                return await _with_retries(call)
            except ValueError:
                if attempt == STRUCTURED_PARSE_ATTEMPTS:
                    raise
                logger.warning("Structured reply not valid for %s (attempt %d of %d); asking again",
                               response_format.__name__, attempt, STRUCTURED_PARSE_ATTEMPTS)
        raise AssertionError("unreachable")

    # ── Stop on a regex ─────────────────────────────────────────────────────

    def complete_with_stop_on_pattern(self, messages: List[Dict[str, Any]], stop_patterns: List[str],
                                      metadata: Optional[Dict[str, Any]] = None,
                                      **kwargs) -> Tuple[str, Optional[str]]:
        """Sync version of :meth:`acomplete_with_stop_on_pattern`."""
        return _run_sync(lambda: self.acomplete_with_stop_on_pattern(messages, stop_patterns, metadata, **kwargs))

    async def acomplete_with_stop_on_pattern(self, messages: List[Dict[str, Any]], stop_patterns: List[str],
                                             metadata: Optional[Dict[str, Any]] = None,
                                             **kwargs) -> Tuple[str, Optional[str]]:
        r"""Stream the reply until a regex of *stop_patterns* (e.g. ``r"Action \d+:"``)
        matches; returns the text before the match and the matched text (None
        when the reply ended first).  On a match the stream is closed, so the
        provider stops generating; that early stop records no token usage (it
        comes with the last chunk only)."""
        request = self._request(messages, metadata, kwargs, stream=True,
                                stream_options={"include_usage": True})

        async def call() -> Tuple[str, Optional[str]]:
            scan = PatternScan(stop_patterns)
            stream = await litellm.acompletion(**request)
            try:
                async for chunk in stream:
                    if scan.feed(chunk, self.token_meter):
                        break
            finally:
                await _close_stream(stream)
            return scan.result()

        return await _with_retries(call)

    # ── Tool calls ──────────────────────────────────────────────────────────

    def complete_with_tools(self, messages: List[Dict[str, Any]], tools: List[Dict[str, Any]],
                            tool_choice: Any = "auto", metadata: Optional[Dict[str, Any]] = None,
                            strip_think: bool = True, **kwargs) -> Tuple[str, List[Dict[str, Any]]]:
        """Sync version of :meth:`acomplete_with_tools`."""
        return _run_sync(lambda: self.acomplete_with_tools(messages, tools, tool_choice, metadata,
                                                           strip_think, **kwargs))

    async def acomplete_with_tools(self, messages: List[Dict[str, Any]], tools: List[Dict[str, Any]],
                                   tool_choice: Any = "auto", metadata: Optional[Dict[str, Any]] = None,
                                   strip_think: bool = True, **kwargs) -> Tuple[str, List[Dict[str, Any]]]:
        """The reply to *messages* with *tools* (OpenAI format) on offer:
        ``(text, [{"name": str, "arguments": dict}, ...])``.  *tool_choice*:
        ``"auto"``, ``"none"``, ``"required"`` or a specific function."""
        request = self._request(messages, metadata, kwargs, tools=tools, tool_choice=tool_choice)

        async def call() -> Tuple[str, List[Dict[str, Any]]]:
            response = await litellm.acompletion(**request)
            self.token_meter.record_usage(getattr(response, "usage", None))
            message = response.choices[0].message
            return reply_text(message, strip_think=strip_think), parse_tool_calls(message)

        return await _with_retries(call)

    # ── Lifecycle ───────────────────────────────────────────────────────────

    def cleanup(self) -> None:
        """Close this thread's persistent event loop and LiteLLM's async HTTP
        clients (those of every client in the process: call it once the
        process is done with LiteLLM, not per client).  Safe to call
        repeatedly: the next call opens a new loop and new clients."""
        loop = getattr(_thread_local, "loop", None)
        if loop is not None and not loop.is_closed() and not loop.is_running():
            _close_loop(loop)
            _thread_local.__dict__.pop("loop", None)

    # ── Internals ───────────────────────────────────────────────────────────

    def _request(self, messages: List[Dict[str, Any]], metadata: Optional[Dict[str, Any]],
                 overrides: Dict[str, Any], **fields: Any) -> Dict[str, Any]:
        """The ``litellm.acompletion`` arguments of one call: the config, the
        caller's *overrides*, then the call's own *fields*."""
        request = {**self.config, **overrides, **fields}
        if "model" not in request:
            raise ValueError("Model must be specified in config")
        request["messages"] = messages
        request["metadata"] = self._metadata(metadata)
        if "request_timeout" in request:
            request["timeout"] = request.pop("request_timeout")
        resolve_token_cap(request, overrides)
        # Claude rejects a request that sets both temperature and top_p.
        model = str(request["model"]).lower()
        if ("claude" in model or "anthropic" in model) and "temperature" in request:
            request.pop("top_p", None)
        # Let LiteLLM drop what the provider does not take (e.g. presence_penalty on Anthropic).
        request["drop_params"] = True
        return request

    def _metadata(self, metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """A fresh metadata dict for one call: the caller's (never mutated), the
        client's on top, and ``trace_id`` prefixed with ``session_id``."""
        merged = {**(metadata or {}), **self.metadata}
        if "trace_id" in merged and "session_id" in merged:
            merged["trace_id"] = f"{merged['session_id']}_{merged['trace_id']}"
        return merged
