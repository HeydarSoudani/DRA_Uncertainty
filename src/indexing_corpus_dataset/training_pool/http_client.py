"""HTTP access shared by the Wikipedia and web fetchers: one session per
thread, an identifying User-Agent, an optional request rate shared by all
threads, and retries with backoff on 429 / 5xx and MediaWiki ``maxlag``
errors (honouring ``Retry-After``).

Wikimedia limits unauthenticated clients per minute across its APIs: 200
requests with a User-Agent that carries contact information, 10 without
(https://www.mediawiki.org/wiki/Wikimedia_APIs/Rate_limits).
"""

import random
import threading
import time
from typing import Optional

import requests

_RETRY_CODES = {429, 500, 502, 503, 504}
_PERMANENT_ERRORS = (requests.exceptions.SSLError, requests.exceptions.TooManyRedirects,
                     requests.exceptions.InvalidURL, requests.exceptions.InvalidSchema,
                     requests.exceptions.MissingSchema)


def error_text(exc: Exception) -> str:
    return f"{type(exc).__name__}: {exc}"[:300]


def is_permanent(exc: Exception) -> bool:
    """A request error a retry cannot fix: unknown host, TLS failure,
    redirect loop, malformed URL."""
    return isinstance(exc, _PERMANENT_ERRORS) or (
        isinstance(exc, requests.ConnectionError) and "NameResolutionError" in str(exc))


class RateLimiter:
    """At most *per_minute* request starts per minute over all threads; a
    ``pause`` (after a 429) delays every thread."""

    def __init__(self, per_minute: float):
        self.interval = 60.0 / per_minute
        self._next = time.monotonic()
        self._lock = threading.Lock()

    def wait(self) -> None:
        with self._lock:
            start = max(self._next, time.monotonic())
            self._next = start + self.interval
        time.sleep(max(0.0, start - time.monotonic()))

    def pause(self, seconds: float) -> None:
        with self._lock:
            self._next = max(self._next, time.monotonic() + seconds)


class HttpClient:
    """``get`` / ``post`` with retries; transport errors are retried, then
    raised (permanent ones, ``is_permanent``, at once).  With a rate, a 429,
    a ``maxlag`` error or a transport error pauses every thread."""

    def __init__(self, contact: str, timeout: float = 30.0, max_retries: int = 4,
                 per_minute: Optional[float] = None):
        self.user_agent = f"DRA-Uncertainty-training-pool/0.1 ({contact}) python-requests/{requests.__version__}"
        self.timeout = timeout
        self.max_retries = max_retries
        self.limiter = RateLimiter(per_minute) if per_minute else None
        self._local = threading.local()
        # Requests sent (retries included), for the progress line.
        self.requests = 0
        self._count_lock = threading.Lock()

    @property
    def session(self) -> requests.Session:
        if not hasattr(self._local, "session"):
            session = requests.Session()
            session.headers["User-Agent"] = self.user_agent
            self._local.session = session
        return self._local.session

    def request(self, method: str, url: str, **kwargs) -> requests.Response:
        kwargs.setdefault("timeout", self.timeout)
        for attempt in range(self.max_retries + 1):
            if self.limiter:
                self.limiter.wait()
            with self._count_lock:
                self.requests += 1
            try:
                response = self.session.request(method, url, **kwargs)
            except (requests.ConnectionError, requests.Timeout) as exc:
                if is_permanent(exc) or attempt == self.max_retries:
                    raise
                # A refused connection is how the Wayback Machine throttles:
                # a rate-limited client backs off on every thread, as on 429.
                delay = self._delay(attempt, None)
                if self.limiter:
                    self.limiter.pause(delay)
                else:
                    time.sleep(delay)
                continue
            # maxlag: the wiki's replicas lag, every client is asked to wait.
            maxlag = response.headers.get("MediaWiki-API-Error") == "maxlag"
            if (response.status_code not in _RETRY_CODES and not maxlag) or attempt == self.max_retries:
                return response
            response.close()
            delay = self._delay(attempt, response.headers.get("Retry-After"))
            if self.limiter and (response.status_code == 429 or maxlag):
                self.limiter.pause(delay)
            else:
                time.sleep(delay)
        raise AssertionError("unreachable")

    def get(self, url: str, **kwargs) -> requests.Response:
        return self.request("GET", url, **kwargs)

    def post(self, url: str, **kwargs) -> requests.Response:
        return self.request("POST", url, **kwargs)

    @staticmethod
    def _delay(attempt: int, retry_after: Optional[str]) -> float:
        try:
            delay = float(retry_after) if retry_after else 0.0
        except ValueError:
            delay = 0.0
        return min(max(delay, 2.0 ** attempt + random.random()), 120.0)
