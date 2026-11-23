"""HTTP access of the web fetcher: one session per thread, an identifying
User-Agent (with the contact of ``web.contact``), an optional request rate
shared by all threads (the Wayback Machine's), and retries with backoff
on 429 / 5xx (honouring ``Retry-After``).
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
    """A request error a retry cannot fix: TLS failure, redirect loop,
    malformed URL.  A DNS failure is not one: resolvers fail transiently."""
    return isinstance(exc, _PERMANENT_ERRORS)


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

    def charge(self, n: int) -> None:
        """Count *n* more requests (redirect hops followed inside one
        ``request``): the next starts come that much later."""
        with self._lock:
            self._next = max(self._next, time.monotonic()) + n * self.interval

    def pause(self, seconds: float) -> None:
        with self._lock:
            self._next = max(self._next, time.monotonic() + seconds)


class HttpClient:
    """``get`` with retries; transport errors are retried, then raised
    (permanent ones, ``is_permanent``, at once).  With a rate, a 429 or a
    transport error pauses every thread."""

    def __init__(self, contact: str, timeout: float = 30.0, max_retries: int = 4,
                 per_minute: Optional[float] = None):
        self.user_agent = f"DRA-Uncertainty-training-datasets/0.1 ({contact}) python-requests/{requests.__version__}"
        self.timeout = timeout
        self.max_retries = max_retries
        self.limiter = RateLimiter(per_minute) if per_minute else None
        self._local = threading.local()

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
            # Redirect hops (e.g. a Wayback replay's 302 to its capture) are
            # requests too.
            if response.history and self.limiter:
                self.limiter.charge(len(response.history))
            if response.status_code not in _RETRY_CODES or attempt == self.max_retries:
                return response
            response.close()
            delay = self._delay(attempt, response.headers.get("Retry-After"))
            if self.limiter and response.status_code == 429:
                self.limiter.pause(delay)
            else:
                time.sleep(delay)
        raise AssertionError("unreachable")

    def get(self, url: str, **kwargs) -> requests.Response:
        return self.request("GET", url, **kwargs)

    @staticmethod
    def _delay(attempt: int, retry_after: Optional[str]) -> float:
        try:
            delay = float(retry_after) if retry_after else 0.0
        except ValueError:
            delay = 0.0
        return min(max(delay, 2.0 ** attempt + random.random()), 120.0)
