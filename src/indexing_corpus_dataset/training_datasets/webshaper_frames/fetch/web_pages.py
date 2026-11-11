"""Non-Wikipedia gold pages: live fetch, text extraction (Trafilatura for
HTML, pypdf for PDF), and the Wayback Machine snapshot closest to a given time
when the live fetch fails.

A fetch succeeds when the response is HTML or PDF and its text has at least
``min_words`` words; otherwise its status says why (``http_<code>``,
``not_html``, ``pdf_error``, ``too_large``, ``too_short``, ``unreachable``
for an error a retry cannot fix, ``error`` for the others).  A failed live
fetch records the fallback's outcome as ``wayback_status`` (``no_snapshot``,
``wayback_error`` when the archive could not be queried, or the status of
the snapshot fetch).

Bodies are streamed: at most ``_MAX_BYTES``, read within ``_MAX_SECONDS``
(the request timeout bounds each read only).

Wayback: the replay URL ``/web/{timestamp}id_/{url}`` redirects to the
closest capture, one call (two requests against the archive's rate: the
redirect hop counts, ``HttpClient.request``).  Only when that capture is not an HTTP 200 one
(e.g. an archived 403 page) does the CDX API look for the closest 200 capture.
"""

import io
import re
import time
from typing import Optional

import pypdf
import requests
import trafilatura

from .http_client import HttpClient, error_text, is_permanent

_WAYBACK = "https://web.archive.org/web"
_CDX_API = "https://web.archive.org/cdx/search/cdx"
_MAX_BYTES = 20_000_000
_MAX_SECONDS = 120.0
_CAPTURE_TIMESTAMP = re.compile(r"/web/(\d{14})id_/")


def extract_text(html, url: Optional[str] = None) -> str:
    return trafilatura.extract(html, url=url, include_tables=True, include_comments=False,
                               favor_recall=True) or ""


def extract_pdf_text(data: bytes) -> str:
    reader = pypdf.PdfReader(io.BytesIO(data))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def _read_body(response: requests.Response) -> Optional[bytes]:
    """The body of a streamed *response*; None above ``_MAX_BYTES``, and
    ``requests.Timeout`` past ``_MAX_SECONDS``."""
    length = response.headers.get("Content-Length", "")
    if length.isdigit() and int(length) > _MAX_BYTES:
        return None
    chunks, size, deadline = [], 0, time.monotonic() + _MAX_SECONDS
    for chunk in response.iter_content(chunk_size=1 << 16):
        size += len(chunk)
        if size > _MAX_BYTES:
            return None
        if time.monotonic() > deadline:
            raise requests.Timeout(f"download longer than {_MAX_SECONDS:.0f} s")
        chunks.append(chunk)
    return b"".join(chunks)


class WebFetcher:
    """*http* fetches the live pages; *wayback_http* (rate-limited) queries
    the archive, or None for no fallback."""

    def __init__(self, http: HttpClient, wayback_http: Optional[HttpClient] = None,
                 min_words: int = 50, wayback_timestamp: str = "20250601"):
        self.http = http
        self.wayback_http = wayback_http
        self.min_words = min_words
        self.wayback_timestamp = wayback_timestamp

    def fetch(self, url: str) -> dict:
        """``{"status", "fetched_via", "format", "text", "http_code", ...}``:
        the live page, else its closest archived snapshot."""
        try:
            with self.http.get(url, allow_redirects=True, stream=True) as response:
                live = self._page(response, url)
        except requests.RequestException as exc:
            live = {"status": "unreachable" if is_permanent(exc) else "error", "error": error_text(exc)}
        live["fetched_via"] = "live"
        if live["status"] == "ok" or self.wayback_http is None:
            return live
        snapshot = self._wayback(url)
        if snapshot["status"] == "ok":
            snapshot["live_status"] = live["status"]
            return snapshot
        live["wayback_status"] = snapshot["status"]
        if snapshot.get("error"):
            live["wayback_error"] = snapshot["error"]
        return live

    def _page(self, response: requests.Response, url: str) -> dict:
        """The outcome of a fetched (streamed) page: its text, or why there
        is none."""
        result = {"http_code": response.status_code, "final_url": response.url}
        if response.status_code != 200:
            return {**result, "status": f"http_{response.status_code}"}
        body = _read_body(response)
        if body is None:
            return {**result, "status": "too_large"}
        content_type = response.headers.get("Content-Type", "").lower()
        if "pdf" in content_type or body[:5] == b"%PDF-":
            try:
                text = extract_pdf_text(body)
            except Exception as exc:  # noqa: BLE001 - encrypted or malformed PDF
                return {**result, "status": "pdf_error", "error": error_text(exc)}
            result["format"] = "pdf"
        elif content_type and "html" not in content_type and "xml" not in content_type:
            return {**result, "status": "not_html", "content_type": content_type[:100]}
        else:
            # Bytes, so that Trafilatura detects the encoding itself.
            text = extract_text(body, url)
            result["format"] = "html"
        words = len(text.split())
        status = "ok" if words >= self.min_words else "too_short"
        return {**result, "status": status, "text": text, "words": words}

    def _wayback(self, url: str) -> dict:
        """The capture of *url* closest to ``wayback_timestamp``; the closest
        HTTP 200 capture (CDX) when that one is not a 200."""
        try:
            with self.wayback_http.get(f"{_WAYBACK}/{self.wayback_timestamp}id_/{url}",
                                       allow_redirects=True, stream=True) as response:
                # A replayed capture carries Memento-Datetime; any other reply
                # is the archive's own (404: no capture).
                if "memento-datetime" not in response.headers:
                    if response.status_code == 429 or response.status_code >= 500:
                        return {"status": "wayback_error", "error": f"HTTP {response.status_code}"}
                    return {"status": "no_snapshot"}
                if response.status_code == 200:
                    return self._snapshot(response, url)
        except requests.RequestException as exc:
            return {"status": "wayback_error", "error": error_text(exc)}
        return self._wayback_cdx(url)

    def _wayback_cdx(self, url: str) -> dict:
        try:
            response = self.wayback_http.get(_CDX_API, params={
                "url": url, "output": "json", "closest": self.wayback_timestamp, "sort": "closest",
                "limit": 1, "filter": "statuscode:200", "fl": "timestamp,original",
            })
            response.raise_for_status()
            rows = response.json()
        except (requests.RequestException, ValueError) as exc:
            return {"status": "wayback_error", "error": error_text(exc)}
        # A JSON CDX reply is a header row followed by the captures.
        if len(rows) < 2:
            return {"status": "no_snapshot"}
        timestamp, original = rows[1][0], rows[1][1]
        try:
            with self.wayback_http.get(f"{_WAYBACK}/{timestamp}id_/{original}",
                                       allow_redirects=True, stream=True) as response:
                return self._snapshot(response, url)
        except requests.RequestException as exc:
            return {"status": "wayback_error", "error": error_text(exc)}

    def _snapshot(self, response: requests.Response, url: str) -> dict:
        """``id_`` serves the capture as archived, without the Wayback toolbar."""
        result = self._page(response, url)
        found = _CAPTURE_TIMESTAMP.search(response.url)
        result.update({"fetched_via": "wayback", "wayback_timestamp": found.group(1) if found else None})
        return result
