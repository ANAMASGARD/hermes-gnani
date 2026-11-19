"""Bounded transport for the Vachana API and its signed S3 transfers."""

from collections.abc import Callable, Iterator
from contextlib import ExitStack, closing, contextmanager
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from time import monotonic, sleep
from typing import Any
from urllib.parse import urlsplit

import re

import httpx
from agent.secret_scope import get_secret

HOST = "api.vachana.ai"
BASE = f"https://{HOST}"
SECRET = "GNANI_API_KEY"
JSON_LIMIT = 16 * 1024 * 1024
AUDIO_LIMIT = 64 * 1024 * 1024


class GnaniError(RuntimeError):
    def __init__(self, message: str, *, code: str = "") -> None:
        super().__init__(message)
        self.code = code


def api_key() -> str:
    key = (get_secret(SECRET) or "").strip()
    if not key or not key.isascii() or any(not 33 <= ord(char) <= 126 for char in key):
        raise GnaniError("Configure a valid GNANI_API_KEY in the active Hermes profile.")
    return key


def available() -> bool:
    try:
        api_key()
    except GnaniError, RuntimeError:
        return False
    return True


def _rate_delay(header: str, fallback: float) -> float:
    try:
        return max(0.0, float(header))
    except ValueError:
        try:
            return max(0.0, (parsedate_to_datetime(header) - datetime.now(UTC)).total_seconds())
        except ValueError, TypeError, OverflowError:
            return fallback


def read_bytes(response: httpx.Response, limit: int, deadline: float) -> Iterator[bytes]:
    total = 0
    for chunk in response.iter_bytes(chunk_size=None):
        total += len(chunk)
        if total > limit:
            raise GnaniError("Gnani response exceeded the size limit.")
        if monotonic() >= deadline:
            raise GnaniError("Gnani request exceeded its time limit.")
        yield chunk


def object_response(raw: bytes) -> dict[str, Any]:
    import json

    try:
        value = json.loads(raw)
    except ValueError, RecursionError:
        raise GnaniError("Gnani returned invalid JSON.") from None
    if not isinstance(value, dict):
        raise GnaniError("Gnani returned an unexpected response.")
    return value


def _failure(response: httpx.Response, deadline: float) -> GnaniError:
    status = response.status_code
    messages = {
        401: "Gnani rejected GNANI_API_KEY. Check the key in this Hermes profile.",
        403: "Gnani denied access. Check the key permissions and model entitlement.",
        429: "Gnani rate limit reached. Wait for active requests to finish and retry.",
    }
    if status == 404 and response.request.url.path == "/stt/v3/batch/v2/jobs":
        return GnaniError(
            "Gnani did not expose presigned Batch upload (HTTP 404). Compress the recording to FLAC "
            "or split it into files under 10 MB, then retry.",
            code="batch_upload_unavailable",
        )
    if status in messages:
        return GnaniError(messages[status])
    if status == 400:
        try:
            error = object_response(b"".join(read_bytes(response, 65536, deadline))).get("error")
        except GnaniError:
            error = None
        if isinstance(error, dict) and error.get("type") == "MAX_AUDIO_DURATION_EXCEEDED":
            return GnaniError("Clip exceeds Gnani's 30-second REST limit.", code="duration")
    return GnaniError(f"Gnani request failed (HTTP {status}). Check the settings or service status.")


def _signed_url(url: Any) -> str:
    if not isinstance(url, str):
        raise GnaniError("Gnani did not provide a signed storage link.")
    parsed = urlsplit(url)
    host = parsed.netloc.lower()
    # The documented result bucket uses virtual-hosted S3 access. Allow its
    # regional S3 endpoints as well, including the presigned upload variant.
    allowed = re.fullmatch(r"gnaniasraudios\.s3(?:[.-][a-z0-9-]+)?\.amazonaws\.com", host)
    if parsed.scheme != "https" or not allowed:
        raise GnaniError("Gnani returned an unsupported signed storage hostname.")
    if not parsed.path or not parsed.query or parsed.fragment or "@" in host or ":" in host:
        raise GnaniError("Gnani returned an invalid signed storage link.")
    return url


class Client:
    """One transport configuration, with independently owned HTTP responses."""

    def __init__(self, transport: httpx.BaseTransport | None = None) -> None:
        self.transport = transport

    @contextmanager
    def response(
        self,
        method: str,
        path: str,
        *,
        deadline: float,
        signed: bool = False,
        content_length: int | None = None,
        upload: Callable[[], Iterator[bytes]] | None = None,
        **kwargs: Any,
    ) -> Iterator[httpx.Response]:
        if signed:
            url, headers = _signed_url(path), {}
        else:
            if not path.startswith("/") or path.startswith("//") or "\\" in path:
                raise GnaniError("Invalid Gnani API path.")
            url, headers = BASE + path, {"X-API-Key-ID": api_key()}
        headers["Accept-Encoding"] = "identity"
        if content_length is not None:
            headers["Content-Length"] = str(content_length)
        # Copy the caller's file offsets so an explicit rate-limit retry can
        # resend the exact same multipart upload.
        files = kwargs.get("files") or {}
        inputs = [
            (part[1], part[1].tell())
            for part in files.values()
            if isinstance(part, tuple) and hasattr(part[1], "tell")
        ]
        try:
            with ExitStack() as resources:
                if signed:
                    # HTTPX's Client logs request URLs at INFO. Send signed
                    # transfers via its transport to keep the query private.
                    transport = resources.enter_context(
                        self.transport or httpx.HTTPTransport(trust_env=False)
                    )
                else:
                    client = resources.enter_context(
                        httpx.Client(transport=self.transport, trust_env=False, follow_redirects=False)
                    )
                for attempt in range(3):
                    remaining = deadline - monotonic()
                    if remaining <= 0:
                        raise GnaniError("Gnani request exceeded its time limit.")
                    delivered = False
                    if upload is not None:
                        kwargs["content"] = upload()
                    try:
                        with ExitStack() as request_scope:
                            timeout = httpx.Timeout(min(remaining, 20), connect=min(remaining, 5))
                            if signed:
                                request = httpx.Request(
                                    method,
                                    url,
                                    headers=headers,
                                    extensions={"timeout": timeout.as_dict()},
                                    **kwargs,
                                )
                                response = request_scope.enter_context(
                                    closing(transport.handle_request(request))
                                )
                                response.request = request
                            else:
                                response = request_scope.enter_context(
                                    client.stream(
                                        method, url, headers=headers, timeout=timeout, **kwargs
                                    )
                                )
                            status = response.status_code
                            if 200 <= status < 300:
                                delivered = True
                                yield response
                                return
                            retry = status == 429 or (method == "GET" and status in {500, 502, 503, 504})
                            if not retry or attempt == 2:
                                raise _failure(response, deadline)
                            delay = _rate_delay(
                                response.headers.get("retry-after", ""), 0.5 * 2**attempt
                            )
                            if delay > 5 or monotonic() + delay >= deadline:
                                raise _failure(response, deadline)
                    except httpx.ConnectError, httpx.ConnectTimeout:
                        if delivered or attempt == 2:
                            raise
                        delay = 0.5 * 2**attempt
                    sleep(delay)
                    for handle, offset in inputs:
                        handle.seek(offset)
        except httpx.TimeoutException:
            raise GnaniError("Gnani request timed out. Check the connection or job status.") from None
        except httpx.HTTPError:
            raise GnaniError("Gnani connection failed. Check the connection or job status.") from None

    def json(self, method: str, path: str, *, deadline: float, **kwargs: Any) -> dict[str, Any]:
        with self.response(method, path, deadline=deadline, **kwargs) as response:
            return object_response(b"".join(read_bytes(response, JSON_LIMIT, deadline)))
