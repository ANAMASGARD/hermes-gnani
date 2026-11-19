"""One bounded Gnani WebSocket per Hermes capture."""

import asyncio
import json
import logging
import queue
import socket
import threading
from collections.abc import Callable
from time import monotonic
from typing import Any

from agent.transcription_provider import TranscriptionStreamSession
from websockets.asyncio.client import connect
from websockets.exceptions import InvalidStatus, WebSocketException

from .client import HOST, GnaniError, object_response

FRAME_BYTES = 1024
FRAME_SECONDS = 0.032
_SESSION_SECONDS = 900
_IDLE_SECONDS = 65
_DRAIN_SECONDS = 8

# websockets DEBUG logs include the authenticated HTTP upgrade headers.
_WIRE_LOG = logging.Logger(__name__ + ".wire", logging.CRITICAL)
_WIRE_LOG.addHandler(logging.NullHandler())


class StreamSession(TranscriptionStreamSession):
    def __init__(
        self, headers: dict[str, str], key: str, connector: Callable[..., Any] = connect
    ) -> None:
        self._headers = {**headers, "X-API-Key-ID": key, "x-sample-rate": "16000"}
        self._connector = connector
        self._audio: queue.Queue[bytes] = queue.Queue(maxsize=256)
        self._lock = threading.Lock()
        self._finish = threading.Event()
        self._stop = threading.Event()
        self._done = threading.Event()
        self._thread: threading.Thread | None = None
        self._result: dict[str, Any] | None = None
        self._last_audio = monotonic()
        self._error = "Gnani live transcription cancelled."

    def push_audio(self, chunk: bytes) -> None:
        with self._lock:
            if self._finish.is_set() or self._stop.is_set() or self._done.is_set() or not chunk:
                return
            if len(chunk) > 128 * 1024:
                self._abort("Gnani live audio chunk exceeded its size limit.")
                return
            try:
                self._audio.put_nowait(bytes(chunk))
            except queue.Full:
                self._abort("Gnani live audio could not be sent in real time.")
                return
            self._last_audio = monotonic()
            if self._thread is None:
                self._thread = threading.Thread(
                    target=self._run, name="gnani-prisma-stream", daemon=True
                )
                self._thread.start()

    def _abort(self, message: str) -> None:
        self._error = message
        self._stop.set()

    def cancel(self) -> None:
        with self._lock:
            self._abort("Gnani live transcription cancelled.")
            self._finish.set()

    def _success(self, text: str, languages: set[str]) -> dict[str, Any]:
        result = {
            "success": True,
            "transcript": text,
            "provider": "gnani",
            "model": self._headers["model"],
        }
        if len(languages) == 1:
            language = next(iter(languages))
            result.update(language=language, detected_language_code=language)
        return result

    def _failure(self, message: str) -> dict[str, Any]:
        return {"success": False, "transcript": "", "provider": "gnani", "error": message}

    def finalize(self) -> dict[str, Any]:
        with self._lock:
            self._finish.set()
            thread = self._thread
            if thread is None and self._result is None:
                self._result = (
                    self._failure(self._error) if self._stop.is_set() else self._success("", set())
                )
        if thread is not None and not self._done.wait(16):
            with self._lock:
                if self._result is None:
                    self._abort("Gnani live transcription timed out during finalization.")
                    self._result = self._failure(self._error)
        if thread is not None:
            thread.join(timeout=2)
        with self._lock:
            return dict(self._result or self._failure(self._error))

    def _run(self) -> None:
        result = self._failure("Gnani live STT failed.")
        try:
            result = asyncio.run(self._serve())
        except InvalidStatus as exc:
            status = exc.response.status_code
            message = {
                401: "Gnani rejected GNANI_API_KEY.",
                403: "Gnani denied live STT access. Check your key permissions.",
                429: "Gnani live STT session limit reached. Wait and retry.",
            }.get(status, "Gnani refused the live STT connection.")
            result = self._failure(message)
        except GnaniError as exc:
            result = self._failure(str(exc))
        except OSError, ValueError, WebSocketException, TimeoutError:
            result = self._failure("Gnani live STT disconnected or timed out. Try recording again.")
        finally:
            self._headers.pop("X-API-Key-ID", None)
            with self._lock:
                if self._result is None:
                    self._result = result
            self._done.set()

    async def _serve(self) -> dict[str, Any]:
        # Supplying the connected TCP socket makes websockets refuse all HTTP
        # redirects. It still performs verified TLS for the fixed URI hostname.
        with socket.socket() as sock:
            sock.setblocking(False)
            async with asyncio.timeout(5):
                await asyncio.get_running_loop().sock_connect(sock, (HOST, 443))
            async with self._connector(
                f"wss://{HOST}/stt/v3/stream",
                sock=sock,
                proxy=None,
                additional_headers=self._headers,
                open_timeout=5,
                close_timeout=1,
                max_size=256 * 1024,
                max_queue=16,
                logger=_WIRE_LOG,
            ) as ws:
                async with asyncio.timeout(5):
                    greeting = await self._receive(ws)
                if greeting.get("type") != "connected":
                    raise GnaniError("Gnani could not start live STT. Check model and account access.")
                return await self._exchange(ws)

    async def _receive(self, ws: Any) -> dict[str, Any]:
        raw = await ws.recv()
        if not isinstance(raw, str):
            raise GnaniError("Gnani returned an unexpected live STT event.")
        event = object_response(raw.encode())
        if event.get("type") == "error":
            raise GnaniError(
                "Gnani reported a live STT error. Check the language, model, and session limits."
            )
        return event

    async def _send_audio(self, ws: Any) -> None:
        pending = bytearray()
        next_frame = monotonic()
        while not self._stop.is_set():
            try:
                chunk = self._audio.get_nowait()
            except queue.Empty:
                if self._finish.is_set():
                    break
                await asyncio.sleep(0.01)
                continue
            pending.extend(chunk)
            sent = 0
            while len(pending) - sent >= FRAME_BYTES:
                if self._stop.is_set():
                    raise GnaniError(self._error)
                await asyncio.sleep(max(0, next_frame - monotonic()))
                async with asyncio.timeout(2):
                    await ws.send(bytes(pending[sent : sent + FRAME_BYTES]))
                next_frame = monotonic() + FRAME_SECONDS
                sent += FRAME_BYTES
            del pending[:sent]
        if self._stop.is_set():
            raise GnaniError(self._error)
        if len(pending) % 2:
            raise GnaniError("Live STT received an incomplete PCM sample.")
        async with asyncio.timeout(2):
            if pending:
                await asyncio.sleep(max(0, next_frame - monotonic()))
                await ws.send(bytes(pending).ljust(FRAME_BYTES, b"\x00"))
            await ws.send(json.dumps({"type": "commit"}))

    async def _exchange(self, ws: Any) -> dict[str, Any]:
        started = monotonic()
        sender = asyncio.create_task(self._send_audio(ws))
        segments: dict[int, tuple[str, str | None]] = {}
        ids: set[str] = set()
        speech_count = processing_count = 0
        committed_at: float | None = None
        try:
            while True:
                now = monotonic()
                if self._stop.is_set():
                    raise GnaniError(self._error)
                if now - started >= _SESSION_SECONDS or now - self._last_audio >= _IDLE_SECONDS:
                    raise GnaniError("Gnani live STT reached its session or idle limit.")
                if sender.done():
                    sender.result()
                    committed_at = committed_at or now
                    outstanding = max(speech_count, processing_count) - len(segments)
                    if outstanding <= 0 and now - committed_at >= 1.5:
                        break
                    if now - committed_at >= _DRAIN_SECONDS:
                        raise GnaniError("Gnani did not finish transcribing the last live segment.")
                try:
                    async with asyncio.timeout(0.1):
                        event = await self._receive(ws)
                except TimeoutError:
                    continue
                kind = event.get("type")
                if kind == "speech_start":
                    speech_count += 1
                elif kind == "processing":
                    processing_count += 1
                elif kind == "transcript":
                    self._segment(event, ids, segments)
        finally:
            sender.cancel()
            await asyncio.gather(sender, return_exceptions=True)
        languages = {language for _, language in segments.values() if language}
        texts = (segments[index][0] for index in sorted(segments))
        return self._success(" ".join(text for text in texts if text), languages)

    @staticmethod
    def _segment(
        event: dict[str, Any], ids: set[str], segments: dict[int, tuple[str, str | None]]
    ) -> None:
        index, identifier, text = (event.get(field) for field in ("segment_index", "segment_id", "text"))
        if (
            type(index) is not int
            or index < 1
            or not isinstance(identifier, str)
            or not isinstance(text, str)
        ):
            raise GnaniError("Gnani returned an invalid live transcript segment.")
        if identifier in ids or index in segments:
            return
        if len(segments) >= 5000:
            raise GnaniError("Gnani live STT produced too many transcript segments.")
        language = event.get("detected_language_code")
        segments[index] = (text.strip(), language if isinstance(language, str) else None)
        ids.add(identifier)
