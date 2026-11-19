"""Hermes file synthesis and low-latency PCM for Gnani Timbre."""

import base64
import binascii
import os
import struct
import tempfile
import wave
from collections.abc import Callable, Iterator
from pathlib import Path
from time import monotonic
from typing import Any

from agent.tts_provider import TTSProvider

from .client import AUDIO_LIMIT, Client, GnaniError, available, object_response, read_bytes
from .models import TTS_LANGUAGES, TTS_MODELS, VOICES, VOICES_V25, setup_schema

_RATE = 24000
_CONTENT_TYPES = {"wav": "audio/wav", "mp3": "audio/mpeg", "ogg": "audio/ogg"}
_EVENT_LIMIT = 2 * 1024 * 1024
_MAX_TEXT = 4000


def _events(chunks: Iterator[bytes]) -> Iterator[tuple[bytes, dict[str, Any]]]:
    pending = bytearray()
    name = b""
    data: list[bytes] = []
    size = 0
    for chunk in chunks:
        pending.extend(chunk)
        while (end := pending.find(b"\n")) >= 0:
            line = bytes(pending[:end]).rstrip(b"\r")
            del pending[: end + 1]
            size += len(line) + 1
            if size > _EVENT_LIMIT:
                raise GnaniError("Gnani SSE event exceeded the size limit.")
            if line.startswith(b"event:"):
                name = line[6:].strip()
            elif line.startswith(b"data:"):
                data.append(line[5:].removeprefix(b" "))
            elif not line:
                if data:
                    yield name, object_response(b"\n".join(data))
                name, data, size = b"", [], 0
        if len(pending) + size > _EVENT_LIMIT:
            raise GnaniError("Gnani SSE event exceeded the size limit.")


class _PCM:
    """Accept raw PCM or parse a RIFF header even across SSE event boundaries."""

    def __init__(self) -> None:
        self.pending = bytearray()
        self.kind = ""
        self.remaining: int | None = None

    def feed(self, chunk: bytes) -> bytes:
        self.pending.extend(chunk)
        if not self.kind and len(self.pending) >= 4:
            self.kind = "wav" if self.pending[:4] == b"RIFF" else "raw"
        if self.kind == "wav":
            self._header()
        if self.kind != "raw":
            return b""
        size = len(self.pending) & ~1
        if self.remaining is not None:
            if len(self.pending) > self.remaining:
                raise GnaniError("Gnani returned bytes past the streamed WAV audio data.")
            self.remaining -= size
        pcm = bytes(self.pending[:size])
        del self.pending[:size]
        return pcm

    def _header(self) -> None:
        if len(self.pending) < 12:
            return
        if self.pending[8:12] != b"WAVE":
            raise GnaniError("Gnani returned an invalid streamed WAV header.")
        at = 12
        valid_format = False
        while at + 8 <= len(self.pending):
            kind, size = struct.unpack_from("<4sI", self.pending, at)
            at += 8
            if kind == b"data":
                if not valid_format:
                    raise GnaniError("Gnani returned an invalid streamed WAV format.")
                self.kind = "raw"
                self.remaining = None if size in (0, 0xFFFFFFFF) else size
                del self.pending[:at]
                return
            if size > 65536 or at > 65536:
                raise GnaniError("Gnani streamed WAV headers exceeded the size limit.")
            if at + size > len(self.pending):
                return
            if kind == b"fmt ":
                valid_format = size >= 16 and struct.unpack_from("<HHIIHH", self.pending, at) == (
                    1,
                    1,
                    _RATE,
                    _RATE * 2,
                    2,
                    16,
                )
            at += size + (size & 1)
        if len(self.pending) > 65536:
            raise GnaniError("Gnani streamed WAV headers exceeded the size limit.")

    def finish(self) -> bytes:
        if not self.kind and len(self.pending) == 2:
            self.kind = "raw"
            return self.feed(b"")
        if self.pending or self.kind != "raw" or self.remaining not in (None, 0):
            raise GnaniError("Gnani returned incomplete streamed PCM.")
        return b""


def _complete_wav_header(path: Path) -> None:
    """Gnani REST can return a streaming RIFF with unknown container and data sizes."""
    with path.open("r+b") as handle:
        header = handle.read(12)
        if len(header) != 12 or header[:4] != b"RIFF" or header[8:12] != b"WAVE":
            return
        riff_size = struct.unpack_from("<I", header, 4)[0]
        if riff_size != 0xFFFFFFFF:
            return
        total = path.stat().st_size
        while handle.tell() < min(total, 65536):
            tag = handle.read(8)
            if len(tag) != 8:
                break
            kind, size = struct.unpack("<4sI", tag)
            if kind != b"data":
                handle.seek(size + (size & 1), 1)
                continue
            data_start = handle.tell()
            if size != 0xFFFFFFFF or total <= data_start or (total - data_start) % 2:
                break
            handle.seek(data_start - 4)
            handle.write(struct.pack("<I", total - data_start))
            handle.seek(4)
            handle.write(struct.pack("<I", total - 8))
            return
        raise GnaniError("Gnani returned an invalid streaming WAV file.")


def _validate_audio(path: Path, fmt: str) -> None:
    with path.open("rb") as handle:
        header = handle.read(256)
    if fmt == "wav":
        try:
            with wave.open(str(path)) as audio:
                size = audio.getnframes() * 2
                if not size or (audio.getnchannels(), audio.getsampwidth(), audio.getframerate()) != (
                    1,
                    2,
                    _RATE,
                ):
                    raise wave.Error
                if path.stat().st_size < size + 44:
                    raise wave.Error
                audio.setpos(audio.getnframes() - 1)
                if len(audio.readframes(1)) != 2:
                    raise wave.Error
        except wave.Error, EOFError:
            raise GnaniError("Gnani returned an invalid WAV file.") from None
    elif fmt == "ogg" and (not header.startswith(b"OggS") or b"OpusHead" not in header):
        raise GnaniError("Gnani returned an invalid OGG Opus file.")
    elif fmt == "mp3" and (
        len(header) < 16
        or not (header.startswith(b"ID3") or (header[0] == 255 and header[1] & 0xE0 == 0xE0))
    ):
        raise GnaniError("Gnani returned an invalid MP3 file.")


class GnaniTTS(TTSProvider):
    name = "gnani"
    display_name = "Gnani AI / Timbre"
    stream_sample_rate = _RATE
    streams_pcm = True
    voice_compatible = True

    def __init__(
        self, setting: Callable[[str, Any], Any], host_config: Callable[[], dict], client: Client
    ) -> None:
        self._setting = setting
        self._host_config = host_config
        self._client = client

    def is_available(self) -> bool:
        return available()

    def get_setup_schema(self) -> dict[str, Any]:
        return setup_schema(self.display_name)

    def list_models(self) -> list[dict[str, Any]]:
        return [{"id": key, "display": name} for key, name in TTS_MODELS.items()]

    def _model(self, model: str | None) -> str:
        section = self._host_config().get("tts") or {}
        model = model or section.get("model") or self._setting("tts_model", "timbre-v2.5")
        if model not in TTS_MODELS:
            raise GnaniError("Select timbre-v2.5 or timbre-v3.0 for Gnani TTS.")
        return str(model)

    def list_voices(self) -> list[dict[str, Any]]:
        try:
            model = self._model(None)
        except GnaniError:
            return []
        return [
            {
                "id": voice,
                "display": voice,
                **({"language": VOICES_V25[voice]} if model == "timbre-v2.5" else {}),
            }
            for voice in VOICES[model]
        ]

    def _payload(
        self, text: str, voice: str | None, model: str | None, speed: float | None, fmt: str
    ) -> dict[str, Any]:
        if not text.strip() or len(text) > _MAX_TEXT:
            raise GnaniError(f"Gnani TTS needs non-empty text up to {_MAX_TEXT} characters per request.")
        model = self._model(model)
        legacy = model == "timbre-v2.5"
        section = self._host_config().get("tts") or {}
        voice = (
            voice
            or section.get("voice")
            or self._setting("timbre_v25_voice" if legacy else "timbre_v30_voice", VOICES[model][0])
        )
        if voice not in VOICES[model]:
            raise GnaniError(f"Choose a voice from the {model} catalog. Voice names are case-sensitive.")
        language = self._setting("tts_language", "auto")
        if language not in TTS_LANGUAGES:
            raise GnaniError("Select a supported Timbre language or auto.")
        if speed is None:
            speed = section.get("speed", 1.0)
        if (
            isinstance(speed, bool)
            or not isinstance(speed, (int, float))
            or not (0.85 <= speed <= 1.15 if legacy else speed in (0.8, 1.0, 1.2))
        ):
            raise GnaniError("Use speed 0.85–1.15 on Timbre v2.5 or 0.8, 1.0, or 1.2 on v3.0.")
        config = {"container": fmt, "sample_rate": _RATE, "num_channels": 1, "sample_width": 2}
        if fmt in ("raw", "wav"):
            config["encoding"] = "linear_pcm"
        return {
            "text": text,
            "voice": voice,
            "model": model,
            "language": language,
            "speed": speed,
            "audio_config": config,
        }

    def synthesize(
        self,
        text: str,
        output_path: str,
        *,
        voice: str | None = None,
        model: str | None = None,
        speed: float | None = None,
        format: str = "mp3",
        **extra: Any,
    ) -> str:
        fmt = "ogg" if format == "opus" else format if format in _CONTENT_TYPES else "wav"
        payload = self._payload(text, voice, model, speed, fmt)
        target = Path(output_path).with_suffix("." + fmt)
        deadline = monotonic() + 60
        staged: Path | None = None
        try:
            with self._client.response(
                "POST", "/api/v1/tts/inference", deadline=deadline, json=payload
            ) as response:
                media_type = response.headers.get("content-type", "").split(";")[0].strip().lower()
                if media_type not in (_CONTENT_TYPES[fmt], "application/octet-stream"):
                    raise GnaniError("Gnani did not return an audio response.")
                with tempfile.NamedTemporaryFile(
                    dir=target.parent, prefix=".gnani-", delete=False
                ) as handle:
                    staged = Path(handle.name)
                    for chunk in read_bytes(response, AUDIO_LIMIT, deadline):
                        handle.write(chunk)
            if fmt == "wav":
                _complete_wav_header(staged)
            _validate_audio(staged, fmt)
            os.replace(staged, target)
        except OSError:
            raise GnaniError("Could not write the Gnani audio file.") from None
        finally:
            if staged is not None:
                staged.unlink(missing_ok=True)
        return str(target)

    def stream(
        self,
        text: str,
        *,
        voice: str | None = None,
        model: str | None = None,
        format: str = "opus",
        **extra: Any,
    ) -> Iterator[bytes]:
        if format in ("opus", "ogg"):
            with tempfile.TemporaryDirectory(prefix="gnani-voice-") as directory:
                path = self.synthesize(
                    text,
                    str(Path(directory) / "speech.ogg"),
                    voice=voice,
                    model=model,
                    speed=extra.get("speed"),
                    format="ogg",
                )
                yield Path(path).read_bytes()
            return
        if format != "pcm":
            raise NotImplementedError("Gnani live TTS streams PCM. Use synthesis for this format.")
        payload = self._payload(text, voice, model, extra.get("speed"), "raw")
        deadline = monotonic() + 60
        parser = _PCM()
        emitted = False
        previous_index = 0
        with self._client.response(
            "POST", "/api/v1/tts/sse", deadline=deadline, json=payload
        ) as response:
            if (
                response.headers.get("content-type", "").split(";")[0].strip().lower()
                != "text/event-stream"
            ):
                raise GnaniError("Gnani did not return a TTS event stream.")
            for event, data in _events(read_bytes(response, AUDIO_LIMIT, deadline)):
                if event == b"error":
                    raise GnaniError(
                        "Gnani streaming synthesis failed. Check the text and voice settings."
                    )
                if event == b"complete":
                    if tail := parser.finish():
                        emitted = True
                        yield tail
                    if not emitted:
                        raise GnaniError("Gnani streaming synthesis returned no audio.")
                    return
                if event == b"chunk":
                    index = data.get("chunk_index")
                    if type(index) is not int or index != previous_index + 1:
                        raise GnaniError("Gnani sent out-of-order audio chunks.")
                    previous_index = index
                    try:
                        audio = base64.b64decode(data["audio"], validate=True)
                    except KeyError, ValueError, TypeError, binascii.Error:
                        raise GnaniError("Gnani returned an invalid streamed audio fragment.") from None
                    if pcm := parser.feed(audio):
                        emitted = True
                        yield pcm
        raise GnaniError("Gnani ended the audio stream before synthesis completed.")
