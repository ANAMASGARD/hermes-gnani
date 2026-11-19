"""Hermes file and live transcription provider for Gnani Prisma."""

import json
import math
import shutil
import subprocess
import wave
from collections.abc import Callable
from pathlib import Path
from time import monotonic
from typing import Any

from agent.transcription_provider import TranscriptionProvider, TranscriptionStreamSession

from . import batch
from .client import Client, GnaniError, api_key, available
from .models import INDIC, STT_LEGACY_LANGUAGES, STT_MODELS, setup_schema
from .stream import StreamSession

_FORMATS = {"wav", "mp3", "ogg", "flac", "aac", "m4a"}
_BATCH_ONLY = {"mp4", "webm", "amr"}
_SHORT_LIMIT = 64 * 1024 * 1024
Setting = Callable[[str, Any], Any]


def failure(message: str) -> dict[str, Any]:
    return {"success": False, "transcript": "", "provider": "gnani", "error": message}


def success(text: str, model: str, language: Any = None, **metadata: Any) -> dict[str, Any]:
    result = {"success": True, "transcript": text, "provider": "gnani", "model": model, **metadata}
    if isinstance(language, str) and language:
        result["language"] = language
        result["detected_language_code"] = language
    return result


def duration(path: Path, fmt: str) -> float:
    if fmt == "wav":
        try:
            with wave.open(str(path)) as reader:
                return reader.getnframes() / reader.getframerate()
        except wave.Error, EOFError:
            pass
    executable = shutil.which("ffprobe")
    if executable is None:
        raise GnaniError(
            "Install FFmpeg (ffprobe) to check this recording's duration, or provide PCM WAV."
        )
    try:
        result = subprocess.run(
            [
                executable,
                "-v",
                "error",
                "-protocol_whitelist",
                "file",
                "-format_whitelist",
                "wav,mp3,ogg,flac,aac,mov,matroska,webm,amr",
                "-show_entries",
                "format=duration:stream=codec_type",
                "-of",
                "json",
                str(path),
            ],
            capture_output=True,
            check=True,
            timeout=10,
        )
        info = json.loads(result.stdout)
        if not any(entry["codec_type"] == "audio" for entry in info["streams"]):
            raise ValueError
        seconds = float(info["format"]["duration"])
        if not math.isfinite(seconds) or seconds <= 0:
            raise ValueError
        return seconds
    except OSError, subprocess.SubprocessError, ValueError, KeyError, TypeError:
        raise GnaniError(
            "Could not determine audio duration. Try re-exporting the recording as WAV."
        ) from None


def _list(value: Any, key: str, limit: int) -> list[str]:
    if (
        not isinstance(value, list)
        or len(value) > limit
        or any(
            not isinstance(item, str)
            or not item.strip()
            or len(item) > 256
            or "\n" in item
            or "\r" in item
            or "<|" in item
            or "|>" in item
            for item in value
        )
    ):
        raise GnaniError(f"Invalid {key}. Provide up to {limit} non-empty single-line entries.")
    return [item.strip() for item in value]


def _wire_fields(options: dict[str, Any], *, headers: bool = False) -> dict[str, str]:
    fields = {
        key: json.dumps(value, ensure_ascii=headers) if isinstance(value, list) else str(value)
        for key, value in options.items()
    }
    if headers and any(
        not value.isascii() or "\n" in value or "\r" in value for value in fields.values()
    ):
        raise GnaniError(
            "Use single-line ASCII topic and previous_text for live STT. REST accepts Unicode."
        )
    return fields


def _batch_config(options: dict[str, Any]) -> dict[str, Any]:
    config = options.copy()
    if config["format"] == "transliterate" or "previous_text" in config:
        raise GnaniError(
            "Batch supports neither transliteration nor previous_text. Use clips up to 30 seconds."
        )
    if config["model"] == "prisma-v2.5":
        if config.pop("format") != "verbatim":
            raise GnaniError(
                "Prisma v2.5 Batch requires verbatim. Change the style or use shorter clips."
            )
    else:
        if config["language_code"] == "indic-auto":
            config["language_code"] = "auto"
        if "speaker_names" in config and {"keyterms", "topic"} & config.keys():
            raise GnaniError("Batch speaker_names must be used without keyterms or topic.")
        if "topic" in config and len(config["topic"]) < 30:
            raise GnaniError("Batch topic must contain at least 30 characters.")
    return config


class GnaniTranscription(TranscriptionProvider):
    name = "gnani"
    display_name = "Gnani AI / Prisma"

    def __init__(self, setting: Setting, host_config: Callable[[], dict], client: Client) -> None:
        self._setting = setting
        self._host_config = host_config
        self._client = client

    def list_models(self) -> list[dict[str, Any]]:
        return [{"id": key, "display": name} for key, name in STT_MODELS.items()]

    def get_setup_schema(self) -> dict[str, Any]:
        return setup_schema(self.display_name)

    def is_available(self) -> bool:
        return available()

    def _options(self, model: str | None, language: str | None, prompt: Any) -> dict[str, Any]:
        section = self._host_config().get("stt") or {}
        config = section.get("gnani") or {}
        model = model or config.get("model") or self._setting("stt_model", "prisma-v3.0")
        if model not in STT_MODELS:
            raise GnaniError("Select prisma-v3.0 or prisma-v2.5 for Gnani STT.")
        modern = model == "prisma-v3.0"
        languages = INDIC if modern else STT_LEGACY_LANGUAGES
        host_language = section.get("language")
        # Hermes seeds the global Whisper hint to "en" even on a fresh profile.
        inherited = host_language if host_language != "en" else None
        if language == "en" and not config.get("language"):
            language = None
        language = language or config.get("language") or self._setting("stt_language", "") or inherited
        language = language or ("indic-auto" if modern else "")
        codes = [part.strip() for part in language.split(",")]
        if not (modern and codes == ["indic-auto"]) and (
            len(codes) > (4 if modern else 1)
            or len(set(codes)) != len(codes)
            or any(code not in languages for code in codes)
        ):
            raise GnaniError(
                "Select a supported Prisma locale. Prisma v2.5 requires one explicit locale."
            )
        style = self._setting("stt_format", "verbatim")
        styles = (
            {"verbatim", "transcribe", "codemix", "transliterate"}
            if modern
            else {"verbatim", "transcribe"}
        )
        if style not in styles:
            raise GnaniError("The selected transcription style is unsupported by this Prisma model.")
        options = {"model": model, "language_code": ",".join(codes), "format": style}
        if modern:
            self._modern_context(options, prompt)
        else:
            self._legacy_context(options, prompt)
        return options

    def _modern_context(self, options: dict[str, Any], prompt: Any) -> None:
        for key, limit in (("keyterms", 200), ("speaker_names", 4)):
            items = self._setting(key, [])
            if key == "keyterms" and not items and isinstance(prompt, str):
                items = [item.strip() for item in prompt.split(",") if item.strip()]
            if items:
                options[key] = _list(items, key, limit)
        for key, limit in (("topic", 1600), ("previous_text", 600)):
            value = self._setting(key, "")
            if not value:
                continue
            if not isinstance(value, str) or not value.strip() or "<|" in value or "|>" in value:
                raise GnaniError(f"Invalid Prisma context field {key}.")
            if key == "topic" and len(value) > limit:
                raise GnaniError("Prisma topic must be at most 1,600 characters.")
            options[key] = value[-limit:]

    def _legacy_context(self, options: dict[str, Any], prompt: Any) -> None:
        if any(
            self._setting(key, None) for key in ("keyterms", "topic", "previous_text", "speaker_names")
        ):
            raise GnaniError("Prisma v3.0 context options require prisma-v3.0. Use bias_list for v2.5.")
        words = self._setting("bias_list", [])
        if not words and isinstance(prompt, str):
            words = [part.strip() for part in prompt.split(",") if part.strip().isalpha()]
        if words:
            words = _list(words, "bias_list", 100)
            score = self._setting("bias_score", 1.0)
            if not all(word.isalpha() for word in words) or score not in (0, 0.5, 1, 1.5, 2):
                raise GnaniError("Prisma bias_list requires single words and an allowed bias_score.")
            options.update(bias_list=words, bias_score=score)

    @property
    def streaming_capable(self) -> bool:
        return self.is_available()

    def open_stream_session(
        self, *, language: str | None = None, prompt: str | None = None
    ) -> TranscriptionStreamSession:
        options = self._options(None, language, prompt)
        return StreamSession(_wire_fields(options, headers=True), api_key())

    def transcribe(
        self, file_path: str, *, model: str | None = None, language: str | None = None, **extra: Any
    ) -> dict[str, Any]:
        try:
            options = self._options(model, language, extra.get("prompt"))
            return self._transcribe(Path(file_path), options)
        except GnaniError as exc:
            return failure(str(exc))
        except OSError, ValueError, TypeError, RuntimeError:
            return failure(
                "Could not transcribe this recording. Check the audio file and active Hermes profile."
            )

    def _transcribe(self, path: Path, options: dict[str, Any]) -> dict[str, Any]:
        path = path.resolve(strict=True)
        fmt = path.suffix.lower().lstrip(".")
        fmt = "ogg" if fmt in ("oga", "opus") else fmt
        if fmt not in _FORMATS | _BATCH_ONLY or not path.is_file():
            raise GnaniError(
                "Provide a supported WAV, MP3, OGG, FLAC, AAC, M4A, MP4, WebM, or AMR file."
            )
        size = path.stat().st_size
        if not 0 < size <= batch.UPLOAD_LIMIT:
            raise GnaniError("Provide a non-empty audio file within Gnani's 5 GiB Batch limit.")
        seconds = duration(path, fmt)
        if seconds > batch.MAX_SECONDS:
            raise GnaniError(
                "Split recordings longer than four hours before sending them to Gnani Batch."
            )
        if seconds <= 0:
            raise GnaniError("The audio file contains no samples.")
        filename = f"audio.{fmt}"
        if seconds <= 30 and fmt in _FORMATS and size <= _SHORT_LIMIT:
            try:
                with path.open("rb") as handle:
                    data = self._client.json(
                        "POST",
                        "/stt/v3",
                        deadline=monotonic() + 45,
                        data=_wire_fields(options),
                        files={"audio_file": (filename, handle)},
                    )
            except GnaniError as exc:
                if exc.code != "duration":
                    raise
            else:
                if data.get("truncated"):
                    raise GnaniError("Prisma cut the transcript short. Try a shorter recording.")
                if data.get("success") is not True or not isinstance(data.get("transcript"), str):
                    raise GnaniError("Gnani REST did not return a valid transcript.")
                return success(data["transcript"], options["model"], data.get("detected_language_code"))
        data, job_id = batch.transcribe(self._client, path, filename, size, _batch_config(options))
        if not isinstance(data.get("full_transcript"), str):
            raise GnaniError("Gnani Batch did not return a valid transcript.")
        segments = data.get("segments")
        detected = (
            {
                segment["language_detected"]
                for segment in segments
                if isinstance(segment, dict) and isinstance(segment.get("language_detected"), str)
            }
            if isinstance(segments, list)
            else set()
        )
        language_code = next(iter(detected)) if len(detected) == 1 else None
        return success(data["full_transcript"], options["model"], language_code, job_id=job_id)
