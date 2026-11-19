"""Opt-in, serial smoke tests for actual billable speech calls."""

import json
import os
import re
import shutil
import subprocess
import wave
from pathlib import Path
from time import sleep

import pytest

from gnani.client import Client
from gnani.stt import GnaniTranscription
from gnani.tts import GnaniTTS

from conftest import write_wav

pytestmark = pytest.mark.live
EN = "The blue bicycle is parked beside the garden gate."
HI = "नमस्ते, आज मौसम बहुत अच्छा है।"
MIXED = "आज मेरी team की meeting है।"
PRISMA = ("prisma-v2.5", "prisma-v3.0")
TIMBRE = (("timbre-v2.5", "Kaveri"), ("timbre-v3.0", "Shivani"))


@pytest.fixture(scope="module")
def evidence(tmp_path_factory):
    root = os.environ.get("GNANI_LIVE_OUTPUT")
    if root:
        folder = Path(root)
        folder.mkdir(mode=0o700, parents=True, exist_ok=True)
        return folder
    return tmp_path_factory.mktemp("gnani-live")


@pytest.fixture(scope="module")
def speech(evidence):
    client = Client()
    files = {}
    for label, text, model, voice, language in [
        ("en", EN, "timbre-v2.5", "Kaveri", "en-IN"),
        ("hi", HI, "timbre-v3.0", "Shivani", "hi-IN"),
        ("mixed", MIXED, "timbre-v2.5", "Poorvi", "hi-en"),
    ]:
        tts = GnaniTTS({"tts_language": language}.get, dict, client)
        path = Path(
            tts.synthesize(text, str(evidence / f"{label}.wav"), voice=voice, model=model, format="wav")
        )
        with wave.open(str(path)) as audio:
            assert audio.getnframes() > 0
            assert (audio.getframerate(), audio.getnchannels(), audio.getsampwidth()) == (24000, 1, 2)
        files[label] = path
    return files


def _assert_phrase(result, words):
    assert result["success"], result.get("error")
    normalized = re.sub(r"[^\w\s]", "", result["transcript"].casefold())
    assert sum(word in normalized for word in words) >= len(words) - 1


def test_real_prisma_rest_and_round_trip(speech, evidence):
    stt = GnaniTranscription({}.get, dict, Client())
    for label, path in speech.items():
        result = stt.transcribe(str(path), model="prisma-v3.0")
        (evidence / f"{label}-prisma-v30.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2)
        )
        assert result["success"] and result["transcript"].strip()
        if label in ("en", "hi"):
            assert result["detected_language_code"] == f"{label}-IN"
        if label == "en":
            _assert_phrase(result, ("blue", "bicycle", "garden", "gate"))
    legacy = stt.transcribe(str(speech["en"]), model="prisma-v2.5", language="en-IN")
    _assert_phrase(legacy, ("blue", "bicycle", "garden", "gate"))
    (evidence / "en-prisma-v25.json").write_text(json.dumps(legacy, ensure_ascii=False, indent=2))


@pytest.mark.parametrize("model", ["timbre-v2.5", "timbre-v3.0"])
def test_real_timbre_sse(evidence, model):
    tts = GnaniTTS({"tts_language": "hi-IN"}.get, dict, Client())
    pcm = b"".join(tts.stream(HI, model=model, format="pcm"))
    assert len(pcm) > 2400 and len(pcm) % 2 == 0
    write_wav(evidence / f"{model}-sse.wav", 0, tts.stream_sample_rate, pcm)


def _english_input(speech):
    given = os.environ.get("GNANI_LIVE_AUDIO_EN")
    return Path(given) if given else speech["en"]


@pytest.mark.parametrize("model", PRISMA)
def test_real_prisma_websocket(speech, evidence, model):
    path = _english_input(speech)
    pcm = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(path),
            "-t",
            "15",
            "-f",
            "s16le",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-",
        ],
        check=True,
        capture_output=True,
        timeout=20,
    ).stdout
    stt = GnaniTranscription({"stt_model": model}.get, dict, Client())
    session = stt.open_stream_session(language="en-IN")
    for at in range(0, len(pcm), 1024):
        session.push_audio(pcm[at : at + 1024])
        sleep(0.032)
    result = session.finalize()
    assert result["success"] and result["transcript"].strip(), result.get("error")
    if not os.environ.get("GNANI_LIVE_AUDIO_EN"):
        _assert_phrase(result, ("blue", "bicycle", "garden", "gate"))
    assert result["model"] == model
    (evidence / f"{model}-websocket.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))


@pytest.mark.parametrize("model", PRISMA)
def test_real_prisma_long_audio(speech, evidence, model):
    path = os.environ.get("GNANI_LIVE_AUDIO_LONG")
    if not path:
        with wave.open(str(speech["en"])) as audio:
            rate = audio.getframerate()
            clip = audio.readframes(audio.getnframes()) + b"\0\0" * (rate // 3)
        path = str(write_wav(evidence / "long.wav", 0, rate, clip * (1 + 64 * rate // len(clip))))
    stt = GnaniTranscription({}.get, dict, Client())
    result = stt.transcribe(path, model=model, language="en-IN")
    assert result["success"] and result["transcript"].strip(), result.get("error")
    assert result["model"] == model
    assert result.get("job_id"), "Supply a recording longer than 30 seconds."
    (evidence / f"{model}-batch.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))


@pytest.mark.parametrize("locale", ["EN", "HI", "MIXED"])
def test_real_natural_speech(locale, evidence):
    path = os.environ.get(f"GNANI_LIVE_AUDIO_{locale}")
    if not path:
        pytest.skip(f"Provide authorized natural speech via GNANI_LIVE_AUDIO_{locale}.")
    settings = {"stt_format": "codemix"} if locale == "MIXED" else {}
    result = GnaniTranscription(settings.get, dict, Client()).transcribe(path)
    assert result["success"] and result["transcript"].strip(), result.get("error")
    (evidence / f"natural-{locale.lower()}.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2)
    )


@pytest.mark.skipif(
    os.environ.get("GNANI_TEST_HERMES") != "1", reason="Install into isolated Hermes first."
)
@pytest.mark.parametrize(
    ("stt_model", "tts_model", "voice"),
    [
        ("prisma-v2.5", "timbre-v2.5", "Kaveri"),
        ("prisma-v3.0", "timbre-v3.0", "Shivani"),
    ],
)
def test_live_hermes_dispatch(evidence, tmp_path, monkeypatch, stt_model, tts_model, voice):
    from hermes_cli.config import save_config
    from hermes_cli.plugins import discover_plugins
    from tools.transcription_streaming import open_streaming_session
    from tools.transcription_tools import transcribe_audio
    from tools.tts_streaming import resolve_streaming_provider
    from tools.tts_tool import text_to_speech_tool

    # Start from the already installed plugin. Keep each model pair's settings
    # isolated from the developer's profile and from the other pair.
    installed = Path(os.environ["HERMES_HOME"]) / "plugins" / "gnani"
    shutil.copytree(installed, tmp_path / "plugins" / "gnani")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    save_config(
        {
            "plugins": {
                "enabled": ["gnani"],
                "entries": {"gnani": {"settings": {"tts_language": "en-IN"}}},
            },
            "stt": {
                "provider": "gnani",
                "streaming": True,
                "gnani": {"model": stt_model, "language": "en-IN"},
            },
            "tts": {"provider": "gnani", "model": tts_model, "voice": voice, "output_format": "wav"},
        },
        strip_defaults=False,
    )
    discover_plugins(force=True)
    target = evidence / f"hermes-{tts_model}-reply.wav"
    spoken = json.loads(text_to_speech_tool(EN, output_path=str(target)))
    assert spoken["success"] and spoken["provider"] == "gnani", spoken.get("error")
    target = Path(spoken["file_path"])
    result = transcribe_audio(str(target), source="gateway")
    assert result["provider"] == "gnani" and result.get("model") == stt_model, result.get("error")
    _assert_phrase(result, ("blue", "bicycle", "garden", "gate"))
    (evidence / f"hermes-{stt_model}-voice-note.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2)
    )

    streamer = resolve_streaming_provider({"provider": "gnani", "model": tts_model, "voice": voice})
    assert streamer is not None
    pcm = b"".join(streamer.stream(EN))
    assert len(pcm) > 2400 and len(pcm) % 2 == 0
    write_wav(evidence / f"hermes-{tts_model}-stream.wav", 0, 24000, pcm)
    host = open_streaming_session()
    assert host is not None
    decoded = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(target), "-f", "s16le", "-ac", "1", "-ar", "16000", "-"],
        check=True,
        capture_output=True,
        timeout=20,
    ).stdout
    for at in range(0, len(decoded), 1024):
        host.push_audio(decoded[at : at + 1024])
        sleep(0.032)
    result = host.finalize()
    _assert_phrase(result, ("blue", "bicycle", "garden", "gate"))
    assert result["model"] == stt_model
    (evidence / f"hermes-{stt_model}-live.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2)
    )


@pytest.mark.parametrize(("model", "voice"), TIMBRE)
@pytest.mark.parametrize("fmt", ["mp3", "ogg"])
def test_real_encoded_synthesis(evidence, fmt, model, voice):
    tts = GnaniTTS({"tts_language": "en-IN"}.get, dict, Client())
    target = tts.synthesize(EN, str(evidence / f"{model}.{fmt}"), model=model, voice=voice, format=fmt)
    subprocess.run(
        ["ffmpeg", "-v", "error", "-i", target, "-f", "null", "-"],
        check=True,
        capture_output=True,
        timeout=20,
    )


@pytest.mark.skipif(
    os.environ.get("GNANI_LIVE_PRESIGNED") != "1", reason="Large signed upload is opt-in."
)
def test_live_presigned_batch(speech, evidence):
    path = evidence / "presigned.wav"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-stream_loop",
            "-1",
            "-i",
            str(speech["en"]),
            "-t",
            "56",
            "-ar",
            "48000",
            "-ac",
            "2",
            str(path),
        ],
        check=True,
        capture_output=True,
        timeout=20,
    )
    assert path.stat().st_size > 10_000_000
    result = GnaniTranscription({}.get, dict, Client()).transcribe(str(path))
    assert result["success"] and result["transcript"].strip(), result.get("error")
    assert result.get("job_id")
    (evidence / "presigned.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
