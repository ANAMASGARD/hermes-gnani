"""Host tests use the source and installed plugin in a disposable Hermes profile."""

import json
import os
from pathlib import Path

import httpx
import pytest

from conftest import write_wav

pytestmark = [
    pytest.mark.hermes,
    pytest.mark.skipif(
        os.environ.get("GNANI_TEST_HERMES") != "1", reason="Isolated host test is opt-in"
    ),
]


def test_discovery_and_file_and_pcm_dispatch(tmp_path, monkeypatch):
    from agent import transcription_registry, tts_registry
    from hermes_cli.plugins import discover_plugins
    from hermes_cli.tools_config_providers import _plugin_tts_providers
    from hermes_cli.web_server_config import _schema_with_dynamic_provider_options
    from tools.transcription_tools import transcribe_audio
    from tools.tts_streaming import resolve_streaming_provider
    from tools.tts_tool import text_to_speech_tool

    monkeypatch.setenv("GNANI_API_KEY", "isolated-host-test-key")
    discover_plugins()
    stt = transcription_registry.get_provider("gnani")
    tts = tts_registry.get_provider("gnani")
    assert stt is not None and tts is not None
    assert stt.display_name == "Gnani AI / Prisma"
    assert tts.display_name == "Gnani AI / Timbre"
    schema = _schema_with_dynamic_provider_options()
    assert "gnani" in schema["tts.provider"]["options"]
    assert any(row["tts_provider"] == "gnani" for row in _plugin_tts_providers())

    wav = write_wav(tmp_path / "fake-tts.wav", 0.1, 24000).read_bytes()
    audio = write_wav(tmp_path / "voice-note.wav", 0.1)
    original = httpx.Client
    calls = []

    def handle(request):
        assert request.headers["x-api-key-id"] == "isolated-host-test-key"
        path = request.url.path
        calls.append(path)
        if path == "/stt/v3":
            assert b"prisma-v3.0" in request.read()
            return httpx.Response(200, json={"success": True, "transcript": "Voice note from Hermes."})
        if path == "/api/v1/tts/inference":
            assert json.loads(request.read())["text"] == "Hermes speaking."
            return httpx.Response(200, content=wav, headers={"content-type": "audio/wav"})
        if path == "/api/v1/tts/sse":
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                content=(
                    'event: chunk\ndata: {"chunk_index": 1, "audio": "AQABAA=="}\n\n'
                    'event: complete\ndata: {"is_final": true}\n\n'
                ),
            )
        pytest.fail(f"Unexpected endpoint {path}")

    def patched_client(*args, **kwargs):
        if kwargs.get("transport") is None and kwargs.get("trust_env") is False:
            kwargs["transport"] = httpx.MockTransport(handle)
        return original(*args, **kwargs)

    monkeypatch.setattr(httpx, "Client", patched_client)
    result = transcribe_audio(str(audio), source="gateway")
    assert result["provider"] == "gnani" and result["transcript"] == "Voice note from Hermes."
    spoken = json.loads(text_to_speech_tool("Hermes speaking.", output_path=str(tmp_path / "reply.wav")))
    assert spoken["success"] and spoken["provider"] == "gnani", spoken
    reply = Path(spoken["file_path"])
    assert reply.suffix == ".ogg" and reply.read_bytes().startswith(b"OggS")
    config = {"provider": "gnani", "streaming": {"provider": "gnani"}}
    streamer = resolve_streaming_provider(config)
    assert streamer is not None and streamer.sample_rate == 24000
    assert b"".join(streamer.stream("Hermes stream.")) == b"\x01\x00\x01\x00"
    assert calls == ["/stt/v3", "/api/v1/tts/inference", "/api/v1/tts/sse"]
