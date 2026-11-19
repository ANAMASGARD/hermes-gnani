import base64
import json
from concurrent.futures import ThreadPoolExecutor

import httpx
import pytest

from gnani.client import Client, GnaniError
from gnani.tts import GnaniTTS

from conftest import write_wav


def provider(handler, **settings):
    return GnaniTTS(settings.get, dict, Client(httpx.MockTransport(handler)))


@pytest.mark.parametrize(
    ("model", "voice", "language", "speed"),
    [
        ("timbre-v2.5", "Kaveri", "en-IN", 0.93),
        ("timbre-v2.5", "Poorvi", "hi-en", 1.15),
        ("timbre-v3.0", "Shivani", "hi-IN", 0.8),
    ],
)
def test_synthesis(tmp_path, model, voice, language, speed):
    pcm = b"\x01\x00" * 192
    audio = write_wav(tmp_path / "source.wav", 0, 24000, pcm).read_bytes()
    text = "<whispers> नमस्ते" if model == "timbre-v3.0" else "Hello, नमस्ते"

    def handler(request):
        assert request.url.path == "/api/v1/tts/inference"
        payload = json.loads(request.read())
        assert (
            payload["text"],
            payload["model"],
            payload["voice"],
            payload["language"],
            payload["speed"],
        ) == (
            text,
            model,
            voice,
            language,
            speed,
        )
        assert payload["audio_config"]["container"] == "wav"
        return httpx.Response(200, content=audio, headers={"content-type": "audio/wav"})

    result = provider(handler, tts_language=language).synthesize(
        text,
        str(tmp_path / "target.mp3"),
        voice=voice,
        model=model,
        speed=speed,
        format="wav",
    )
    assert result.endswith(".wav") and (tmp_path / "target.wav").read_bytes() == audio


@pytest.mark.parametrize(
    "kwargs",
    [
        {"model": "timbre-v3.0", "voice": "Nalini"},
        {"voice": "nalini"},
        {"model": "timbre-v3.0", "speed": 1.1},
        {"speed": 1.2},
    ],
)
def test_model_constraints(tmp_path, kwargs):
    with pytest.raises(GnaniError):
        provider(lambda _: pytest.fail("No network expected")).synthesize(
            "Hello", str(tmp_path / "x"), **kwargs
        )


def event(name, data):
    return f"event: {name}\r\ndata: {json.dumps(data)}\r\n\r\n".encode()


def chunks(encoded):
    return b"".join(
        event("chunk", {"chunk_index": i, "audio": base64.b64encode(piece).decode()})
        for i, piece in enumerate(encoded, 1)
    )


class Stream(httpx.SyncByteStream):
    def __init__(self, data):
        self.data = data
        self.closed = False

    def __iter__(self):
        for index in range(0, len(self.data), 5):
            yield self.data[index : index + 5]

    def close(self):
        self.closed = True


@pytest.mark.parametrize("wrapped", [True, False])
def test_sse_pcm(tmp_path, wrapped):
    pcm = b"\0\1\2\3\4\5\6\7"
    body = write_wav(tmp_path / "sample.wav", 0, 24000, pcm).read_bytes() if wrapped else pcm
    stream = Stream(chunks([body[:1], body[1:7], body[7:]]) + event("complete", {"is_final": True}))

    def handler(request):
        assert request.url.path == "/api/v1/tts/sse"
        assert json.loads(request.read())["audio_config"] == {
            "container": "raw",
            "sample_rate": 24000,
            "num_channels": 1,
            "sample_width": 2,
            "encoding": "linear_pcm",
        }
        return httpx.Response(200, stream=stream, headers={"content-type": "text/event-stream"})

    assert b"".join(provider(handler).stream("Hello", format="pcm")) == pcm
    assert stream.closed


def test_stream_close_releases_response():
    stream = Stream(chunks([b"012345", b"6789"]))

    def handler(_):
        return httpx.Response(200, stream=stream, headers={"content-type": "text/event-stream"})

    iterator = provider(handler).stream("Hello", format="pcm")
    assert next(iterator) == b"012345"
    iterator.close()
    assert stream.closed


def test_unterminated_stream():
    stream = Stream(chunks([b"1234"]))

    def handler(_):
        return httpx.Response(200, stream=stream, headers={"content-type": "text/event-stream"})

    with pytest.raises(GnaniError, match="before synthesis completed"):
        list(provider(handler).stream("Hello", format="pcm"))


def test_concurrent_streams():
    def handler(request):
        text = json.loads(request.read())["text"].encode()
        data = chunks([text]) + event("complete", {"is_final": True})
        return httpx.Response(200, stream=Stream(data), headers={"content-type": "text/event-stream"})

    tts = provider(handler)
    with ThreadPoolExecutor(max_workers=3) as executor:
        outputs = list(
            executor.map(lambda text: b"".join(tts.stream(text, format="pcm")), ["test", "okay", "good"])
        )
    assert outputs == [b"test", b"okay", b"good"]


def test_rejects_json_audio_without_replacing_target(tmp_path):
    target = tmp_path / "target.wav"
    target.write_bytes(b"previous")
    tts = provider(lambda _: httpx.Response(200, json={"error": "private"}))
    with pytest.raises(GnaniError, match="audio response"):
        tts.synthesize("Hello", str(target), format="wav")
    assert target.read_bytes() == b"previous"


def test_rest_fixes_streaming_wav_sizes(tmp_path):
    pcm = b"\x03\x00" * 512
    wav = bytearray(write_wav(tmp_path / "source.wav", 0, 24000, pcm).read_bytes())
    wav[4:8] = wav[40:44] = b"\xff" * 4
    tts = provider(
        lambda _: httpx.Response(200, content=bytes(wav), headers={"content-type": "audio/wav"})
    )
    output = tts.synthesize("Test", str(tmp_path / "valid.wav"), format="wav")
    from pathlib import Path

    assert Path(output).read_bytes() == (tmp_path / "source.wav").read_bytes()
