import json
from time import monotonic

import httpx
import pytest

from gnani import batch, client
from gnani.client import Client, GnaniError
from gnani.stt import GnaniTranscription

from conftest import write_wav

JOB = "019fa79e-81f9-7a8a-a446-6eff16ddec30"
SIGNED = "https://gnaniasraudios.s3.amazonaws.com/recording?X-Amz-Signature=secret-link"


def provider(handler, **settings):
    return GnaniTranscription(settings.get, dict, Client(httpx.MockTransport(handler)))


@pytest.mark.parametrize(
    ("model", "language", "style"),
    [
        ("prisma-v3.0", "indic-auto", "verbatim"),
        ("prisma-v3.0", "hi-IN,en-IN", "codemix"),
        ("prisma-v3.0", "hi-IN", "transliterate"),
        ("prisma-v3.0", "en-IN", "transcribe"),
        ("prisma-v2.5", "en-IN", "transcribe"),
    ],
)
def test_multipart(tmp_path, model, language, style):
    path = write_wav(tmp_path / "private-filename.wav", 30)

    def handler(request):
        assert request.url.path == "/stt/v3"
        assert request.headers["x-api-key-id"] == "offline-test-key"
        body = request.read()
        for name in ("audio_file", "model", "language_code", "format"):
            assert f'name="{name}"'.encode() in body
        for value in (model, language, style):
            assert value.encode() in body
        assert b"private-filename" not in body
        return httpx.Response(
            200, json={"success": True, "transcript": "Hello", "detected_language_code": "en-IN"}
        )

    result = provider(handler, stt_format=style).transcribe(str(path), model=model, language=language)
    assert result["success"] and result["transcript"] == "Hello"
    assert result["language"] == "en-IN" and result["model"] == model


@pytest.mark.parametrize("status", [401, 403, 429, 500, 503])
def test_http_errors_are_safe(tmp_path, monkeypatch, status):
    monkeypatch.setattr(client, "sleep", lambda _: None)
    calls = []

    def handler(request):
        calls.append(request.read())
        return httpx.Response(status, text="offline-test-key and private transcript")

    path = write_wav(tmp_path / "clip.wav", 1)
    result = provider(handler).transcribe(str(path))
    assert not result["success"]
    assert "offline-test-key" not in result["error"]
    assert "private transcript" not in result["error"]
    assert len(calls) == (3 if status == 429 else 1)
    assert all(b"RIFF" in body for body in calls)


def no_request(_):
    pytest.fail("Unexpected network request")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"model": "unknown"},
        {"model": "prisma-v2.5"},
        {"language": "bogus"},
        {"language": "hi-IN,hi-IN"},
        {"language": "as-IN"},
    ],
)
def test_bad_configuration(tmp_path, kwargs):
    path = write_wav(tmp_path / "clip.wav", 1)
    assert not provider(no_request).transcribe(str(path), **kwargs)["success"]


def test_missing_and_empty_files(tmp_path):
    stt = provider(no_request)
    path = tmp_path / "clip.wav"
    assert not stt.transcribe(str(path))["success"]
    assert not stt.transcribe(str(write_wav(path, 0)))["success"]


@pytest.mark.parametrize(
    ("payload", "accepted"),
    [
        ({"success": True, "transcript": ""}, True),
        ({"success": True, "transcript": "partial", "truncated": True}, False),
        ({"success": True, "transcript": []}, False),
    ],
)
def test_transcript_validation(tmp_path, payload, accepted):
    path = write_wav(tmp_path / "clip.wav", 1)
    result = provider(lambda _: httpx.Response(200, json=payload)).transcribe(str(path))
    assert result["success"] is accepted


def test_ambiguous_timeout_not_replayed(tmp_path):
    calls = []

    def handler(request):
        calls.append(request)
        raise httpx.ReadTimeout("private")

    result = provider(handler).transcribe(str(write_wav(tmp_path / "clip.wav", 1)))
    assert "timed out" in result["error"] and len(calls) == 1


@pytest.mark.parametrize("large", [False, True])
def test_batch_lifecycle(tmp_path, monkeypatch, large):
    if large:
        monkeypatch.setattr(batch, "DIRECT_LIMIT", 1)
    path = write_wav(tmp_path / "clip.wav", 31)
    calls = []

    def handler(request):
        route = (request.method, request.url.path)
        calls.append(route)
        if request.url.host != client.HOST:
            assert "x-api-key-id" not in request.headers
            if route[0] == "PUT":
                assert request.read() == path.read_bytes()
                assert int(request.headers["content-length"]) == path.stat().st_size
                return httpx.Response(200)
            return httpx.Response(200, json={"full_transcript": "A long transcript"})
        assert request.headers["x-api-key-id"] == "offline-test-key"
        if route == ("POST", "/stt/v3/batch/v2/jobs"):
            assert json.loads(request.read())["config"]["language_code"] == "auto"
            return httpx.Response(201, json={"job_id": JOB, "uploads": [{"url": SIGNED}]})
        if route == ("POST", batch.JOBS):
            assert b'name="files"' in request.read()
            assert b"application/json" in request.read()
            return httpx.Response(201, json={"job_id": JOB})
        if route == ("POST", f"{batch.JOBS}/{JOB}/start"):
            return httpx.Response(202, json={})
        if route == ("GET", f"{batch.JOBS}/{JOB}"):
            return httpx.Response(200, json={"status": "COMPLETED"})
        if route == ("GET", f"{batch.JOBS}/{JOB}/files"):
            return httpx.Response(
                200, json={"data": [{"status": "COMPLETED", "transcript_url": SIGNED}]}
            )
        pytest.fail(f"Unexpected route {route}")

    result = provider(handler).transcribe(str(path))
    assert result["transcript"] == "A long transcript"
    assert result["job_id"] == JOB
    assert len(calls) == (6 if large else 5)


def test_pending_and_cancelled():
    for status, message in [("QUEUED", "pending"), ("CANCELLED", "CANCELLED")]:
        api = Client(httpx.MockTransport(lambda _: httpx.Response(200, json={"status": status})))
        with pytest.raises(GnaniError, match=message):
            batch.wait(api, JOB, monotonic() + 2)


@pytest.mark.parametrize(
    "url",
    [
        "http://gnaniasraudios.s3.amazonaws.com/x?token=a",
        "https://example.com/x?token=a",
        "https://gnaniasraudios.s3.amazonaws.com.example.com/x?token=a",
        "https://gnaniasraudios.s3.amazonaws.com@evil.com/x?token=a",
    ],
)
def test_untrusted_signed_url(url):
    with pytest.raises(GnaniError, match="storage hostname"):
        Client(httpx.MockTransport(no_request)).json("GET", url, signed=True, deadline=monotonic() + 5)


def test_no_redirect_followed():
    calls = []

    def handler(request):
        calls.append(request.url.host)
        return httpx.Response(307, headers={"location": "https://example.com/credential"})

    with pytest.raises(GnaniError, match="HTTP 307"):
        Client(httpx.MockTransport(handler)).json("GET", "/stt/v3", deadline=monotonic() + 5)
    assert calls == [client.HOST]


def test_vendor_duration_rejection_selects_batch(tmp_path, monkeypatch):
    path = write_wav(tmp_path / "limit.wav", 30)

    def handler(request):
        assert request.url.path == "/stt/v3"
        return httpx.Response(400, json={"error": {"type": "MAX_AUDIO_DURATION_EXCEEDED"}})

    monkeypatch.setattr(batch, "transcribe", lambda *_: ({"full_transcript": "Batch result"}, JOB))
    result = provider(handler).transcribe(str(path))
    assert result["transcript"] == "Batch result"


def test_batch_settings_fail_before_upload(tmp_path):
    path = write_wav(tmp_path / "long.wav", 31)
    result = provider(no_request, stt_format="transliterate").transcribe(str(path))
    assert "Batch supports neither" in result["error"]


def test_compressed_ogg_duration(tmp_path):
    import shutil
    import subprocess

    if not (shutil.which("ffmpeg") and shutil.which("ffprobe")):
        pytest.skip("FFmpeg and ffprobe are required for the compressed audio test.")
    path = tmp_path / "note.ogg"
    source = write_wav(tmp_path / "note.wav", 0.5)
    subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(source), "-c:a", "libopus", str(path)],
        capture_output=True,
        check=True,
        timeout=15,
    )

    def handler(request):
        assert b'filename="audio.ogg"' in request.read()
        return httpx.Response(200, json={"success": True, "transcript": "Voice note"})

    assert provider(handler).transcribe(str(path))["transcript"] == "Voice note"


def test_presigned_route_unavailable_is_actionable(tmp_path, monkeypatch):
    monkeypatch.setattr(batch, "DIRECT_LIMIT", 1)
    path = write_wav(tmp_path / "long.wav", 31)

    def handler(request):
        assert request.url.path == "/stt/v3/batch/v2/jobs"
        return httpx.Response(404)

    result = provider(handler).transcribe(str(path))
    assert not result["success"]
    assert "HTTP 404" in result["error"] and "under 10 MB" in result["error"]
