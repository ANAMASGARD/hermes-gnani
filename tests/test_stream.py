import asyncio
import json
import queue

from gnani.stream import StreamSession


class Wire:
    def __init__(self):
        self.sent = []
        self.events = queue.Queue()

    async def send(self, item):
        self.sent.append(item)
        if isinstance(item, str):
            for index, text in [(2, "world"), (1, "hello"), (2, "world")]:
                self.events.put(
                    {
                        "type": "transcript",
                        "text": text,
                        "segment_id": str(index),
                        "segment_index": index,
                        "detected_language_code": "en-IN",
                    }
                )

    async def recv(self):
        while self.events.empty():
            await asyncio.sleep(0.001)
        return json.dumps(self.events.get_nowait())


def make_session(monkeypatch, wire):
    class Session(StreamSession):
        async def _serve(self):
            return await self._exchange(wire)

    return Session({"model": "prisma-v3.0"}, "offline-test-key")


def test_frames_commit_order_and_worker_cleanup(monkeypatch):
    wire = Wire()
    session = make_session(monkeypatch, wire)
    session.push_audio(b"\1" * 600)
    session.push_audio(b"\2" * 500)
    result = session.finalize()
    assert result["success"] and result["transcript"] == "hello world"
    assert result["language"] == "en-IN"
    assert session.finalize() == result
    assert len(wire.sent[0]) == len(wire.sent[1]) == 1024
    assert wire.sent[0] == b"\1" * 600 + b"\2" * 424
    assert wire.sent[1] == b"\2" * 76 + b"\0" * 948
    assert json.loads(wire.sent[2]) == {"type": "commit"}
    assert not session._thread.is_alive()


def test_cancel_closes_worker(monkeypatch):
    session = make_session(monkeypatch, Wire())
    session.push_audio(b"00" * 512)
    session.cancel()
    assert "cancelled" in session.finalize()["error"]
    assert not session._thread.is_alive()


def test_empty_audio_avoids_connection(monkeypatch):
    session = make_session(monkeypatch, Wire())
    assert session.finalize()["transcript"] == ""
    assert session._thread is None


def test_disconnect_during_capture(monkeypatch):
    class Disconnected(Wire):
        async def recv(self):
            raise OSError("unexpected disconnect with private content")

    session = make_session(monkeypatch, Disconnected())
    session.push_audio(b"00" * 512)
    result = session.finalize()
    assert not result["success"] and "disconnected" in result["error"]
    assert "private" not in result["error"]
    assert not session._thread.is_alive()


def test_finalization_reports_unresolved_speech(monkeypatch):
    from gnani import stream

    monkeypatch.setattr(stream, "_DRAIN_SECONDS", 0.05)

    class NoTranscript(Wire):
        async def send(self, item):
            if isinstance(item, str):
                self.events.put({"type": "processing"})

    session = make_session(monkeypatch, NoTranscript())
    session.push_audio(b"00" * 512)
    result = session.finalize()
    assert not result["success"] and "finish transcribing" in result["error"]
    assert not session._thread.is_alive()
