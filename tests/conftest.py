import os
import wave
from pathlib import Path

import pytest

from gnani import client


@pytest.fixture(autouse=True)
def scoped_key(monkeypatch, request):
    if request.node.get_closest_marker("live"):
        return
    monkeypatch.delenv("GNANI_API_KEY", raising=False)
    monkeypatch.setattr(client, "get_secret", lambda _: "offline-test-key")


def write_wav(path: Path, seconds: float, rate: int = 16000, pcm: bytes | None = None) -> Path:
    with wave.open(str(path), "wb") as audio:
        audio.setparams((1, 2, rate, 0, "NONE", "not compressed"))
        audio.writeframes(pcm if pcm is not None else b"\x00\x00" * int(seconds * rate))
    return path


def pytest_collection_modifyitems(config, items):
    if os.environ.get("GNANI_RUN_LIVE_TESTS") == "1" and os.environ.get("GNANI_API_KEY"):
        return
    marker = pytest.mark.skip(
        reason="Set GNANI_RUN_LIVE_TESTS=1 and supply GNANI_API_KEY to run paid tests."
    )
    for item in items:
        if "live" in item.keywords:
            item.add_marker(marker)
