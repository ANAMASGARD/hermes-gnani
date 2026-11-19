from types import SimpleNamespace

import pytest

from gnani import register
from gnani import client


def test_register_catalogs_without_credentials(monkeypatch):
    monkeypatch.setattr(client, "get_secret", lambda _: None)
    stt, tts = [], []
    register(
        SimpleNamespace(
            get_config=lambda _, default: default,
            register_transcription_provider=stt.append,
            register_tts_provider=tts.append,
        )
    )
    assert len(stt) == len(tts) == 1
    assert stt[0].name == tts[0].name == "gnani"
    assert not stt[0].is_available() and not tts[0].is_available()
    assert [model["id"] for model in stt[0].list_models()] == ["prisma-v3.0", "prisma-v2.5"]
    assert [model["id"] for model in tts[0].list_models()] == ["timbre-v2.5", "timbre-v3.0"]
    assert len(tts[0].list_voices()) == 42
    assert tts[0].default_voice() == "Nalini"


def test_key_rotation_resolved_via_host(monkeypatch):
    keys = iter(["old", "new"])
    seen = []

    def resolve(name):
        seen.append(name)
        return next(keys)

    monkeypatch.setattr(client, "get_secret", resolve)
    assert client.api_key() == "old"
    assert client.api_key() == "new"
    assert seen == ["GNANI_API_KEY", "GNANI_API_KEY"]


@pytest.mark.parametrize("value", ["", "line\nbreak", "य"])
def test_invalid_key(monkeypatch, value):
    monkeypatch.setattr(client, "get_secret", lambda _: value)
    assert not client.available()


def test_manifest_matches_project_and_voice_catalog():
    import tomllib
    from pathlib import Path

    from ruamel.yaml import YAML

    from gnani.models import VOICES

    root = Path(__file__).resolve().parents[1]
    project = tomllib.loads((root / "pyproject.toml").read_text())["project"]
    manifest = YAML(typ="safe").load((root / "gnani/plugin.yaml").read_text())
    assert manifest["python_dependencies"] == project["dependencies"]
    assert manifest["version"] == project["version"]
    settings = manifest["config_schema"]
    assert tuple(settings["timbre_v25_voice"]["choices"]) == VOICES["timbre-v2.5"]
    assert tuple(settings["timbre_v30_voice"]["choices"]) == VOICES["timbre-v3.0"]
