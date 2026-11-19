"""Native Hermes speech providers for Gnani AI."""

from typing import Any

from agent.transcription_provider import TranscriptionProvider
from hermes_cli.config import load_config_readonly

from .client import Client
from .stt import GnaniTranscription
from .tts import GnaniTTS

__all__ = ["register"]


def register(ctx: Any) -> None:
    client = Client()
    stt: TranscriptionProvider = GnaniTranscription(ctx.get_config, load_config_readonly, client)
    ctx.register_transcription_provider(stt)
    ctx.register_tts_provider(GnaniTTS(ctx.get_config, load_config_readonly, client))
