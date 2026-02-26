# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""TTS provider factory."""

from __future__ import annotations

from typing import Any

from core.voice.tts_base import BaseTTSProvider
from core.voice.tts_elevenlabs import ElevenLabsTTS
from core.voice.tts_sbv2 import StyleBertVits2TTS
from core.voice.tts_voicevox import VoicevoxTTS

# ── Factory ────────────────────────────────────────────────────


def create_tts_provider(
    provider_name: str,
    voice_config: Any,
) -> BaseTTSProvider:
    """Create TTS provider instance by name.

    Args:
        provider_name: One of "voicevox", "style_bert_vits2", "elevenlabs".
        voice_config: Voice configuration object (e.g. config.voice).

    Returns:
        Instantiated TTS provider.

    Raises:
        ValueError: If provider_name is unknown.
    """
    providers: dict[str, type[BaseTTSProvider]] = {
        "voicevox": VoicevoxTTS,
        "style_bert_vits2": StyleBertVits2TTS,
        "elevenlabs": ElevenLabsTTS,
    }
    cls = providers.get(provider_name.lower())
    if cls is None:
        raise ValueError(f"Unknown TTS provider: {provider_name}")
    return cls(voice_config)
