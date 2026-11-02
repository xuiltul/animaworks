from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Phone speech rendering using each Anima's configured TTS voice."""

from collections import OrderedDict
from pathlib import Path
from typing import Any

from core.voice.audio_codec import wav_to_mulaw_wav
from core.voice.speech_text import prepare_speech
from core.voice.tts_factory import create_tts_provider
from core.voice.voice_config import load_per_anima_voice

_STATIC_AUDIO_CACHE: OrderedDict[tuple[str, str, str, str], bytes] = OrderedDict()
_STATIC_AUDIO_CACHE_LIMIT = 128


def _to_telephone_audio(audio: bytes) -> bytes:
    """Shrink a WAV to phone quality so Twilio fetches it quickly over a slow tunnel.

    Anything that is not PCM WAV, such as MP3 from a provider, plays as it is.
    """
    try:
        return wav_to_mulaw_wav(audio)
    except ValueError:
        return audio


async def synthesize_speech(
    anima: str,
    text: str,
    *,
    animas_dir: Path | None = None,
    voice_config: Any | None = None,
    cache_key: str | None = None,
    max_chars: int = 400,
    link_placeholder: str | None = None,
) -> bytes:
    """Convert prepared phone text to WAV with the Anima's configured TTS voice."""
    from core.config import load_config
    from core.paths import get_animas_dir

    if voice_config is None:
        voice_config = load_config().voice
    effective_animas_dir = animas_dir or get_animas_dir()
    if link_placeholder is None:
        from core.i18n import t

        link_placeholder = t("phone.link_placeholder")
    tts_config = load_per_anima_voice(effective_animas_dir, anima, voice_config)
    speech_text = prepare_speech(
        text,
        provider=tts_config.provider,
        link_placeholder=link_placeholder,
        max_chars=max_chars,
    )
    if not speech_text.spoken:
        raise ValueError("phone speech text is empty")

    key = None
    if cache_key:
        key = (
            str(effective_animas_dir),
            anima,
            cache_key,
            f"{tts_config.provider}:{tts_config.voice_id}:{max_chars}:{link_placeholder!r}",
        )
        cached = _STATIC_AUDIO_CACHE.get(key)
        if cached is not None:
            _STATIC_AUDIO_CACHE.move_to_end(key)
            return cached

    provider = create_tts_provider(tts_config.provider, voice_config)
    audio = await provider.synthesize_full(speech_text.spoken, tts_config)
    if not audio:
        raise ValueError("TTS returned empty audio")
    audio = _to_telephone_audio(bytes(audio))

    if key is not None:
        _STATIC_AUDIO_CACHE[key] = audio
        _STATIC_AUDIO_CACHE.move_to_end(key)
        while len(_STATIC_AUDIO_CACHE) > _STATIC_AUDIO_CACHE_LIMIT:
            _STATIC_AUDIO_CACHE.popitem(last=False)
    return audio


def clear_static_audio_cache() -> None:
    """Clear the process-local phrase cache (used during tests/shutdown)."""
    _STATIC_AUDIO_CACHE.clear()


__all__ = ["clear_static_audio_cache", "synthesize_speech"]
