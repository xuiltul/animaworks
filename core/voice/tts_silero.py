# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Silero TTS provider for AnimaWorks."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from typing import Any

import httpx

from core.voice.tts_base import BaseTTSProvider, TTSConfig, TTSSynthesisError

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "http://localhost:8000"
HTTP_TIMEOUT = 60.0


class SileroTTS(BaseTTSProvider):
    """Silero TTS API provider.

    Provides local text-to-speech synthesis with support for Russian
    (and other) languages.
    """

    def __init__(self, voice_config: Any) -> None:
        """Initialize the Silero TTS provider.

        Args:
            voice_config: Configuration object containing the base_url.
        """
        # Пытаемся достать base_url из конфига, иначе используем дефолтный
        base = None
        if isinstance(voice_config, dict):
            silero = voice_config.get("silero", {})
            base = silero.get("base_url") if isinstance(silero, dict) else None
        else:
            silero = getattr(voice_config, "silero", None)
            base = getattr(silero, "base_url", None) if silero else None

        self._base_url = (base or DEFAULT_BASE_URL).rstrip("/")

    async def synthesize(self, text: str, config: TTSConfig) -> AsyncIterator[bytes]:
        """Stream TTS audio chunks.

        Args:
            text: The text to synthesize.
            config: The TTS configuration (voice, speed, etc).

        Yields:
            Chunks of WAV audio bytes.
        """
        audio = await self.synthesize_full(text, config)
        yield audio

    async def synthesize_full(self, text: str, config: TTSConfig) -> bytes:
        """Generate complete WAV audio for the given text.

        Args:
            text: The text to synthesize.
            config: The TTS configuration.

        Returns:
            Complete WAV audio bytes.

        Raises:
            TTSSynthesisError: If the synthesis fails.
        """
        voice_id = (config.voice_id or "xenia").strip()

        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT, trust_env=False) as client:
            try:
                r = await client.post(
                    f"{self._base_url}/process",
                    data={
                        "VOICE": voice_id,
                        "INPUT_TEXT": text,
                    },
                )
                r.raise_for_status()
                if not r.content:
                    raise TTSSynthesisError("Silero TTS: empty audio response")
                return r.content
            except httpx.HTTPError as e:
                logger.warning("Silero TTS synthesis failed: %s", e)
                raise TTSSynthesisError(f"Silero TTS synthesis failed: {e}") from e

    async def list_voices(self) -> list[dict]:
        """List available voices from the Silero API.

        Returns:
            A list of dictionaries containing voice IDs and names.
        """
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT, trust_env=False) as client:
            try:
                r = await client.get(f"{self._base_url}/voices")
                r.raise_for_status()

                try:
                    data = r.json()
                    voices = data if isinstance(data, list) else data.get("voices", [])
                except ValueError:
                    voices = [line.strip() for line in r.text.split("\n") if line.strip()]

                result = []
                for v in voices:
                    if isinstance(v, str):
                        result.append({"id": v, "name": v})
                    elif isinstance(v, dict):
                        vid = v.get("id", v.get("name", str(v)))
                        result.append({"id": vid, "name": vid})

                return result if result else [{"id": "xenia", "name": "xenia"}]
            except Exception:
                return [{"id": "xenia", "name": "xenia"}]

    async def health_check(self) -> bool:
        """Check if the Silero API is available.

        Returns:
            True if the API responds, False otherwise.
        """
        async with httpx.AsyncClient(timeout=10.0, trust_env=False) as client:
            try:
                r = await client.get(f"{self._base_url}/voices")
                return r.status_code == 200
            except Exception:
                return False
