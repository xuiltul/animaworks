# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""ElevenLabs TTS provider — REST API streaming."""

from __future__ import annotations

import logging
import os
from typing import Any, AsyncIterator

import httpx

from core.voice.tts_base import BaseTTSProvider, TTSConfig, TTSSynthesisError

logger = logging.getLogger(__name__)

BASE_URL = "https://api.elevenlabs.io/v1"
DEFAULT_API_KEY_ENV = "ELEVENLABS_API_KEY"
DEFAULT_MODEL_ID = "eleven_flash_v2_5"
HTTP_TIMEOUT = 60.0


# ── ElevenLabsTTS ──────────────────────────────────────────────


class ElevenLabsTTS(BaseTTSProvider):
    """ElevenLabs REST API TTS provider."""

    def __init__(self, voice_config: Any) -> None:
        """Initialize with voice config.

        Args:
            voice_config: Config with elevenlabs.api_key_env, elevenlabs.model_id.
        """
        el = getattr(voice_config, "elevenlabs", None) or {}
        self._api_key_env = (
            el.get("api_key_env") if isinstance(el, dict) else getattr(el, "api_key_env", None)
        ) or DEFAULT_API_KEY_ENV
        self._default_model_id = (
            el.get("model_id") if isinstance(el, dict) else getattr(el, "model_id", None)
        ) or DEFAULT_MODEL_ID

    def _get_api_key(self) -> str | None:
        return os.environ.get(self._api_key_env)

    async def synthesize(
        self, text: str, config: TTSConfig
    ) -> AsyncIterator[bytes]:
        """Stream TTS audio chunks via ElevenLabs streaming endpoint.

        Raises:
            TTSSynthesisError: On HTTP errors or missing API key.
        """
        api_key = self._get_api_key()
        if not api_key:
            raise TTSSynthesisError(
                f"ElevenLabs: API key not configured (env: {self._api_key_env})"
            )
        voice_id = await self._resolve_voice_id(config)
        model_id = config.extra.get("model_id") or self._default_model_id
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
            try:
                async with client.stream(
                    "POST",
                    f"{BASE_URL}/text-to-speech/{voice_id}/stream",
                    headers={
                        "xi-api-key": api_key,
                        "Content-Type": "application/json",
                    },
                    json={"text": text, "model_id": model_id},
                ) as resp:
                    resp.raise_for_status()
                    async for chunk in resp.aiter_bytes():
                        if chunk:
                            yield chunk
            except httpx.HTTPError as e:
                logger.warning("ElevenLabs synthesis failed: %s", e)
                raise TTSSynthesisError(f"ElevenLabs synthesis failed: {e}") from e

    async def synthesize_full(self, text: str, config: TTSConfig) -> bytes:
        """Generate complete audio (non-streaming).

        Raises:
            TTSSynthesisError: On synthesis failure.
        """
        chunks: list[bytes] = []
        async for chunk in self.synthesize(text, config):
            chunks.append(chunk)
        if not chunks:
            raise TTSSynthesisError("ElevenLabs: empty response from TTS")
        return b"".join(chunks)

    async def list_voices(self) -> list[dict]:
        """List available ElevenLabs voices."""
        api_key = self._get_api_key()
        if not api_key:
            return []
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
            try:
                r = await client.get(
                    f"{BASE_URL}/voices",
                    headers={"xi-api-key": api_key},
                )
                r.raise_for_status()
                data = r.json()
                return [
                    {"id": v.get("voice_id", ""), "name": v.get("name", "")}
                    for v in data.get("voices", [])
                ]
            except httpx.HTTPError:
                return []

    async def health_check(self) -> bool:
        """Check if ElevenLabs API is available."""
        if not self._get_api_key():
            return False
        voices = await self.list_voices()
        return len(voices) > 0

    def supported_formats(self) -> list[str]:
        return ["mp3"]

    async def _resolve_voice_id(self, config: TTSConfig) -> str:
        """Resolve voice_id; fallback to first voice from list if empty."""
        vid = (config.voice_id or "").strip()
        if vid:
            return vid
        voices = await self.list_voices()
        if voices:
            fallback = voices[0].get("id", "")
            logger.warning(
                "ElevenLabs: empty voice_id, falling back to first voice %r",
                fallback,
            )
            return fallback
        logger.warning("ElevenLabs: no voices available, using empty id")
        return ""
