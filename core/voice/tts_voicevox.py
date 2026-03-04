# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""VOICEVOX TTS provider — Engine HTTP API."""

from __future__ import annotations

import logging
from typing import Any, AsyncIterator

import httpx

from core.voice.tts_base import BaseTTSProvider, TTSConfig, TTSSynthesisError

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "http://localhost:50021"
HTTP_TIMEOUT = 30.0


# ── VoicevoxTTS ────────────────────────────────────────────────


class VoicevoxTTS(BaseTTSProvider):
    """VOICEVOX Engine HTTP API TTS provider."""

    def __init__(self, voice_config: Any) -> None:
        """Initialize with voice config.

        Args:
            voice_config: Config object with voicevox.base_url (default localhost:50021).
        """
        vv = getattr(voice_config, "voicevox", None) or {}
        base = vv.get("base_url") if isinstance(vv, dict) else getattr(vv, "base_url", None)
        self._base_url = (base or DEFAULT_BASE_URL).rstrip("/")

    async def synthesize(
        self, text: str, config: TTSConfig
    ) -> AsyncIterator[bytes]:
        """Stream TTS audio chunks. VOICEVOX does not support streaming; yields full WAV."""
        audio = await self.synthesize_full(text, config)
        yield audio

    async def synthesize_full(self, text: str, config: TTSConfig) -> bytes:
        """Generate complete WAV audio for given text."""
        speaker_id = self._resolve_speaker_id(config)
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
            try:
                r = await client.post(
                    f"{self._base_url}/audio_query",
                    params={"text": text, "speaker": speaker_id},
                )
                r.raise_for_status()
                audio_query = r.json()
                # Apply speed if needed
                if config.speed != 1.0:
                    audio_query.setdefault("speedScale", 1.0)
                    audio_query["speedScale"] = config.speed
                r2 = await client.post(
                    f"{self._base_url}/synthesis",
                    params={"speaker": speaker_id},
                    json=audio_query,
                )
                r2.raise_for_status()
                if not r2.content:
                    raise TTSSynthesisError("VOICEVOX: empty audio response")
                return r2.content
            except httpx.HTTPError as e:
                logger.warning("VOICEVOX synthesis failed: %s", e)
                raise TTSSynthesisError(f"VOICEVOX synthesis failed: {e}") from e

    async def list_voices(self) -> list[dict]:
        """List available VOICEVOX speakers."""
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
            try:
                r = await client.get(f"{self._base_url}/speakers")
                r.raise_for_status()
                data = r.json()
                result: list[dict] = []
                for sp in data:
                    for style in sp.get("styles", []):
                        result.append({
                            "id": str(style.get("id", "")),
                            "name": f"{sp.get('name', '')} / {style.get('name', '')}",
                        })
                return result
            except httpx.HTTPError:
                return []

    async def health_check(self) -> bool:
        """Check if VOICEVOX Engine is available."""
        async with httpx.AsyncClient(timeout=5.0) as client:
            try:
                r = await client.get(f"{self._base_url}/version")
                return r.status_code == 200
            except Exception:
                return False

    def _resolve_speaker_id(self, config: TTSConfig) -> int:
        """Resolve speaker ID; fallback to 0 if invalid."""
        vid = config.voice_id.strip()
        if vid:
            try:
                return int(vid)
            except ValueError:
                pass
        logger.warning("VOICEVOX: invalid voice_id=%r, falling back to speaker_id=0", config.voice_id)
        return 0
