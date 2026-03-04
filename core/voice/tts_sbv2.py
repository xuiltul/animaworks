# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Style-BERT-VITS2 / AivisSpeech TTS provider."""

from __future__ import annotations

import logging
from typing import Any, AsyncIterator

import httpx

from core.voice.tts_base import BaseTTSProvider, TTSConfig, TTSSynthesisError

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "http://localhost:5000"
HTTP_TIMEOUT = 60.0


# ── StyleBertVits2TTS ───────────────────────────────────────────


class StyleBertVits2TTS(BaseTTSProvider):
    """Style-BERT-VITS2 / AivisSpeech FastAPI server TTS provider."""

    def __init__(self, voice_config: Any) -> None:
        """Initialize with voice config.

        Args:
            voice_config: Config with style_bert_vits2.base_url.
        """
        sbv2 = getattr(voice_config, "style_bert_vits2", None) or {}
        base = sbv2.get("base_url") if isinstance(sbv2, dict) else getattr(sbv2, "base_url", None)
        self._base_url = (base or DEFAULT_BASE_URL).rstrip("/")

    async def synthesize(
        self, text: str, config: TTSConfig
    ) -> AsyncIterator[bytes]:
        """Stream TTS audio chunks. SBV2 returns full WAV; yield as single chunk."""
        audio = await self.synthesize_full(text, config)
        yield audio

    async def synthesize_full(self, text: str, config: TTSConfig) -> bytes:
        """Generate complete WAV audio for given text."""
        model_id, speaker_id, style = self._parse_voice_id(config)
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
            try:
                r = await client.get(
                    f"{self._base_url}/voice",
                    params={
                        "text": text,
                        "model_id": model_id,
                        "speaker_id": speaker_id,
                        "style": style or "",
                    },
                )
                r.raise_for_status()
                if not r.content:
                    raise TTSSynthesisError("Style-BERT-VITS2: empty audio response")
                return r.content
            except httpx.HTTPError as e:
                logger.warning("Style-BERT-VITS2 synthesis failed: %s", e)
                raise TTSSynthesisError(f"Style-BERT-VITS2 synthesis failed: {e}") from e

    async def list_voices(self) -> list[dict]:
        """List available models/speakers from SBV2 API."""
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
            try:
                r = await client.get(f"{self._base_url}/models/info")
                r.raise_for_status()
                data = r.json()
                result: list[dict] = []
                if isinstance(data, list):
                    for i, m in enumerate(data):
                        name = m.get("name", m.get("id", str(i)))
                        result.append({"id": f"{i}:0", "name": str(name)})
                elif isinstance(data, dict):
                    models = data.get("models", data.get("models_info", []))
                    for i, m in enumerate(models):
                        name = m.get("name", m.get("id", str(i))) if isinstance(m, dict) else str(i)
                        result.append({"id": f"{i}:0", "name": str(name)})
                return result if result else [{"id": "0:0", "name": "default"}]
            except httpx.HTTPError:
                return [{"id": "0:0", "name": "default"}]

    async def health_check(self) -> bool:
        """Check if SBV2 API is available."""
        async with httpx.AsyncClient(timeout=5.0) as client:
            try:
                r = await client.get(f"{self._base_url}/models/info")
                return r.status_code == 200
            except Exception:
                return False

    def _parse_voice_id(self, config: TTSConfig) -> tuple[int, int, str]:
        """Parse voice_id as 'model_id:speaker_id' or 'model_id:speaker_id:style'."""
        vid = (config.voice_id or "").strip()
        if ":" in vid:
            parts = vid.split(":", 2)
            try:
                model_id = int(parts[0])
                speaker_id = int(parts[1]) if len(parts) > 1 else 0
                style = parts[2] if len(parts) > 2 else ""
                return model_id, speaker_id, style
            except (ValueError, IndexError):
                pass
        logger.warning(
            "Style-BERT-VITS2: invalid voice_id=%r, using model_id=0 speaker_id=0",
            config.voice_id,
        )
        return 0, 0, ""
