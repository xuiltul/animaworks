# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""TTS abstract base — provider interface and config."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import AsyncIterator

from pydantic import BaseModel, Field


class TTSConfig(BaseModel):
    """Per-provider TTS configuration."""

    provider: str  # "voicevox", "style_bert_vits2", "elevenlabs"
    voice_id: str = ""
    speed: float = 1.0
    pitch: float = 0.0
    extra: dict = Field(default_factory=dict)


class BaseTTSProvider(ABC):
    """Abstract TTS provider interface."""

    @abstractmethod
    async def synthesize(
        self, text: str, config: TTSConfig
    ) -> AsyncIterator[bytes]:
        """Stream TTS audio chunks for given text."""

    @abstractmethod
    async def synthesize_full(self, text: str, config: TTSConfig) -> bytes:
        """Generate complete audio for given text."""

    @abstractmethod
    async def list_voices(self) -> list[dict]:
        """List available voices from this provider."""

    @abstractmethod
    async def health_check(self) -> bool:
        """Check if the TTS provider is available."""

    def supported_formats(self) -> list[str]:
        """Audio formats this provider can produce."""
        return ["wav"]
