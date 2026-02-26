# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Voice STT — in-memory PCM transcription via faster-whisper."""

from __future__ import annotations

import asyncio
import logging
import shutil
from typing import Any

try:
    from faster_whisper import WhisperModel
except ImportError:
    WhisperModel = None  # type: ignore[assignment, misc]

logger = logging.getLogger(__name__)

# ── Whisper singleton ──────────────────────────────────────────

_whisper_model: WhisperModel | None = None


# ── VoiceSTT ───────────────────────────────────────────────────


class VoiceSTT:
    """In-memory speech-to-text using faster-whisper core."""

    def __init__(
        self,
        model_name: str = "large-v3-turbo",
        device: str = "auto",
        compute_type: str = "default",
    ) -> None:
        """Initialize STT engine.

        Args:
            model_name: Whisper model name.
            device: Device ("auto", "cuda", "cpu").
            compute_type: Compute type ("default", "float16", "int8", etc.).
        """
        self._model_name = model_name
        self._device = device
        self._compute_type = compute_type
        self._model: WhisperModel | None = None

    def _ensure_model(self) -> WhisperModel:
        """Lazy-load WhisperModel singleton."""
        global _whisper_model
        if _whisper_model is None:
            if WhisperModel is None:
                raise ImportError(
                    "Voice STT requires 'faster-whisper'. "
                    "Install with: pip install animaworks[transcribe]"
                )
            device = self._device
            if device == "auto":
                device = "cuda" if shutil.which("nvidia-smi") else "cpu"
            compute = self._compute_type
            if compute == "default":
                compute = "float16" if device == "cuda" else "int8"
            _whisper_model = WhisperModel(
                self._model_name, device=device, compute_type=compute
            )
        return _whisper_model

    def transcribe_buffer(
        self,
        audio_data: bytes,
        sample_rate: int = 16000,
        language: str | None = None,
        vad_filter: bool = True,
    ) -> dict[str, Any]:
        """Transcribe in-memory PCM audio buffer.

        Args:
            audio_data: Raw PCM 16-bit mono audio bytes.
            sample_rate: Sample rate (default 16kHz).
            language: Language code or None for auto-detect.
            vad_filter: Apply VAD filtering.

        Returns:
            Dict with raw_text, language, duration, segments.
        """
        import numpy as np

        if WhisperModel is None:
            raise ImportError(
                "Voice STT requires 'faster-whisper'. "
                "Install with: pip install animaworks[transcribe]"
            )
        model = self._ensure_model()
        audio_np = (
            np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
        )
        segments, info = model.transcribe(
            audio_np,
            beam_size=5,
            language=language,
            vad_filter=vad_filter,
        )
        segments_list = list(segments)
        raw_text = " ".join(seg.text.strip() for seg in segments_list)
        return {
            "raw_text": raw_text,
            "language": info.language,
            "duration": info.duration,
            "segments": [
                {"start": s.start, "end": s.end, "text": s.text.strip()}
                for s in segments_list
            ],
        }

    async def transcribe_buffer_async(
        self,
        audio_data: bytes,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Async wrapper running transcribe in thread pool.

        Args:
            audio_data: Raw PCM 16-bit mono audio bytes.
            **kwargs: Passed to transcribe_buffer.

        Returns:
            Dict with raw_text, language, duration, segments.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.transcribe_buffer(audio_data, **kwargs),
        )
