# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for voice chat (STT, TTS, session, sentence splitter, config)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.config.models import AnimaWorksConfig, VoiceConfig
from core.voice.sentence_splitter import StreamingSentenceSplitter, split_sentences
from core.voice.tts_base import TTSConfig
from core.voice.tts_elevenlabs import ElevenLabsTTS
from core.voice.tts_factory import create_tts_provider
from core.voice.tts_sbv2 import StyleBertVits2TTS
from core.voice.tts_voicevox import VoicevoxTTS


# ── TestSplitSentences ──────────────────────────────────────────


class TestSplitSentences:
    def test_basic_japanese(self) -> None:
        result = split_sentences("こんにちは。お元気ですか？元気です！")
        assert result == ["こんにちは。", "お元気ですか？", "元気です！"]

    def test_no_punctuation(self) -> None:
        result = split_sentences("テスト文")
        assert result == ["テスト文"]

    def test_empty(self) -> None:
        result = split_sentences("")
        assert result == []

    def test_english(self) -> None:
        result = split_sentences("Hello! How are you? I'm fine.")
        assert len(result) == 3

    def test_newline_split(self) -> None:
        result = split_sentences("Line 1\nLine 2")
        assert len(result) == 2


# ── TestStreamingSentenceSplitter ────────────────────────────────


class TestStreamingSentenceSplitter:
    def test_feed_complete_sentence(self) -> None:
        s = StreamingSentenceSplitter()
        result = s.feed("こんにちは。")
        assert result == ["こんにちは。"]

    def test_feed_partial(self) -> None:
        s = StreamingSentenceSplitter()
        result = s.feed("こんに")
        assert result == []
        result = s.feed("ちは。")
        assert result == ["こんにちは。"]

    def test_feed_multiple(self) -> None:
        s = StreamingSentenceSplitter()
        result = s.feed("一つ。二つ。")
        assert len(result) == 2

    def test_flush(self) -> None:
        s = StreamingSentenceSplitter()
        s.feed("残りのテキスト")
        remaining = s.flush()
        assert remaining == "残りのテキスト"

    def test_flush_empty(self) -> None:
        s = StreamingSentenceSplitter()
        assert s.flush() is None


# ── TestTTSConfig ────────────────────────────────────────────────


class TestTTSConfig:
    def test_defaults(self) -> None:
        config = TTSConfig(provider="voicevox")
        assert config.voice_id == ""
        assert config.speed == 1.0
        assert config.pitch == 0.0

    def test_custom(self) -> None:
        config = TTSConfig(provider="elevenlabs", voice_id="abc", speed=1.5)
        assert config.provider == "elevenlabs"
        assert config.voice_id == "abc"
        assert config.speed == 1.5


# ── TestTTSFactory ───────────────────────────────────────────────


class TestTTSFactory:
    def test_voicevox(self) -> None:
        provider = create_tts_provider("voicevox", VoiceConfig())
        assert isinstance(provider, VoicevoxTTS)

    def test_elevenlabs(self) -> None:
        provider = create_tts_provider("elevenlabs", VoiceConfig())
        assert isinstance(provider, ElevenLabsTTS)

    def test_sbv2(self) -> None:
        provider = create_tts_provider("style_bert_vits2", VoiceConfig())
        assert isinstance(provider, StyleBertVits2TTS)

    def test_unknown_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown TTS provider"):
            create_tts_provider("unknown", VoiceConfig())


# ── TestVoiceSTT ──────────────────────────────────────────────────


class TestVoiceSTT:
    def test_init(self) -> None:
        from core.voice.stt import VoiceSTT

        stt = VoiceSTT(model_name="tiny", device="cpu", compute_type="int8")
        assert stt._model_name == "tiny"
        assert stt._model is None  # lazy

    @patch("core.voice.stt.WhisperModel")
    def test_transcribe_buffer(self, mock_whisper_cls: MagicMock) -> None:
        from core.voice.stt import VoiceSTT

        # Mock model
        mock_model = MagicMock()
        mock_segment = MagicMock()
        mock_segment.text = "テスト"
        mock_segment.start = 0.0
        mock_segment.end = 1.0

        mock_info = MagicMock()
        mock_info.language = "ja"
        mock_info.language_probability = 0.95
        mock_info.duration = 1.0

        mock_model.transcribe.return_value = ([mock_segment], mock_info)
        mock_whisper_cls.return_value = mock_model

        stt = VoiceSTT(model_name="tiny", device="cpu", compute_type="int8")

        import numpy as np

        pcm = np.zeros(16000, dtype=np.int16).tobytes()

        result = stt.transcribe_buffer(pcm)
        assert "raw_text" in result
        assert result["raw_text"] == "テスト"
        assert result["language"] == "ja"

    @pytest.mark.asyncio
    @patch("core.voice.stt.WhisperModel")
    async def test_transcribe_buffer_async(self, mock_whisper_cls: MagicMock) -> None:
        from core.voice import stt

        # Reset singleton so this test uses its own mock (previous test may have set it)
        stt._whisper_model = None

        from core.voice.stt import VoiceSTT

        mock_model = MagicMock()
        mock_segment = MagicMock()
        mock_segment.text = "async test"
        mock_segment.start = 0.0
        mock_segment.end = 0.5
        mock_info = MagicMock()
        mock_info.language = "en"
        mock_info.language_probability = 0.99
        mock_info.duration = 0.5
        mock_model.transcribe.return_value = ([mock_segment], mock_info)
        mock_whisper_cls.return_value = mock_model

        stt_instance = VoiceSTT(model_name="tiny", device="cpu", compute_type="int8")

        import numpy as np

        pcm = np.zeros(8000, dtype=np.int16).tobytes()
        result = await stt_instance.transcribe_buffer_async(pcm)
        assert result["raw_text"] == "async test"


# ── TestVoiceSession ────────────────────────────────────────────


class TestVoiceSession:
    @pytest.mark.asyncio
    async def test_handle_audio_chunk(self) -> None:
        from core.voice.session import VoiceSession

        ws = AsyncMock()
        stt = MagicMock()
        tts = MagicMock()
        tts_config = TTSConfig(provider="voicevox")
        supervisor = MagicMock()
        voice_config = MagicMock(stt_refine_enabled=False)

        session = VoiceSession(
            "test", ws, stt, tts, tts_config, supervisor, voice_config
        )
        await session.handle_audio_chunk(b"\x00\x01\x02\x03")
        assert len(session._audio_buffer) == 4

    @pytest.mark.asyncio
    async def test_handle_interrupt(self) -> None:
        from core.voice.session import VoiceSession

        ws = AsyncMock()
        stt = MagicMock()
        tts = MagicMock()
        tts_config = TTSConfig(provider="voicevox")
        supervisor = MagicMock()
        voice_config = MagicMock(stt_refine_enabled=False)

        session = VoiceSession(
            "test", ws, stt, tts, tts_config, supervisor, voice_config
        )
        session._audio_buffer.extend(b"\x00\x01")
        await session.handle_interrupt()
        assert session._interrupted is True
        assert len(session._audio_buffer) == 0

    @pytest.mark.asyncio
    async def test_handle_speech_end_empty_buffer(self) -> None:
        from core.voice.session import VoiceSession

        ws = AsyncMock()
        stt = MagicMock()
        tts = MagicMock()
        tts_config = TTSConfig(provider="voicevox")
        supervisor = MagicMock()
        voice_config = MagicMock(stt_refine_enabled=False)

        session = VoiceSession(
            "test", ws, stt, tts, tts_config, supervisor, voice_config
        )
        await session.handle_speech_end()
        ws.send_json.assert_not_called()

    @pytest.mark.asyncio
    async def test_tts_health_check_caching(self) -> None:
        """M1: _check_tts_health caches result and sends error on failure."""
        from core.voice.session import VoiceSession

        ws = AsyncMock()
        stt = MagicMock()
        tts = AsyncMock()
        tts.health_check = AsyncMock(return_value=False)
        tts_config = TTSConfig(provider="voicevox")
        supervisor = MagicMock()
        voice_config = MagicMock(stt_refine_enabled=False)

        session = VoiceSession(
            "test", ws, stt, tts, tts_config, supervisor, voice_config
        )
        result = await session._check_tts_health()
        assert result is False
        ws.send_json.assert_called_with(
            {"type": "error", "message": "TTS unavailable"}
        )
        # Second call uses cache, no additional health_check call
        tts.health_check.reset_mock()
        ws.send_json.reset_mock()
        result2 = await session._check_tts_health()
        assert result2 is False
        tts.health_check.assert_not_called()

    @pytest.mark.asyncio
    async def test_tts_health_check_success(self) -> None:
        """M1: _check_tts_health returns True when provider is available."""
        from core.voice.session import VoiceSession

        ws = AsyncMock()
        stt = MagicMock()
        tts = AsyncMock()
        tts.health_check = AsyncMock(return_value=True)
        tts_config = TTSConfig(provider="voicevox")
        supervisor = MagicMock()
        voice_config = MagicMock(stt_refine_enabled=False)

        session = VoiceSession(
            "test", ws, stt, tts, tts_config, supervisor, voice_config
        )
        result = await session._check_tts_health()
        assert result is True
        ws.send_json.assert_not_called()

    @pytest.mark.asyncio
    async def test_invalidate_tts_health(self) -> None:
        """M1: invalidate_tts_health resets cache."""
        from core.voice.session import VoiceSession

        ws = AsyncMock()
        stt = MagicMock()
        tts = AsyncMock()
        tts.health_check = AsyncMock(return_value=True)
        tts_config = TTSConfig(provider="voicevox")
        supervisor = MagicMock()
        voice_config = MagicMock(stt_refine_enabled=False)

        session = VoiceSession(
            "test", ws, stt, tts, tts_config, supervisor, voice_config
        )
        await session._check_tts_health()
        assert session._tts_available is True
        session.invalidate_tts_health()
        assert session._tts_available is None

    @pytest.mark.asyncio
    async def test_concurrent_speech_end_guard(self) -> None:
        """Concurrent speech_end calls are rejected while processing."""
        from core.voice.session import VoiceSession

        ws = AsyncMock()
        stt = MagicMock()
        tts = AsyncMock()
        tts_config = TTSConfig(provider="voicevox")
        supervisor = MagicMock()
        voice_config = MagicMock(stt_refine_enabled=False)

        session = VoiceSession(
            "test", ws, stt, tts, tts_config, supervisor, voice_config
        )
        session._processing = True
        session._audio_buffer.extend(b"\x00" * 100)
        await session.handle_speech_end()
        # Buffer should NOT be cleared since processing was skipped
        assert len(session._audio_buffer) == 100


# ── TestElevenLabsApiKeyCheck ────────────────────────────────────


class TestElevenLabsApiKeyCheck:
    @pytest.mark.asyncio
    async def test_synthesize_no_api_key_yields_nothing(self) -> None:
        """M2: synthesize() returns immediately when API key is not set."""
        vc = VoiceConfig()
        provider = ElevenLabsTTS(vc)
        config = TTSConfig(provider="elevenlabs", voice_id="test-id")

        chunks: list[bytes] = []
        async for chunk in provider.synthesize("hello", config):
            chunks.append(chunk)
        assert chunks == []


# ── TestVoiceConfig ──────────────────────────────────────────────


class TestVoiceConfig:
    def test_voice_config_defaults(self) -> None:
        vc = VoiceConfig()
        assert vc.stt_model == "large-v3-turbo"
        assert vc.default_tts_provider == "voicevox"
        assert vc.stt_refine_enabled is False
        assert vc.voicevox.base_url == "http://localhost:50021"
        assert vc.elevenlabs.api_key_env == "ELEVENLABS_API_KEY"
        assert vc.style_bert_vits2.base_url == "http://localhost:5000"

    def test_animaworks_config_has_voice(self) -> None:
        config = AnimaWorksConfig()
        assert hasattr(config, "voice")
        assert config.voice.stt_model == "large-v3-turbo"


# ── TestPerAnimaVoice ────────────────────────────────────────────


class TestPerAnimaVoice:
    def test_load_per_anima_voice_with_voice_section(self, tmp_path: Path) -> None:
        from server.routes.voice import _load_per_anima_voice

        animas_dir = tmp_path / "animas"
        anima_dir = animas_dir / "test_anima"
        anima_dir.mkdir(parents=True)
        (anima_dir / "status.json").write_text(
            json.dumps({
                "enabled": True,
                "model": "claude-sonnet-4-6",
                "voice": {
                    "tts_provider": "elevenlabs",
                    "voice_id": "abc123",
                    "speed": 1.2,
                },
            })
        )

        tts_config = _load_per_anima_voice(animas_dir, "test_anima", VoiceConfig())
        assert tts_config.provider == "elevenlabs"
        assert tts_config.voice_id == "abc123"
        assert tts_config.speed == 1.2

    def test_load_per_anima_voice_without_voice_section(
        self, tmp_path: Path
    ) -> None:
        from server.routes.voice import _load_per_anima_voice

        animas_dir = tmp_path / "animas"
        anima_dir = animas_dir / "test_anima"
        anima_dir.mkdir(parents=True)
        (anima_dir / "status.json").write_text(
            json.dumps({"enabled": True, "model": "claude-sonnet-4-6"})
        )

        tts_config = _load_per_anima_voice(animas_dir, "test_anima", VoiceConfig())
        assert tts_config.provider == "voicevox"
        assert tts_config.voice_id == ""

    def test_load_per_anima_voice_no_status(self, tmp_path: Path) -> None:
        from server.routes.voice import _load_per_anima_voice

        animas_dir = tmp_path / "animas"
        tts_config = _load_per_anima_voice(
            animas_dir, "nonexistent", VoiceConfig()
        )
        assert tts_config.provider == "voicevox"
