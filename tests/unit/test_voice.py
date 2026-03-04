# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for voice chat (STT, TTS, session, sentence splitter, config)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
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


# ── TestSanitizeForTTS ────────────────────────────────────────────


class TestSanitizeForTTS:
    def test_strip_emotion_tag(self) -> None:
        from core.voice.session import sanitize_for_tts

        text = 'こんにちは！\n<!-- emotion: {"emotion": "smile"} -->'
        assert sanitize_for_tts(text) == "こんにちは！"

    def test_strip_emotion_tag_multiline(self) -> None:
        from core.voice.session import sanitize_for_tts

        text = 'はい。\n<!-- emotion: {\n"emotion": "laugh"\n} -->'
        assert sanitize_for_tts(text) == "はい。"

    def test_strip_markdown_heading(self) -> None:
        from core.voice.session import sanitize_for_tts

        assert sanitize_for_tts("## 見出し") == "見出し"
        assert sanitize_for_tts("# H1\n## H2") == "H1\nH2"

    def test_strip_bold_italic(self) -> None:
        from core.voice.session import sanitize_for_tts

        assert sanitize_for_tts("これは**太字**です") == "これは太字です"
        assert sanitize_for_tts("これは*斜体*です") == "これは斜体です"

    def test_strip_code_block(self) -> None:
        from core.voice.session import sanitize_for_tts

        text = "説明:\n```python\nprint('hello')\n```\n以上です。"
        assert sanitize_for_tts(text) == "説明:\n\n以上です。"

    def test_strip_inline_code(self) -> None:
        from core.voice.session import sanitize_for_tts

        assert sanitize_for_tts("`command` を実行") == "command を実行"

    def test_strip_link(self) -> None:
        from core.voice.session import sanitize_for_tts

        assert sanitize_for_tts("[リンク](https://example.com)") == "リンク"

    def test_strip_list_markers(self) -> None:
        from core.voice.session import sanitize_for_tts

        text = "- 項目1\n- 項目2\n1. 番号1"
        result = sanitize_for_tts(text)
        assert "- " not in result
        assert "1. " not in result
        assert "項目1" in result
        assert "番号1" in result

    def test_strip_table_pipes(self) -> None:
        from core.voice.session import sanitize_for_tts

        text = "| 列1 | 列2 |"
        result = sanitize_for_tts(text)
        assert "|" not in result
        assert "列1" in result

    def test_strip_horizontal_rule(self) -> None:
        from core.voice.session import sanitize_for_tts

        text = "上\n---\n下"
        result = sanitize_for_tts(text)
        assert "---" not in result
        assert "上" in result
        assert "下" in result

    def test_empty_after_sanitize(self) -> None:
        from core.voice.session import sanitize_for_tts

        assert sanitize_for_tts('<!-- emotion: {"emotion": "neutral"} -->') == ""

    def test_plain_text_unchanged(self) -> None:
        from core.voice.session import sanitize_for_tts

        text = "普通のテキストです。変わりません。"
        assert sanitize_for_tts(text) == text

    def test_combined(self) -> None:
        from core.voice.session import sanitize_for_tts

        text = (
            "## 回答\n\n"
            "これは**重要**なポイントです。\n"
            "- 項目A\n"
            "- 項目B\n\n"
            '<!-- emotion: {"emotion": "smile"} -->'
        )
        result = sanitize_for_tts(text)
        assert "##" not in result
        assert "**" not in result
        assert "- " not in result
        assert "<!--" not in result
        assert "重要" in result
        assert "項目A" in result


    def test_strip_trailing_html_comment(self) -> None:
        from core.voice.session import sanitize_for_tts

        text = '了解しました。\n<!-- emothion: {"emotion": "smile"} -->'
        assert sanitize_for_tts(text) == "了解しました。"


# ── TestVoiceModeSuffix ──────────────────────────────────────────


class TestVoiceModeSuffix:
    def test_suffix_constant_exists(self) -> None:
        from core.voice.session import VOICE_MODE_SUFFIX

        assert "voice-mode" in VOICE_MODE_SUFFIX
        assert "200文字以内" in VOICE_MODE_SUFFIX
        assert "Markdown" in VOICE_MODE_SUFFIX


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
        """M1: _check_tts_health retries on failure and sends error."""
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
        # Second call retries (failure is not cached)
        tts.health_check.reset_mock()
        ws.send_json.reset_mock()
        result2 = await session._check_tts_health()
        assert result2 is False
        tts.health_check.assert_awaited_once()

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
    async def test_synthesize_no_api_key_raises(self) -> None:
        """synthesize() raises TTSSynthesisError when API key is not set."""
        from core.voice.tts_base import TTSSynthesisError

        vc = VoiceConfig()
        provider = ElevenLabsTTS(vc)
        config = TTSConfig(provider="elevenlabs", voice_id="test-id")

        with pytest.raises(TTSSynthesisError, match="API key not configured"):
            async for _ in provider.synthesize("hello", config):
                pass


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


# ── TestTTSSynthesisError ────────────────────────────────────────


class TestTTSSynthesisError:
    """RC-1: TTSSynthesisError is raised from all providers on failure."""

    def test_error_class_in_tts_base(self) -> None:
        from core.voice.tts_base import TTSSynthesisError

        err = TTSSynthesisError("test error")
        assert isinstance(err, Exception)
        assert str(err) == "test error"

    @pytest.mark.asyncio
    async def test_voicevox_raises_on_http_error(self) -> None:
        provider = VoicevoxTTS(VoiceConfig())
        config = TTSConfig(provider="voicevox", voice_id="0")

        from core.voice.tts_base import TTSSynthesisError

        with patch("core.voice.tts_voicevox.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.post = AsyncMock(side_effect=httpx.ConnectError("connection refused"))
            mock_cls.return_value = mock_client

            with pytest.raises(TTSSynthesisError, match="VOICEVOX synthesis failed"):
                await provider.synthesize_full("hello", config)

    @pytest.mark.asyncio
    async def test_sbv2_raises_on_http_error(self) -> None:
        provider = StyleBertVits2TTS(VoiceConfig())
        config = TTSConfig(provider="style_bert_vits2", voice_id="0:0")

        from core.voice.tts_base import TTSSynthesisError

        with patch("core.voice.tts_sbv2.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.get = AsyncMock(side_effect=httpx.ConnectError("connection refused"))
            mock_cls.return_value = mock_client

            with pytest.raises(TTSSynthesisError, match="Style-BERT-VITS2 synthesis failed"):
                await provider.synthesize_full("hello", config)

    @pytest.mark.asyncio
    async def test_elevenlabs_raises_on_no_api_key(self) -> None:
        from core.voice.tts_base import TTSSynthesisError

        provider = ElevenLabsTTS(VoiceConfig())
        config = TTSConfig(provider="elevenlabs", voice_id="test-id")

        with pytest.raises(TTSSynthesisError, match="API key not configured"):
            chunks = []
            async for chunk in provider.synthesize("hello", config):
                chunks.append(chunk)


# ── TestConsecutiveTTSFailures ───────────────────────────────────


class TestConsecutiveTTSFailures:
    """RC-1: Consecutive failure counter + health invalidation."""

    @pytest.mark.asyncio
    async def test_success_resets_counter(self) -> None:
        from core.voice.session import VoiceSession
        from core.voice.tts_base import TTSSynthesisError

        ws = AsyncMock()
        stt = MagicMock()
        tts = AsyncMock()
        tts_config = TTSConfig(provider="voicevox")
        supervisor = MagicMock()
        voice_config = MagicMock(stt_refine_enabled=False)

        async def mock_synthesize(text, config):
            yield b"\x00\x01\x02\x03"

        tts.synthesize = mock_synthesize

        session = VoiceSession("test", ws, stt, tts, tts_config, supervisor, voice_config)
        session._consecutive_tts_failures = 2
        await session._synthesize_and_send("hello")
        assert session._consecutive_tts_failures == 0

    @pytest.mark.asyncio
    async def test_tts_error_increments_counter(self) -> None:
        from core.voice.session import VoiceSession
        from core.voice.tts_base import TTSSynthesisError

        ws = AsyncMock()
        stt = MagicMock()
        tts = AsyncMock()
        tts_config = TTSConfig(provider="voicevox")
        supervisor = MagicMock()
        voice_config = MagicMock(stt_refine_enabled=False)

        async def mock_synthesize_fail(text, config):
            raise TTSSynthesisError("test error")
            yield  # noqa: unreachable — makes this an async generator

        tts.synthesize = mock_synthesize_fail

        session = VoiceSession("test", ws, stt, tts, tts_config, supervisor, voice_config)
        await session._synthesize_and_send("hello")
        assert session._consecutive_tts_failures == 1

        await session._synthesize_and_send("hello again")
        assert session._consecutive_tts_failures == 2

    @pytest.mark.asyncio
    async def test_three_failures_invalidates_health(self) -> None:
        from core.voice.session import VoiceSession
        from core.voice.tts_base import TTSSynthesisError

        ws = AsyncMock()
        stt = MagicMock()
        tts = AsyncMock()
        tts_config = TTSConfig(provider="voicevox")
        supervisor = MagicMock()
        voice_config = MagicMock(stt_refine_enabled=False)

        async def mock_synthesize_fail(text, config):
            raise TTSSynthesisError("provider down")
            yield  # noqa: unreachable

        tts.synthesize = mock_synthesize_fail

        session = VoiceSession("test", ws, stt, tts, tts_config, supervisor, voice_config)
        session._tts_available = True

        for _ in range(3):
            await session._synthesize_and_send("test")

        assert session._consecutive_tts_failures == 3
        assert session._tts_available is None

    @pytest.mark.asyncio
    async def test_tts_error_sends_sanitized_message(self) -> None:
        """tts_error message must not leak internal URLs."""
        from core.voice.session import VoiceSession
        from core.voice.tts_base import TTSSynthesisError

        ws = AsyncMock()
        stt = MagicMock()
        tts = AsyncMock()
        tts_config = TTSConfig(provider="voicevox")
        supervisor = MagicMock()
        voice_config = MagicMock(stt_refine_enabled=False)

        async def mock_synthesize_fail(text, config):
            raise TTSSynthesisError("VOICEVOX synthesis failed: http://localhost:50021/synthesis 500")
            yield  # noqa: unreachable

        tts.synthesize = mock_synthesize_fail

        session = VoiceSession("test", ws, stt, tts, tts_config, supervisor, voice_config)
        await session._synthesize_and_send("hello")

        tts_error_calls = [
            c for c in ws.send_json.call_args_list
            if c.args and isinstance(c.args[0], dict) and c.args[0].get("type") == "tts_error"
        ]
        assert len(tts_error_calls) == 1
        msg = tts_error_calls[0].args[0]["message"]
        assert "localhost" not in msg
        assert msg == "TTS synthesis failed"

    @pytest.mark.asyncio
    async def test_ws_error_does_not_increment_tts_counter(self) -> None:
        """WebSocket send errors should not count as TTS failures."""
        from core.voice.session import VoiceSession

        ws = AsyncMock()
        ws.send_json = AsyncMock(side_effect=ConnectionResetError("ws closed"))
        stt = MagicMock()
        tts = AsyncMock()
        tts_config = TTSConfig(provider="voicevox")
        supervisor = MagicMock()
        voice_config = MagicMock(stt_refine_enabled=False)

        async def mock_synthesize(text, config):
            yield b"\x00\x01"

        tts.synthesize = mock_synthesize

        session = VoiceSession("test", ws, stt, tts, tts_config, supervisor, voice_config)
        await session._synthesize_and_send("hello")
        assert session._consecutive_tts_failures == 0


# ── TestResponseDoneGuarantee ────────────────────────────────────


class TestResponseDoneGuarantee:
    """RC-6: response_done is always sent, even on interrupt or exception."""

    @pytest.mark.asyncio
    async def test_response_done_on_ipc_exception(self) -> None:
        from core.voice.session import VoiceSession

        ws = AsyncMock()
        stt = MagicMock()
        stt.transcribe_buffer_async = AsyncMock(return_value={
            "raw_text": "test",
            "language": "en",
        })
        tts = AsyncMock()
        tts.health_check = AsyncMock(return_value=True)
        tts_config = TTSConfig(provider="voicevox")
        supervisor = MagicMock()
        voice_config = MagicMock(stt_refine_enabled=False)

        async def mock_stream_error(*args, **kwargs):
            raise RuntimeError("IPC connection lost")
            yield  # noqa: unreachable

        supervisor.send_request_stream = mock_stream_error

        session = VoiceSession("test", ws, stt, tts, tts_config, supervisor, voice_config)
        import numpy as np
        pcm = np.random.randint(-1000, 1000, 16000, dtype=np.int16).tobytes()
        session._audio_buffer.extend(pcm)
        await session._do_speech_end("human")

        response_done_calls = [
            c for c in ws.send_json.call_args_list
            if c.args and isinstance(c.args[0], dict) and c.args[0].get("type") == "response_done"
        ]
        assert len(response_done_calls) >= 1

    @pytest.mark.asyncio
    async def test_response_done_on_interrupt(self) -> None:
        from core.supervisor.ipc import IPCResponse
        from core.voice.session import VoiceSession

        ws = AsyncMock()
        stt = MagicMock()
        stt.transcribe_buffer_async = AsyncMock(return_value={
            "raw_text": "test speech",
            "language": "en",
        })
        tts = AsyncMock()
        tts.health_check = AsyncMock(return_value=True)
        tts_config = TTSConfig(provider="voicevox")
        supervisor = MagicMock()
        voice_config = MagicMock(stt_refine_enabled=False)

        async def mock_stream(*args, **kwargs):
            yield IPCResponse(id="m1", done=False, chunk='{"type":"text_delta","text":"hello"}', result=None)
            yield IPCResponse(id="m2", done=False, chunk='{"type":"text_delta","text":" world."}', result=None)

        supervisor.send_request_stream = mock_stream

        session = VoiceSession("test", ws, stt, tts, tts_config, supervisor, voice_config)
        session._interrupted = True
        import numpy as np
        pcm = np.random.randint(-1000, 1000, 16000, dtype=np.int16).tobytes()
        session._audio_buffer.extend(pcm)
        await session._do_speech_end("human")

        response_done_calls = [
            c for c in ws.send_json.call_args_list
            if c.args and isinstance(c.args[0], dict) and c.args[0].get("type") == "response_done"
        ]
        assert len(response_done_calls) == 1
