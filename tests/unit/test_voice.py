# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for voice chat (STT, TTS, session, sentence splitter, config)."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import numpy as np
import pytest

from core.config.models import AnimaWorksConfig, VoiceConfig
from core.voice.sentence_splitter import StreamingSentenceSplitter, split_sentences
from core.voice.tts_base import TTSConfig
from core.voice.tts_elevenlabs import ElevenLabsTTS
from core.voice.tts_factory import create_tts_provider
from core.voice.tts_irodori import IrodoriTTS
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

    def test_emotion_tag_not_split_and_sanitized(self) -> None:
        # Regression: "!" in "<!--" must not split the comment, or
        # sanitize_for_tts can't strip it and TTS reads it aloud.
        from core.voice.session import sanitize_for_tts

        tag = ' <!-- emotion: {"emotion": "smile"} -->'
        s = StreamingSentenceSplitter()
        sentences = s.feed("こんにちは！元気です。" + tag)
        remaining = s.flush()
        if remaining:
            sentences.append(remaining)
        spoken = [t for t in (sanitize_for_tts(x) for x in sentences) if t]
        assert spoken == ["こんにちは！", "元気です。"]

    def test_unterminated_comment_sanitized(self) -> None:
        from core.voice.session import sanitize_for_tts

        assert sanitize_for_tts('本文です。<!-- emotion: {"emo') == "本文です。"


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

    def test_irodori(self) -> None:
        provider = create_tts_provider("irodori", VoiceConfig())
        assert isinstance(provider, IrodoriTTS)

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

    def test_keep_emoji_filters_to_allowlist(self) -> None:
        from core.voice.session import sanitize_for_tts

        # 😊 is an Irodori annotation emoji, 😃 is not.
        assert sanitize_for_tts("😊😃こんにちは", keep_emoji=True) == "😊こんにちは"
        # ZWJ sequence in the allowlist survives intact.
        assert sanitize_for_tts("😮‍💨ふう", keep_emoji=True) == "😮‍💨ふう"

    def test_keep_emoji_false_strips_all(self) -> None:
        from core.voice.session import sanitize_for_tts

        assert sanitize_for_tts("😊😃こんにちは") == "こんにちは"

    def test_year_kana_conversion(self) -> None:
        from core.voice.session import apply_reading_rules, sanitize_for_tts

        assert apply_reading_rules("2003年です") == "にせんさんねんです"
        assert apply_reading_rules("二〇二六年") == "にせんにじゅうろくねん"
        # sanitize keeps the display copy readable — no kana conversion.
        assert sanitize_for_tts("2003年です", keep_emoji=True) == "2003年です"

    def test_ruby_reading_for_alphabet_terms(self) -> None:
        from core.voice.session import apply_reading_rules, resolve_ruby, strip_ruby

        text = "GitHub（ギットハブ）のPR(ピーアール)とClaude Code（クロード コード）を見た"
        assert resolve_ruby(text) == "ギットハブのピーアールとクロード コードを見た"
        assert strip_ruby(text) == "GitHubのPRとClaude Codeを見た"
        # Ruby feeds the TTS copy through apply_reading_rules too.
        assert apply_reading_rules("API（エーピーアイ）です") == "エーピーアイです"
        # Japanese parentheticals are not ruby.
        assert resolve_ruby("東京（とうきょう）") == "東京（とうきょう）"
        assert strip_ruby("GitHub（設定）") == "GitHub（設定）"

    def test_yomi_dict_substitution(self, tmp_path, monkeypatch) -> None:
        import core.voice.session as vs

        (tmp_path / "voice_yomi.tsv").write_text(
            "# comment\n小鳥遊\tたかなし\nRAG\tラグ\n", encoding="utf-8"
        )
        monkeypatch.setenv("ANIMAWORKS_DATA_DIR", str(tmp_path))
        monkeypatch.setattr(vs, "_yomi_cache", None)
        monkeypatch.setattr(vs, "_yomi_mtime", 0.0)
        assert vs.apply_reading_rules("小鳥遊さんとRAGの話") == "たかなしさんとラグの話"
        # Display copy stays untouched.
        assert vs.sanitize_for_tts("小鳥遊さんとRAGの話", keep_emoji=True) == "小鳥遊さんとRAGの話"

    def test_yomi_dict_missing_is_noop(self, tmp_path, monkeypatch) -> None:
        import core.voice.session as vs

        monkeypatch.setenv("ANIMAWORKS_DATA_DIR", str(tmp_path))
        monkeypatch.setattr(vs, "_yomi_cache", None)
        monkeypatch.setattr(vs, "_yomi_mtime", 0.0)
        assert vs.apply_reading_rules("そのまま") == "そのまま"

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

    def test_strip_emoji(self) -> None:
        from core.voice.session import sanitize_for_tts

        assert sanitize_for_tts("こんにちは😊") == "こんにちは"
        assert sanitize_for_tts("嬉しい🎉です👍") == "嬉しいです"
        assert sanitize_for_tts("普通のテキスト") == "普通のテキスト"

    def test_strip_emoji_and_emotion_comment(self) -> None:
        from core.voice.session import sanitize_for_tts

        text = 'やったね✨\n<!-- emotion: {"emotion": "laugh"} -->'
        assert sanitize_for_tts(text) == "やったね"


# ── TestVoiceModeSuffix ──────────────────────────────────────────


class TestVoiceModeSuffix:
    def test_suffix_constant_exists(self) -> None:
        from core.voice.session import VOICE_MODE_SUFFIX

        assert "voice-mode" in VOICE_MODE_SUFFIX
        assert "200文字以内" in VOICE_MODE_SUFFIX
        assert "Markdown" in VOICE_MODE_SUFFIX

    def test_suffix_requires_emotion_tag(self) -> None:
        from core.voice.session import VOICE_MODE_SUFFIX

        assert "emotion" in VOICE_MODE_SUFFIX
        assert "<!-- emotion:" in VOICE_MODE_SUFFIX
        assert "smile" in VOICE_MODE_SUFFIX
        assert "絵文字" in VOICE_MODE_SUFFIX

    def test_suffix_defers_long_running_requests_to_task(self) -> None:
        from core.voice.session import VOICE_MODE_SUFFIX

        assert "自分宛てにタスクを作成" in VOICE_MODE_SUFFIX
        assert "タスクに積んでやっておきますね" in VOICE_MODE_SUFFIX


class TestVoiceThinkingEffortConfig:
    def test_schema_defaults_and_override(self) -> None:
        from core.config.schemas import AnimaDefaults
        from core.schemas import ModelConfig

        assert ModelConfig().voice_thinking_effort is None
        assert AnimaDefaults(voice_thinking_effort="medium").voice_thinking_effort == "medium"

    def test_voice_effort_loaded_from_status(self, tmp_path: Path) -> None:
        from core.config.resolver import _load_status_json

        anima_dir = tmp_path / "animas" / "voice"
        anima_dir.mkdir(parents=True)
        (anima_dir / "status.json").write_text(
            json.dumps({"voice_thinking_effort": "medium"}),
            encoding="utf-8",
        )

        assert _load_status_json(anima_dir)["voice_thinking_effort"] == "medium"


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
        assert vc.irodori.base_url == "http://localhost:7861"

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
    async def test_irodori_raises_on_http_error(self) -> None:
        provider = IrodoriTTS(VoiceConfig())
        config = TTSConfig(provider="irodori", voice_id="default")

        from core.voice.tts_base import TTSSynthesisError

        with patch("core.voice.tts_irodori.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.post = AsyncMock(side_effect=httpx.ConnectError("connection refused"))
            mock_cls.return_value = mock_client

            with pytest.raises(TTSSynthesisError, match="Irodori-TTS synthesis failed"):
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


# ── TestIrodoriTTS ────────────────────────────────────────────────


class TestIrodoriTTS:
    def test_default_base_url(self) -> None:
        provider = IrodoriTTS(VoiceConfig())
        assert provider._base_url == "http://localhost:7861"

    def test_custom_base_url_from_config(self) -> None:
        vc = VoiceConfig(irodori={"base_url": "http://localhost:9999/"})
        provider = IrodoriTTS(vc)
        assert provider._base_url == "http://localhost:9999"

    def test_custom_base_url_from_dict_attr(self) -> None:
        vc = MagicMock()
        vc.irodori = {"base_url": "http://example:7861"}
        provider = IrodoriTTS(vc)
        assert provider._base_url == "http://example:7861"

    @pytest.mark.asyncio
    async def test_synthesize_posts_json_payload(self) -> None:
        provider = IrodoriTTS(VoiceConfig())
        config = TTSConfig(provider="irodori", voice_id="v1", speed=1.2)
        wav = b"RIFF....WAVEfmt "

        with patch("core.voice.tts_irodori.httpx.AsyncClient") as mock_cls:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.content = wav
            mock_resp.raise_for_status = MagicMock()

            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.post = AsyncMock(return_value=mock_resp)
            mock_cls.return_value = mock_client

            result = await provider.synthesize_full("こんにちは", config)

        assert result == wav
        mock_client.post.assert_awaited_once()
        call_args = mock_client.post.call_args
        assert call_args.args[0] == "http://localhost:7861/voice"
        assert call_args.kwargs["json"] == {
            "text": "こんにちは",
            "voice_id": "v1",
            "speed": 1.2,
        }

    @pytest.mark.asyncio
    async def test_synthesize_null_voice_id_and_speed(self) -> None:
        provider = IrodoriTTS(VoiceConfig())
        config = TTSConfig(provider="irodori", voice_id="", speed=0.0)
        wav = b"audio"

        with patch("core.voice.tts_irodori.httpx.AsyncClient") as mock_cls:
            mock_resp = MagicMock()
            mock_resp.content = wav
            mock_resp.raise_for_status = MagicMock()

            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.post = AsyncMock(return_value=mock_resp)
            mock_cls.return_value = mock_client

            await provider.synthesize_full("hi", config)

        assert mock_client.post.call_args.kwargs["json"] == {
            "text": "hi",
            "voice_id": None,
            "speed": None,
        }

    @pytest.mark.asyncio
    async def test_synthesize_empty_response_raises(self) -> None:
        from core.voice.tts_base import TTSSynthesisError

        provider = IrodoriTTS(VoiceConfig())
        config = TTSConfig(provider="irodori")

        with patch("core.voice.tts_irodori.httpx.AsyncClient") as mock_cls:
            mock_resp = MagicMock()
            mock_resp.content = b""
            mock_resp.raise_for_status = MagicMock()

            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.post = AsyncMock(return_value=mock_resp)
            mock_cls.return_value = mock_client

            with pytest.raises(TTSSynthesisError, match="empty audio"):
                await provider.synthesize_full("hi", config)

    @pytest.mark.asyncio
    async def test_health_check_ok(self) -> None:
        provider = IrodoriTTS(VoiceConfig())

        with patch("core.voice.tts_irodori.httpx.AsyncClient") as mock_cls:
            mock_resp = MagicMock()
            mock_resp.status_code = 200

            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.get = AsyncMock(return_value=mock_resp)
            mock_cls.return_value = mock_client

            assert await provider.health_check() is True
            mock_client.get.assert_awaited_once_with("http://localhost:7861/health")

    @pytest.mark.asyncio
    async def test_health_check_connection_error(self) -> None:
        provider = IrodoriTTS(VoiceConfig())

        with patch("core.voice.tts_irodori.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.get = AsyncMock(side_effect=httpx.ConnectError("refused"))
            mock_cls.return_value = mock_client

            assert await provider.health_check() is False


# ── TestConsecutiveTTSFailures ───────────────────────────────────


class TestConsecutiveTTSFailures:
    """RC-1: Consecutive failure counter + health invalidation."""

    @pytest.mark.asyncio
    async def test_success_resets_counter(self) -> None:
        from core.voice.session import VoiceSession

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
            yield  # noqa

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
            yield  # noqa

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
            yield  # noqa

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
            yield  # noqa

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

        requests = []

        async def mock_stream(*args, **kwargs):
            requests.append(kwargs)
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
        assert requests[0]["params"]["voice_mode"] is True


# ── TestTTSPrefetchPipeline ───────────────────────────────────────


def _pcm_speech(seconds: float = 1.0) -> bytes:
    """Non-silent PCM16 mono @16kHz for speech_end gate checks."""
    n = int(16_000 * seconds)
    return np.random.randint(-3000, 3000, n, dtype=np.int16).tobytes()


def _make_session(*, ws=None, stt=None, tts=None, supervisor=None):
    from core.voice.session import VoiceSession

    ws = ws or AsyncMock()
    stt = stt or MagicMock()
    tts = tts or AsyncMock()
    tts_config = TTSConfig(provider="voicevox")
    supervisor = supervisor or MagicMock()
    voice_config = MagicMock(stt_refine_enabled=False)
    return VoiceSession("test", ws, stt, tts, tts_config, supervisor, voice_config)


class TestBargeProbe:
    def test_self_echo_containment(self) -> None:
        from core.voice.session import _is_self_echo

        recent = ["今日はいい天気ですね。散歩に行きませんか"]
        assert _is_self_echo("こんにちは、今日はいい天気ですね", recent) is True
        assert _is_self_echo("ちょっと待って", recent) is False
        assert _is_self_echo("今日はいい天気ですね", []) is False

    def test_self_echo_short_text(self) -> None:
        from core.voice.session import _is_self_echo

        assert _is_self_echo("天気", ["今日はいい天気です"]) is True
        assert _is_self_echo("待て", ["今日はいい天気です"]) is False
        assert _is_self_echo("", ["今日はいい天気です"]) is False

    @pytest.mark.asyncio
    async def test_probe_accepts_audio_while_tts_playing(self) -> None:
        session = _make_session()
        session._tts_playing = True
        session._processing = True

        await session.handle_barge_probe()
        await session.handle_audio_chunk(b"\x00\x01\x02\x03")

        assert bytes(session._audio_buffer) == b"\x00\x01\x02\x03"
        await session.close()

    @pytest.mark.asyncio
    async def test_false_probe_verdict_discards_audio(self) -> None:
        ws = AsyncMock()
        session = _make_session(ws=ws)
        await session.handle_barge_probe()
        session._recent_tts_text.append("今日はいい天気ですね。散歩に行きませんか")
        session._audio_buffer.extend(b"probe audio")

        interrupted = await session._judge_probe("こんにちは、今日はいい天気ですね")

        assert interrupted is False
        ws.send_json.assert_awaited_with({"type": "barge_verdict", "interrupt": False})
        assert session._audio_buffer == bytearray()
        assert session._interrupted is False

    @pytest.mark.asyncio
    async def test_true_probe_verdict_preserves_audio(self) -> None:
        ws = AsyncMock()
        session = _make_session(ws=ws)
        await session.handle_barge_probe()
        session._recent_tts_text.append("今日はいい天気ですね。散歩に行きませんか")
        session._audio_buffer.extend(b"probe audio")

        interrupted = await session._judge_probe("ちょっと待って")

        assert interrupted is True
        ws.send_json.assert_awaited_with({"type": "barge_verdict", "interrupt": True})
        assert session._interrupted is True
        assert bytes(session._audio_buffer) == b"probe audio"

    @pytest.mark.asyncio
    async def test_short_partial_keeps_probe_open(self) -> None:
        ws = AsyncMock()
        session = _make_session(ws=ws)
        session._tts_playing = True
        session._processing = True
        session._recent_tts_text.append("今日はいい天気ですね")

        await session.handle_barge_probe()
        # "うん" alone can't be judged — the probe must stay open for more text.
        assert await session._judge_probe("うん", final=False) is False
        assert session._probe_active is True
        ws.send_json.assert_not_called()
        # Enough non-echo text on a later partial confirms the interruption.
        assert await session._judge_probe("うん、ちょっと待って", final=False) is True
        assert session._probe_active is False
        ws.send_json.assert_called_with({"type": "barge_verdict", "interrupt": True})
        await session.close()

    @pytest.mark.asyncio
    async def test_probe_timeout_resumes_tts(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import core.voice.session as voice_session

        monkeypatch.setattr(voice_session, "PROBE_TIMEOUT_SEC", 0.01)
        ws = AsyncMock()
        session = _make_session(ws=ws)

        await session.handle_barge_probe()
        await asyncio.sleep(0.03)

        ws.send_json.assert_awaited_with({"type": "barge_verdict", "interrupt": False})
        assert session._probe_active is False
        assert session._interrupted is False


class TestTTSPrefetchPipeline:
    """Sentence TTS producer/consumer queue (prefetch pipeline)."""

    @pytest.mark.asyncio
    async def test_sentences_synthesized_in_order(self) -> None:
        """Multiple sentences are synthesized and sent in enqueue order."""
        from core.supervisor.ipc import IPCResponse

        ws = AsyncMock()
        stt = MagicMock()
        stt.transcribe_buffer_async = AsyncMock(
            return_value={"raw_text": "hello there", "language": "en"}
        )
        tts = AsyncMock()
        tts.health_check = AsyncMock(return_value=True)
        synth_order: list[str] = []
        started = asyncio.Event()

        async def mock_synthesize(text, config):
            synth_order.append(text)
            if len(synth_order) == 1:
                started.set()
            await asyncio.sleep(0.02)
            yield b"\x00\x01"

        tts.synthesize = mock_synthesize
        supervisor = MagicMock()

        async def mock_stream(*args, **kwargs):
            yield IPCResponse(
                id="1",
                done=False,
                chunk=json.dumps({"type": "text_delta", "text": "第一文。"}),
                result=None,
            )
            yield IPCResponse(
                id="2",
                done=False,
                chunk=json.dumps({"type": "text_delta", "text": "第二文。"}),
                result=None,
            )
            yield IPCResponse(
                id="3",
                done=False,
                chunk=json.dumps({"type": "text_delta", "text": "第三文。"}),
                result=None,
            )
            yield IPCResponse(
                id="4",
                done=False,
                chunk=json.dumps(
                    {"type": "cycle_done", "cycle_result": {"emotion": "smile"}}
                ),
                result=None,
            )

        supervisor.send_request_stream = mock_stream
        session = _make_session(ws=ws, stt=stt, tts=tts, supervisor=supervisor)
        session._audio_buffer.extend(_pcm_speech())
        await session._do_speech_end("human")

        assert synth_order == ["第一文。", "第二文。", "第三文。"]
        types = [
            c.args[0].get("type")
            for c in ws.send_json.call_args_list
            if c.args and isinstance(c.args[0], dict)
        ]
        assert types.index("response_done") > types.index("tts_done")
        # last tts_done before response_done
        last_tts_done = max(i for i, t in enumerate(types) if t == "tts_done")
        response_done_idx = types.index("response_done")
        assert last_tts_done < response_done_idx

    @pytest.mark.asyncio
    async def test_producer_does_not_wait_for_synthesis(self) -> None:
        """IPC consumer enqueues sentences while first synthesis is still running."""
        from core.supervisor.ipc import IPCResponse

        ws = AsyncMock()
        stt = MagicMock()
        stt.transcribe_buffer_async = AsyncMock(
            return_value={"raw_text": "hello there", "language": "en"}
        )
        tts = AsyncMock()
        tts.health_check = AsyncMock(return_value=True)
        first_entered = asyncio.Event()
        release_first = asyncio.Event()
        enqueue_progress: list[str] = []

        async def mock_synthesize(text, config):
            if text == "第一文。":
                first_entered.set()
                await release_first.wait()
            yield b"\x00\x01"

        tts.synthesize = mock_synthesize
        supervisor = MagicMock()

        async def mock_stream(*args, **kwargs):
            yield IPCResponse(
                id="1",
                done=False,
                chunk=json.dumps({"type": "text_delta", "text": "第一文。"}),
                result=None,
            )
            # Wait until consumer has started first synthesis, then feed more
            # while the first synth is still held (proves producer is decoupled).
            await first_entered.wait()
            yield IPCResponse(
                id="2",
                done=False,
                chunk=json.dumps({"type": "text_delta", "text": "第二文。"}),
                result=None,
            )
            enqueue_progress.append("second_fed")
            yield IPCResponse(
                id="3",
                done=False,
                chunk=json.dumps({"type": "text_delta", "text": "第三文。"}),
                result=None,
            )
            enqueue_progress.append("third_fed")
            # Release held synth before cycle_done so drain cannot deadlock the
            # test harness; producer already advanced past synthesis.
            release_first.set()
            yield IPCResponse(
                id="4",
                done=False,
                chunk=json.dumps(
                    {"type": "cycle_done", "cycle_result": {"emotion": "neutral"}}
                ),
                result=None,
            )

        supervisor.send_request_stream = mock_stream
        session = _make_session(ws=ws, stt=stt, tts=tts, supervisor=supervisor)
        session._audio_buffer.extend(_pcm_speech())
        await session._do_speech_end("human")

        # Producer fed 2nd/3rd while 1st synth was still held → not sequential.
        assert enqueue_progress == ["second_fed", "third_fed"]

    @pytest.mark.asyncio
    async def test_interrupt_discards_queued_sentences(self) -> None:
        """Barge-in clears the queue; later sentences must not be synthesized."""
        from core.supervisor.ipc import IPCResponse

        ws = AsyncMock()
        stt = MagicMock()
        stt.transcribe_buffer_async = AsyncMock(
            return_value={"raw_text": "hello there", "language": "en"}
        )
        tts = AsyncMock()
        tts.health_check = AsyncMock(return_value=True)
        synth_order: list[str] = []
        first_started = asyncio.Event()
        hold_first = asyncio.Event()

        async def mock_synthesize(text, config):
            synth_order.append(text)
            if text == "第一文。":
                first_started.set()
                await hold_first.wait()
            yield b"\x00\x01"

        tts.synthesize = mock_synthesize
        supervisor = MagicMock()

        async def mock_stream(*args, **kwargs):
            yield IPCResponse(
                id="1",
                done=False,
                chunk=json.dumps({"type": "text_delta", "text": "第一文。"}),
                result=None,
            )
            yield IPCResponse(
                id="2",
                done=False,
                chunk=json.dumps({"type": "text_delta", "text": "第二文。"}),
                result=None,
            )
            yield IPCResponse(
                id="3",
                done=False,
                chunk=json.dumps({"type": "text_delta", "text": "第三文。"}),
                result=None,
            )
            await first_started.wait()
            # Simulate client barge-in while first sentence is still synthesizing.
            await session.handle_interrupt()
            # Unblock in-flight synth so the worker can observe _interrupted and exit.
            hold_first.set()
            # After interrupt the speech loop breaks; further yields are unused.
            yield IPCResponse(
                id="4",
                done=False,
                chunk=json.dumps(
                    {"type": "cycle_done", "cycle_result": {"emotion": "neutral"}}
                ),
                result=None,
            )

        supervisor.send_request_stream = mock_stream
        session = _make_session(ws=ws, stt=stt, tts=tts, supervisor=supervisor)
        session._audio_buffer.extend(_pcm_speech())
        await session._do_speech_end("human")

        assert "第二文。" not in synth_order
        assert "第三文。" not in synth_order
        # At most the in-flight first sentence was started.
        assert synth_order == ["第一文。"] or synth_order == []

    @pytest.mark.asyncio
    async def test_response_done_after_all_tts(self) -> None:
        """response_done is emitted only after the TTS queue is fully drained."""
        from core.supervisor.ipc import IPCResponse

        ws = AsyncMock()
        stt = MagicMock()
        stt.transcribe_buffer_async = AsyncMock(
            return_value={"raw_text": "hello there", "language": "en"}
        )
        tts = AsyncMock()
        tts.health_check = AsyncMock(return_value=True)
        timeline: list[str] = []

        async def mock_synthesize(text, config):
            timeline.append(f"synth:{text}")
            await asyncio.sleep(0.01)
            timeline.append(f"synth_done:{text}")
            yield b"\x00\x01"

        tts.synthesize = mock_synthesize

        original_send_json = AsyncMock()

        async def tracking_send_json(payload):
            if isinstance(payload, dict) and payload.get("type") == "response_done":
                timeline.append("response_done")
            await original_send_json(payload)

        ws.send_json = tracking_send_json
        supervisor = MagicMock()

        async def mock_stream(*args, **kwargs):
            yield IPCResponse(
                id="1",
                done=False,
                chunk=json.dumps({"type": "text_delta", "text": "A。"}),
                result=None,
            )
            yield IPCResponse(
                id="2",
                done=False,
                chunk=json.dumps({"type": "text_delta", "text": "B。"}),
                result=None,
            )
            yield IPCResponse(
                id="3",
                done=False,
                chunk=json.dumps(
                    {"type": "cycle_done", "cycle_result": {"emotion": "laugh"}}
                ),
                result=None,
            )

        supervisor.send_request_stream = mock_stream
        session = _make_session(ws=ws, stt=stt, tts=tts, supervisor=supervisor)
        session._audio_buffer.extend(_pcm_speech())
        await session._do_speech_end("human")

        assert "response_done" in timeline
        done_idx = timeline.index("response_done")
        assert timeline.index("synth_done:A。") < done_idx
        assert timeline.index("synth_done:B。") < done_idx

    @pytest.mark.asyncio
    async def test_greet_uses_tts_worker_queue(self) -> None:
        """greet_and_speak routes sentences through the same worker queue."""
        ws = AsyncMock()
        tts = AsyncMock()
        tts.health_check = AsyncMock(return_value=True)
        synth_order: list[str] = []

        async def mock_synthesize(text, config):
            synth_order.append(text)
            await asyncio.sleep(0.01)
            yield b"\x00\x01"

        tts.synthesize = mock_synthesize
        supervisor = MagicMock()
        supervisor.send_request = AsyncMock(
            return_value={
                "response": "こんにちは。元気ですか？",
                "emotion": "smile",
            }
        )
        session = _make_session(ws=ws, tts=tts, supervisor=supervisor)
        await session.greet_and_speak()

        assert synth_order == ["こんにちは。", "元気ですか？"]
        types = [
            c.args[0].get("type")
            for c in ws.send_json.call_args_list
            if c.args and isinstance(c.args[0], dict)
        ]
        assert "response_done" in types
        assert session._tts_worker is None
        assert session._tts_queue is None

    @pytest.mark.asyncio
    async def test_close_cancels_worker(self) -> None:
        """close() cancels the consumer task and drops the queue."""
        session = _make_session()
        await session._start_tts_worker()
        assert session._tts_worker is not None
        assert not session._tts_worker.done()
        await session.close()
        assert session._tts_worker is None
        assert session._tts_queue is None

    @pytest.mark.asyncio
    async def test_pipeline_tts_failure_counter(self) -> None:
        """Failure counter / health invalidation still work via the worker path."""
        from core.supervisor.ipc import IPCResponse
        from core.voice.tts_base import TTSSynthesisError

        ws = AsyncMock()
        stt = MagicMock()
        stt.transcribe_buffer_async = AsyncMock(
            return_value={"raw_text": "hello there", "language": "en"}
        )
        tts = AsyncMock()
        tts.health_check = AsyncMock(return_value=True)

        async def mock_synthesize_fail(text, config):
            raise TTSSynthesisError("provider down")
            yield  # noqa

        tts.synthesize = mock_synthesize_fail
        supervisor = MagicMock()

        async def mock_stream(*args, **kwargs):
            for i, sentence in enumerate(["一。", "二。", "三。"], start=1):
                yield IPCResponse(
                    id=str(i),
                    done=False,
                    chunk=json.dumps({"type": "text_delta", "text": sentence}),
                    result=None,
                )
            yield IPCResponse(
                id="done",
                done=False,
                chunk=json.dumps(
                    {"type": "cycle_done", "cycle_result": {"emotion": "neutral"}}
                ),
                result=None,
            )

        supervisor.send_request_stream = mock_stream
        session = _make_session(ws=ws, stt=stt, tts=tts, supervisor=supervisor)
        session._tts_available = True
        session._audio_buffer.extend(_pcm_speech())
        await session._do_speech_end("human")

        assert session._consecutive_tts_failures == 3
        assert session._tts_available is None
