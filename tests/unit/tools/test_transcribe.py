"""Tests for core/tools/transcribe.py — Whisper speech-to-text tool."""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from core.tools.transcribe import (
    _load_prompt,
    _prompt_cache,
    get_tool_schemas,
    refine_with_llm,
    transcribe,
    process_audio,
)


# ── _load_prompt ──────────────────────────────────────────────────


class TestLoadPrompt:
    @pytest.fixture(autouse=True)
    def _clear_cache(self):
        _prompt_cache.clear()
        yield
        _prompt_cache.clear()

    def test_fallback_prompt(self, tmp_path: Path):
        """When no prompt file exists, returns a minimal fallback."""
        with patch("core.tools.transcribe._PROMPTS_DIR", tmp_path / "nonexistent"):
            result = _load_prompt("xx")
        assert "system_prompt" in result
        assert "user_template" in result

    def test_loads_from_file(self, tmp_path: Path):
        prompts_dir = tmp_path / "prompts"
        prompts_dir.mkdir()
        prompt_data = {
            "system_prompt": "Custom system prompt",
            "user_template": "Text: {text}",
        }
        (prompts_dir / "en.json").write_text(json.dumps(prompt_data), encoding="utf-8")

        with patch("core.tools.transcribe._PROMPTS_DIR", prompts_dir):
            result = _load_prompt("en")
        assert result["system_prompt"] == "Custom system prompt"

    def test_caches_result(self, tmp_path: Path):
        with patch("core.tools.transcribe._PROMPTS_DIR", tmp_path / "none"):
            result1 = _load_prompt("zz")
            result2 = _load_prompt("zz")
        assert result1 is result2

    def test_language_normalization(self, tmp_path: Path):
        """Language code 'en-US' should match 'en.json'."""
        prompts_dir = tmp_path / "prompts"
        prompts_dir.mkdir()
        (prompts_dir / "en.json").write_text(json.dumps({
            "system_prompt": "English prompt",
            "user_template": "{text}",
        }), encoding="utf-8")

        with patch("core.tools.transcribe._PROMPTS_DIR", prompts_dir):
            result = _load_prompt("en-US")
        assert "English" in result["system_prompt"]


# ── transcribe ────────────────────────────────────────────────────


class TestTranscribe:
    def test_requires_faster_whisper(self):
        with patch("core.tools.transcribe.WhisperModel", None):
            with pytest.raises(ImportError, match="faster-whisper"):
                transcribe("/fake/audio.wav")

    def test_transcribe_success(self):
        mock_segment = MagicMock()
        mock_segment.text = " Hello world "
        mock_segment.start = 0.0
        mock_segment.end = 2.0

        mock_info = MagicMock()
        mock_info.language = "en"
        mock_info.language_probability = 0.95
        mock_info.duration = 5.0

        mock_model = MagicMock()
        mock_model.transcribe.return_value = ([mock_segment], mock_info)

        with patch("core.tools.transcribe._get_whisper_model", return_value=mock_model):
            with patch("core.tools.transcribe.WhisperModel", MagicMock()):
                result = transcribe("/fake/audio.wav")

        assert result["raw_text"] == "Hello world"
        assert result["language"] == "en"
        assert result["duration"] == 5.0
        assert len(result["segments"]) == 1
        assert result["segments"][0]["text"] == "Hello world"

    def test_transcribe_empty_segments(self):
        mock_info = MagicMock()
        mock_info.language = "ja"
        mock_info.language_probability = 0.8
        mock_info.duration = 1.0

        mock_model = MagicMock()
        mock_model.transcribe.return_value = ([], mock_info)

        with patch("core.tools.transcribe._get_whisper_model", return_value=mock_model):
            with patch("core.tools.transcribe.WhisperModel", MagicMock()):
                result = transcribe("/fake/silence.wav")
        assert result["raw_text"] == ""
        assert result["segments"] == []


# ── refine_with_llm ───────────────────────────────────────────────


class TestRefineWithLlm:
    @pytest.fixture(autouse=True)
    def _clear_cache(self):
        _prompt_cache.clear()
        yield
        _prompt_cache.clear()

    def test_refine_success(self):
        mock_client = MagicMock()
        mock_client.chat.return_value = "Refined text output"

        with patch("core.tools.local_llm.OllamaClient", return_value=mock_client):
            result = refine_with_llm("raw text input", model="test-model")
        assert result["refined_text"] == "Refined text output"
        assert result["model"] == "test-model"

    def test_refine_with_custom_prompt(self):
        mock_client = MagicMock()
        mock_client.chat.return_value = "Custom refined text"

        with patch("core.tools.local_llm.OllamaClient", return_value=mock_client):
            result = refine_with_llm(
                "raw",
                custom_prompt="Also fix names",
            )
        assert result["refined_text"] == "Custom refined text"

    def test_refine_with_context_hint(self):
        mock_client = MagicMock()
        mock_client.chat.return_value = "Contextual refined text"

        with patch("core.tools.local_llm.OllamaClient", return_value=mock_client):
            result = refine_with_llm(
                "raw text",
                context_hint="This is a tech meeting",
            )
        assert result["refined_text"] == "Contextual refined text"

    def test_refine_too_short_fallback(self):
        """If refined text is drastically shorter, fall back to raw."""
        mock_client = MagicMock()
        mock_client.chat.return_value = "X"  # Much shorter than raw

        raw = "This is a long raw text that should not be replaced by something too short"
        with patch("core.tools.local_llm.OllamaClient", return_value=mock_client):
            result = refine_with_llm(raw)
        assert result["refined_text"] == raw  # falls back


# ── process_audio ─────────────────────────────────────────────────


class TestProcessAudio:
    def test_process_raw_only(self):
        mock_transcribe_result = {
            "raw_text": "Hello from whisper",
            "language": "en",
            "language_probability": 0.9,
            "duration": 3.0,
            "load_time": 0.1,
            "transcribe_time": 0.5,
            "speed": 6.0,
            "segments": [],
        }

        with patch("core.tools.transcribe.transcribe", return_value=mock_transcribe_result):
            result = process_audio("/fake.wav", raw_only=True, quiet=True)
        assert result["raw_text"] == "Hello from whisper"
        assert result["refined_text"] is None

    def test_process_with_refinement(self):
        mock_transcribe_result = {
            "raw_text": "some raw text",
            "language": "ja",
            "language_probability": 0.9,
            "duration": 5.0,
            "load_time": 0.1,
            "transcribe_time": 1.0,
            "speed": 5.0,
            "segments": [],
        }
        mock_refine_result = {
            "refined_text": "polished text",
            "model": "test-model",
            "refine_time": 0.3,
        }

        with patch("core.tools.transcribe.transcribe", return_value=mock_transcribe_result):
            with patch("core.tools.transcribe.refine_with_llm", return_value=mock_refine_result):
                result = process_audio("/fake.wav", quiet=True)
        assert result["refined_text"] == "polished text"
        assert result["refine_model"] == "test-model"

    def test_process_skip_refine_on_empty(self):
        mock_transcribe_result = {
            "raw_text": "",
            "language": "en",
            "language_probability": 0.5,
            "duration": 0.5,
            "load_time": 0.1,
            "transcribe_time": 0.1,
            "speed": 5.0,
            "segments": [],
        }

        with patch("core.tools.transcribe.transcribe", return_value=mock_transcribe_result):
            result = process_audio("/fake.wav", quiet=True)
        assert result["refined_text"] is None


# ── get_tool_schemas ──────────────────────────────────────────────


class TestGetToolSchemas:
    def test_returns_schemas(self):
        schemas = get_tool_schemas()
        assert isinstance(schemas, list)
        assert len(schemas) == 1
        assert schemas[0]["name"] == "transcribe_audio"

    def test_requires_audio_path(self):
        schema = get_tool_schemas()[0]
        assert "audio_path" in schema["input_schema"]["required"]
