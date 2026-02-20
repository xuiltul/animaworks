"""Unit tests for core/memory/config_reader.py — ConfigReader."""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from core.memory.config_reader import ConfigReader
from core.schemas import ModelConfig


# ── Fixtures ──────────────────────────────────────────────


@pytest.fixture
def anima_dir(tmp_path: Path) -> Path:
    d = tmp_path / "anima"
    d.mkdir()
    return d


@pytest.fixture
def reader(anima_dir: Path) -> ConfigReader:
    return ConfigReader(anima_dir)


# ── _read_model_config_from_md ────────────────────────────


class TestReadModelConfigFromMd:
    """Tests for the legacy config.md parser."""

    def test_returns_defaults_on_missing_file(self, reader: ConfigReader) -> None:
        """When config.md does not exist, returns default ModelConfig."""
        mc = reader._read_model_config_from_md()

        assert isinstance(mc, ModelConfig)
        assert mc.model == "claude-sonnet-4-20250514"
        assert mc.max_tokens == 4096
        assert mc.max_turns == 20

    def test_returns_defaults_on_empty_file(
        self, reader: ConfigReader, anima_dir: Path,
    ) -> None:
        """When config.md is empty, returns default ModelConfig."""
        (anima_dir / "config.md").write_text("", encoding="utf-8")

        mc = reader._read_model_config_from_md()

        assert isinstance(mc, ModelConfig)
        assert mc.model == "claude-sonnet-4-20250514"

    def test_parses_all_fields(
        self, reader: ConfigReader, anima_dir: Path,
    ) -> None:
        """Parses model, max_tokens, max_turns, api_key_env, api_base_url."""
        (anima_dir / "config.md").write_text(
            "# Config\n"
            "- model: gpt-4o\n"
            "- max_tokens: 8192\n"
            "- max_turns: 10\n"
            "- api_key_env: OPENAI_API_KEY\n"
            "- api_base_url: http://localhost:8000\n",
            encoding="utf-8",
        )

        mc = reader._read_model_config_from_md()

        assert mc.model == "gpt-4o"
        assert mc.max_tokens == 8192
        assert mc.max_turns == 10
        assert mc.api_key_env == "OPENAI_API_KEY"
        assert mc.api_base_url == "http://localhost:8000"

    def test_parses_partial_fields(
        self, reader: ConfigReader, anima_dir: Path,
    ) -> None:
        """Missing fields fall back to defaults."""
        (anima_dir / "config.md").write_text(
            "- model: custom-model\n",
            encoding="utf-8",
        )

        mc = reader._read_model_config_from_md()

        assert mc.model == "custom-model"
        assert mc.max_tokens == 4096  # default
        assert mc.max_turns == 20  # default

    def test_ignores_biko_section(
        self, reader: ConfigReader, anima_dir: Path,
    ) -> None:
        """Lines after '## 備考' are excluded to avoid matching example values."""
        (anima_dir / "config.md").write_text(
            "- model: real\n"
            "\n"
            "## 備考\n"
            "- model: fake-example\n",
            encoding="utf-8",
        )

        mc = reader._read_model_config_from_md()

        assert mc.model == "real"

    def test_ignores_settings_example_section(
        self, reader: ConfigReader, anima_dir: Path,
    ) -> None:
        """Lines after '### 設定例' are excluded."""
        (anima_dir / "config.md").write_text(
            "- model: actual-model\n"
            "- max_tokens: 2048\n"
            "\n"
            "### 設定例\n"
            "- model: example-model\n"
            "- max_tokens: 9999\n",
            encoding="utf-8",
        )

        mc = reader._read_model_config_from_md()

        assert mc.model == "actual-model"
        assert mc.max_tokens == 2048

    def test_fallback_model_defaults_to_none(
        self, reader: ConfigReader, anima_dir: Path,
    ) -> None:
        """When fallback_model is not specified, it remains None."""
        (anima_dir / "config.md").write_text(
            "- model: some-model\n",
            encoding="utf-8",
        )

        mc = reader._read_model_config_from_md()

        assert mc.fallback_model is None

    def test_fallback_model_parsed(
        self, reader: ConfigReader, anima_dir: Path,
    ) -> None:
        """Explicit fallback_model is parsed correctly."""
        (anima_dir / "config.md").write_text(
            "- model: primary\n"
            "- fallback_model: secondary\n",
            encoding="utf-8",
        )

        mc = reader._read_model_config_from_md()

        assert mc.fallback_model == "secondary"


# ── resolve_api_key ───────────────────────────────────────


class TestResolveApiKey:
    """Tests for API key resolution (direct value vs env var fallback)."""

    def test_uses_config_api_key_when_available(
        self, reader: ConfigReader,
    ) -> None:
        """When config.api_key is set, it is returned directly."""
        config = ModelConfig(api_key="sk-direct-key", api_key_env="SHOULD_NOT_USE")

        result = reader.resolve_api_key(config)

        assert result == "sk-direct-key"

    def test_falls_back_to_env_var(
        self, reader: ConfigReader, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """When config.api_key is None, falls back to env var."""
        config = ModelConfig(api_key=None, api_key_env="TEST_RESOLVE_KEY")
        monkeypatch.setenv("TEST_RESOLVE_KEY", "sk-from-env")

        result = reader.resolve_api_key(config)

        assert result == "sk-from-env"

    def test_returns_none_when_no_key(
        self, reader: ConfigReader, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """When neither config.api_key nor env var is set, returns None."""
        config = ModelConfig(api_key=None, api_key_env="NONEXISTENT_KEY_XYZ_123")
        monkeypatch.delenv("NONEXISTENT_KEY_XYZ_123", raising=False)

        result = reader.resolve_api_key(config)

        assert result is None

    def test_empty_string_api_key_is_falsy(
        self, reader: ConfigReader, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """An empty string api_key is falsy, so env var fallback is used."""
        config = ModelConfig(api_key="", api_key_env="TEST_FALLBACK_KEY")
        monkeypatch.setenv("TEST_FALLBACK_KEY", "sk-fallback")

        result = reader.resolve_api_key(config)

        assert result == "sk-fallback"

    def test_reads_config_when_none_passed(self, reader: ConfigReader) -> None:
        """When config argument is None, read_model_config() is called."""
        mock_config = ModelConfig(api_key="sk-auto-read")
        with patch.object(reader, "read_model_config", return_value=mock_config) as m:
            result = reader.resolve_api_key()

        m.assert_called_once()
        assert result == "sk-auto-read"


# ── read_model_config ─────────────────────────────────────


class TestReadModelConfig:
    """Tests for the unified config.json loader with config.md fallback."""

    def test_loads_from_config_json(self, reader: ConfigReader) -> None:
        """When config.json exists, model config is resolved from it."""
        mock_config = MagicMock()
        mock_resolved = MagicMock()
        mock_resolved.model = "gpt-4o"
        mock_resolved.fallback_model = None
        mock_resolved.max_tokens = 4096
        mock_resolved.max_turns = 20
        mock_resolved.credential = "openai"
        mock_resolved.context_threshold = 0.50
        mock_resolved.max_chains = 2
        mock_resolved.conversation_history_threshold = 0.30
        mock_resolved.execution_mode = None
        mock_resolved.supervisor = None
        mock_resolved.speciality = None
        mock_resolved.thinking = None
        mock_resolved.llm_timeout = None

        mock_credential = MagicMock()
        mock_credential.api_key = "sk-test"
        mock_credential.base_url = None

        config_path = MagicMock()
        config_path.exists.return_value = True

        with patch("core.config.get_config_path", return_value=config_path), \
             patch("core.config.load_config", return_value=mock_config), \
             patch("core.config.resolve_anima_config", return_value=(mock_resolved, mock_credential)), \
             patch("core.config.resolve_execution_mode", return_value="A2"):
            mc = reader.read_model_config()

        assert isinstance(mc, ModelConfig)
        assert mc.model == "gpt-4o"
        assert mc.api_key == "sk-test"
        assert mc.api_key_env == "OPENAI_API_KEY"
        assert mc.resolved_mode == "A2"

    def test_falls_back_to_config_md(
        self, reader: ConfigReader, anima_dir: Path,
    ) -> None:
        """When config.json does not exist, falls back to config.md parser."""
        (anima_dir / "config.md").write_text(
            "- model: legacy-model\n"
            "- max_tokens: 2048\n",
            encoding="utf-8",
        )

        config_path = MagicMock()
        config_path.exists.return_value = False

        with patch("core.config.get_config_path", return_value=config_path):
            mc = reader.read_model_config()

        assert mc.model == "legacy-model"
        assert mc.max_tokens == 2048

    def test_uses_anima_dir_name_as_anima_name(self, tmp_path: Path) -> None:
        """The anima name is derived from the directory name."""
        anima_dir = tmp_path / "my-anima"
        anima_dir.mkdir()
        reader = ConfigReader(anima_dir)

        mock_config = MagicMock()
        mock_resolved = MagicMock()
        mock_resolved.model = "claude-sonnet-4-20250514"
        mock_resolved.fallback_model = None
        mock_resolved.max_tokens = 4096
        mock_resolved.max_turns = 20
        mock_resolved.credential = "anthropic"
        mock_resolved.context_threshold = 0.50
        mock_resolved.max_chains = 2
        mock_resolved.conversation_history_threshold = 0.30
        mock_resolved.execution_mode = None
        mock_resolved.supervisor = None
        mock_resolved.speciality = None
        mock_resolved.thinking = None
        mock_resolved.llm_timeout = None

        mock_credential = MagicMock()
        mock_credential.api_key = ""
        mock_credential.base_url = None

        config_path = MagicMock()
        config_path.exists.return_value = True

        with patch("core.config.get_config_path", return_value=config_path), \
             patch("core.config.load_config", return_value=mock_config), \
             patch("core.config.resolve_anima_config", return_value=(mock_resolved, mock_credential)) as mock_resolve, \
             patch("core.config.resolve_execution_mode", return_value="A1"):
            reader.read_model_config()

        # Verify anima_name was passed correctly
        call_args = mock_resolve.call_args
        assert call_args[0][1] == "my-anima"
