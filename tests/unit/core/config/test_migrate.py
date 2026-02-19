"""Unit tests for core/config/migrate.py — legacy config.md migration."""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from core.config.migrate import (
    _env_name_to_credential_name,
    _parse_config_md,
    migrate_to_config_json,
)
from core.config.models import invalidate_cache


# ── _parse_config_md ──────────────────────────────────────


class TestParseConfigMd:
    def test_basic_parsing(self, tmp_path):
        md = tmp_path / "config.md"
        md.write_text(
            "# Config\n\n- model: gpt-4o\n- max_tokens: 8192\n- api_key_env: OPENAI_API_KEY\n",
            encoding="utf-8",
        )
        result = _parse_config_md(md)
        assert result["model"] == "gpt-4o"
        assert result["max_tokens"] == "8192"
        assert result["api_key_env"] == "OPENAI_API_KEY"

    def test_ignores_biko_section(self, tmp_path):
        md = tmp_path / "config.md"
        md.write_text(
            "- model: claude-sonnet-4\n\n## 備考\n- model: this_should_be_ignored\n",
            encoding="utf-8",
        )
        result = _parse_config_md(md)
        assert result["model"] == "claude-sonnet-4"

    def test_ignores_example_section(self, tmp_path):
        md = tmp_path / "config.md"
        md.write_text(
            "- model: real-model\n\n### 設定例\n- model: example-model\n",
            encoding="utf-8",
        )
        result = _parse_config_md(md)
        assert result["model"] == "real-model"

    def test_empty_file(self, tmp_path):
        md = tmp_path / "config.md"
        md.write_text("", encoding="utf-8")
        result = _parse_config_md(md)
        assert result == {}


# ── _env_name_to_credential_name ──────────────────────────


class TestEnvNameToCredentialName:
    def test_anthropic(self):
        assert _env_name_to_credential_name("ANTHROPIC_API_KEY") == "anthropic"

    def test_anthropic_sakura(self):
        assert _env_name_to_credential_name("ANTHROPIC_API_KEY_SAKURA") == "anthropic_sakura"

    def test_ollama(self):
        assert _env_name_to_credential_name("OLLAMA_API_KEY") == "ollama"

    def test_openai(self):
        assert _env_name_to_credential_name("OPENAI_API_KEY") == "openai"

    def test_bare_api_key(self):
        # "API_KEY" -> lowercase "api_key" -> remove "_api_key" suffix -> empty -> "default"
        # Actually: "api_key" -> re.sub(r"_api_key$", "", "api_key") -> "" -> "default"?
        # Let's check: "api_key".lower() = "api_key"
        # re.sub(r"_api_key$", "", "api_key") = "api_key" (doesn't match because no leading _)
        # Wait: the regex is _api_key$ which matches the _ before api_key
        # "api_key" does NOT start with _, so r"_api_key$" won't match
        # The result would be "api_key" not "default"
        result = _env_name_to_credential_name("API_KEY")
        assert result == "api_key"


# ── migrate_to_config_json ────────────────────────────────


class TestMigrateToConfigJson:
    @pytest.fixture(autouse=True)
    def _clear(self):
        invalidate_cache()
        yield
        invalidate_cache()

    def test_migrate_no_animas(self, tmp_path):
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        # No animas dir
        with patch("core.config.models.get_config_path", return_value=data_dir / "config.json"):
            migrate_to_config_json(data_dir)
        config_path = data_dir / "config.json"
        assert config_path.exists()
        data = json.loads(config_path.read_text(encoding="utf-8"))
        assert data["animas"] == {}

    def test_migrate_with_anima(self, tmp_path):
        data_dir = tmp_path / "data"
        animas_dir = data_dir / "animas"
        alice_dir = animas_dir / "alice"
        alice_dir.mkdir(parents=True)
        (alice_dir / "config.md").write_text(
            "- model: gpt-4o\n- max_tokens: 2048\n- api_key_env: OPENAI_API_KEY\n",
            encoding="utf-8",
        )

        with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test123"}, clear=False), \
             patch("core.config.models.get_config_path", return_value=data_dir / "config.json"):
            invalidate_cache()
            migrate_to_config_json(data_dir)

        config_path = data_dir / "config.json"
        data = json.loads(config_path.read_text(encoding="utf-8"))
        assert "alice" in data["animas"]
        assert data["animas"]["alice"]["model"] == "gpt-4o"
        assert data["animas"]["alice"]["max_tokens"] == 2048
        assert "openai" in data["credentials"]
        assert data["credentials"]["openai"]["api_key"] == "sk-test123"

    def test_ensures_anthropic_credential(self, tmp_path):
        data_dir = tmp_path / "data"
        animas_dir = data_dir / "animas"
        bob_dir = animas_dir / "bob"
        bob_dir.mkdir(parents=True)
        (bob_dir / "config.md").write_text(
            "- model: custom\n- api_key_env: CUSTOM_API_KEY\n",
            encoding="utf-8",
        )

        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "sk-anth"}, clear=False), \
             patch("core.config.models.get_config_path", return_value=data_dir / "config.json"):
            invalidate_cache()
            migrate_to_config_json(data_dir)

        data = json.loads((data_dir / "config.json").read_text(encoding="utf-8"))
        assert "anthropic" in data["credentials"]

    def test_skips_non_directory(self, tmp_path):
        data_dir = tmp_path / "data"
        animas_dir = data_dir / "animas"
        animas_dir.mkdir(parents=True)
        (animas_dir / "not_a_dir.txt").write_text("file", encoding="utf-8")

        with patch("core.config.models.get_config_path", return_value=data_dir / "config.json"):
            invalidate_cache()
            migrate_to_config_json(data_dir)

        data = json.loads((data_dir / "config.json").read_text(encoding="utf-8"))
        assert data["animas"] == {}

    def test_skips_anima_without_config_md(self, tmp_path):
        data_dir = tmp_path / "data"
        animas_dir = data_dir / "animas"
        (animas_dir / "alice").mkdir(parents=True)
        # No config.md

        with patch("core.config.models.get_config_path", return_value=data_dir / "config.json"):
            invalidate_cache()
            migrate_to_config_json(data_dir)

        data = json.loads((data_dir / "config.json").read_text(encoding="utf-8"))
        assert data["animas"] == {}
