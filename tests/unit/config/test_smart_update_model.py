"""Tests for smart_update_model and related helpers in core.config.model_config."""

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from core.config.model_config import (
    _FAMILY_CREDENTIAL_MAP,
    _model_family,
    infer_mode_s_auth,
    smart_update_model,
)
from core.config.schemas import AnimaDefaults, AnimaWorksConfig, CredentialConfig

# ── Helpers ──────────────────────────────────────────────────


def _write_status(anima_dir: Path, data: dict) -> Path:
    """Write a minimal status.json and return its path."""
    status_path = anima_dir / "status.json"
    status_path.write_text(json.dumps(data, ensure_ascii=False) + "\n", encoding="utf-8")
    return status_path


def _make_config(
    credentials: dict[str, CredentialConfig] | None = None,
    anima_defaults: AnimaDefaults | None = None,
) -> AnimaWorksConfig:
    """Build a minimal AnimaWorksConfig for tests."""
    return AnimaWorksConfig(
        credentials=credentials or {"anthropic": CredentialConfig(type="claude_code_login")},
        anima_defaults=anima_defaults or AnimaDefaults(credential="anthropic"),
    )


# ── _model_family ────────────────────────────────────────────


class TestModelFamily:
    def test_claude_models(self):
        assert _model_family("claude-sonnet-4-6") == "claude"
        assert _model_family("claude-opus-4-6") == "claude"
        assert _model_family("claude") == "claude"

    def test_slash_prefix(self):
        assert _model_family("openai/gpt-4.1") == "openai"
        assert _model_family("ollama/qwen3:14b") == "ollama"
        assert _model_family("google/gemini-2.5-pro") == "google"
        assert _model_family("vertex_ai/claude-sonnet-4-6") == "vertex_ai"
        assert _model_family("codex/codex-mini-latest") == "codex"

    def test_unknown_model(self):
        assert _model_family("some-custom-model") == "some-custom-model"

    def test_empty_string(self):
        assert _model_family("") == ""


# ── infer_mode_s_auth ────────────────────────────────────────


class TestInferModeSAuth:
    def test_non_s_mode_returns_none(self):
        config = _make_config()
        assert infer_mode_s_auth(mode="A", credential_name="anthropic", config=config) is None
        assert infer_mode_s_auth(mode="B", credential_name="anthropic", config=config) is None

    def test_claude_code_login_returns_max(self):
        config = _make_config(
            credentials={"anthropic": CredentialConfig(type="claude_code_login")},
        )
        assert infer_mode_s_auth(mode="S", credential_name="anthropic", config=config) == "max"

    def test_api_key_returns_api(self):
        config = _make_config(
            credentials={"anthropic": CredentialConfig(type="api_key", api_key="sk-test")},
        )
        assert infer_mode_s_auth(mode="S", credential_name="anthropic", config=config) == "api"

    def test_unknown_credential_falls_back_to_defaults(self):
        config = _make_config(
            credentials={"custom": CredentialConfig(type="custom_type")},
            anima_defaults=AnimaDefaults(mode_s_auth="max"),
        )
        assert infer_mode_s_auth(mode="S", credential_name="custom", config=config) == "max"

    def test_missing_credential_falls_back_to_defaults(self):
        config = _make_config(
            credentials={},
            anima_defaults=AnimaDefaults(mode_s_auth="api"),
        )
        assert infer_mode_s_auth(mode="S", credential_name="nonexistent", config=config) == "api"


# ── smart_update_model ───────────────────────────────────────


class TestSmartUpdateModel:
    """Tests for smart_update_model with mocked resolve_execution_mode."""

    @pytest.fixture()
    def anima_dir(self, tmp_path: Path) -> Path:
        d = tmp_path / "animas" / "test_anima"
        d.mkdir(parents=True)
        return d

    @pytest.fixture()
    def config_anthropic(self) -> AnimaWorksConfig:
        return _make_config(
            credentials={
                "anthropic": CredentialConfig(type="claude_code_login"),
                "ollama": CredentialConfig(type="api_key", base_url="http://localhost:11434"),
                "openai": CredentialConfig(type="api_key", api_key="sk-test"),
            },
            anima_defaults=AnimaDefaults(credential="anthropic", mode_s_auth="max"),
        )

    def _mock_resolve(self, model_name: str) -> str:
        """Simple mock for resolve_execution_mode."""
        if model_name.startswith("claude-"):
            return "S"
        if model_name.startswith("ollama/"):
            return "B"
        if model_name.startswith("openai/"):
            return "A"
        return "B"

    # -- Family change: ollama → claude --

    @patch("core.config.model_mode.resolve_execution_mode")
    def test_family_change_ollama_to_claude(self, mock_resolve, anima_dir, config_anthropic):
        mock_resolve.side_effect = lambda cfg, m: self._mock_resolve(m)
        _write_status(
            anima_dir,
            {
                "model": "ollama/qwen3:14b",
                "credential": "vllm-local",
                "thinking": False,
                "max_tokens": 4096,
                "enabled": True,
            },
        )

        result = smart_update_model(anima_dir, model="claude-sonnet-4-6", config=config_anthropic)

        assert result["model"] == "claude-sonnet-4-6"
        assert result["credential"] == "anthropic"
        assert result["execution_mode"] == "S"
        assert result["mode_s_auth"] == "max"
        assert result["family_changed"] is True
        assert "thinking" in result["cleared_fields"]
        assert "max_tokens" in result["cleared_fields"]

        data = json.loads((anima_dir / "status.json").read_text())
        assert data["model"] == "claude-sonnet-4-6"
        assert data["credential"] == "anthropic"
        assert data["execution_mode"] == "S"
        assert "thinking" not in data
        assert "max_tokens" not in data

    # -- Same family: claude → claude --

    @patch("core.config.model_mode.resolve_execution_mode")
    def test_same_family_claude_to_claude(self, mock_resolve, anima_dir, config_anthropic):
        mock_resolve.side_effect = lambda cfg, m: self._mock_resolve(m)
        _write_status(
            anima_dir,
            {
                "model": "claude-sonnet-4-6",
                "credential": "anthropic",
                "thinking": True,
                "max_tokens": 16384,
                "enabled": True,
            },
        )

        result = smart_update_model(anima_dir, model="claude-opus-4-6", config=config_anthropic)

        assert result["family_changed"] is False
        assert result["credential"] == "anthropic"
        assert result["cleared_fields"] == []

        data = json.loads((anima_dir / "status.json").read_text())
        assert data["thinking"] is True
        assert data["max_tokens"] == 16384

    # -- Explicit credential overrides auto-inference --

    @patch("core.config.model_mode.resolve_execution_mode")
    def test_explicit_credential_skips_auto(self, mock_resolve, anima_dir, config_anthropic):
        mock_resolve.side_effect = lambda cfg, m: self._mock_resolve(m)
        _write_status(
            anima_dir,
            {
                "model": "ollama/qwen3:14b",
                "credential": "vllm-local",
                "enabled": True,
            },
        )

        result = smart_update_model(
            anima_dir,
            model="claude-sonnet-4-6",
            credential="custom-cred",
            config=config_anthropic,
        )

        assert result["credential"] == "custom-cred"
        assert result["cleared_fields"] == []

    # -- Family change: claude → openai --

    @patch("core.config.model_mode.resolve_execution_mode")
    def test_family_change_claude_to_openai(self, mock_resolve, anima_dir, config_anthropic):
        mock_resolve.side_effect = lambda cfg, m: self._mock_resolve(m)
        _write_status(
            anima_dir,
            {
                "model": "claude-sonnet-4-6",
                "credential": "anthropic",
                "mode_s_auth": "max",
                "thinking": True,
                "max_tokens": 16384,
                "enabled": True,
            },
        )

        result = smart_update_model(anima_dir, model="openai/gpt-4.1", config=config_anthropic)

        assert result["model"] == "openai/gpt-4.1"
        assert result["credential"] == "openai"
        assert result["execution_mode"] == "A"
        assert result["mode_s_auth"] is None
        assert result["family_changed"] is True

        data = json.loads((anima_dir / "status.json").read_text())
        assert "mode_s_auth" not in data
        assert "thinking" not in data
        assert "max_tokens" not in data

    # -- Credential fallback to anima_defaults --

    @patch("core.config.model_mode.resolve_execution_mode")
    def test_credential_fallback_to_defaults(self, mock_resolve, anima_dir):
        mock_resolve.side_effect = lambda cfg, m: "B"
        config = _make_config(
            credentials={"anthropic": CredentialConfig(type="claude_code_login")},
            anima_defaults=AnimaDefaults(credential="anthropic"),
        )
        _write_status(
            anima_dir,
            {
                "model": "claude-sonnet-4-6",
                "credential": "anthropic",
                "enabled": True,
            },
        )

        result = smart_update_model(anima_dir, model="xai/grok-3", config=config)

        assert result["family_changed"] is True
        assert result["credential"] == "anthropic"

    # -- Credential: no matching credential, keep current --

    @patch("core.config.model_mode.resolve_execution_mode")
    def test_credential_keep_current_when_no_match(self, mock_resolve, anima_dir):
        mock_resolve.side_effect = lambda cfg, m: "B"
        config = _make_config(
            credentials={},
            anima_defaults=AnimaDefaults(credential=""),
        )
        _write_status(
            anima_dir,
            {
                "model": "claude-sonnet-4-6",
                "credential": "my-custom",
                "enabled": True,
            },
        )

        result = smart_update_model(anima_dir, model="xai/grok-3", config=config)

        assert result["credential"] == "my-custom"

    # -- mode_s_auth cleared for non-S modes --

    @patch("core.config.model_mode.resolve_execution_mode")
    def test_mode_s_auth_cleared_for_non_s(self, mock_resolve, anima_dir, config_anthropic):
        mock_resolve.side_effect = lambda cfg, m: self._mock_resolve(m)
        _write_status(
            anima_dir,
            {
                "model": "claude-sonnet-4-6",
                "credential": "anthropic",
                "mode_s_auth": "max",
                "enabled": True,
            },
        )

        result = smart_update_model(anima_dir, model="ollama/qwen3:14b", config=config_anthropic)

        assert result["mode_s_auth"] is None
        data = json.loads((anima_dir / "status.json").read_text())
        assert "mode_s_auth" not in data

    # -- Atomic write: status.json is valid after update --

    @patch("core.config.model_mode.resolve_execution_mode")
    def test_atomic_write_produces_valid_json(self, mock_resolve, anima_dir, config_anthropic):
        mock_resolve.side_effect = lambda cfg, m: "S"
        _write_status(anima_dir, {"model": "claude-sonnet-4-6", "credential": "anthropic"})

        smart_update_model(anima_dir, model="claude-opus-4-6", config=config_anthropic)

        data = json.loads((anima_dir / "status.json").read_text())
        assert isinstance(data, dict)
        assert data["model"] == "claude-opus-4-6"

    # -- FileNotFoundError when status.json missing --

    def test_missing_status_json_raises(self, anima_dir, config_anthropic):
        with pytest.raises(FileNotFoundError):
            smart_update_model(anima_dir, model="claude-sonnet-4-6", config=config_anthropic)


# ── _FAMILY_CREDENTIAL_MAP ──────────────────────────────────


class TestFamilyCredentialMap:
    def test_known_families(self):
        assert _FAMILY_CREDENTIAL_MAP["claude"] == "anthropic"
        assert _FAMILY_CREDENTIAL_MAP["openai"] == "openai"
        assert _FAMILY_CREDENTIAL_MAP["ollama"] == "ollama"
        assert _FAMILY_CREDENTIAL_MAP["bedrock"] == "anthropic"

    def test_unknown_family_not_in_map(self):
        assert "xai" not in _FAMILY_CREDENTIAL_MAP
