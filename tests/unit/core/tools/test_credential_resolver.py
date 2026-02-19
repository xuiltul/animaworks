"""Unit tests for get_credential() unified resolver."""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from core.config.models import (
    AnimaWorksConfig,
    CredentialConfig,
    invalidate_cache,
    save_config,
)
from core.tools._base import ToolConfigError, get_credential


@pytest.fixture(autouse=True)
def _clean_config_cache():
    """Invalidate config cache before and after each test."""
    invalidate_cache()
    yield
    invalidate_cache()


@pytest.fixture
def config_dir(tmp_path, monkeypatch):
    """Create isolated config directory."""
    monkeypatch.setenv("ANIMAWORKS_DATA_DIR", str(tmp_path))
    return tmp_path


def _write_config(config_dir: Path, credentials: dict) -> None:
    """Write a config.json with given credentials."""
    creds = {k: CredentialConfig(**v) for k, v in credentials.items()}
    config = AnimaWorksConfig(credentials=creds)
    save_config(config, config_dir / "config.json")


class TestConfigJsonPriority:
    """config.json should be checked first."""

    def test_resolves_from_config_json(self, config_dir):
        _write_config(config_dir, {
            "chatwork": {"type": "api_token", "api_key": "cwt-from-config"},
        })
        result = get_credential("chatwork", "chatwork", env_var="CHATWORK_API_TOKEN")
        assert result == "cwt-from-config"

    def test_config_json_wins_over_env(self, config_dir, monkeypatch):
        _write_config(config_dir, {
            "chatwork": {"type": "api_token", "api_key": "cwt-from-config"},
        })
        monkeypatch.setenv("CHATWORK_API_TOKEN", "cwt-from-env")
        result = get_credential("chatwork", "chatwork", env_var="CHATWORK_API_TOKEN")
        assert result == "cwt-from-config"


class TestEnvVarFallback:
    """Environment variable should be used when config.json has no value."""

    def test_falls_back_to_env(self, config_dir, monkeypatch):
        _write_config(config_dir, {})  # No chatwork credential
        monkeypatch.setenv("CHATWORK_API_TOKEN", "cwt-from-env")
        result = get_credential("chatwork", "chatwork", env_var="CHATWORK_API_TOKEN")
        assert result == "cwt-from-env"

    def test_falls_back_when_api_key_empty(self, config_dir, monkeypatch):
        _write_config(config_dir, {
            "chatwork": {"type": "api_token", "api_key": ""},
        })
        monkeypatch.setenv("CHATWORK_API_TOKEN", "cwt-from-env")
        result = get_credential("chatwork", "chatwork", env_var="CHATWORK_API_TOKEN")
        assert result == "cwt-from-env"


class TestErrorCase:
    """Should raise ToolConfigError with guidance when neither source has a value."""

    def test_raises_when_nothing_set(self, config_dir):
        _write_config(config_dir, {})
        with pytest.raises(ToolConfigError, match="chatwork"):
            get_credential("chatwork", "chatwork", env_var="CHATWORK_API_TOKEN")

    def test_error_mentions_config_json(self, config_dir):
        _write_config(config_dir, {})
        with pytest.raises(ToolConfigError, match="config.json"):
            get_credential("chatwork", "chatwork", env_var="CHATWORK_API_TOKEN")

    def test_error_mentions_env_var(self, config_dir):
        _write_config(config_dir, {})
        with pytest.raises(ToolConfigError, match="CHATWORK_API_TOKEN"):
            get_credential("chatwork", "chatwork", env_var="CHATWORK_API_TOKEN")

    def test_error_without_env_var(self, config_dir):
        _write_config(config_dir, {})
        with pytest.raises(ToolConfigError, match="config.json"):
            get_credential("chatwork", "chatwork")


class TestMultiKeyCredential:
    """Test keys dict for credentials with multiple keys."""

    def test_resolve_from_keys_dict(self, config_dir):
        _write_config(config_dir, {
            "some_tool": {
                "type": "oauth_client",
                "keys": {"client_id": "id-123", "client_secret": "sec-456"},
            },
        })
        cid = get_credential("some_tool", "some_tool", key_name="client_id")
        assert cid == "id-123"
        csec = get_credential("some_tool", "some_tool", key_name="client_secret")
        assert csec == "sec-456"

    def test_keys_not_found_falls_to_env(self, config_dir, monkeypatch):
        _write_config(config_dir, {
            "some_tool": {"type": "api_key", "keys": {}},
        })
        monkeypatch.setenv("SOME_TOOL_SECRET", "from-env")
        result = get_credential(
            "some_tool", "some_tool",
            key_name="secret_key", env_var="SOME_TOOL_SECRET",
        )
        assert result == "from-env"

    def test_keys_empty_string_falls_to_env(self, config_dir, monkeypatch):
        _write_config(config_dir, {
            "some_tool": {"type": "api_key", "keys": {"secret_key": ""}},
        })
        monkeypatch.setenv("SOME_TOOL_SECRET", "from-env")
        result = get_credential(
            "some_tool", "some_tool",
            key_name="secret_key", env_var="SOME_TOOL_SECRET",
        )
        assert result == "from-env"


class TestSharedCredentialsJson:
    """shared/credentials.json should be checked between config.json and env."""

    def _write_shared_creds(self, config_dir: Path, creds: dict) -> None:
        shared_dir = config_dir / "shared"
        shared_dir.mkdir(parents=True, exist_ok=True)
        cred_file = shared_dir / "credentials.json"
        cred_file.write_text(json.dumps(creds), encoding="utf-8")

    def test_resolves_from_shared_credentials(self, config_dir):
        _write_config(config_dir, {})  # No chatwork in config.json
        self._write_shared_creds(config_dir, {"CHATWORK_API_TOKEN": "cwt-shared"})
        result = get_credential("chatwork", "chatwork", env_var="CHATWORK_API_TOKEN")
        assert result == "cwt-shared"

    def test_config_json_wins_over_shared(self, config_dir):
        _write_config(config_dir, {
            "chatwork": {"type": "api_token", "api_key": "cwt-from-config"},
        })
        self._write_shared_creds(config_dir, {"CHATWORK_API_TOKEN": "cwt-shared"})
        result = get_credential("chatwork", "chatwork", env_var="CHATWORK_API_TOKEN")
        assert result == "cwt-from-config"

    def test_shared_wins_over_env(self, config_dir, monkeypatch):
        _write_config(config_dir, {})
        self._write_shared_creds(config_dir, {"CHATWORK_API_TOKEN": "cwt-shared"})
        monkeypatch.setenv("CHATWORK_API_TOKEN", "cwt-from-env")
        result = get_credential("chatwork", "chatwork", env_var="CHATWORK_API_TOKEN")
        assert result == "cwt-shared"

    def test_falls_through_when_key_missing(self, config_dir, monkeypatch):
        _write_config(config_dir, {})
        self._write_shared_creds(config_dir, {"OTHER_KEY": "irrelevant"})
        monkeypatch.setenv("CHATWORK_API_TOKEN", "cwt-from-env")
        result = get_credential("chatwork", "chatwork", env_var="CHATWORK_API_TOKEN")
        assert result == "cwt-from-env"

    def test_falls_through_when_value_empty(self, config_dir, monkeypatch):
        _write_config(config_dir, {})
        self._write_shared_creds(config_dir, {"CHATWORK_API_TOKEN": ""})
        monkeypatch.setenv("CHATWORK_API_TOKEN", "cwt-from-env")
        result = get_credential("chatwork", "chatwork", env_var="CHATWORK_API_TOKEN")
        assert result == "cwt-from-env"

    def test_falls_through_when_file_missing(self, config_dir, monkeypatch):
        _write_config(config_dir, {})
        # No shared/credentials.json created
        monkeypatch.setenv("CHATWORK_API_TOKEN", "cwt-from-env")
        result = get_credential("chatwork", "chatwork", env_var="CHATWORK_API_TOKEN")
        assert result == "cwt-from-env"

    def test_falls_through_when_file_invalid_json(self, config_dir, monkeypatch):
        shared_dir = config_dir / "shared"
        shared_dir.mkdir(parents=True, exist_ok=True)
        (shared_dir / "credentials.json").write_text("not json", encoding="utf-8")
        _write_config(config_dir, {})
        monkeypatch.setenv("CHATWORK_API_TOKEN", "cwt-from-env")
        result = get_credential("chatwork", "chatwork", env_var="CHATWORK_API_TOKEN")
        assert result == "cwt-from-env"

    def test_no_env_var_skips_shared_lookup(self, config_dir):
        """When env_var is None, shared/credentials.json is not consulted."""
        _write_config(config_dir, {})
        self._write_shared_creds(config_dir, {"CHATWORK_API_TOKEN": "cwt-shared"})
        with pytest.raises(ToolConfigError):
            get_credential("chatwork", "chatwork")  # no env_var

    def test_error_message_mentions_shared(self, config_dir):
        _write_config(config_dir, {})
        # No shared/credentials.json, no env var
        with pytest.raises(ToolConfigError, match="shared/credentials.json"):
            get_credential("chatwork", "chatwork", env_var="CHATWORK_API_TOKEN")


class TestAllToolCredentials:
    """Verify each migrated tool's credential resolution."""

    @pytest.mark.parametrize("cred_name,tool_name,env_var,token", [
        ("chatwork", "chatwork", "CHATWORK_API_TOKEN", "cwt-test"),
        ("slack", "slack", "SLACK_BOT_TOKEN", "xoxb-test"),
        ("brave", "web_search", "BRAVE_API_KEY", "BSA-test"),
        ("x_twitter", "x_search", "TWITTER_BEARER_TOKEN", "AAAA-test"),
        ("novelai", "image_gen", "NOVELAI_TOKEN", "nai-test"),
        ("fal", "image_gen", "FAL_KEY", "fal-test"),
        ("meshy", "image_gen", "MESHY_API_KEY", "meshy-test"),
    ])
    def test_from_config_json(self, config_dir, cred_name, tool_name, env_var, token):
        _write_config(config_dir, {
            cred_name: {"api_key": token},
        })
        result = get_credential(cred_name, tool_name, env_var=env_var)
        assert result == token

    @pytest.mark.parametrize("cred_name,tool_name,env_var,token", [
        ("chatwork", "chatwork", "CHATWORK_API_TOKEN", "cwt-env"),
        ("slack", "slack", "SLACK_BOT_TOKEN", "xoxb-env"),
        ("brave", "web_search", "BRAVE_API_KEY", "BSA-env"),
        ("x_twitter", "x_search", "TWITTER_BEARER_TOKEN", "AAAA-env"),
        ("novelai", "image_gen", "NOVELAI_TOKEN", "nai-env"),
        ("fal", "image_gen", "FAL_KEY", "fal-env"),
        ("meshy", "image_gen", "MESHY_API_KEY", "meshy-env"),
    ])
    def test_from_env_fallback(self, config_dir, monkeypatch, cred_name, tool_name, env_var, token):
        _write_config(config_dir, {})
        monkeypatch.setenv(env_var, token)
        result = get_credential(cred_name, tool_name, env_var=env_var)
        assert result == token
