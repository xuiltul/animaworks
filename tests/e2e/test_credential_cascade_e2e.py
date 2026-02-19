# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
"""E2E tests for credential cascade: config.json → env → error.

Validates that actual tool client constructors use the unified
get_credential() resolver and that the cascade works end-to-end.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from core.config.models import (
    AnimaWorksConfig,
    CredentialConfig,
    invalidate_cache,
    save_config,
)
from core.tools._base import ToolConfigError


@pytest.fixture(autouse=True)
def _clean_cache():
    invalidate_cache()
    yield
    invalidate_cache()


@pytest.fixture
def config_dir(tmp_path, monkeypatch):
    """Isolated config directory with ANIMAWORKS_DATA_DIR set."""
    monkeypatch.setenv("ANIMAWORKS_DATA_DIR", str(tmp_path))
    return tmp_path


def _save_credentials(config_dir: Path, credentials: dict) -> None:
    creds = {}
    for name, conf in credentials.items():
        creds[name] = CredentialConfig(**conf)
    config = AnimaWorksConfig(credentials=creds)
    save_config(config, config_dir / "config.json")


class TestChatworkCredentialCascade:
    """Chatwork client should resolve via config.json → env → error."""

    def test_config_json_resolution(self, config_dir, monkeypatch):
        _save_credentials(config_dir, {
            "chatwork": {"type": "api_token", "api_key": "cwt-config-test"},
        })
        # Remove env var to ensure config.json is used
        monkeypatch.delenv("CHATWORK_API_TOKEN", raising=False)

        from core.tools.chatwork import ChatworkClient
        client = ChatworkClient()
        assert client.api_token == "cwt-config-test"

    def test_env_fallback(self, config_dir, monkeypatch):
        _save_credentials(config_dir, {})  # No chatwork in config
        monkeypatch.setenv("CHATWORK_API_TOKEN", "cwt-env-test")

        from core.tools.chatwork import ChatworkClient
        client = ChatworkClient()
        assert client.api_token == "cwt-env-test"

    def test_error_when_missing(self, config_dir, monkeypatch):
        _save_credentials(config_dir, {})
        monkeypatch.delenv("CHATWORK_API_TOKEN", raising=False)

        from core.tools.chatwork import ChatworkClient
        with pytest.raises(ToolConfigError, match="chatwork"):
            ChatworkClient()


class TestSlackCredentialCascade:
    """Slack client should resolve via config.json → env → error."""

    def test_config_json_resolution(self, config_dir, monkeypatch):
        _save_credentials(config_dir, {
            "slack": {"type": "api_token", "api_key": "xoxb-config-test"},
        })
        monkeypatch.delenv("SLACK_BOT_TOKEN", raising=False)

        from core.tools.slack import SlackClient
        # SlackClient needs slack_sdk, mock it
        with patch("core.tools.slack._require_slack_sdk"):
            with patch("core.tools.slack.WebClient") as mock_wc:
                client = SlackClient()
                mock_wc.assert_called_with(token="xoxb-config-test")

    def test_env_fallback(self, config_dir, monkeypatch):
        _save_credentials(config_dir, {})
        monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-env-test")

        from core.tools.slack import SlackClient
        with patch("core.tools.slack._require_slack_sdk"):
            with patch("core.tools.slack.WebClient") as mock_wc:
                client = SlackClient()
                mock_wc.assert_called_with(token="xoxb-env-test")


class TestWebSearchCredentialCascade:
    """Brave search should resolve via config.json → env → error."""

    def test_config_json_resolution(self, config_dir, monkeypatch):
        from core.tools._base import get_credential
        _save_credentials(config_dir, {
            "brave": {"type": "api_key", "api_key": "BSA-config-test"},
        })
        monkeypatch.delenv("BRAVE_API_KEY", raising=False)
        result = get_credential("brave", "web_search", env_var="BRAVE_API_KEY")
        assert result == "BSA-config-test"

    def test_env_fallback(self, config_dir, monkeypatch):
        from core.tools._base import get_credential
        _save_credentials(config_dir, {})
        monkeypatch.setenv("BRAVE_API_KEY", "BSA-env-test")
        result = get_credential("brave", "web_search", env_var="BRAVE_API_KEY")
        assert result == "BSA-env-test"


class TestXSearchCredentialCascade:
    """X/Twitter client should resolve via config.json → env → error."""

    def test_config_json_resolution(self, config_dir, monkeypatch):
        _save_credentials(config_dir, {
            "x_twitter": {"type": "bearer_token", "api_key": "AAAA-config-test"},
        })
        monkeypatch.delenv("TWITTER_BEARER_TOKEN", raising=False)

        from core.tools.x_search import XSearchClient
        client = XSearchClient()
        assert client.bearer_token == "AAAA-config-test"

    def test_env_fallback(self, config_dir, monkeypatch):
        _save_credentials(config_dir, {})
        monkeypatch.setenv("TWITTER_BEARER_TOKEN", "AAAA-env-test")

        from core.tools.x_search import XSearchClient
        client = XSearchClient()
        assert client.bearer_token == "AAAA-env-test"


class TestImageGenCredentialCascade:
    """Image generation clients should resolve via config.json → env → error."""

    def test_novelai_config_json(self, config_dir, monkeypatch):
        from core.tools._base import get_credential
        _save_credentials(config_dir, {
            "novelai": {"type": "api_token", "api_key": "nai-config-test"},
        })
        monkeypatch.delenv("NOVELAI_TOKEN", raising=False)
        result = get_credential("novelai", "image_gen", env_var="NOVELAI_TOKEN")
        assert result == "nai-config-test"

    def test_fal_config_json(self, config_dir, monkeypatch):
        from core.tools._base import get_credential
        _save_credentials(config_dir, {
            "fal": {"type": "api_key", "api_key": "fal-config-test"},
        })
        monkeypatch.delenv("FAL_KEY", raising=False)
        result = get_credential("fal", "image_gen", env_var="FAL_KEY")
        assert result == "fal-config-test"

    def test_meshy_config_json(self, config_dir, monkeypatch):
        from core.tools._base import get_credential
        _save_credentials(config_dir, {
            "meshy": {"type": "api_key", "api_key": "meshy-config-test"},
        })
        monkeypatch.delenv("MESHY_API_KEY", raising=False)
        result = get_credential("meshy", "image_gen", env_var="MESHY_API_KEY")
        assert result == "meshy-config-test"


class TestCascadePriority:
    """When both config.json and env var are set, config.json wins."""

    @pytest.mark.parametrize("cred_name,env_var", [
        ("chatwork", "CHATWORK_API_TOKEN"),
        ("slack", "SLACK_BOT_TOKEN"),
        ("brave", "BRAVE_API_KEY"),
        ("x_twitter", "TWITTER_BEARER_TOKEN"),
        ("novelai", "NOVELAI_TOKEN"),
        ("fal", "FAL_KEY"),
        ("meshy", "MESHY_API_KEY"),
    ])
    def test_config_json_takes_priority(self, config_dir, monkeypatch, cred_name, env_var):
        from core.tools._base import get_credential
        _save_credentials(config_dir, {
            cred_name: {"api_key": "from-config"},
        })
        monkeypatch.setenv(env_var, "from-env")
        result = get_credential(cred_name, "test_tool", env_var=env_var)
        assert result == "from-config"


class TestSharedCredentialsCascadeE2E:
    """shared/credentials.json should integrate into the cascade end-to-end."""

    def _write_shared_creds(self, config_dir: Path, creds: dict) -> None:
        shared_dir = config_dir / "shared"
        shared_dir.mkdir(parents=True, exist_ok=True)
        cred_file = shared_dir / "credentials.json"
        cred_file.write_text(json.dumps(creds), encoding="utf-8")

    def test_shared_resolves_when_config_empty(self, config_dir, monkeypatch):
        from core.tools._base import get_credential
        _save_credentials(config_dir, {})
        self._write_shared_creds(config_dir, {"CHATWORK_API_TOKEN": "cwt-shared-e2e"})
        monkeypatch.delenv("CHATWORK_API_TOKEN", raising=False)
        result = get_credential("chatwork", "chatwork", env_var="CHATWORK_API_TOKEN")
        assert result == "cwt-shared-e2e"

    def test_config_json_wins_over_shared(self, config_dir, monkeypatch):
        from core.tools._base import get_credential
        _save_credentials(config_dir, {
            "chatwork": {"type": "api_token", "api_key": "cwt-config-e2e"},
        })
        self._write_shared_creds(config_dir, {"CHATWORK_API_TOKEN": "cwt-shared-e2e"})
        monkeypatch.delenv("CHATWORK_API_TOKEN", raising=False)
        result = get_credential("chatwork", "chatwork", env_var="CHATWORK_API_TOKEN")
        assert result == "cwt-config-e2e"

    def test_shared_wins_over_env(self, config_dir, monkeypatch):
        from core.tools._base import get_credential
        _save_credentials(config_dir, {})
        self._write_shared_creds(config_dir, {"SLACK_BOT_TOKEN": "xoxb-shared-e2e"})
        monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-env-e2e")
        result = get_credential("slack", "slack", env_var="SLACK_BOT_TOKEN")
        assert result == "xoxb-shared-e2e"

    @pytest.mark.parametrize("env_var,token", [
        ("CHATWORK_API_TOKEN", "cwt-shared"),
        ("SLACK_BOT_TOKEN", "xoxb-shared"),
        ("BRAVE_API_KEY", "BSA-shared"),
        ("FAL_KEY", "fal-shared"),
        ("MESHY_API_KEY", "meshy-shared"),
    ])
    def test_multiple_tools_from_shared(self, config_dir, monkeypatch, env_var, token):
        from core.tools._base import get_credential
        _save_credentials(config_dir, {})
        self._write_shared_creds(config_dir, {env_var: token})
        monkeypatch.delenv(env_var, raising=False)
        cred_name = env_var.lower().rsplit("_", maxsplit=2)[0]
        # Map env_var to credential_name used in get_credential
        cred_map = {
            "CHATWORK_API_TOKEN": "chatwork",
            "SLACK_BOT_TOKEN": "slack",
            "BRAVE_API_KEY": "brave",
            "FAL_KEY": "fal",
            "MESHY_API_KEY": "meshy",
        }
        result = get_credential(cred_map[env_var], "test_tool", env_var=env_var)
        assert result == token

    def test_full_cascade_priority(self, config_dir, monkeypatch):
        """Full 3-tier cascade: config.json > shared > env."""
        from core.tools._base import get_credential

        _save_credentials(config_dir, {
            "brave": {"type": "api_key", "api_key": "BSA-config"},
        })
        self._write_shared_creds(config_dir, {"BRAVE_API_KEY": "BSA-shared"})
        monkeypatch.setenv("BRAVE_API_KEY", "BSA-env")

        # config.json wins
        result = get_credential("brave", "web_search", env_var="BRAVE_API_KEY")
        assert result == "BSA-config"

        # Remove config.json entry → shared wins
        _save_credentials(config_dir, {})
        invalidate_cache()
        result = get_credential("brave", "web_search", env_var="BRAVE_API_KEY")
        assert result == "BSA-shared"

        # Remove shared entry → env wins
        self._write_shared_creds(config_dir, {})
        result = get_credential("brave", "web_search", env_var="BRAVE_API_KEY")
        assert result == "BSA-env"


class TestMultiKeyE2E:
    """Test multi-key credential resolution end-to-end."""

    def test_resolve_client_id_and_secret(self, config_dir):
        from core.tools._base import get_credential
        _save_credentials(config_dir, {
            "custom_oauth": {
                "type": "oauth_client",
                "keys": {
                    "client_id": "my-client-id",
                    "client_secret": "my-client-secret",
                },
            },
        })
        cid = get_credential("custom_oauth", "custom_tool", key_name="client_id")
        csec = get_credential("custom_oauth", "custom_tool", key_name="client_secret")
        assert cid == "my-client-id"
        assert csec == "my-client-secret"

    def test_multi_key_with_primary(self, config_dir):
        from core.tools._base import get_credential
        _save_credentials(config_dir, {
            "hybrid": {
                "type": "api_key",
                "api_key": "primary-key",
                "keys": {"secondary": "secondary-key"},
            },
        })
        primary = get_credential("hybrid", "hybrid_tool")
        secondary = get_credential("hybrid", "hybrid_tool", key_name="secondary")
        assert primary == "primary-key"
        assert secondary == "secondary-key"
