"""E2E tests for send_message → outbound routing through ToolHandler."""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from core.config.models import (
    AnimaWorksConfig,
    ExternalMessagingConfig,
    UserAliasConfig,
    save_config,
)
from core.messenger import Messenger
from core.tooling.handler import ToolHandler


# ── Fixtures ──────────────────────────────────────────────


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    d = tmp_path / ".animaworks"
    d.mkdir()
    return d


@pytest.fixture
def animas_dir(data_dir: Path) -> Path:
    d = data_dir / "animas"
    d.mkdir()
    return d


@pytest.fixture
def make_anima(animas_dir: Path):
    def _make(name: str) -> Path:
        d = animas_dir / name
        d.mkdir(exist_ok=True)
        (d / "permissions.md").write_text("", encoding="utf-8")
        return d
    return _make


@pytest.fixture
def shared_dir(data_dir: Path) -> Path:
    d = data_dir / "shared"
    d.mkdir()
    return d


@pytest.fixture
def config_with_aliases(data_dir: Path) -> AnimaWorksConfig:
    cfg = AnimaWorksConfig(
        external_messaging=ExternalMessagingConfig(
            preferred_channel="slack",
            user_aliases={
                "user": UserAliasConfig(slack_user_id="U06MJKLV0TG"),
            },
        ),
    )
    save_config(cfg, data_dir / "config.json")
    return cfg


@pytest.fixture
def make_handler(shared_dir: Path):
    def _make(anima_dir: Path, messenger: Messenger | None = None) -> ToolHandler:
        memory = MagicMock()
        memory.read_permissions.return_value = ""
        if messenger is None:
            messenger = Messenger(shared_dir, anima_dir.name)
        return ToolHandler(
            anima_dir=anima_dir,
            memory=memory,
            messenger=messenger,
            tool_registry=[],
        )
    return _make


# ── TestSendMessageToAlias ───────────────────────────────


class TestSendMessageToAlias:
    @patch("core.outbound._send_via_slack")
    def test_send_to_user_alias_routes_to_slack(
        self, mock_slack, make_anima, animas_dir, config_with_aliases,
        data_dir, make_handler,
    ):
        mock_slack.return_value = json.dumps({
            "status": "sent", "channel": "slack", "recipient": "U06MJKLV0TG",
            "message": "Message sent via Slack DM to U06MJKLV0TG",
        })
        sakura_dir = make_anima("sakura")
        handler = make_handler(sakura_dir)

        with patch("core.paths.get_animas_dir", return_value=animas_dir), \
             patch("core.config.models.load_config", return_value=config_with_aliases):
            result = handler.handle("send_message", {"to": "user", "content": "hello", "intent": "report"})

        data = json.loads(result)
        assert data["status"] == "sent"
        mock_slack.assert_called_once_with("U06MJKLV0TG", "hello", "sakura")

    def test_send_to_internal_anima_unchanged(
        self, make_anima, animas_dir, config_with_aliases,
        data_dir, make_handler,
    ):
        sakura_dir = make_anima("sakura")
        kotoha_dir = make_anima("kotoha")
        handler = make_handler(sakura_dir)

        with patch("core.paths.get_animas_dir", return_value=animas_dir), \
             patch("core.config.models.load_config", return_value=config_with_aliases):
            result = handler.handle("send_message", {"to": "kotoha", "content": "hi", "intent": "report"})

        assert "Message sent to kotoha" in result

    @patch("core.outbound._send_via_slack")
    def test_send_to_slack_prefix(
        self, mock_slack, make_anima, animas_dir, config_with_aliases,
        data_dir, make_handler,
    ):
        mock_slack.return_value = json.dumps({
            "status": "sent", "channel": "slack", "recipient": "U06MJKLV0TG",
            "message": "OK",
        })
        sakura_dir = make_anima("sakura")
        handler = make_handler(sakura_dir)

        with patch("core.paths.get_animas_dir", return_value=animas_dir), \
             patch("core.config.models.load_config", return_value=config_with_aliases):
            result = handler.handle(
                "send_message", {"to": "slack:U06MJKLV0TG", "content": "hi", "intent": "report"},
            )

        data = json.loads(result)
        assert data["status"] == "sent"

    def test_send_to_unknown_returns_error(
        self, make_anima, animas_dir, config_with_aliases,
        data_dir, make_handler,
    ):
        sakura_dir = make_anima("sakura")
        handler = make_handler(sakura_dir)

        with patch("core.paths.get_animas_dir", return_value=animas_dir), \
             patch("core.config.models.load_config", return_value=config_with_aliases):
            result = handler.handle(
                "send_message", {"to": "nobody", "content": "hi", "intent": "report"},
            )

        data = json.loads(result)
        assert data["error_type"] == "UnknownRecipient"


# ── TestRobustRecipientHandling ──────────────────────────


class TestRobustRecipientHandling:
    @pytest.mark.parametrize("variant", ["USER", "User", "uSeR"])
    @patch("core.outbound._send_via_slack")
    def test_case_insensitive_alias(
        self, mock_slack, variant, make_anima, animas_dir,
        config_with_aliases, make_handler,
    ):
        mock_slack.return_value = json.dumps({"status": "sent"})
        sakura_dir = make_anima("sakura")
        handler = make_handler(sakura_dir)

        with patch("core.paths.get_animas_dir", return_value=animas_dir), \
             patch("core.config.models.load_config", return_value=config_with_aliases):
            result = handler.handle("send_message", {"to": variant, "content": "hi", "intent": "report"})

        data = json.loads(result)
        assert data["status"] == "sent"

    @patch("core.outbound._send_via_slack")
    def test_bare_slack_user_id(
        self, mock_slack, make_anima, animas_dir, config_with_aliases,
        make_handler,
    ):
        mock_slack.return_value = json.dumps({"status": "sent"})
        sakura_dir = make_anima("sakura")
        handler = make_handler(sakura_dir)

        with patch("core.paths.get_animas_dir", return_value=animas_dir), \
             patch("core.config.models.load_config", return_value=config_with_aliases):
            result = handler.handle(
                "send_message", {"to": "U06MJKLV0TG", "content": "hi", "intent": "report"},
            )

        data = json.loads(result)
        assert data["status"] == "sent"

    def test_case_insensitive_anima_name(
        self, make_anima, animas_dir, config_with_aliases, make_handler,
    ):
        sakura_dir = make_anima("sakura")
        handler = make_handler(sakura_dir)

        with patch("core.paths.get_animas_dir", return_value=animas_dir), \
             patch("core.config.models.load_config", return_value=config_with_aliases):
            result = handler.handle(
                "send_message", {"to": "SAKURA", "content": "hi", "intent": "report"},
            )

        assert "Message sent to sakura" in result

    @patch("core.outbound._send_via_slack")
    def test_whitespace_in_recipient(
        self, mock_slack, make_anima, animas_dir, config_with_aliases,
        make_handler,
    ):
        mock_slack.return_value = json.dumps({"status": "sent"})
        sakura_dir = make_anima("sakura")
        handler = make_handler(sakura_dir)

        with patch("core.paths.get_animas_dir", return_value=animas_dir), \
             patch("core.config.models.load_config", return_value=config_with_aliases):
            result = handler.handle(
                "send_message", {"to": "  user  ", "content": "hi", "intent": "report"},
            )

        data = json.loads(result)
        assert data["status"] == "sent"


# ── TestFallbackBehavior ─────────────────────────────────


class TestFallbackBehavior:
    def test_config_load_failure_returns_error_json(
        self, make_anima, make_handler,
    ):
        sakura_dir = make_anima("sakura")
        handler = make_handler(sakura_dir)

        with patch("core.config.models.load_config", side_effect=RuntimeError("boom")):
            result = handler.handle(
                "send_message", {"to": "user", "content": "hi", "intent": "report"},
            )

        # Should return a RecipientResolutionError instead of silently
        # falling back to internal messaging
        parsed = json.loads(result)
        assert parsed["status"] == "error"
        assert parsed["error_type"] == "RecipientResolutionError"

    @patch("core.outbound._send_via_slack")
    def test_activity_timeline_log_on_external(
        self, mock_slack, make_anima, animas_dir, config_with_aliases,
        shared_dir, make_handler,
    ):
        """ToolHandler logs external sends to the unified activity log
        (dm_logs are no longer written by messenger.send)."""
        mock_slack.return_value = json.dumps({"status": "sent"})
        sakura_dir = make_anima("sakura")
        messenger = Messenger(shared_dir, "sakura")
        handler = make_handler(sakura_dir, messenger=messenger)

        with patch("core.paths.get_animas_dir", return_value=animas_dir), \
             patch("core.config.models.load_config", return_value=config_with_aliases):
            handler.handle("send_message", {"to": "user", "content": "hello", "intent": "report"})

        # ToolHandler._log_tool_activity records dm_sent in unified activity log
        from core.memory.activity import ActivityLogger
        activity = ActivityLogger(sakura_dir)
        entries = activity.recent(days=1, types=["dm_sent"])
        assert len(entries) >= 1
        assert entries[0].to_person == "user"
