# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
"""E2E tests for the human notification subsystem.

Tests the full flow: config → notifier creation → tool handler → notification dispatch.
Uses mock HTTP to verify actual channel formatting without making real API calls.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.config.models import (
    AnimaWorksConfig,
    HumanNotificationConfig,
    NotificationChannelConfig,
)
from core.notification.notifier import HumanNotifier
from core.tooling.handler import ToolHandler


# ── Fixtures ──────────────────────────────────────────────────


@pytest.fixture
def notification_config() -> HumanNotificationConfig:
    return HumanNotificationConfig(
        enabled=True,
        channels=[
            NotificationChannelConfig(
                type="ntfy",
                config={
                    "server_url": "https://ntfy.example.com",
                    "topic": "e2e-test",
                },
            ),
        ],
    )


@pytest.fixture
def anima_dir(tmp_path: Path) -> Path:
    d = tmp_path / "animas" / "e2e-leader"
    d.mkdir(parents=True)
    (d / "permissions.md").write_text("", encoding="utf-8")
    return d


@pytest.fixture
def memory() -> MagicMock:
    m = MagicMock()
    m.read_permissions.return_value = ""
    return m


# ── E2E Tests ─────────────────────────────────────────────────


class TestNotificationE2EFlow:
    """Test the complete notification flow from config to dispatch."""

    def test_config_to_notifier_creation(self, notification_config):
        """HumanNotifier can be created from config with valid channels."""
        notifier = HumanNotifier.from_config(notification_config)
        assert notifier.channel_count == 1

    def test_config_to_handler_to_notify(
        self, notification_config, anima_dir, memory,
    ):
        """Full flow: config → notifier → handler → notify_human tool call."""
        notifier = HumanNotifier.from_config(notification_config)
        handler = ToolHandler(
            anima_dir=anima_dir,
            memory=memory,
            human_notifier=notifier,
        )

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()

        with patch("core.notification.channels.ntfy.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_cls.return_value = mock_client

            result = handler.handle("call_human", {
                "subject": "E2E Test Alert",
                "body": "This is a full end-to-end test",
                "priority": "high",
            })

        parsed = json.loads(result)
        assert parsed["status"] == "sent"
        assert any("ntfy: OK" in r for r in parsed["results"])

        # Verify the actual HTTP call
        call_args = mock_client.post.call_args
        assert call_args[0][0] == "https://ntfy.example.com/e2e-test"
        assert call_args[1]["headers"]["Priority"] == "4"  # "high" → 4
        assert "(from e2e-leader)" in call_args[1]["headers"]["Title"]

    def test_multi_channel_e2e(self, anima_dir, memory, monkeypatch):
        """Test notification to multiple channels simultaneously."""
        monkeypatch.setenv("CHATWORK_API_TOKEN", "cw-test-token")

        config = HumanNotificationConfig(
            enabled=True,
            channels=[
                NotificationChannelConfig(
                    type="slack",
                    config={"webhook_url": "https://hooks.slack.com/e2e"},
                ),
                NotificationChannelConfig(
                    type="ntfy",
                    config={
                        "server_url": "https://ntfy.sh",
                        "topic": "multi-test",
                    },
                ),
                NotificationChannelConfig(
                    type="chatwork",
                    config={
                        "api_token_env": "CHATWORK_API_TOKEN",
                        "room_id": "999",
                    },
                ),
            ],
        )

        notifier = HumanNotifier.from_config(config)
        assert notifier.channel_count == 3

        handler = ToolHandler(
            anima_dir=anima_dir,
            memory=memory,
            human_notifier=notifier,
        )

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()

        # Patch all three channel HTTP clients
        with patch("core.notification.channels.slack.httpx.AsyncClient") as slack_cls, \
             patch("core.notification.channels.ntfy.httpx.AsyncClient") as ntfy_cls, \
             patch("core.notification.channels.chatwork.httpx.AsyncClient") as cw_cls:

            for cls in (slack_cls, ntfy_cls, cw_cls):
                mock_client = AsyncMock()
                mock_client.post.return_value = mock_response
                mock_client.__aenter__ = AsyncMock(return_value=mock_client)
                mock_client.__aexit__ = AsyncMock(return_value=False)
                cls.return_value = mock_client

            result = handler.handle("call_human", {
                "subject": "Multi-channel Test",
                "body": "Sent to all channels",
                "priority": "urgent",
            })

        parsed = json.loads(result)
        assert parsed["status"] == "sent"
        assert len(parsed["results"]) == 3

    def test_disabled_channels_skipped(self, anima_dir, memory):
        """Disabled channels in config are not included in the notifier."""
        config = HumanNotificationConfig(
            enabled=True,
            channels=[
                NotificationChannelConfig(
                    type="ntfy",
                    enabled=True,
                    config={"topic": "active"},
                ),
                NotificationChannelConfig(
                    type="ntfy",
                    enabled=False,
                    config={"topic": "inactive"},
                ),
            ],
        )
        notifier = HumanNotifier.from_config(config)
        assert notifier.channel_count == 1


class TestNotificationPromptIntegration:
    """Test that the prompt builder correctly injects notification guidance."""

    def test_top_level_anima_gets_notification_guidance(self, data_dir, make_anima):
        """Top-level Anima with notification enabled gets guidance in prompt."""
        anima_dir = make_anima("top-leader")

        # Enable human_notification in config
        config_path = data_dir / "config.json"
        config_data = json.loads(config_path.read_text(encoding="utf-8"))
        config_data["human_notification"] = {
            "enabled": True,
            "channels": [
                {"type": "ntfy", "config": {"topic": "test"}},
            ],
        }
        config_path.write_text(
            json.dumps(config_data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        from core.config import invalidate_cache
        invalidate_cache()

        from core.memory import MemoryManager
        from core.prompt.builder import build_system_prompt

        memory = MemoryManager(anima_dir)
        prompt = build_system_prompt(memory)

        assert "call_human" in prompt
        assert "人間への連絡" in prompt

    def test_supervised_anima_no_notification_guidance(self, data_dir, make_anima):
        """Anima with supervisor does NOT get notification guidance."""
        make_anima("boss")
        anima_dir = make_anima("worker", supervisor="boss")

        # Enable human_notification in config
        config_path = data_dir / "config.json"
        config_data = json.loads(config_path.read_text(encoding="utf-8"))
        config_data["human_notification"] = {
            "enabled": True,
            "channels": [
                {"type": "ntfy", "config": {"topic": "test"}},
            ],
        }
        config_path.write_text(
            json.dumps(config_data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        from core.config import invalidate_cache
        invalidate_cache()

        from core.memory import MemoryManager
        from core.prompt.builder import build_system_prompt

        memory = MemoryManager(anima_dir)
        prompt = build_system_prompt(memory)

        # Supervised anima should not have the human notification section
        assert "人間への連絡" not in prompt

    def test_notification_disabled_no_guidance(self, data_dir, make_anima):
        """Top-level Anima with notification disabled gets no guidance."""
        anima_dir = make_anima("top-leader-disabled")

        # human_notification disabled (default)
        from core.config import invalidate_cache
        invalidate_cache()

        from core.memory import MemoryManager
        from core.prompt.builder import build_system_prompt

        memory = MemoryManager(anima_dir)
        prompt = build_system_prompt(memory)

        assert "人間への連絡" not in prompt


class TestNotificationToolSchema:
    """Test that notify_human tool schema is properly defined."""

    def test_notify_human_in_notification_tools(self):
        from core.tooling.schemas import NOTIFICATION_TOOLS

        assert len(NOTIFICATION_TOOLS) == 1
        schema = NOTIFICATION_TOOLS[0]
        assert schema["name"] == "call_human"
        assert "subject" in schema["parameters"]["properties"]
        assert "body" in schema["parameters"]["properties"]
        assert "priority" in schema["parameters"]["properties"]
        assert "subject" in schema["parameters"]["required"]
        assert "body" in schema["parameters"]["required"]

    def test_build_tool_list_includes_notification(self):
        from core.tooling.schemas import build_tool_list

        tools = build_tool_list(include_notification_tools=True)
        names = [t["name"] for t in tools]
        assert "call_human" in names

    def test_build_tool_list_excludes_notification_by_default(self):
        from core.tooling.schemas import build_tool_list

        tools = build_tool_list()
        names = [t["name"] for t in tools]
        assert "call_human" not in names
