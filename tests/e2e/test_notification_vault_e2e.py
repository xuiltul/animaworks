"""E2E tests for notification vault/shared credential resolution.

Tests the full flow through HumanNotifier with vault credential fallback,
partial failure reporting, and chatwork/telegram robustness fixes.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.config.models import (
    HumanNotificationConfig,
    NotificationChannelConfig,
)
from core.notification.notifier import HumanNotifier
from core.tooling.handler import ToolHandler


# ── Fixtures ──────────────────────────────────────────────────


@pytest.fixture
def anima_dir(tmp_path: Path) -> Path:
    d = tmp_path / "animas" / "vault-test"
    d.mkdir(parents=True)
    (d / "permissions.md").write_text("", encoding="utf-8")
    return d


@pytest.fixture
def memory() -> MagicMock:
    m = MagicMock()
    m.read_permissions.return_value = ""
    return m


def _mock_http(cls_mock):
    """Configure an httpx.AsyncClient mock to return a successful response."""
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_client = AsyncMock()
    mock_client.post.return_value = mock_response
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    cls_mock.return_value = mock_client
    return mock_client


# ── E2E: vault credential resolution across channels ─────────


class TestVaultCredentialE2E:
    def test_chatwork_int_room_id_full_flow(self, anima_dir, memory, monkeypatch):
        """Full flow: chatwork with int room_id from JSON config doesn't crash."""
        monkeypatch.setenv("CW_API_TOKEN", "cw-test-token")

        config = HumanNotificationConfig(
            enabled=True,
            channels=[
                NotificationChannelConfig(
                    type="chatwork",
                    config={
                        "api_token_env": "CW_API_TOKEN",
                        "room_id": 99999,  # int, not str
                    },
                ),
            ],
        )

        notifier = HumanNotifier.from_config(config)
        handler = ToolHandler(
            anima_dir=anima_dir,
            memory=memory,
            human_notifier=notifier,
        )

        with patch("core.notification.channels.chatwork.httpx.AsyncClient") as mock_cls:
            _mock_http(mock_cls)
            result = handler.handle("call_human", {
                "subject": "Room ID Test",
                "body": "int room_id should work",
                "priority": "normal",
            })

        parsed = json.loads(result)
        assert parsed["status"] == "sent"
        assert any("chatwork: OK" in r for r in parsed["results"])

    def test_telegram_long_body_full_flow(self, anima_dir, memory, monkeypatch):
        """Full flow: telegram with very long body truncates before escape."""
        monkeypatch.setenv("TG_BOT_TOKEN", "tg-test-token")

        config = HumanNotificationConfig(
            enabled=True,
            channels=[
                NotificationChannelConfig(
                    type="telegram",
                    config={
                        "bot_token_env": "TG_BOT_TOKEN",
                        "chat_id": "123456",
                    },
                ),
            ],
        )

        notifier = HumanNotifier.from_config(config)
        handler = ToolHandler(
            anima_dir=anima_dir,
            memory=memory,
            human_notifier=notifier,
        )

        captured: dict[str, Any] = {}

        async def capture_post(url, json=None):
            captured.update(json or {})
            resp = MagicMock()
            resp.raise_for_status = MagicMock()
            return resp

        with patch("core.notification.channels.telegram.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_client.post = capture_post
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_cls.return_value = mock_client

            long_body = "Check A & B situation! " * 500
            result = handler.handle("call_human", {
                "subject": "Telegram Truncate Test",
                "body": long_body,
                "priority": "high",
            })

        parsed = json.loads(result)
        assert parsed["status"] == "sent"
        text = captured.get("text", "")
        assert len(text) <= 4096
        # No broken HTML entities
        for i, c in enumerate(text):
            if c == "&":
                rest = text[i:]
                assert (
                    rest.startswith("&amp;")
                    or rest.startswith("&lt;")
                    or rest.startswith("&gt;")
                    or rest.startswith("&quot;")
                    or rest.startswith("&#")
                ), f"Broken entity at position {i}: {text[i:i+10]}"


class TestPartialFailureE2E:
    def test_mixed_success_failure_reports_correctly(
        self, anima_dir, memory, monkeypatch, caplog,
    ):
        """When one channel succeeds and another fails, notifier logs warning."""
        config = HumanNotificationConfig(
            enabled=True,
            channels=[
                NotificationChannelConfig(
                    type="ntfy",
                    config={
                        "server_url": "https://ntfy.sh",
                        "topic": "ok-topic",
                    },
                ),
                NotificationChannelConfig(
                    type="ntfy",
                    config={
                        "server_url": "https://ntfy.sh",
                        "topic": "fail-topic",
                    },
                ),
            ],
        )

        notifier = HumanNotifier.from_config(config)
        handler = ToolHandler(
            anima_dir=anima_dir,
            memory=memory,
            human_notifier=notifier,
        )

        call_count = 0

        async def _alternating_post(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                resp = MagicMock()
                resp.raise_for_status = MagicMock()
                return resp
            raise ConnectionError("network down")

        with patch("httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_client.post = _alternating_post
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_cls.return_value = mock_client

            with caplog.at_level(logging.WARNING, logger="animaworks.notification"):
                result = handler.handle("call_human", {
                    "subject": "Partial Fail",
                    "body": "One channel fails",
                })

        parsed = json.loads(result)
        assert parsed["status"] == "sent"
        assert any("ntfy: OK" in r for r in parsed["results"])
        assert any("ERROR" in r for r in parsed["results"])
        assert any("partial failure" in r.message for r in caplog.records)
