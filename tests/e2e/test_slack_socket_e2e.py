# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
"""E2E tests for Slack Socket Mode — message routing to anima inbox.

Tests the handler logic end-to-end using real filesystem and Messenger
(no mocks on message routing), with only the Slack SDK objects mocked.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.config.models import (
    AnimaWorksConfig,
    ExternalMessagingChannelConfig,
    ExternalMessagingConfig,
    save_config,
)
from core.schemas import Message


# ── Helpers ──────────────────────────────────────────────


def _make_socket_manager(data_dir, anima_mapping, monkeypatch):
    """Create a SlackSocketModeManager with captured event handlers.

    Returns (manager, captured_handlers) where captured_handlers maps
    event type → handler function for direct invocation.
    """
    from server.slack_socket import SlackSocketModeManager

    # Patch config to enable socket mode with mapping
    config = AnimaWorksConfig.model_validate(
        json.loads((data_dir / "config.json").read_text(encoding="utf-8")),
    )
    config.external_messaging = ExternalMessagingConfig(
        slack=ExternalMessagingChannelConfig(
            enabled=True,
            mode="socket",
            anima_mapping=anima_mapping,
        ),
    )
    save_config(config, data_dir / "config.json")

    # Patch get_credential to return test tokens
    monkeypatch.setattr(
        "server.slack_socket.get_credential",
        lambda name, tool, **kw: f"xoxb-test" if "bot" in (kw.get("env_var") or "").lower() else "xapp-test",
    )

    # Capture event handler registrations instead of connecting to Slack
    captured_handlers: dict[str, list] = {}
    mock_async_app = MagicMock()

    def _capture_event(event_type):
        def decorator(func):
            captured_handlers.setdefault(event_type, []).append(func)
            return func
        return decorator

    mock_async_app.event = _capture_event

    monkeypatch.setattr("server.slack_socket.AsyncApp", lambda **kw: mock_async_app)
    monkeypatch.setattr(
        "server.slack_socket.AsyncSocketModeHandler",
        lambda app, token: AsyncMock(),
    )

    mgr = SlackSocketModeManager()
    return mgr, captured_handlers


# ── E2E Tests ────────────────────────────────────────────


class TestSlackSocketModeE2E:
    """E2E: Socket Mode message → Messenger → inbox file on disk."""

    async def test_full_flow_message_to_inbox(self, data_dir, make_anima, monkeypatch):
        """A Socket Mode message creates a correct inbox file via real Messenger."""
        make_anima("sakura")

        mgr, handlers = _make_socket_manager(
            data_dir,
            anima_mapping={"C_SOCKET_E2E": "sakura"},
            monkeypatch=monkeypatch,
        )
        await mgr.start()

        assert "message" in handlers
        handler_fn = handlers["message"][0]

        # Simulate incoming Slack message
        event = {
            "channel": "C_SOCKET_E2E",
            "user": "U_E2E_SOCKET",
            "text": "Hello via Socket Mode",
            "ts": "8888888888.000001",
        }
        await handler_fn(event=event, say=AsyncMock())

        # Verify inbox file was created
        inbox = data_dir / "shared" / "inbox" / "sakura"
        files = list(inbox.glob("*.json"))
        assert len(files) == 1

        msg = Message.model_validate_json(files[0].read_text(encoding="utf-8"))
        assert msg.content == "Hello via Socket Mode"
        assert msg.source == "slack"
        assert msg.external_user_id == "U_E2E_SOCKET"
        assert msg.external_channel_id == "C_SOCKET_E2E"
        assert msg.source_message_id == "8888888888.000001"
        assert msg.to_person == "sakura"
        assert "slack:" in msg.from_person

    async def test_message_delivered_to_inbox(self, data_dir, make_anima, monkeypatch):
        """Socket Mode messages are placed in the anima's inbox."""
        make_anima("kotoha")

        mgr, handlers = _make_socket_manager(
            data_dir,
            anima_mapping={"C_LOG_TEST": "kotoha"},
            monkeypatch=monkeypatch,
        )
        await mgr.start()

        handler_fn = handlers["message"][0]
        await handler_fn(
            event={
                "channel": "C_LOG_TEST",
                "user": "U_LOG",
                "text": "Log this message",
                "ts": "7777777777.000001",
            },
            say=AsyncMock(),
        )

        inbox = data_dir / "shared" / "inbox" / "kotoha"
        files = list(inbox.glob("*.json"))
        assert len(files) == 1
        msg = Message.model_validate_json(files[0].read_text(encoding="utf-8"))
        assert msg.content == "Log this message"

    async def test_unmapped_channel_no_inbox(self, data_dir, make_anima, monkeypatch):
        """Messages from unmapped channels do NOT create inbox files."""
        make_anima("sakura")

        mgr, handlers = _make_socket_manager(
            data_dir,
            anima_mapping={"C_KNOWN": "sakura"},
            monkeypatch=monkeypatch,
        )
        await mgr.start()

        handler_fn = handlers["message"][0]
        await handler_fn(
            event={
                "channel": "C_UNKNOWN",
                "user": "U_X",
                "text": "Should not arrive",
                "ts": "1.1",
            },
            say=AsyncMock(),
        )

        inbox = data_dir / "shared" / "inbox" / "sakura"
        files = list(inbox.glob("*.json"))
        assert len(files) == 0

    async def test_subtype_message_ignored(self, data_dir, make_anima, monkeypatch):
        """Messages with subtype (edits, bot, etc.) are ignored."""
        make_anima("sakura")

        mgr, handlers = _make_socket_manager(
            data_dir,
            anima_mapping={"C_SUB": "sakura"},
            monkeypatch=monkeypatch,
        )
        await mgr.start()

        handler_fn = handlers["message"][0]
        await handler_fn(
            event={
                "channel": "C_SUB",
                "user": "U_BOT",
                "text": "Edited message",
                "ts": "2.2",
                "subtype": "message_changed",
            },
            say=AsyncMock(),
        )

        inbox = data_dir / "shared" / "inbox" / "sakura"
        files = list(inbox.glob("*.json"))
        assert len(files) == 0

    async def test_multiple_messages_to_different_animas(
        self, data_dir, make_anima, monkeypatch
    ):
        """Messages to different channels route to the correct animas."""
        make_anima("sakura")
        make_anima("kotoha")

        mgr, handlers = _make_socket_manager(
            data_dir,
            anima_mapping={"C_SAKURA": "sakura", "C_KOTOHA": "kotoha"},
            monkeypatch=monkeypatch,
        )
        await mgr.start()

        handler_fn = handlers["message"][0]

        await handler_fn(
            event={"channel": "C_SAKURA", "user": "U1", "text": "For sakura", "ts": "10.1"},
            say=AsyncMock(),
        )
        await handler_fn(
            event={"channel": "C_KOTOHA", "user": "U2", "text": "For kotoha", "ts": "10.2"},
            say=AsyncMock(),
        )

        sakura_inbox = data_dir / "shared" / "inbox" / "sakura"
        kotoha_inbox = data_dir / "shared" / "inbox" / "kotoha"

        sakura_files = list(sakura_inbox.glob("*.json"))
        kotoha_files = list(kotoha_inbox.glob("*.json"))

        assert len(sakura_files) == 1
        assert len(kotoha_files) == 1

        sakura_msg = Message.model_validate_json(
            sakura_files[0].read_text(encoding="utf-8"),
        )
        kotoha_msg = Message.model_validate_json(
            kotoha_files[0].read_text(encoding="utf-8"),
        )

        assert sakura_msg.content == "For sakura"
        assert sakura_msg.to_person == "sakura"
        assert kotoha_msg.content == "For kotoha"
        assert kotoha_msg.to_person == "kotoha"

    async def test_messages_readable_via_messenger_receive(
        self, data_dir, make_anima, monkeypatch
    ):
        """Socket Mode messages can be read back via Messenger.receive()."""
        make_anima("sakura")

        mgr, handlers = _make_socket_manager(
            data_dir,
            anima_mapping={"C_READ": "sakura"},
            monkeypatch=monkeypatch,
        )
        await mgr.start()

        handler_fn = handlers["message"][0]
        await handler_fn(
            event={"channel": "C_READ", "user": "U_R", "text": "Readable", "ts": "3.3"},
            say=AsyncMock(),
        )

        from core.messenger import Messenger

        messenger = Messenger(data_dir / "shared", "sakura")
        messages = messenger.receive()

        assert len(messages) == 1
        assert messages[0].content == "Readable"
        assert messages[0].source == "slack"


class TestSocketModeConfigE2E:
    """E2E: config.json mode field integration."""

    async def test_webhook_mode_does_not_connect(self, data_dir, monkeypatch):
        """When mode='webhook', Socket Mode is not started."""
        config = AnimaWorksConfig.model_validate(
            json.loads((data_dir / "config.json").read_text(encoding="utf-8")),
        )
        config.external_messaging = ExternalMessagingConfig(
            slack=ExternalMessagingChannelConfig(
                enabled=True,
                mode="webhook",
                anima_mapping={"C1": "sakura"},
            ),
        )
        save_config(config, data_dir / "config.json")

        from core.config.models import invalidate_cache
        invalidate_cache()

        from server.slack_socket import SlackSocketModeManager

        connect_called = False
        original_connect = AsyncMock()

        monkeypatch.setattr(
            "server.slack_socket.get_credential",
            lambda name, tool, **kw: "token",
        )
        monkeypatch.setattr(
            "server.slack_socket.AsyncApp",
            lambda **kw: MagicMock(),
        )
        monkeypatch.setattr(
            "server.slack_socket.AsyncSocketModeHandler",
            lambda app, token: original_connect,
        )

        mgr = SlackSocketModeManager()
        await mgr.start()

        assert not mgr.is_connected

    def test_config_mode_field_persisted(self, data_dir):
        """mode field round-trips through save/load."""
        config = AnimaWorksConfig.model_validate(
            json.loads((data_dir / "config.json").read_text(encoding="utf-8")),
        )
        config.external_messaging = ExternalMessagingConfig(
            slack=ExternalMessagingChannelConfig(
                enabled=True,
                mode="socket",
                anima_mapping={"C1": "sakura"},
            ),
        )
        save_config(config, data_dir / "config.json")

        from core.config.models import invalidate_cache
        invalidate_cache()

        raw = json.loads((data_dir / "config.json").read_text(encoding="utf-8"))
        assert raw["external_messaging"]["slack"]["mode"] == "socket"
