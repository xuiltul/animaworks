from __future__ import annotations
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.

"""Tests for error visibility improvements in ToolHandler and outbound.

Covers:
1. Change 1: resolve_recipient exception -> RecipientResolutionError JSON
3. Change 3: send_external empty channel early return -> NoChannelConfigured
5. Change 5: _log_tool_activity logs warnings instead of silently passing
"""

import json
import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ── Change 1: resolve_recipient exception -> RecipientResolutionError ──


class TestResolveRecipientError:
    """Test that resolve_recipient exceptions return error JSON to LLM."""

    def _make_handler(self, tmp_path: Path):
        """Create a ToolHandler with minimal mocks."""
        from core.tooling.handler import ToolHandler

        memory = MagicMock()
        messenger = MagicMock()
        messenger.anima_name = "test_anima"
        messenger.send.return_value = MagicMock(id="msg1", thread_id="t1")

        handler = ToolHandler(
            anima_dir=tmp_path,
            memory=memory,
            messenger=messenger,
        )
        return handler

    @patch("core.config.models.load_config")
    @patch("core.paths.get_animas_dir")
    @patch("core.outbound.resolve_recipient")
    def test_resolve_recipient_unexpected_exception_returns_error_json(
        self, mock_resolve, mock_animas_dir, mock_config, tmp_path
    ):
        """When resolve_recipient raises an unexpected Exception (not ValueError),
        the handler should return a RecipientResolutionError JSON instead of
        silently falling back to internal routing."""
        mock_animas_dir.return_value = tmp_path / "animas"
        (tmp_path / "animas").mkdir()
        mock_config.return_value = MagicMock()
        mock_resolve.side_effect = RuntimeError("Unexpected DB error")

        handler = self._make_handler(tmp_path)
        result = handler._handle_send_message({"to": "someone", "content": "hello"})

        parsed = json.loads(result)
        assert parsed["status"] == "error"
        assert parsed["error_type"] == "RecipientResolutionError"
        assert "someone" in parsed["message"]
        assert "suggestion" in parsed

    @patch("core.config.models.load_config")
    @patch("core.paths.get_animas_dir")
    @patch("core.outbound.resolve_recipient")
    def test_resolve_recipient_value_error_returns_unknown_recipient(
        self, mock_resolve, mock_animas_dir, mock_config, tmp_path
    ):
        """ValueError from resolve_recipient should return UnknownRecipient error."""
        mock_animas_dir.return_value = tmp_path / "animas"
        (tmp_path / "animas").mkdir()
        mock_config.return_value = MagicMock()
        mock_resolve.side_effect = ValueError("Unknown recipient 'ghost'")

        handler = self._make_handler(tmp_path)
        result = handler._handle_send_message({"to": "ghost", "content": "hello"})

        parsed = json.loads(result)
        assert parsed["status"] == "error"
        assert parsed["error_type"] == "UnknownRecipient"


# ── Change 3: send_external empty channel early return ──


class TestSendExternalEmptyChannel:
    """Test that send_external returns an error when no valid channels are configured."""

    def test_empty_channel_no_ids_returns_delivery_failed(self):
        """When channel is empty and no slack/chatwork IDs are set,
        send_external should return a DeliveryFailed error since no
        valid channel can be used for delivery."""
        from core.outbound import send_external, ResolvedRecipient

        resolved = ResolvedRecipient(
            is_internal=False,
            name="admin",
            channel="",
            slack_user_id="",
            chatwork_room_id="",
        )
        result = send_external(resolved, "hello", sender_name="test")
        parsed = json.loads(result)
        assert parsed["status"] == "error"
        assert parsed["error_type"] == "DeliveryFailed"
        assert "admin" in parsed["message"]

    @patch("core.outbound._build_channel_order", return_value=[])
    def test_no_channels_returns_no_channel_configured(self, mock_build):
        """When _build_channel_order returns an empty list,
        send_external should return NoChannelConfigured error."""
        from core.outbound import send_external, ResolvedRecipient

        resolved = ResolvedRecipient(
            is_internal=False,
            name="admin",
            channel="",
            slack_user_id="",
            chatwork_room_id="",
        )
        result = send_external(resolved, "hello", sender_name="test")
        parsed = json.loads(result)
        assert parsed["status"] == "error"
        assert parsed["error_type"] == "NoChannelConfigured"
        assert "admin" in parsed["message"]


# ── Change 5: _log_tool_activity logs warnings ──


class TestLogToolActivityWarning:
    """Test that _log_tool_activity logs warnings instead of silently passing."""

    def test_activity_logging_failure_logs_warning(self, tmp_path, caplog):
        from core.tooling.handler import ToolHandler

        memory = MagicMock()
        handler = ToolHandler(anima_dir=tmp_path, memory=memory)

        handler._activity = MagicMock()
        handler._activity.log = MagicMock(side_effect=RuntimeError("DB error"))
        with caplog.at_level(logging.WARNING, logger="animaworks.tool_handler"):
            handler._log_tool_activity("test_tool", {"key": "value"})

        assert any("Activity logging failed" in r.message for r in caplog.records)
