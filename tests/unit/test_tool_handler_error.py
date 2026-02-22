"""Unit tests for ToolHandler top-level catch, output truncation, and depth-limit error handling."""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from core.schemas import Message
from core.tooling.handler import ToolHandler


def _make_handler(tmp_path: Path) -> ToolHandler:
    """Create a ToolHandler with minimal mocked dependencies."""
    anima_dir = tmp_path / "animas" / "test"
    anima_dir.mkdir(parents=True)
    (anima_dir / "permissions.md").write_text("", encoding="utf-8")

    memory = MagicMock()
    memory.read_permissions.return_value = ""
    memory.search_memory_text.return_value = []

    handler = ToolHandler(
        anima_dir=anima_dir,
        memory=memory,
        messenger=None,
        tool_registry=[],
    )
    return handler


class TestToolHandlerTopLevelCatch:
    def test_unhandled_exception_returns_error_string(self, tmp_path):
        handler = _make_handler(tmp_path)
        # Patch the dispatch dict entry directly (bound method refs are
        # captured at __init__ time, so patch.object on the instance
        # attribute does not affect the dict).
        handler._dispatch["search_memory"] = MagicMock(side_effect=RuntimeError("boom"))
        result = handler.handle("search_memory", {"query": "test"})
        assert "Tool execution failed" in result
        assert "search_memory" in result
        assert "boom" in result


class TestToolHandlerDepthLimitError:
    """Test that _handle_send_message checks for depth-limit error from send()."""

    def test_depth_limit_error_returns_error_string(self, tmp_path):
        """When send() returns type='error', handler should return Error string."""
        handler = _make_handler(tmp_path)

        # Create a messenger mock that returns an error Message (depth limit exceeded)
        messenger = MagicMock()
        messenger.anima_name = "test"
        error_msg = Message(
            from_person="system",
            to_person="test",
            type="error",
            content="ConversationDepthExceeded: bob",
        )
        messenger.send.return_value = error_msg
        handler._messenger = messenger

        # Mock config and resolve_recipient to reach internal send path
        mock_config = MagicMock()
        mock_config.external_messaging = MagicMock()
        with (
            patch("core.config.models.load_config", return_value=mock_config),
            patch("core.paths.get_animas_dir", return_value=tmp_path / "animas"),
            patch("core.outbound.resolve_recipient", return_value=None),
        ):
            result = handler._handle_send_message({"to": "bob", "content": "hello", "intent": "report"})

        assert "Error:" in result
        assert "ConversationDepthExceeded" in result

    def test_depth_limit_error_does_not_track_replied_to(self, tmp_path):
        """When send() is blocked, replied_to should NOT be updated."""
        handler = _make_handler(tmp_path)

        messenger = MagicMock()
        messenger.anima_name = "test"
        error_msg = Message(
            from_person="system",
            to_person="test",
            type="error",
            content="ConversationDepthExceeded: bob",
        )
        messenger.send.return_value = error_msg
        handler._messenger = messenger

        mock_config = MagicMock()
        mock_config.external_messaging = MagicMock()
        with (
            patch("core.config.models.load_config", return_value=mock_config),
            patch("core.paths.get_animas_dir", return_value=tmp_path / "animas"),
            patch("core.outbound.resolve_recipient", return_value=None),
        ):
            handler._handle_send_message({"to": "bob", "content": "hello", "intent": "report"})

        assert "bob" not in handler._replied_to

    def test_successful_send_still_tracks_replied_to(self, tmp_path):
        """Normal send should still track replied_to as before."""
        handler = _make_handler(tmp_path)

        messenger = MagicMock()
        messenger.anima_name = "test"
        success_msg = Message(
            from_person="test",
            to_person="bob",
            type="message",
            content="hello",
        )
        messenger.send.return_value = success_msg
        handler._messenger = messenger

        mock_config = MagicMock()
        mock_config.external_messaging = MagicMock()
        with (
            patch("core.config.models.load_config", return_value=mock_config),
            patch("core.paths.get_animas_dir", return_value=tmp_path / "animas"),
            patch("core.outbound.resolve_recipient", return_value=None),
        ):
            result = handler._handle_send_message({"to": "bob", "content": "hello", "intent": "report"})

        assert "bob" in handler._replied_to
        assert "Message sent to bob" in result


class TestToolOutputTruncation:
    def test_output_within_limit_not_truncated(self, tmp_path):
        handler = _make_handler(tmp_path)
        short_output = "a" * 1000
        result = handler._truncate_output(short_output)
        assert result == short_output

    def test_output_over_limit_is_truncated(self, tmp_path):
        handler = _make_handler(tmp_path)
        large_output = "a" * 600_000
        result = handler._truncate_output(large_output)
        assert len(result.encode("utf-8")) < len(large_output.encode("utf-8"))

    def test_truncated_output_includes_message(self, tmp_path):
        handler = _make_handler(tmp_path)
        large_output = "a" * 600_000
        result = handler._truncate_output(large_output)
        assert "トランケーション" in result
        assert "500KB" in result
        assert "600000" in result
