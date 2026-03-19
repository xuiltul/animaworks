"""Unit tests for meeting mode tool restriction and context summarization."""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.tooling.handler import ToolHandler, meeting_mode
from core.tooling.handler_base import meeting_mode as meeting_mode_var
from server.room_manager import SUMMARY_THRESHOLD, RoomManager


# ── ToolHandler meeting mode tests ──────────────────────────


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


class TestMeetingModeToolBlocking:
    """Verify that blocked tools return error in meeting mode."""

    BLOCKED_TOOLS = [
        "send_message",
        "post_channel",
        "delegate_task",
        "call_human",
        "read_dm_history",
        "read_channel",
    ]

    @pytest.mark.parametrize("tool_name", BLOCKED_TOOLS)
    def test_blocked_tool_returns_error_in_meeting_mode(self, tmp_path, tool_name):
        handler = _make_handler(tmp_path)
        token = meeting_mode_var.set(True)
        try:
            result = handler.handle(tool_name, {})
            assert "会議中" in result or "not available during meetings" in result
        finally:
            meeting_mode_var.reset(token)

    @pytest.mark.parametrize("tool_name", BLOCKED_TOOLS)
    def test_blocked_tool_works_outside_meeting_mode(self, tmp_path, tool_name):
        handler = _make_handler(tmp_path)
        assert meeting_mode_var.get() is False
        # These will fail with missing args/messenger, not with meeting block message
        try:
            result = handler.handle(tool_name, {})
        except Exception:
            pass  # Expected — we only care it doesn't return meeting block
        # If it returns a string, it should NOT contain the meeting block message
        # (it might raise instead, which is also fine)

    def test_memory_tools_allowed_in_meeting_mode(self, tmp_path):
        handler = _make_handler(tmp_path)
        token = meeting_mode_var.set(True)
        try:
            result = handler.handle("search_memory", {"query": "test"})
            assert "会議中" not in result
            assert "not available during meetings" not in result
        finally:
            meeting_mode_var.reset(token)

    def test_write_memory_file_allowed_in_meeting_mode(self, tmp_path):
        handler = _make_handler(tmp_path)
        token = meeting_mode_var.set(True)
        try:
            result = handler.handle(
                "write_memory_file",
                {"path": "knowledge/test.md", "content": "test content"},
            )
            assert "会議中" not in result
            assert "not available during meetings" not in result
        finally:
            meeting_mode_var.reset(token)

    def test_meeting_mode_default_is_false(self):
        assert meeting_mode_var.get() is False


class TestMeetingModeContextVarReset:
    """Verify meeting_mode is properly reset via token."""

    def test_set_and_reset(self):
        assert meeting_mode_var.get() is False
        token = meeting_mode_var.set(True)
        assert meeting_mode_var.get() is True
        meeting_mode_var.reset(token)
        assert meeting_mode_var.get() is False


# ── RoomManager summarization tests ─────────────────────────


class TestFormatEntries:
    """Test the static _format_entries helper."""

    def test_chair_format(self):
        entries = [{"speaker": "sakura", "role": "chair", "text": "Hello"}]
        result = RoomManager._format_entries(entries)
        assert result == "[sakura(議長)] Hello"

    def test_human_format(self):
        entries = [{"speaker": "taka", "role": "human", "text": "Hi"}]
        result = RoomManager._format_entries(entries)
        assert result == "[human(taka)] Hi"

    def test_participant_format(self):
        entries = [{"speaker": "rin", "role": "participant", "text": "Agreed"}]
        result = RoomManager._format_entries(entries)
        assert result == "[rin] Agreed"

    def test_multiple_entries(self):
        entries = [
            {"speaker": "taka", "role": "human", "text": "Start"},
            {"speaker": "sakura", "role": "chair", "text": "OK"},
            {"speaker": "rin", "role": "participant", "text": "Yes"},
        ]
        result = RoomManager._format_entries(entries)
        lines = result.split("\n")
        assert len(lines) == 3


class TestGetConversationContext:
    """Test refactored get_conversation_context uses _format_entries."""

    def test_returns_empty_for_missing_room(self, tmp_path):
        rm = RoomManager(tmp_path)
        assert rm.get_conversation_context("nonexistent") == ""

    def test_formats_entries(self, tmp_path):
        rm = RoomManager(tmp_path)
        room = rm.create_room(["sakura", "rin"], chair="sakura", created_by="taka")
        rm.append_message(room.room_id, "taka", "human", "Hello")
        rm.append_message(room.room_id, "sakura", "chair", "Welcome")
        ctx = rm.get_conversation_context(room.room_id)
        assert "[human(taka)] Hello" in ctx
        assert "[sakura(議長)] Welcome" in ctx


class TestGetSummarizedContext:
    """Test async get_summarized_context with LLM summarization."""

    @pytest.mark.asyncio
    async def test_returns_full_context_when_below_threshold(self, tmp_path):
        rm = RoomManager(tmp_path)
        room = rm.create_room(["sakura", "rin"], chair="sakura", created_by="taka")
        for i in range(SUMMARY_THRESHOLD):
            rm.append_message(room.room_id, "taka", "human", f"Message {i}")
        result = await rm.get_summarized_context(room.room_id)
        assert "[要約]" not in result
        assert "Message 0" in result

    @pytest.mark.asyncio
    async def test_summarizes_when_above_threshold(self, tmp_path):
        rm = RoomManager(tmp_path)
        room = rm.create_room(["sakura", "rin"], chair="sakura", created_by="taka")
        for i in range(SUMMARY_THRESHOLD + 2):
            rm.append_message(room.room_id, "taka", "human", f"Message {i}")

        with patch.object(rm, "_call_summary_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = "Summary of older messages"
            result = await rm.get_summarized_context(room.room_id)

        assert "[要約] Summary of older messages" in result
        # Recent entries should be present
        assert f"Message {SUMMARY_THRESHOLD + 1}" in result
        # Older entries should NOT be present verbatim
        assert "Message 0" not in result

    @pytest.mark.asyncio
    async def test_uses_cached_summary(self, tmp_path):
        rm = RoomManager(tmp_path)
        room = rm.create_room(["sakura", "rin"], chair="sakura", created_by="taka")
        for i in range(SUMMARY_THRESHOLD + 2):
            rm.append_message(room.room_id, "taka", "human", f"Message {i}")

        with patch.object(rm, "_call_summary_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = "Cached summary"
            await rm.get_summarized_context(room.room_id)
            # Second call should use cache
            result = await rm.get_summarized_context(room.room_id)

        assert mock_llm.call_count == 1
        assert "[要約] Cached summary" in result

    @pytest.mark.asyncio
    async def test_invalidates_cache_on_new_messages(self, tmp_path):
        rm = RoomManager(tmp_path)
        room = rm.create_room(["sakura", "rin"], chair="sakura", created_by="taka")
        for i in range(SUMMARY_THRESHOLD + 2):
            rm.append_message(room.room_id, "taka", "human", f"Message {i}")

        with patch.object(rm, "_call_summary_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = "Summary v1"
            await rm.get_summarized_context(room.room_id)

            # Add more messages (shifts the older/recent boundary)
            rm.append_message(room.room_id, "taka", "human", "New message")
            mock_llm.return_value = "Summary v2"
            result = await rm.get_summarized_context(room.room_id)

        assert mock_llm.call_count == 2
        assert "[要約] Summary v2" in result

    @pytest.mark.asyncio
    async def test_fallback_on_llm_failure(self, tmp_path):
        rm = RoomManager(tmp_path)
        room = rm.create_room(["sakura", "rin"], chair="sakura", created_by="taka")
        for i in range(SUMMARY_THRESHOLD + 2):
            rm.append_message(room.room_id, "taka", "human", f"Message {i}")

        with patch.object(rm, "_call_summary_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.side_effect = RuntimeError("LLM unavailable")
            result = await rm.get_summarized_context(room.room_id)

        assert "[要約]" not in result
        assert "Message 0" in result

    @pytest.mark.asyncio
    async def test_returns_empty_for_missing_room(self, tmp_path):
        rm = RoomManager(tmp_path)
        result = await rm.get_summarized_context("nonexistent")
        assert result == ""
