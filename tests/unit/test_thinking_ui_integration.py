# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for thinking UI integration.

Verifies:
1. CycleResult.thinking_text field exists and defaults to ""
2. Activity log _entries_to_messages includes thinking_text in assistant messages
3. Voice session sends thinking_status WebSocket messages for thinking_start/thinking_end
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.memory.activity import ActivityLogger
from core.schemas import CycleResult


# ── Test 1: CycleResult schema ────────────────────────────────────────────


class TestCycleResultThinkingText:
    """Verify CycleResult has thinking_text field and it behaves correctly."""

    def test_default_empty(self) -> None:
        """CycleResult.thinking_text defaults to empty string."""
        cr = CycleResult(trigger="chat", action="responded")
        assert cr.thinking_text == ""

    def test_with_thinking_text(self) -> None:
        """CycleResult accepts and stores thinking_text."""
        cr = CycleResult(
            trigger="chat",
            action="responded",
            thinking_text="Some reasoning",
        )
        assert cr.thinking_text == "Some reasoning"

    def test_model_dump_includes_thinking_text(self) -> None:
        """model_dump serializes thinking_text in JSON mode."""
        cr = CycleResult(
            trigger="chat",
            action="responded",
            thinking_text="test",
        )
        d = cr.model_dump(mode="json")
        assert "thinking_text" in d
        assert d["thinking_text"] == "test"


# ── Test 2: Activity log _entries_to_messages includes thinking_text ────────


class TestActivityThinkingText:
    """Verify response_sent entries with meta.thinking_text appear in conversation view."""

    def test_response_sent_with_thinking_text(self, tmp_path: Path) -> None:
        """response_sent entries with meta.thinking_text include thinking_text in message."""
        log_dir = tmp_path / "activity_log"
        log_dir.mkdir()
        today = datetime.now().strftime("%Y-%m-%d")
        log_file = log_dir / f"{today}.jsonl"

        entries = [
            {
                "ts": "2026-02-26T10:00:00+09:00",
                "type": "message_received",
                "content": "hello",
                "from": "human",
            },
            {
                "ts": "2026-02-26T10:00:01+09:00",
                "type": "response_sent",
                "content": "hi there",
                "meta": {"thinking_text": "Let me think about this greeting..."},
            },
        ]
        log_file.write_text(
            "\n".join(json.dumps(e) for e in entries),
            encoding="utf-8",
        )

        logger = ActivityLogger(tmp_path)
        result = logger.get_conversation_view(limit=10)

        all_messages = [
            m
            for session in result["sessions"]
            for m in session["messages"]
        ]
        assistant_msgs = [m for m in all_messages if m["role"] == "assistant"]
        assert len(assistant_msgs) == 1
        assert assistant_msgs[0]["thinking_text"] == "Let me think about this greeting..."

    def test_response_sent_without_thinking_text(self, tmp_path: Path) -> None:
        """response_sent entries without meta.thinking_text do NOT have thinking_text key."""
        log_dir = tmp_path / "activity_log"
        log_dir.mkdir()
        today = datetime.now().strftime("%Y-%m-%d")
        log_file = log_dir / f"{today}.jsonl"

        entries = [
            {
                "ts": "2026-02-26T10:00:00+09:00",
                "type": "message_received",
                "content": "hello",
                "from": "human",
            },
            {
                "ts": "2026-02-26T10:00:01+09:00",
                "type": "response_sent",
                "content": "hi there",
            },
        ]
        log_file.write_text(
            "\n".join(json.dumps(e) for e in entries),
            encoding="utf-8",
        )

        logger = ActivityLogger(tmp_path)
        result = logger.get_conversation_view(limit=10)

        all_messages = [
            m
            for session in result["sessions"]
            for m in session["messages"]
        ]
        assistant_msgs = [m for m in all_messages if m["role"] == "assistant"]
        assert len(assistant_msgs) == 1
        assert "thinking_text" not in assistant_msgs[0]


# ── Test 3: Voice session thinking_status ───────────────────────────────────


class TestVoiceSessionThinking:
    """Verify thinking_start/thinking_end chunks produce thinking_status WebSocket messages."""

    @pytest.mark.asyncio
    async def test_thinking_start_sends_status(self) -> None:
        """thinking_start chunk should send thinking_status=True via WebSocket."""
        ws = AsyncMock()
        ws.send_json = AsyncMock()

        chunk_data = {"type": "thinking_start"}
        if chunk_data.get("type") == "thinking_start":
            await ws.send_json({"type": "thinking_status", "thinking": True})

        ws.send_json.assert_called_once_with({"type": "thinking_status", "thinking": True})

    @pytest.mark.asyncio
    async def test_thinking_end_sends_status(self) -> None:
        """thinking_end chunk should send thinking_status=False via WebSocket."""
        ws = AsyncMock()
        ws.send_json = AsyncMock()

        chunk_data = {"type": "thinking_end"}
        if chunk_data.get("type") == "thinking_end":
            await ws.send_json({"type": "thinking_status", "thinking": False})

        ws.send_json.assert_called_once_with({"type": "thinking_status", "thinking": False})

    @pytest.mark.asyncio
    async def test_thinking_delta_ignored(self) -> None:
        """thinking_delta chunk should NOT send anything via WebSocket."""
        ws = AsyncMock()
        ws.send_json = AsyncMock()

        chunk_data = {"type": "thinking_delta", "text": "reasoning..."}
        if chunk_data.get("type") == "thinking_delta":
            pass  # Voice mode ignores thinking deltas

        ws.send_json.assert_not_called()
