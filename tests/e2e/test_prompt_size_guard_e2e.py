# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
"""E2E test: prompt size guard prevents Agent SDK buffer overflow.

Verifies that large messages and accumulated conversation history
do not crash the Agent SDK by testing the complete flow through
ConversationMemory, prompt building, and pre-flight size checks.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.memory.conversation import (
    ConversationMemory,
    _MAX_HUMAN_CHARS_IN_HISTORY,
    _MAX_STORED_CONTENT_CHARS,
)
from core.schemas import ModelConfig


@pytest.fixture
def anima_dir(tmp_path):
    """Create a minimal anima directory structure."""
    d = tmp_path / "animas" / "yuki"
    d.mkdir(parents=True)
    (d / "state").mkdir()
    (d / "episodes").mkdir()
    (d / "knowledge").mkdir()
    (d / "procedures").mkdir()
    (d / "skills").mkdir()
    (d / "shortterm").mkdir()
    (d / "activity_log").mkdir()
    return d


class TestLargeMessageHandling:
    """A message with 5000+ chars should not corrupt conversation state."""

    def test_large_human_message_stored_truncated(self, anima_dir):
        """Large human message is truncated at storage time."""
        mem = ConversationMemory(anima_dir, ModelConfig())
        big_msg = "AWS log output: " + "ERROR something " * 500  # ~8000 chars
        mem.append_turn("human", big_msg)
        mem.save()

        # Reload from disk to verify persistence
        mem2 = ConversationMemory(anima_dir, ModelConfig())
        state = mem2.load()
        assert len(state.turns) == 1
        assert len(state.turns[0].content) <= _MAX_STORED_CONTENT_CHARS + 100  # +margin for truncation notice
        assert "[...truncated" in state.turns[0].content

    def test_large_human_message_truncated_in_history(self, anima_dir):
        """Large human message in history display is capped at 800 chars."""
        mem = ConversationMemory(anima_dir, ModelConfig())
        big_msg = "x" * 2000
        mem.append_turn("human", big_msg)
        state = mem.load()
        history = mem._format_history(state)
        # History should contain the truncated version
        assert "..." in history
        # Full 2000-char string should NOT appear
        assert big_msg not in history

    def test_conversation_json_size_bounded(self, anima_dir):
        """50 turns of conversation should produce bounded JSON size."""
        mem = ConversationMemory(anima_dir, ModelConfig())

        for i in range(25):
            mem.append_turn("human", f"Question {i}: " + "detail " * 100)
            mem.append_turn("assistant", f"Answer {i}: " + "explanation " * 200)

        mem.save()

        # conversation.json should not exceed a reasonable size
        conv_path = anima_dir / "state" / "conversation.json"
        size = conv_path.stat().st_size
        # 50 turns * 5000 chars max + JSON overhead = ~280KB max
        # Without truncation this would be ~400KB+
        assert size < 350_000, f"conversation.json too large: {size} bytes"

    def test_history_with_mixed_lengths(self, anima_dir):
        """Mix of short and long messages produces bounded history."""
        mem = ConversationMemory(anima_dir, ModelConfig())

        # Short messages
        mem.append_turn("human", "Hi")
        mem.append_turn("assistant", "Hello!")

        # Long message (command output paste)
        mem.append_turn("human", "$ aws logs filter-log-events ...\n" + "log line\n" * 500)
        mem.append_turn("assistant", "I see the error is on line 42." + " detail" * 300)

        state = mem.load()
        history = mem._format_history(state)

        # History should be bounded
        assert len(history) < 10_000, f"History too long: {len(history)} chars"


class TestPreflightConstants:
    """Pre-flight size check constants are properly defined."""

    def test_soft_limit_exists(self):
        from core.agent import _PROMPT_SOFT_LIMIT_BYTES
        assert _PROMPT_SOFT_LIMIT_BYTES > 0

    def test_hard_limit_exists(self):
        from core.agent import _PROMPT_HARD_LIMIT_BYTES
        assert _PROMPT_HARD_LIMIT_BYTES > 0

    def test_sdk_buffer_size(self):
        from core.execution.agent_sdk import _SDK_MAX_BUFFER_SIZE
        assert _SDK_MAX_BUFFER_SIZE == 4 * 1024 * 1024

    def test_limits_ordering(self):
        """Soft < Hard < SDK buffer."""
        from core.agent import _PROMPT_SOFT_LIMIT_BYTES, _PROMPT_HARD_LIMIT_BYTES
        from core.execution.agent_sdk import _SDK_MAX_BUFFER_SIZE
        assert _PROMPT_SOFT_LIMIT_BYTES < _PROMPT_HARD_LIMIT_BYTES < _SDK_MAX_BUFFER_SIZE


class TestStderrLogRotation:
    """stderr.log rotation on process start."""

    def test_rotation_preserves_content(self, tmp_path):
        """Rotated file should contain the original content."""
        stderr_dir = tmp_path / "logs" / "animas" / "test"
        stderr_dir.mkdir(parents=True)
        stderr_log = stderr_dir / "stderr.log"
        backup = stderr_dir / "stderr.log.1"

        # Create 6MB file with marker
        content = b"MARKER" + b"x" * (6 * 1024 * 1024)
        stderr_log.write_bytes(content)

        # Simulate rotation
        if stderr_log.stat().st_size > 5 * 1024 * 1024:
            if backup.exists():
                backup.unlink()
            stderr_log.rename(backup)

        assert backup.exists()
        assert backup.read_bytes().startswith(b"MARKER")
        assert not stderr_log.exists()
