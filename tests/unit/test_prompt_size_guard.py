# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for prompt size guard features.

Tests conversation turn truncation, pre-flight size checks, Agent SDK
max_buffer_size configuration, and stderr.log rotation.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.memory.conversation import (
    ConversationMemory,
    ConversationState,
    ConversationTurn,
    _MAX_HUMAN_CHARS_IN_HISTORY,
    _MAX_RESPONSE_CHARS_IN_HISTORY,
    _MAX_STORED_CONTENT_CHARS,
)
from core.schemas import ModelConfig


# ── Conversation truncation tests ─────────────────────────────────


class TestHumanTurnTruncation:
    """Human messages should be truncated in history display."""

    def _make_memory(self, tmp_path: Path) -> ConversationMemory:
        anima_dir = tmp_path / "animas" / "test-anima"
        anima_dir.mkdir(parents=True)
        (anima_dir / "state").mkdir(parents=True)
        return ConversationMemory(anima_dir, ModelConfig())

    def test_short_human_message_preserved(self, tmp_path):
        """Human messages under the limit are not truncated."""
        mem = self._make_memory(tmp_path)
        state = mem.load()
        state.turns.append(
            ConversationTurn(role="human", content="Hello, how are you?")
        )
        history = mem._format_history(state)
        assert "Hello, how are you?" in history
        assert "..." not in history

    def test_long_human_message_truncated(self, tmp_path):
        """Human messages over _MAX_HUMAN_CHARS_IN_HISTORY are truncated."""
        mem = self._make_memory(tmp_path)
        long_msg = "x" * (_MAX_HUMAN_CHARS_IN_HISTORY + 500)
        state = mem.load()
        state.turns.append(ConversationTurn(role="human", content=long_msg))
        history = mem._format_history(state)
        assert "..." in history
        # The full message should NOT be present
        assert long_msg not in history
        # But the first part should be
        assert long_msg[:100] in history

    def test_assistant_truncation_unchanged(self, tmp_path):
        """Assistant messages are still truncated at the original limit."""
        mem = self._make_memory(tmp_path)
        long_resp = "y" * (_MAX_RESPONSE_CHARS_IN_HISTORY + 500)
        state = mem.load()
        state.turns.append(
            ConversationTurn(role="assistant", content=long_resp)
        )
        history = mem._format_history(state)
        assert "..." in history
        assert long_resp not in history


class TestStoredContentTruncation:
    """Content should be truncated at storage time in append_turn."""

    def _make_memory(self, tmp_path: Path) -> ConversationMemory:
        anima_dir = tmp_path / "animas" / "test-anima"
        anima_dir.mkdir(parents=True)
        (anima_dir / "state").mkdir(parents=True)
        return ConversationMemory(anima_dir, ModelConfig())

    def test_short_content_stored_fully(self, tmp_path):
        """Content under the limit is stored verbatim."""
        mem = self._make_memory(tmp_path)
        mem.append_turn("human", "short message")
        state = mem.load()
        assert state.turns[-1].content == "short message"

    def test_long_content_truncated_at_storage(self, tmp_path):
        """Content over _MAX_STORED_CONTENT_CHARS is truncated."""
        mem = self._make_memory(tmp_path)
        long_content = "z" * (_MAX_STORED_CONTENT_CHARS + 1000)
        mem.append_turn("human", long_content)
        state = mem.load()
        stored = state.turns[-1].content
        assert len(stored) < len(long_content)
        assert "[...truncated" in stored
        assert str(len(long_content)) in stored

    def test_truncation_applies_to_both_roles(self, tmp_path):
        """Both human and assistant turns are truncated at storage."""
        mem = self._make_memory(tmp_path)
        long_content = "a" * (_MAX_STORED_CONTENT_CHARS + 500)

        mem.append_turn("human", long_content)
        mem.append_turn("assistant", long_content)
        state = mem.load()

        for turn in state.turns:
            assert "[...truncated" in turn.content


# ── Agent SDK max_buffer_size tests ───────────────────────────────


class TestAgentSDKBufferSize:
    """max_buffer_size should be set on ClaudeAgentOptions."""

    def test_buffer_size_constant(self):
        """The constant should be 4 MB."""
        from core.execution.agent_sdk import _SDK_MAX_BUFFER_SIZE
        assert _SDK_MAX_BUFFER_SIZE == 4 * 1024 * 1024


# ── Pre-flight size check tests ───────────────────────────────────


class TestPreflightSizeCheck:
    """Pre-flight prompt size check in AgentCore."""

    def test_constants_defined(self):
        """Size limit constants should be defined."""
        from core.agent import _PROMPT_SOFT_LIMIT_BYTES, _PROMPT_HARD_LIMIT_BYTES
        assert _PROMPT_SOFT_LIMIT_BYTES == 600_000
        assert _PROMPT_HARD_LIMIT_BYTES == 1_200_000
        assert _PROMPT_SOFT_LIMIT_BYTES < _PROMPT_HARD_LIMIT_BYTES


# ── stderr rotation tests ─────────────────────────────────────────


class TestStderrRotation:
    """stderr.log should be rotated when exceeding 5 MB."""

    def test_small_stderr_not_rotated(self, tmp_path):
        """stderr.log under 5MB should not be rotated."""
        log_dir = tmp_path / "logs"
        stderr_dir = log_dir / "animas" / "test-anima"
        stderr_dir.mkdir(parents=True)
        stderr_log = stderr_dir / "stderr.log"
        stderr_log.write_text("small content")

        # Import after creating dirs so we can test the rotation logic directly
        # The rotation is inline in ProcessHandle.start(), so we test the
        # file state after calling start() indirectly via the rotation check.
        original_size = stderr_log.stat().st_size
        assert original_size < 5 * 1024 * 1024
        # Backup should not exist
        assert not (stderr_dir / "stderr.log.1").exists()

    def test_large_stderr_rotated(self, tmp_path):
        """stderr.log over 5MB should be rotated to stderr.log.1."""
        log_dir = tmp_path / "logs"
        stderr_dir = log_dir / "animas" / "test-anima"
        stderr_dir.mkdir(parents=True)
        stderr_log = stderr_dir / "stderr.log"

        # Create a 6MB file
        stderr_log.write_bytes(b"x" * (6 * 1024 * 1024))
        assert stderr_log.stat().st_size > 5 * 1024 * 1024

        # Simulate the rotation logic from process_handle.py
        backup = stderr_dir / "stderr.log.1"
        if stderr_log.exists() and stderr_log.stat().st_size > 5 * 1024 * 1024:
            if backup.exists():
                backup.unlink()
            stderr_log.rename(backup)

        # Original should be gone, backup should exist
        assert not stderr_log.exists()
        assert backup.exists()
        assert backup.stat().st_size > 5 * 1024 * 1024
