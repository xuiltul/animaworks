"""Tests for agent_sdk tool record helper functions.

Tests for _handle_tool_use_block(), _handle_tool_result_block(),
and _finalize_pending_records() extracted from core.execution.agent_sdk.
"""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from core.execution.agent_sdk import (
    _finalize_pending_records,
    _handle_tool_result_block,
    _handle_tool_use_block,
)
from core.execution.base import ToolCallRecord


# ── Fake block types for testing ─────────────────────────────────


class FakeToolUseBlock:
    """Minimal stand-in for claude_agent_sdk.ToolUseBlock."""

    def __init__(self, name: str, id: str, input: str = "") -> None:
        self.name = name
        self.id = id
        self.input = input


class FakeToolResultBlock:
    """Minimal stand-in for claude_agent_sdk.ToolResultBlock."""

    def __init__(
        self,
        tool_use_id: str,
        content: str | list = "",
        is_error: bool | None = None,
    ) -> None:
        self.tool_use_id = tool_use_id
        self.content = content
        self.is_error = is_error


# ── _handle_tool_use_block ───────────────────────────────────────


class TestHandleToolUseBlock:
    """Test _handle_tool_use_block() record creation and registration."""

    MODEL = "claude-sonnet-4-20250514"

    def test_creates_record_with_correct_fields(self):
        """Record should have correct tool_name, tool_id, and input_summary."""
        pending: dict[str, ToolCallRecord] = {}
        block = FakeToolUseBlock(name="web_search", id="toolu_abc", input="query=test")

        record = _handle_tool_use_block(block, pending, None, self.MODEL)

        assert record.tool_name == "web_search"
        assert record.tool_id == "toolu_abc"
        assert "query=test" in record.input_summary
        assert record.result_summary == ""
        assert record.is_error is False

    def test_registers_record_in_pending_dict(self):
        """The record should be stored in pending_records keyed by block.id."""
        pending: dict[str, ToolCallRecord] = {}
        block = FakeToolUseBlock(name="Bash", id="toolu_xyz")

        record = _handle_tool_use_block(block, pending, None, self.MODEL)

        assert "toolu_xyz" in pending
        assert pending["toolu_xyz"] is record

    def test_writes_to_journal_when_provided(self):
        """Should call journal.write_tool_start when journal is provided."""
        pending: dict[str, ToolCallRecord] = {}
        block = FakeToolUseBlock(name="Read", id="toolu_j1", input="/path/to/file")
        mock_journal = MagicMock()

        _handle_tool_use_block(block, pending, mock_journal, self.MODEL)

        mock_journal.write_tool_start.assert_called_once()
        call_args = mock_journal.write_tool_start.call_args
        assert call_args[0][0] == "Read"  # tool name
        assert call_args[1]["tool_id"] == "toolu_j1"

    def test_does_not_write_to_journal_when_none(self):
        """Should not raise when journal is None."""
        pending: dict[str, ToolCallRecord] = {}
        block = FakeToolUseBlock(name="Bash", id="toolu_n1")

        # Should not raise
        record = _handle_tool_use_block(block, pending, None, self.MODEL)
        assert record is not None


# ── _handle_tool_result_block ────────────────────────────────────


class TestHandleToolResultBlock:
    """Test _handle_tool_result_block() result matching and error tracking."""

    MODEL = "claude-sonnet-4-20250514"

    def _make_pending(
        self, tool_name: str = "web_search", tool_id: str = "toolu_1",
    ) -> dict[str, ToolCallRecord]:
        """Create a pending_records dict with a single pre-registered record."""
        record = ToolCallRecord(
            tool_name=tool_name,
            tool_id=tool_id,
            input_summary="test input",
        )
        return {tool_id: record}

    def test_updates_result_summary_from_string_content(self):
        """Should set result_summary from string content."""
        pending = self._make_pending()
        block = FakeToolResultBlock(
            tool_use_id="toolu_1",
            content="search results here",
        )

        _handle_tool_result_block(block, pending, None, self.MODEL)

        assert "search results here" in pending["toolu_1"].result_summary

    def test_updates_result_summary_from_list_content(self):
        """Should concatenate text fields from list content."""
        pending = self._make_pending()
        block = FakeToolResultBlock(
            tool_use_id="toolu_1",
            content=[
                {"text": "line 1"},
                {"text": "line 2"},
            ],
        )

        _handle_tool_result_block(block, pending, None, self.MODEL)

        assert "line 1" in pending["toolu_1"].result_summary
        assert "line 2" in pending["toolu_1"].result_summary

    def test_sets_is_error_from_block(self):
        """Should set is_error from block.is_error when True."""
        pending = self._make_pending()
        block = FakeToolResultBlock(
            tool_use_id="toolu_1",
            content="error message",
            is_error=True,
        )

        _handle_tool_result_block(block, pending, None, self.MODEL)

        assert pending["toolu_1"].is_error is True

    def test_defaults_is_error_to_false_when_none(self):
        """Should default is_error to False when block.is_error is None."""
        pending = self._make_pending()
        block = FakeToolResultBlock(
            tool_use_id="toolu_1",
            content="ok",
            is_error=None,
        )

        _handle_tool_result_block(block, pending, None, self.MODEL)

        assert pending["toolu_1"].is_error is False

    def test_logs_warning_for_unknown_tool_use_id(self, caplog):
        """Should log a warning when tool_use_id is not in pending_records."""
        pending: dict[str, ToolCallRecord] = {}
        block = FakeToolResultBlock(
            tool_use_id="toolu_unknown",
            content="orphan result",
        )

        with caplog.at_level("WARNING", logger="animaworks.execution.agent_sdk"):
            _handle_tool_result_block(block, pending, None, self.MODEL)

        assert "unknown tool_use_id" in caplog.text.lower() or "toolu_unknown" in caplog.text

    def test_writes_to_journal_when_provided(self):
        """Should call journal.write_tool_end when journal is provided."""
        pending = self._make_pending()
        block = FakeToolResultBlock(
            tool_use_id="toolu_1",
            content="result data",
        )
        mock_journal = MagicMock()

        _handle_tool_result_block(block, pending, mock_journal, self.MODEL)

        mock_journal.write_tool_end.assert_called_once()
        call_args = mock_journal.write_tool_end.call_args
        assert call_args[0][0] == "web_search"  # tool name
        assert call_args[1]["tool_id"] == "toolu_1"

    def test_journal_write_for_unknown_id(self):
        """Should still write to journal even for unknown tool_use_id."""
        pending: dict[str, ToolCallRecord] = {}
        block = FakeToolResultBlock(
            tool_use_id="toolu_orphan",
            content="orphan content",
        )
        mock_journal = MagicMock()

        _handle_tool_result_block(block, pending, mock_journal, self.MODEL)

        mock_journal.write_tool_end.assert_called_once()
        call_args = mock_journal.write_tool_end.call_args
        assert call_args[0][0] == "unknown"  # tool name fallback


# ── _finalize_pending_records ────────────────────────────────────


class TestFinalizePendingRecords:
    """Test _finalize_pending_records() collection and error marking."""

    def test_returns_all_records_as_list(self):
        """Should return all records from pending_records."""
        pending = {
            "toolu_1": ToolCallRecord(
                tool_name="Bash", tool_id="toolu_1",
                input_summary="ls", result_summary="file listing",
            ),
            "toolu_2": ToolCallRecord(
                tool_name="Read", tool_id="toolu_2",
                input_summary="/file", result_summary="content",
            ),
        }

        records = _finalize_pending_records(pending)

        assert len(records) == 2
        tool_names = {r.tool_name for r in records}
        assert tool_names == {"Bash", "Read"}

    def test_marks_records_without_result_as_error(self):
        """Records with empty result_summary should be marked is_error=True."""
        pending = {
            "toolu_1": ToolCallRecord(
                tool_name="Bash", tool_id="toolu_1",
                input_summary="ls", result_summary="",
            ),
        }

        records = _finalize_pending_records(pending)

        assert len(records) == 1
        assert records[0].is_error is True

    def test_preserves_is_error_for_records_with_result(self):
        """Records with result_summary should keep their original is_error value."""
        pending = {
            "toolu_1": ToolCallRecord(
                tool_name="Bash", tool_id="toolu_1",
                input_summary="ls", result_summary="ok",
                is_error=False,
            ),
            "toolu_2": ToolCallRecord(
                tool_name="web_search", tool_id="toolu_2",
                input_summary="q", result_summary="error output",
                is_error=True,
            ),
        }

        records = _finalize_pending_records(pending)

        by_id = {r.tool_id: r for r in records}
        assert by_id["toolu_1"].is_error is False
        assert by_id["toolu_2"].is_error is True

    def test_empty_pending_returns_empty_list(self):
        """Empty pending_records should return an empty list."""
        records = _finalize_pending_records({})
        assert records == []

    def test_mixed_completed_and_incomplete(self):
        """Mix of records with and without results."""
        pending = {
            "toolu_ok": ToolCallRecord(
                tool_name="Read", tool_id="toolu_ok",
                input_summary="/f", result_summary="data",
                is_error=False,
            ),
            "toolu_no_result": ToolCallRecord(
                tool_name="Bash", tool_id="toolu_no_result",
                input_summary="cmd", result_summary="",
            ),
        }

        records = _finalize_pending_records(pending)

        by_id = {r.tool_id: r for r in records}
        assert by_id["toolu_ok"].is_error is False
        assert by_id["toolu_no_result"].is_error is True
