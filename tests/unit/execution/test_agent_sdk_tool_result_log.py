from __future__ import annotations

"""Tests for _log_tool_result() and _handle_tool_result_block() with anima_dir."""

import json
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest


class TestLogToolResult:
    """Test the _log_tool_result() helper function."""

    def test_writes_tool_result_entry(self, tmp_path: Path) -> None:
        from core.execution.agent_sdk import _log_tool_result

        _log_tool_result(
            tmp_path, "web_search", "tu_abc123",
            "search result content", is_error=False,
        )

        today = date.today().isoformat()
        log_file = tmp_path / "activity_log" / f"{today}.jsonl"
        assert log_file.exists()

        lines = log_file.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 1

        raw = json.loads(lines[0])
        assert raw["type"] == "tool_result"
        assert raw["tool"] == "web_search"
        assert raw["content"] == "search result content"
        assert raw["meta"]["tool_use_id"] == "tu_abc123"
        assert raw["meta"]["is_error"] is False

    def test_writes_error_flag(self, tmp_path: Path) -> None:
        from core.execution.agent_sdk import _log_tool_result

        _log_tool_result(
            tmp_path, "Bash", "tu_err456",
            "command not found", is_error=True,
        )

        today = date.today().isoformat()
        log_file = tmp_path / "activity_log" / f"{today}.jsonl"
        raw = json.loads(log_file.read_text(encoding="utf-8").strip())
        assert raw["meta"]["is_error"] is True

    def test_never_raises(self, tmp_path: Path) -> None:
        """_log_tool_result should silently swallow errors."""
        from core.execution.agent_sdk import _log_tool_result

        # Use a path that will cause an error (e.g., read-only or missing dir)
        with patch("core.memory.activity.ActivityLogger.log", side_effect=RuntimeError("disk full")):
            # Should not raise
            _log_tool_result(
                tmp_path, "Read", "tu_xyz",
                "some content",
            )


class TestHandleToolResultBlockWithAnimaDir:
    """Test that _handle_tool_result_block logs to activity log when anima_dir is provided."""

    def _make_block(
        self,
        tool_use_id: str = "tu_test",
        content: str = "result text",
        is_error: bool = False,
    ) -> SimpleNamespace:
        return SimpleNamespace(
            tool_use_id=tool_use_id,
            content=content,
            is_error=is_error,
        )

    def test_logs_to_activity_with_anima_dir(self, tmp_path: Path) -> None:
        from core.execution.agent_sdk import _handle_tool_result_block
        from core.execution.base import ToolCallRecord

        pending = {
            "tu_test": ToolCallRecord(
                tool_name="Grep",
                tool_id="tu_test",
                input_summary="pattern",
                result_summary="",
                is_error=False,
            ),
        }

        block = self._make_block(content="grep results here")
        _handle_tool_result_block(
            block, pending, None, "claude-sonnet-4-20250514",
            anima_dir=tmp_path,
        )

        today = date.today().isoformat()
        log_file = tmp_path / "activity_log" / f"{today}.jsonl"
        assert log_file.exists()

        raw = json.loads(log_file.read_text(encoding="utf-8").strip())
        assert raw["type"] == "tool_result"
        assert raw["tool"] == "Grep"

    def test_no_activity_log_without_anima_dir(self, tmp_path: Path) -> None:
        from core.execution.agent_sdk import _handle_tool_result_block
        from core.execution.base import ToolCallRecord

        pending = {
            "tu_test2": ToolCallRecord(
                tool_name="Read",
                tool_id="tu_test2",
                input_summary="/some/file",
                result_summary="",
                is_error=False,
            ),
        }

        block = self._make_block(tool_use_id="tu_test2")
        _handle_tool_result_block(
            block, pending, None, "claude-sonnet-4-20250514",
            # anima_dir not passed (default None)
        )

        log_dir = tmp_path / "activity_log"
        assert not log_dir.exists()
