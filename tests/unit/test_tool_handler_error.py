"""Unit tests for ToolHandler top-level catch and output truncation."""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

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
        with patch.object(handler, "_handle_search_memory", side_effect=RuntimeError("boom")):
            result = handler.handle("search_memory", {"query": "test"})
        assert "Tool execution failed" in result
        assert "search_memory" in result
        assert "boom" in result


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
