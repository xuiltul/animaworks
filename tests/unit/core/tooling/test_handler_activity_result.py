from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from core.tooling.handler import ToolHandler


def test_tool_result_activity_writes_normalized_error_flag(tmp_path: Path) -> None:
    anima_dir = tmp_path / "animas" / "test-anima"
    anima_dir.mkdir(parents=True)
    handler = ToolHandler(anima_dir=anima_dir, memory=MagicMock())
    handler._activity = MagicMock()

    handler._log_tool_result_activity("Bash", "Error: command failed", tool_use_id="tool-1")

    handler._activity.log.assert_called_once()
    args, kwargs = handler._activity.log.call_args
    assert args == ("tool_result",)
    assert kwargs["meta"]["result_status"] == "fail"
    assert kwargs["meta"]["is_error"] is True


def test_tool_result_activity_writes_success_flag(tmp_path: Path) -> None:
    anima_dir = tmp_path / "animas" / "test-anima"
    anima_dir.mkdir(parents=True)
    handler = ToolHandler(anima_dir=anima_dir, memory=MagicMock())
    handler._activity = MagicMock()

    handler._log_tool_result_activity("Read", "contents", tool_use_id="tool-2")

    meta = handler._activity.log.call_args.kwargs["meta"]
    assert meta["result_status"] == "ok"
    assert meta["is_error"] is False
