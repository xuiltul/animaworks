from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from core.execution.session_context import RuntimeSessionContext
from core.memory.activity import ActivityLogger
from core.tooling.handler import ToolHandler


@pytest.mark.e2e
def test_runtime_session_bind_reaches_activity_jsonl_and_reader(tmp_path: Path) -> None:
    """Runtime session -> ToolHandler -> JSONL -> ActivityLogger reader."""
    anima_dir = tmp_path / "animas" / "worker"
    handler = ToolHandler(anima_dir=anima_dir, memory=MagicMock())
    handler.bind_runtime_session(
        RuntimeSessionContext.create(
            session_type="task",
            thread_id="parallel-7",
            trigger="task:parallel-7",
        )
    )

    handler._log_tool_activity("search_memory", {"query": "synthetic fixture"})
    handler._log_tool_result_activity("search_memory", json.dumps({"count": 1}))

    entries = ActivityLogger(anima_dir).recent(days=1)
    assert [(entry.type, entry.ctx) for entry in entries] == [
        ("tool_use", "task:parallel-7"),
        ("tool_result", "task:parallel-7"),
    ]
    assert all(entry.to_api_dict("worker")["ctx"] == "task:parallel-7" for entry in entries)
