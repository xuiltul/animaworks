"""Unit tests for ToolHandler._handle_todo_write (session-scoped TodoWrite tool)."""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

from core.tooling.handler import ToolHandler


def _make_handler(tmp_path: Path) -> ToolHandler:
    """Create a ToolHandler with minimal mocked dependencies."""
    anima_dir = tmp_path / "animas" / "test"
    anima_dir.mkdir(parents=True, exist_ok=True)
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
    handler._activity = MagicMock()
    return handler


class TestTodoWriteBasic:
    """Basic CRUD operations."""

    def test_create_todo_list(self, tmp_path: Path) -> None:
        handler = _make_handler(tmp_path)
        result = json.loads(
            handler._handle_todo_write(
                {
                    "todos": [
                        {"id": "t1", "content": "Task one", "status": "pending"},
                        {"id": "t2", "content": "Task two", "status": "in_progress"},
                    ],
                }
            )
        )
        assert result["status"] == "ok"
        assert "0/2 completed" in result["progress"]
        assert len(handler._session_todos) == 2

    def test_replace_todo_list(self, tmp_path: Path) -> None:
        handler = _make_handler(tmp_path)
        handler._handle_todo_write(
            {
                "todos": [{"id": "t1", "content": "Old", "status": "pending"}],
            }
        )
        assert len(handler._session_todos) == 1

        handler._handle_todo_write(
            {
                "todos": [
                    {"id": "t2", "content": "New", "status": "pending"},
                    {"id": "t3", "content": "Also new", "status": "completed"},
                ],
                "merge": False,
            }
        )
        assert len(handler._session_todos) == 2
        assert handler._session_todos[0]["id"] == "t2"

    def test_completed_progress(self, tmp_path: Path) -> None:
        handler = _make_handler(tmp_path)
        result = json.loads(
            handler._handle_todo_write(
                {
                    "todos": [
                        {"id": "t1", "content": "Done", "status": "completed"},
                        {"id": "t2", "content": "Also done", "status": "completed"},
                        {"id": "t3", "content": "Not done", "status": "pending"},
                    ],
                }
            )
        )
        assert "2/3 completed" in result["progress"]


class TestTodoWriteMerge:
    """Merge mode (merge=true)."""

    def test_merge_updates_existing(self, tmp_path: Path) -> None:
        handler = _make_handler(tmp_path)
        handler._handle_todo_write(
            {
                "todos": [
                    {"id": "t1", "content": "Step 1", "status": "pending"},
                    {"id": "t2", "content": "Step 2", "status": "pending"},
                ],
            }
        )

        handler._handle_todo_write(
            {
                "todos": [{"id": "t1", "content": "Step 1", "status": "completed"}],
                "merge": True,
            }
        )

        assert len(handler._session_todos) == 2
        t1 = next(t for t in handler._session_todos if t["id"] == "t1")
        assert t1["status"] == "completed"
        t2 = next(t for t in handler._session_todos if t["id"] == "t2")
        assert t2["status"] == "pending"

    def test_merge_adds_new(self, tmp_path: Path) -> None:
        handler = _make_handler(tmp_path)
        handler._handle_todo_write(
            {
                "todos": [{"id": "t1", "content": "Existing", "status": "pending"}],
            }
        )

        handler._handle_todo_write(
            {
                "todos": [{"id": "t2", "content": "New item", "status": "in_progress"}],
                "merge": True,
            }
        )

        assert len(handler._session_todos) == 2
        ids = {t["id"] for t in handler._session_todos}
        assert ids == {"t1", "t2"}

    def test_merge_preserves_content_when_empty(self, tmp_path: Path) -> None:
        handler = _make_handler(tmp_path)
        handler._handle_todo_write(
            {
                "todos": [{"id": "t1", "content": "Original content", "status": "pending"}],
            }
        )

        handler._handle_todo_write(
            {
                "todos": [{"id": "t1", "content": "", "status": "completed"}],
                "merge": True,
            }
        )

        t1 = handler._session_todos[0]
        assert t1["content"] == "Original content"
        assert t1["status"] == "completed"


class TestTodoWriteConstraints:
    """Validation and constraint enforcement."""

    def test_max_20_items(self, tmp_path: Path) -> None:
        handler = _make_handler(tmp_path)
        todos = [{"id": f"t{i}", "content": f"Task {i}", "status": "pending"} for i in range(25)]
        handler._handle_todo_write({"todos": todos})
        assert len(handler._session_todos) == 20

    def test_in_progress_warning(self, tmp_path: Path) -> None:
        handler = _make_handler(tmp_path)
        result = json.loads(
            handler._handle_todo_write(
                {
                    "todos": [
                        {"id": "t1", "content": "A", "status": "in_progress"},
                        {"id": "t2", "content": "B", "status": "in_progress"},
                    ],
                }
            )
        )
        assert "Warning" in result["message"]
        assert "2 tasks in_progress" in result["message"]

    def test_single_in_progress_no_warning(self, tmp_path: Path) -> None:
        handler = _make_handler(tmp_path)
        result = json.loads(
            handler._handle_todo_write(
                {
                    "todos": [
                        {"id": "t1", "content": "A", "status": "in_progress"},
                        {"id": "t2", "content": "B", "status": "pending"},
                    ],
                }
            )
        )
        assert "Warning" not in result["message"]

    def test_invalid_status_defaults_to_pending(self, tmp_path: Path) -> None:
        handler = _make_handler(tmp_path)
        handler._handle_todo_write(
            {
                "todos": [{"id": "t1", "content": "Test", "status": "invalid_status"}],
            }
        )
        assert handler._session_todos[0]["status"] == "pending"

    def test_empty_id_skipped(self, tmp_path: Path) -> None:
        handler = _make_handler(tmp_path)
        handler._handle_todo_write(
            {
                "todos": [
                    {"id": "", "content": "No ID", "status": "pending"},
                    {"id": "t1", "content": "Has ID", "status": "pending"},
                ],
            }
        )
        assert len(handler._session_todos) == 1
        assert handler._session_todos[0]["id"] == "t1"


class TestTodoWriteErrors:
    """Error handling."""

    def test_missing_todos_returns_error(self, tmp_path: Path) -> None:
        handler = _make_handler(tmp_path)
        result = json.loads(handler._handle_todo_write({}))
        assert "error" in result

    def test_empty_todos_returns_error(self, tmp_path: Path) -> None:
        handler = _make_handler(tmp_path)
        result = json.loads(handler._handle_todo_write({"todos": []}))
        assert "error" in result

    def test_all_invalid_items_returns_error(self, tmp_path: Path) -> None:
        handler = _make_handler(tmp_path)
        result = json.loads(
            handler._handle_todo_write(
                {
                    "todos": [{"id": "", "content": "", "status": "pending"}],
                }
            )
        )
        assert "error" in result

    def test_non_dict_items_skipped(self, tmp_path: Path) -> None:
        handler = _make_handler(tmp_path)
        result = json.loads(
            handler._handle_todo_write(
                {
                    "todos": ["not a dict", {"id": "t1", "content": "Valid", "status": "pending"}],
                }
            )
        )
        assert result["status"] == "ok"
        assert len(handler._session_todos) == 1


class TestTodoWriteActivityLog:
    """Activity log integration."""

    def test_activity_log_called(self, tmp_path: Path) -> None:
        handler = _make_handler(tmp_path)
        handler._handle_todo_write(
            {
                "todos": [{"id": "t1", "content": "Task", "status": "pending"}],
            }
        )
        handler._activity.log.assert_called_once()
        call_kwargs = handler._activity.log.call_args
        assert call_kwargs[1]["event_type"] == "tool_use"
        assert "todo_write" in call_kwargs[1]["content"]


class TestTodoWriteOutput:
    """Output format verification."""

    def test_output_contains_marks(self, tmp_path: Path) -> None:
        handler = _make_handler(tmp_path)
        result = json.loads(
            handler._handle_todo_write(
                {
                    "todos": [
                        {"id": "t1", "content": "Completed", "status": "completed"},
                        {"id": "t2", "content": "In progress", "status": "in_progress"},
                        {"id": "t3", "content": "Pending", "status": "pending"},
                    ],
                }
            )
        )
        assert "[x] t1: Completed" in result["todos"]
        assert "[>] t2: In progress" in result["todos"]
        assert "[ ] t3: Pending" in result["todos"]

    def test_session_scoped_isolation(self, tmp_path: Path) -> None:
        """Two handlers don't share state."""
        h1 = _make_handler(tmp_path)
        h2 = _make_handler(tmp_path)
        h1._handle_todo_write(
            {
                "todos": [{"id": "t1", "content": "H1 task", "status": "pending"}],
            }
        )
        assert len(h1._session_todos) == 1
        assert len(h2._session_todos) == 0


class TestTodoWriteSchema:
    """Schema integration test."""

    def test_schema_in_unified_list(self) -> None:
        from core.tooling.schemas.builder import build_unified_tool_list

        tools = build_unified_tool_list()
        names = [t["name"] for t in tools]
        assert "todo_write" in names

    def test_schema_has_required_fields(self) -> None:
        from core.tooling.schemas.session_todo import _session_todo_tools

        schema = _session_todo_tools()[0]
        assert schema["name"] == "todo_write"
        params = schema["parameters"]
        assert "todos" in params["properties"]
        assert "merge" in params["properties"]
        assert "todos" in params["required"]
