from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from core.tooling.schemas import TASK_TOOLS, build_tool_list


class TestTaskToolSchemas:
    def test_task_tools_defined(self):
        assert len(TASK_TOOLS) == 3
        names = {t["name"] for t in TASK_TOOLS}
        assert names == {"add_task", "update_task", "list_tasks"}

    def test_add_task_schema(self):
        add_task = next(t for t in TASK_TOOLS if t["name"] == "add_task")
        required = add_task["parameters"]["required"]
        assert "source" in required
        assert "original_instruction" in required
        assert "assignee" in required
        assert "summary" in required
        assert "deadline" in required

    def test_update_task_schema(self):
        update_task = next(t for t in TASK_TOOLS if t["name"] == "update_task")
        required = update_task["parameters"]["required"]
        assert "task_id" in required
        assert "status" in required

    def test_list_tasks_schema(self):
        list_tasks = next(t for t in TASK_TOOLS if t["name"] == "list_tasks")
        # status is optional
        assert "required" not in list_tasks["parameters"] or "status" not in list_tasks["parameters"].get("required", [])

    def test_build_tool_list_includes_task_tools(self):
        tools = build_tool_list(include_task_tools=True)
        names = {t["name"] for t in tools}
        assert "add_task" in names
        assert "update_task" in names
        assert "list_tasks" in names

    def test_build_tool_list_excludes_task_tools_by_default(self):
        tools = build_tool_list()
        names = {t["name"] for t in tools}
        assert "add_task" not in names


class TestTaskToolHandler:
    @pytest.fixture
    def handler(self, tmp_path):
        from core.memory import MemoryManager
        from core.tooling.handler import ToolHandler

        anima_dir = tmp_path / "animas" / "test"
        for d in ["state", "episodes", "knowledge", "procedures", "skills"]:
            (anima_dir / d).mkdir(parents=True)
        memory = MemoryManager(anima_dir)
        return ToolHandler(anima_dir, memory)

    def test_handle_add_task(self, handler):
        result = handler.handle("add_task", {
            "source": "human",
            "original_instruction": "Test instruction",
            "assignee": "rin",
            "summary": "Test summary",
            "deadline": "1h",
        })
        data = json.loads(result)
        assert data["source"] == "human"
        assert data["assignee"] == "rin"
        assert data["status"] == "pending"
        assert "task_id" in data

    def test_handle_add_task_missing_instruction(self, handler):
        result = handler.handle("add_task", {
            "source": "human",
            "assignee": "rin",
            "summary": "s",
            "deadline": "1h",
        })
        data = json.loads(result)
        assert data["status"] == "error"

    def test_handle_update_task(self, handler):
        # First add a task
        add_result = json.loads(handler.handle("add_task", {
            "source": "human",
            "original_instruction": "test",
            "assignee": "rin",
            "summary": "s",
            "deadline": "1h",
        }))
        task_id = add_result["task_id"]

        # Update it
        result = handler.handle("update_task", {
            "task_id": task_id,
            "status": "in_progress",
        })
        data = json.loads(result)
        assert data["status"] == "in_progress"

    def test_handle_update_nonexistent(self, handler):
        result = handler.handle("update_task", {
            "task_id": "nonexistent",
            "status": "done",
        })
        data = json.loads(result)
        assert data["status"] == "error"

    def test_handle_list_tasks(self, handler):
        handler.handle("add_task", {
            "source": "human",
            "original_instruction": "t1",
            "assignee": "a",
            "summary": "s1",
            "deadline": "1h",
        })
        handler.handle("add_task", {
            "source": "anima",
            "original_instruction": "t2",
            "assignee": "b",
            "summary": "s2",
            "deadline": "2h",
        })
        result = json.loads(handler.handle("list_tasks", {}))
        assert len(result) == 2

    def test_handle_list_tasks_with_filter(self, handler):
        add_result = json.loads(handler.handle("add_task", {
            "source": "human",
            "original_instruction": "t1",
            "assignee": "a",
            "summary": "s1",
            "deadline": "1h",
        }))
        handler.handle("update_task", {
            "task_id": add_result["task_id"],
            "status": "done",
        })
        handler.handle("add_task", {
            "source": "human",
            "original_instruction": "t2",
            "assignee": "b",
            "summary": "s2",
            "deadline": "2h",
        })
        result = json.loads(handler.handle("list_tasks", {"status": "pending"}))
        assert len(result) == 1
