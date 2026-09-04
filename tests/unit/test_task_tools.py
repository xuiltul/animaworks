from __future__ import annotations

import json

import pytest

from core.tooling.schemas import _task_tools, build_tool_list


class TestTaskToolSchemas:
    def test_task_tools_defined(self):
        task_tools = _task_tools()
        assert len(task_tools) == 3
        names = {t["name"] for t in task_tools}
        assert names == {"backlog_task", "update_task", "list_tasks"}

    def test_backlog_task_schema(self):
        """deadline was retired (A1 task-model teardown): no longer a param at all."""
        task_tools = _task_tools()
        backlog_task = next(t for t in task_tools if t["name"] == "backlog_task")
        required = backlog_task["parameters"]["required"]
        assert "source" in required
        assert "original_instruction" in required
        assert "assignee" in required
        assert "summary" in required
        assert "deadline" not in required
        assert "deadline" not in backlog_task["parameters"]["properties"]

    def test_update_task_schema(self):
        """blocked/failed/unblock_check were retired (A1 task-model teardown)."""
        task_tools = _task_tools()
        update_task = next(t for t in task_tools if t["name"] == "update_task")
        required = update_task["parameters"]["required"]
        assert "task_id" in required
        assert "status" in required
        assert update_task["parameters"]["properties"]["result"]["type"] == "string"
        assert update_task["parameters"]["properties"]["status"]["enum"] == [
            "pending",
            "in_progress",
            "done",
            "cancelled",
        ]
        assert "unblock_check" not in update_task["parameters"]["properties"]
        assert "unblock_check" not in required

    def test_list_tasks_schema(self):
        task_tools = _task_tools()
        list_tasks = next(t for t in task_tools if t["name"] == "list_tasks")
        assert "required" not in list_tasks["parameters"] or "status" not in list_tasks["parameters"].get(
            "required", []
        )

    def test_build_tool_list_includes_task_tools(self):
        tools = build_tool_list(include_task_tools=True)
        names = {t["name"] for t in tools}
        assert "backlog_task" in names
        assert "update_task" in names
        assert "list_tasks" in names

    def test_build_tool_list_excludes_task_tools_by_default(self):
        tools = build_tool_list()
        names = {t["name"] for t in tools}
        assert "backlog_task" not in names


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

    def test_handle_backlog_task(self, handler):
        result = handler.handle(
            "backlog_task",
            {
                "source": "human",
                "original_instruction": "Test instruction",
                "assignee": "rin",
                "summary": "Test summary",
            },
        )
        data = json.loads(result)
        assert data["source"] == "human"
        assert data["assignee"] == "rin"
        assert data["status"] == "pending"
        assert "task_id" in data

    def test_handle_backlog_task_missing_instruction(self, handler):
        result = handler.handle(
            "backlog_task",
            {
                "source": "human",
                "assignee": "rin",
                "summary": "s",
            },
        )
        data = json.loads(result)
        assert data["status"] == "error"

    def test_handle_update_task(self, handler):
        # First add a task
        add_result = json.loads(
            handler.handle(
                "backlog_task",
                {
                    "source": "human",
                    "original_instruction": "test",
                    "assignee": "rin",
                    "summary": "s",
                },
            )
        )
        task_id = add_result["task_id"]

        # Update it to an externally-writable status ('in_progress' is written
        # only by the running TaskExec, so it is rejected from here).
        result = handler.handle(
            "update_task",
            {
                "task_id": task_id,
                "status": "done",
            },
        )
        data = json.loads(result)
        assert data["status"] == "done"

    @pytest.mark.parametrize("status", ["blocked", "failed"])
    def test_handle_update_task_retired_status_rejected(self, handler, status):
        """blocked/failed/unblock_check were retired (A1 task-model teardown):
        update_task now rejects them instead of persisting a blocked/failed state."""
        task = json.loads(
            handler.handle(
                "backlog_task",
                {
                    "source": "human",
                    "original_instruction": "test",
                    "assignee": "rin",
                    "summary": "s",
                },
            )
        )

        result = json.loads(
            handler.handle(
                "update_task",
                {
                    "task_id": task["task_id"],
                    "status": status,
                    "unblock_check": "test -w .",
                },
            )
        )

        assert result["status"] == "error"
        assert result["error_type"] == "InvalidArguments"

        # The task itself must remain untouched (still pending).
        reloaded = json.loads(handler.handle("list_tasks", {"status": "pending"}))
        assert any(t["task_id"] == task["task_id"] for t in reloaded)

    def test_handle_update_nonexistent(self, handler):
        result = handler.handle(
            "update_task",
            {
                "task_id": "nonexistent",
                "status": "done",
            },
        )
        data = json.loads(result)
        assert data["status"] == "error"

    def test_handle_list_tasks(self, handler):
        handler.handle(
            "backlog_task",
            {
                "source": "human",
                "original_instruction": "t1",
                "assignee": "a",
                "summary": "s1",
            },
        )
        handler.handle(
            "backlog_task",
            {
                "source": "anima",
                "original_instruction": "t2",
                "assignee": "b",
                "summary": "s2",
            },
        )
        result = json.loads(handler.handle("list_tasks", {}))
        assert len(result) == 2

    def test_handle_list_tasks_with_filter(self, handler):
        add_result = json.loads(
            handler.handle(
                "backlog_task",
                {
                    "source": "human",
                    "original_instruction": "t1",
                    "assignee": "a",
                    "summary": "s1",
                },
            )
        )
        handler.handle(
            "update_task",
            {
                "task_id": add_result["task_id"],
                "status": "done",
            },
        )
        handler.handle(
            "backlog_task",
            {
                "source": "human",
                "original_instruction": "t2",
                "assignee": "b",
                "summary": "s2",
            },
        )
        result = json.loads(handler.handle("list_tasks", {"status": "pending"}))
        assert len(result) == 1
