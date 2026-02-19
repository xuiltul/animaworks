"""Tests for core/background.py — BackgroundTaskManager."""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from core.background import (
    BackgroundTask,
    BackgroundTaskManager,
    TaskStatus,
    _DEFAULT_ELIGIBLE_TOOLS,
)


# ── Fixtures ──────────────────────────────────────────────────


@pytest.fixture
def anima_dir(tmp_path: Path) -> Path:
    d = tmp_path / "animas" / "test-anima"
    d.mkdir(parents=True)
    return d


@pytest.fixture
def manager(anima_dir: Path) -> BackgroundTaskManager:
    return BackgroundTaskManager(anima_dir, anima_name="test-anima")


# ── Eligibility ──────────────────────────────────────────────


class TestEligibility:
    def test_is_eligible_default_tools(self, manager: BackgroundTaskManager):
        """Default eligible tools include image_gen schema names, local_llm, run_command."""
        assert manager.is_eligible("generate_character_assets") is True
        assert manager.is_eligible("generate_fullbody") is True
        assert manager.is_eligible("generate_bustup") is True
        assert manager.is_eligible("generate_chibi") is True
        assert manager.is_eligible("generate_3d_model") is True
        assert manager.is_eligible("generate_rigged_model") is True
        assert manager.is_eligible("generate_animations") is True
        assert manager.is_eligible("local_llm") is True
        assert manager.is_eligible("run_command") is True

    def test_is_eligible_custom_tools(self, anima_dir: Path):
        """Custom eligible tools passed to constructor are used."""
        custom = {"my_tool": 10, "another_tool": 20}
        mgr = BackgroundTaskManager(
            anima_dir, anima_name="test-anima", eligible_tools=custom,
        )
        assert mgr.is_eligible("my_tool") is True
        assert mgr.is_eligible("another_tool") is True
        # Default tools should NOT be eligible when custom set is provided
        assert mgr.is_eligible("generate_character_assets") is False

    def test_is_not_eligible(self, manager: BackgroundTaskManager):
        """Tools not in the eligible set return False."""
        assert manager.is_eligible("send_message") is False
        assert manager.is_eligible("search_memory") is False
        assert manager.is_eligible("nonexistent_tool") is False
        assert manager.is_eligible("image_generation") is False


# ── Submit ───────────────────────────────────────────────────


class TestSubmit:
    async def test_submit_creates_task(self, manager: BackgroundTaskManager):
        """submit() returns a task_id and the task status is RUNNING."""
        execute_fn = MagicMock(return_value="done")
        task_id = manager.submit("image_generation", {"prompt": "cat"}, execute_fn)

        assert isinstance(task_id, str)
        assert len(task_id) == 12  # uuid4 hex[:12]

        task = manager.get_task(task_id)
        assert task is not None
        assert task.status == TaskStatus.RUNNING
        assert task.tool_name == "image_generation"
        assert task.tool_args == {"prompt": "cat"}
        assert task.anima_name == "test-anima"

    async def test_submit_saves_to_disk(
        self, manager: BackgroundTaskManager, anima_dir: Path,
    ):
        """Task JSON file is created in state/background_tasks/."""
        execute_fn = MagicMock(return_value="done")
        task_id = manager.submit("local_llm", {"query": "hello"}, execute_fn)

        task_path = anima_dir / "state" / "background_tasks" / f"{task_id}.json"
        assert task_path.exists()

        data = json.loads(task_path.read_text(encoding="utf-8"))
        assert data["task_id"] == task_id
        assert data["tool_name"] == "local_llm"
        assert data["status"] == "running"

    async def test_task_completes_successfully(
        self, manager: BackgroundTaskManager,
    ):
        """Submit with a mock execute_fn, await the async task, verify COMPLETED."""
        def execute_fn(name: str, args: dict) -> str:
            return "result text"

        task_id = manager.submit("image_generation", {"prompt": "cat"}, execute_fn)

        # Wait for the background asyncio.Task to finish
        async_task = manager._async_tasks.get(task_id)
        assert async_task is not None
        await async_task

        task = manager.get_task(task_id)
        assert task is not None
        assert task.status == TaskStatus.COMPLETED
        assert task.result == "result text"
        assert task.completed_at is not None
        assert task.error is None

    async def test_task_fails_on_exception(
        self, manager: BackgroundTaskManager,
    ):
        """Submit with a failing execute_fn, verify FAILED status."""
        def execute_fn(name: str, args: dict) -> str:
            raise ValueError("something broke")

        task_id = manager.submit("run_command", {"cmd": "test"}, execute_fn)

        async_task = manager._async_tasks.get(task_id)
        assert async_task is not None
        await async_task

        task = manager.get_task(task_id)
        assert task is not None
        assert task.status == TaskStatus.FAILED
        assert "ValueError" in task.error
        assert "something broke" in task.error
        assert task.completed_at is not None
        assert task.result is None

    async def test_on_complete_callback_called(
        self, manager: BackgroundTaskManager,
    ):
        """Verify the on_complete callback is called after completion."""
        callback = AsyncMock()
        manager.on_complete = callback

        def execute_fn(name: str, args: dict) -> str:
            return "ok"

        task_id = manager.submit("image_generation", {"p": "cat"}, execute_fn)
        async_task = manager._async_tasks.get(task_id)
        await async_task

        callback.assert_called_once()
        called_task = callback.call_args[0][0]
        assert called_task.task_id == task_id
        assert called_task.status == TaskStatus.COMPLETED


# ── Get / List / Count ───────────────────────────────────────


class TestGetTask:
    async def test_get_task_from_memory(self, manager: BackgroundTaskManager):
        """get_task returns the in-memory task."""
        execute_fn = MagicMock(return_value="done")
        task_id = manager.submit("image_generation", {"p": "cat"}, execute_fn)

        task = manager.get_task(task_id)
        assert task is not None
        assert task.task_id == task_id

    def test_get_task_from_disk(
        self, manager: BackgroundTaskManager, anima_dir: Path,
    ):
        """get_task loads from disk when not in memory."""
        # Manually write a task file to disk
        task_data = {
            "task_id": "diskonly12345",
            "anima_name": "test-anima",
            "tool_name": "local_llm",
            "tool_args": {"q": "hi"},
            "status": "completed",
            "created_at": time.time() - 100,
            "completed_at": time.time(),
            "result": "disk result",
            "error": None,
        }
        storage_dir = anima_dir / "state" / "background_tasks"
        storage_dir.mkdir(parents=True, exist_ok=True)
        (storage_dir / "diskonly12345.json").write_text(
            json.dumps(task_data, ensure_ascii=False), encoding="utf-8",
        )

        # Ensure it's not in memory
        assert "diskonly12345" not in manager._tasks

        task = manager.get_task("diskonly12345")
        assert task is not None
        assert task.task_id == "diskonly12345"
        assert task.status == TaskStatus.COMPLETED
        assert task.result == "disk result"

    def test_get_task_nonexistent(self, manager: BackgroundTaskManager):
        """get_task returns None for unknown task_id."""
        assert manager.get_task("nonexistent") is None


class TestListTasks:
    async def test_list_tasks_all(self, manager: BackgroundTaskManager):
        """list_tasks returns all tasks."""
        execute_fn = MagicMock(return_value="done")
        id1 = manager.submit("image_generation", {"p": "a"}, execute_fn)
        id2 = manager.submit("local_llm", {"q": "b"}, execute_fn)

        tasks = manager.list_tasks()
        ids = {t.task_id for t in tasks}
        assert id1 in ids
        assert id2 in ids

    async def test_list_tasks_filtered_by_status(
        self, manager: BackgroundTaskManager,
    ):
        """list_tasks with status filter returns only matching tasks."""
        def succeed(name: str, args: dict) -> str:
            return "ok"

        def fail(name: str, args: dict) -> str:
            raise RuntimeError("fail")

        id_ok = manager.submit("image_generation", {}, succeed)
        id_fail = manager.submit("local_llm", {}, fail)

        # Wait for both to finish
        for tid in [id_ok, id_fail]:
            at = manager._async_tasks.get(tid)
            if at:
                await at

        completed = manager.list_tasks(status=TaskStatus.COMPLETED)
        failed = manager.list_tasks(status=TaskStatus.FAILED)
        running = manager.list_tasks(status=TaskStatus.RUNNING)

        assert any(t.task_id == id_ok for t in completed)
        assert any(t.task_id == id_fail for t in failed)
        assert len(running) == 0


class TestActiveCount:
    async def test_active_count(self, manager: BackgroundTaskManager):
        """active_count reflects running tasks."""
        # Before any submission
        assert manager.active_count() == 0

        # Use an event to keep the task in RUNNING state
        blocker = asyncio.Event()

        def blocking_fn(name: str, args: dict) -> str:
            # This runs in a thread executor. We need a way to block it
            # long enough to check active_count. Use a short sleep.
            import time as _time
            _time.sleep(0.2)
            return "done"

        manager.submit("image_generation", {}, blocking_fn)

        # Immediately after submit, the task status is RUNNING
        assert manager.active_count() == 1

        # Wait for the task to complete
        async_task = list(manager._async_tasks.values())[0]
        await async_task

        assert manager.active_count() == 0


# ── Cleanup ──────────────────────────────────────────────────


class TestCleanup:
    def test_cleanup_old_tasks(
        self, manager: BackgroundTaskManager, anima_dir: Path,
    ):
        """cleanup_old_tasks removes old completed tasks."""
        storage_dir = anima_dir / "state" / "background_tasks"
        storage_dir.mkdir(parents=True, exist_ok=True)

        # Create an old completed task (48 hours ago)
        old_time = time.time() - 48 * 3600
        old_data = {
            "task_id": "old_task_001",
            "anima_name": "test-anima",
            "tool_name": "image_generation",
            "tool_args": {},
            "status": "completed",
            "created_at": old_time - 100,
            "completed_at": old_time,
            "result": "old result",
            "error": None,
        }
        (storage_dir / "old_task_001.json").write_text(
            json.dumps(old_data), encoding="utf-8",
        )

        # Create a recent completed task (1 hour ago)
        recent_time = time.time() - 3600
        recent_data = {
            "task_id": "new_task_001",
            "anima_name": "test-anima",
            "tool_name": "local_llm",
            "tool_args": {},
            "status": "completed",
            "created_at": recent_time - 10,
            "completed_at": recent_time,
            "result": "new result",
            "error": None,
        }
        (storage_dir / "new_task_001.json").write_text(
            json.dumps(recent_data), encoding="utf-8",
        )

        removed = manager.cleanup_old_tasks(max_age_hours=24)
        assert removed == 1
        assert not (storage_dir / "old_task_001.json").exists()
        assert (storage_dir / "new_task_001.json").exists()


# ── BackgroundTask data model ────────────────────────────────


class TestBackgroundTaskModel:
    def test_task_to_dict(self):
        """BackgroundTask.to_dict() serialization."""
        task = BackgroundTask(
            task_id="abc123",
            anima_name="sakura",
            tool_name="image_generation",
            tool_args={"prompt": "flower"},
            status=TaskStatus.COMPLETED,
            created_at=1000.0,
            completed_at=1010.0,
            result="image_url_here",
            error=None,
        )
        d = task.to_dict()
        assert d["task_id"] == "abc123"
        assert d["anima_name"] == "sakura"
        assert d["tool_name"] == "image_generation"
        assert d["tool_args"] == {"prompt": "flower"}
        assert d["status"] == "completed"
        assert d["created_at"] == 1000.0
        assert d["completed_at"] == 1010.0
        assert d["result"] == "image_url_here"
        assert d["error"] is None

    def test_task_summary_completed(self):
        """summary() for completed task includes tool name and result preview."""
        task = BackgroundTask(
            task_id="abc123",
            anima_name="sakura",
            tool_name="image_generation",
            tool_args={},
            status=TaskStatus.COMPLETED,
            result="Generated image at /path/to/image.png",
        )
        summary = task.summary()
        assert "[image_generation]" in summary
        assert "completed" in summary
        assert "Generated image" in summary

    def test_task_summary_completed_truncated(self):
        """summary() truncates long results to 200 chars."""
        long_result = "x" * 500
        task = BackgroundTask(
            task_id="abc123",
            anima_name="sakura",
            tool_name="local_llm",
            tool_args={},
            status=TaskStatus.COMPLETED,
            result=long_result,
        )
        summary = task.summary()
        # The preview is at most 200 chars of the result
        assert len(summary) < 250  # [tool_name] completed: + 200

    def test_task_summary_failed(self):
        """summary() for failed task includes error message."""
        task = BackgroundTask(
            task_id="abc123",
            anima_name="sakura",
            tool_name="run_command",
            tool_args={},
            status=TaskStatus.FAILED,
            error="TimeoutError: command timed out",
        )
        summary = task.summary()
        assert "[run_command]" in summary
        assert "failed" in summary
        assert "TimeoutError" in summary

    def test_task_summary_running(self):
        """summary() for running task shows status."""
        task = BackgroundTask(
            task_id="abc123",
            anima_name="sakura",
            tool_name="image_generation",
            tool_args={},
            status=TaskStatus.RUNNING,
        )
        summary = task.summary()
        assert "[image_generation]" in summary
        assert "running" in summary

    def test_task_summary_pending(self):
        """summary() for pending task shows status."""
        task = BackgroundTask(
            task_id="abc123",
            anima_name="sakura",
            tool_name="image_generation",
            tool_args={},
            status=TaskStatus.PENDING,
        )
        summary = task.summary()
        assert "pending" in summary


# ── Default eligible tools constant ──────────────────────────


class TestDefaultEligibleTools:
    def test_default_tools_content(self):
        """_DEFAULT_ELIGIBLE_TOOLS contains expected tools with timeouts."""
        # Image gen schema names (all threshold 30)
        for name in (
            "generate_character_assets", "generate_fullbody", "generate_bustup",
            "generate_chibi", "generate_3d_model", "generate_rigged_model",
            "generate_animations",
        ):
            assert name in _DEFAULT_ELIGIBLE_TOOLS, f"{name} missing"
            assert _DEFAULT_ELIGIBLE_TOOLS[name] == 30

        # Other background tools
        assert "local_llm" in _DEFAULT_ELIGIBLE_TOOLS
        assert "run_command" in _DEFAULT_ELIGIBLE_TOOLS
        assert _DEFAULT_ELIGIBLE_TOOLS["local_llm"] == 60
        assert _DEFAULT_ELIGIBLE_TOOLS["run_command"] == 60

        # Old category name must NOT be present
        assert "image_generation" not in _DEFAULT_ELIGIBLE_TOOLS


# ── SubmitAsync ─────────────────────────────────────────────


class TestSubmitAsync:
    async def test_submit_async_creates_task(
        self, manager: BackgroundTaskManager,
    ):
        """submit_async() returns a task_id and the task status is RUNNING."""
        async def execute_fn(name: str, args: dict) -> str:
            await asyncio.sleep(5)
            return "async done"

        task_id = await manager.submit_async(
            "image_generation", {"prompt": "cat"}, execute_fn,
        )

        assert isinstance(task_id, str)
        assert len(task_id) == 12

        task = manager.get_task(task_id)
        assert task is not None
        assert task.status == TaskStatus.RUNNING
        assert task.tool_name == "image_generation"
        assert task.tool_args == {"prompt": "cat"}
        assert task.anima_name == "test-anima"

        # Cancel the long-running task to avoid warnings
        async_task = manager._async_tasks.get(task_id)
        if async_task:
            async_task.cancel()
            try:
                await async_task
            except asyncio.CancelledError:
                pass

    async def test_submit_async_completes_successfully(
        self, manager: BackgroundTaskManager,
    ):
        """submit_async with a succeeding async execute_fn transitions to COMPLETED."""
        async def execute_fn(name: str, args: dict) -> str:
            return "async result"

        task_id = await manager.submit_async(
            "image_generation", {"prompt": "dog"}, execute_fn,
        )

        # Wait for the background asyncio.Task to finish
        async_task = manager._async_tasks.get(task_id)
        assert async_task is not None
        await async_task

        task = manager.get_task(task_id)
        assert task is not None
        assert task.status == TaskStatus.COMPLETED
        assert task.result == "async result"
        assert task.completed_at is not None
        assert task.error is None

    async def test_submit_async_fails_on_exception(
        self, manager: BackgroundTaskManager,
    ):
        """submit_async with a failing async execute_fn sets FAILED status."""
        async def execute_fn(name: str, args: dict) -> str:
            raise RuntimeError("async kaboom")

        task_id = await manager.submit_async(
            "run_command", {"cmd": "fail"}, execute_fn,
        )

        async_task = manager._async_tasks.get(task_id)
        assert async_task is not None
        await async_task

        task = manager.get_task(task_id)
        assert task is not None
        assert task.status == TaskStatus.FAILED
        assert "RuntimeError" in task.error
        assert "async kaboom" in task.error
        assert task.completed_at is not None
        assert task.result is None

    async def test_submit_async_on_complete_callback_called(
        self, manager: BackgroundTaskManager,
    ):
        """Verify the on_complete callback fires after async task completion."""
        callback = AsyncMock()
        manager.on_complete = callback

        async def execute_fn(name: str, args: dict) -> str:
            return "callback test"

        task_id = await manager.submit_async(
            "local_llm", {"q": "hello"}, execute_fn,
        )
        async_task = manager._async_tasks.get(task_id)
        await async_task

        callback.assert_called_once()
        called_task = callback.call_args[0][0]
        assert called_task.task_id == task_id
        assert called_task.status == TaskStatus.COMPLETED


# ── LoadTask error handling ─────────────────────────────────


class TestLoadTaskErrors:
    def test_load_task_handles_corrupt_json(
        self, manager: BackgroundTaskManager, anima_dir: Path,
    ):
        """get_task returns None when the task file contains invalid JSON."""
        storage_dir = anima_dir / "state" / "background_tasks"
        storage_dir.mkdir(parents=True, exist_ok=True)

        # Write corrupt (non-parseable) JSON
        (storage_dir / "corrupt001.json").write_text(
            "{{not valid json!!", encoding="utf-8",
        )

        assert manager.get_task("corrupt001") is None

    def test_load_task_handles_missing_keys(
        self, manager: BackgroundTaskManager, anima_dir: Path,
    ):
        """get_task returns None when required keys are missing from JSON."""
        storage_dir = anima_dir / "state" / "background_tasks"
        storage_dir.mkdir(parents=True, exist_ok=True)

        # Valid JSON but missing required keys (e.g. task_id, anima_name)
        (storage_dir / "badkeys001.json").write_text(
            json.dumps({"some_field": "value"}), encoding="utf-8",
        )

        assert manager.get_task("badkeys001") is None

    def test_load_task_handles_invalid_status(
        self, manager: BackgroundTaskManager, anima_dir: Path,
    ):
        """get_task returns None when status value is not a valid TaskStatus."""
        storage_dir = anima_dir / "state" / "background_tasks"
        storage_dir.mkdir(parents=True, exist_ok=True)

        bad_data = {
            "task_id": "badstat001",
            "anima_name": "test-anima",
            "tool_name": "local_llm",
            "tool_args": {},
            "status": "invalid_status_value",
            "created_at": time.time(),
        }
        (storage_dir / "badstat001.json").write_text(
            json.dumps(bad_data), encoding="utf-8",
        )

        assert manager.get_task("badstat001") is None


# ── Cleanup edge cases ──────────────────────────────────────


class TestCleanupEdgeCases:
    def test_cleanup_skips_corrupt_json(
        self, manager: BackgroundTaskManager, anima_dir: Path,
    ):
        """cleanup_old_tasks skips files with corrupt JSON gracefully."""
        storage_dir = anima_dir / "state" / "background_tasks"
        storage_dir.mkdir(parents=True, exist_ok=True)

        # Write a corrupt JSON file
        (storage_dir / "corrupt_cleanup.json").write_text(
            "not json at all", encoding="utf-8",
        )

        # Should not raise, should return 0 removed
        removed = manager.cleanup_old_tasks(max_age_hours=0)
        assert removed == 0
        # The corrupt file should still exist (not deleted)
        assert (storage_dir / "corrupt_cleanup.json").exists()
