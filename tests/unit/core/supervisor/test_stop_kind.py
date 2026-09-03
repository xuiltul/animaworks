from __future__ import annotations

import asyncio
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.memory.task_queue import TaskQueueManager
from core.supervisor.pending_executor import _SENTINEL_CANCELLED, PendingTaskExecutor


def _make_executor(tmp_path: Path, stop_kind: str = "normal") -> PendingTaskExecutor:
    anima_dir = tmp_path / "animas" / "test-anima"
    (anima_dir / "state").mkdir(parents=True)
    anima = MagicMock()
    anima._background_lock = asyncio.Lock()
    anima._task_semaphore = None
    anima._status_slots = {"background": "idle"}
    anima._task_slots = {"background": ""}
    anima._active_parallel_tasks = {}
    anima._active_background_workers = {}

    async def stream(*_args, **_kwargs):
        yield {"type": "text_delta", "text": "output"}
        yield {
            "type": "cycle_done",
            "cycle_result": {
                "summary": "result",
                "action": "skipped" if stop_kind == "budget_skipped" else "responded",
                "stop_kind": stop_kind,
                "tool_call_records": [],
            },
        }

    anima.agent.run_cycle_streaming = stream
    return PendingTaskExecutor(
        anima=anima,
        anima_name="test-anima",
        anima_dir=anima_dir,
        shutdown_event=asyncio.Event(),
    )


def _task(task_id: str, **overrides) -> dict:
    task = {
        "task_type": "llm",
        "task_id": task_id,
        "title": "Stop kind test",
        "description": "finish the task",
        "context": "original context",
        "reply_to": None,
        "submitted_at": "",
    }
    task.update(overrides)
    return task


def _queue_task(executor: PendingTaskExecutor, task_id: str, *, status: str = "in_progress") -> TaskQueueManager:
    manager = TaskQueueManager(executor._anima_dir)
    manager.add_task(
        source="anima",
        original_instruction="finish the task",
        assignee="test-anima",
        summary="stop kind test",
        task_id=task_id,
        status=status,
    )
    return manager


@contextmanager
def _execution_patches():
    with (
        patch("core.paths.load_prompt", return_value="prompt"),
        patch("core.memory.activity.ActivityLogger"),
    ):
        yield


@pytest.mark.asyncio
async def test_interrupted_without_declaration_returns_to_pending(tmp_path: Path) -> None:
    """An interrupted run that never declared goes back to pending, not re-enqueued."""
    executor = _make_executor(tmp_path, "interrupted")
    manager = _queue_task(executor, "interrupted")

    with _execution_patches():
        await executor._execute_llm_task(_task("interrupted"))

    entry = manager.get_task_by_id("interrupted")
    assert entry is not None
    assert entry.status == "pending"
    assert entry.meta["last_run_stop_kind"] == "interrupted"
    assert entry.meta["last_run_ended_at"]
    assert not (executor._anima_dir / "state" / "pending" / "interrupted.json").exists()


@pytest.mark.asyncio
async def test_normal_stop_without_declaration_returns_to_pending(tmp_path: Path) -> None:
    """A run that finishes normally but never declares is handed back as pending."""
    executor = _make_executor(tmp_path)
    manager = _queue_task(executor, "undeclared")

    with _execution_patches():
        await executor._execute_llm_task(_task("undeclared"))

    entry = manager.get_task_by_id("undeclared")
    assert entry is not None
    assert entry.status == "pending"
    assert entry.summary == "stop kind test"
    assert entry.meta["last_run_note"] == "run ended without a completion declaration"
    assert entry.meta["last_run_stop_kind"] == "normal"
    assert entry.meta["last_run_ended_at"]
    # No descriptor is regenerated: nothing re-runs it on the anima's behalf.
    assert not list((executor._anima_dir / "state" / "pending").glob("*.json"))
    executor._anima.messenger.send.assert_not_called()


@pytest.mark.asyncio
async def test_interrupted_after_declaration_stays_done(tmp_path: Path) -> None:
    """A run that declared done before being interrupted keeps its declaration."""
    executor = _make_executor(tmp_path, "interrupted")
    manager = _queue_task(executor, "declared")

    async def declared_then_interrupted(*_args, **_kwargs):
        manager.update_status("declared", "done")
        manager.update_meta("declared", {"completed_by": "agent_declaration", "result_note": "verified"})
        yield {"type": "text_delta", "text": "output"}
        yield {
            "type": "cycle_done",
            "cycle_result": {
                "summary": "result",
                "action": "responded",
                "stop_kind": "interrupted",
                "tool_call_records": [],
            },
        }

    executor._anima.agent.run_cycle_streaming = declared_then_interrupted

    with _execution_patches():
        await executor._execute_llm_task(_task("declared"))

    entry = manager.get_task_by_id("declared")
    assert entry is not None
    assert entry.status == "done"
    assert not (executor._anima_dir / "state" / "pending" / "declared.json").exists()


@pytest.mark.asyncio
async def test_budget_skipped_keeps_queue_pending_and_records_activity(tmp_path: Path) -> None:
    executor = _make_executor(tmp_path, "budget_skipped")
    manager = _queue_task(executor, "budget")

    with (
        patch("core.paths.load_prompt", return_value="prompt"),
        patch("core.memory.activity.ActivityLogger") as activity,
    ):
        await executor._execute_llm_task(_task("budget"))

    entry = manager.get_task_by_id("budget")
    assert entry is not None
    assert entry.status == "pending"
    assert activity.return_value.log.call_args_list[-1].kwargs["meta"]["status"] == "budget_skipped"


@pytest.mark.asyncio
async def test_cancelled_batch_result_does_not_start_dependent(tmp_path: Path) -> None:
    executor = _make_executor(tmp_path)
    _queue_task(executor, "parent")
    manager = _queue_task(executor, "child")
    executor._run_task_in_worker = AsyncMock(return_value=_SENTINEL_CANCELLED)

    await executor._dispatch_batch(
        "batch",
        [
            {"task_id": "parent", "task_type": "llm", "parallel": False},
            {"task_id": "child", "task_type": "llm", "parallel": False, "depends_on": ["parent"]},
        ],
    )

    assert executor._run_task_in_worker.await_count == 1
    child = manager.get_task_by_id("child")
    assert child is not None
    assert child.status == "pending"
    assert child.summary == "stop kind test"  # title survives; the reason goes to meta
    assert child.meta["last_run_note"] == "a task this one depends on did not complete"
    assert child.meta["last_run_stop_kind"] == "dependency"


@pytest.mark.asyncio
async def test_normal_stop_with_declaration_completes(tmp_path: Path) -> None:
    executor = _make_executor(tmp_path)
    manager = _queue_task(executor, "normal")
    manager.update_meta("normal", {"completed_by": "agent_declaration", "result_note": "verified"})
    manager.update_status("normal", "done")

    with _execution_patches():
        await executor._execute_llm_task(_task("normal"))

    entry = manager.get_task_by_id("normal")
    assert entry is not None
    assert entry.status == "done"
    assert entry.summary == "verified"


@pytest.mark.asyncio
async def test_stream_error_is_not_suppressed_without_declaration(tmp_path: Path) -> None:
    executor = _make_executor(tmp_path)
    manager = _queue_task(executor, "stream-error")

    async def error_stream(*_args, **_kwargs):
        yield {"type": "error", "message": "connection lost"}
        yield {
            "type": "cycle_done",
            "cycle_result": {"summary": "partial", "action": "responded", "stop_kind": "normal"},
        }

    executor._anima.agent.run_cycle_streaming = error_stream
    with _execution_patches():
        await executor._execute_llm_task(_task("stream-error"))

    entry = manager.get_task_by_id("stream-error")
    assert entry is not None
    assert entry.status == "pending"
    assert "streaming error" in entry.meta["last_run_note"]
    assert entry.meta["last_run_stop_kind"] == "crash"
