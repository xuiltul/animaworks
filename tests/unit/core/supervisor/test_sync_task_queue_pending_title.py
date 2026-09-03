"""A pending sync must keep the task title and file the reason in meta."""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import MagicMock

from core.memory.task_queue import TaskQueueManager
from core.supervisor.pending_executor import PendingTaskExecutor, _classify_task_result


def _make_executor(tmp_path: Path) -> PendingTaskExecutor:
    anima_dir = tmp_path / "animas" / "test-anima"
    anima_dir.mkdir(parents=True, exist_ok=True)
    mock_anima = MagicMock()
    mock_anima._background_lock = asyncio.Lock()
    mock_anima._status_slots = {"background": "idle"}
    mock_anima._task_slots = {"background": ""}
    mock_anima._task_semaphore = None
    return PendingTaskExecutor(
        anima=mock_anima,
        anima_name="test-anima",
        anima_dir=anima_dir,
        shutdown_event=asyncio.Event(),
    )


def _add(manager: TaskQueueManager, title: str) -> str:
    entry = manager.add_task(source="anima", original_instruction="do the thing", assignee="test-anima", summary=title)
    manager.update_status(entry.task_id, "in_progress")
    return entry.task_id


def test_undeclared_end_keeps_title_and_records_note(tmp_path: Path) -> None:
    executor = _make_executor(tmp_path)
    manager = TaskQueueManager(executor._anima_dir)
    task_id = _add(manager, "PR #5215 review corrective")

    status, summary = _classify_task_result("(undeclared)")
    assert status == "pending"
    executor._sync_task_queue(task_id, status, summary=summary)

    entry = TaskQueueManager(executor._anima_dir).get_task_by_id(task_id)
    assert entry is not None
    assert entry.status == "pending"
    assert entry.summary == "PR #5215 review corrective"
    assert entry.meta["last_run_note"] == "run ended without a completion declaration"


def test_done_sync_still_writes_result_summary(tmp_path: Path) -> None:
    executor = _make_executor(tmp_path)
    manager = TaskQueueManager(executor._anima_dir)
    task_id = _add(manager, "PR #5215 review corrective")

    executor._sync_task_queue(task_id, "done", summary="pushed fix")

    entry = TaskQueueManager(executor._anima_dir).get_task_by_id(task_id)
    assert entry is not None
    assert entry.status == "done"
    assert entry.summary == "pushed fix"
