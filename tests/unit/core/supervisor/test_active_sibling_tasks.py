"""Tests for sibling-worker visibility injection (_format_active_sibling_tasks)."""

from __future__ import annotations

import asyncio
from pathlib import Path

from core.memory.task_queue import TaskQueueManager
from core.supervisor.pending_executor import PendingTaskExecutor


def _executor(tmp_path: Path) -> PendingTaskExecutor:
    anima_dir = tmp_path / "anima"
    anima_dir.mkdir(exist_ok=True)
    return PendingTaskExecutor(
        anima=None,
        anima_name="sibling-test",
        anima_dir=anima_dir,
        shutdown_event=asyncio.Event(),
    )


def _add_task(
    anima_dir: Path,
    task_id: str,
    summary: str,
    status: str = "in_progress",
) -> None:
    manager = TaskQueueManager(anima_dir)
    manager.add_task(
        source="anima",
        original_instruction=f"instruction for {task_id}",
        assignee="sibling-test",
        summary=summary,
        task_id=task_id,
        status="in_progress" if status == "in_progress" else "pending",
    )
    if status not in ("pending", "in_progress"):
        manager.update_status(task_id, status)


def test_no_siblings_returns_empty(tmp_path: Path) -> None:
    executor = _executor(tmp_path)
    assert executor._format_active_sibling_tasks("current-task") == ""


def test_excludes_current_task(tmp_path: Path) -> None:
    executor = _executor(tmp_path)
    _add_task(executor._anima_dir, "current-task", "[PR #3440] fixing")
    assert executor._format_active_sibling_tasks("current-task") == ""


def test_formats_sibling_one_liners(tmp_path: Path) -> None:
    executor = _executor(tmp_path)
    _add_task(executor._anima_dir, "current-task", "[PR #3440] fixing")
    _add_task(executor._anima_dir, "sibling-one", "[PR #3442] addressing review")
    _add_task(executor._anima_dir, "sibling-two", "line one\nline two")

    result = executor._format_active_sibling_tasks("current-task")
    lines = result.splitlines()

    assert len(lines) == 2
    assert any("[sibling-" in line and "[PR #3442] addressing review" in line for line in lines)
    # Multi-line summaries collapse to their first line
    assert any("line one" in line and "line two" not in line for line in lines)
    # Current task is never listed
    assert "[PR #3440]" not in result


def test_only_in_progress_tasks_listed(tmp_path: Path) -> None:
    executor = _executor(tmp_path)
    _add_task(executor._anima_dir, "done-task", "[PR #3444] merged", status="done")
    _add_task(executor._anima_dir, "pending-task", "[PR #3445] queued", status="pending")
    _add_task(executor._anima_dir, "running-task", "[PR #3446] running")

    result = executor._format_active_sibling_tasks("current-task")

    assert "[PR #3446]" in result
    assert "[PR #3444]" not in result
    assert "[PR #3445]" not in result


def test_respects_limit_and_truncates_long_summary(tmp_path: Path) -> None:
    executor = _executor(tmp_path)
    for i in range(10):
        _add_task(executor._anima_dir, f"sibling-{i:02d}", f"[PR #{3400 + i}] " + "x" * 300)

    result = executor._format_active_sibling_tasks("current-task", limit=8)
    lines = result.splitlines()

    assert len(lines) == 8
    assert all("..." in line for line in lines)
    assert all(len(line) < 250 for line in lines)
