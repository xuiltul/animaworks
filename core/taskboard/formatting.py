"""Formatting helpers for TaskBoard prompt sections."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from core.i18n import t
from core.memory.task_queue import (
    _STALE_TASK_THRESHOLD_SEC,
    TaskQueueManager,
    _elapsed_seconds,
    _format_elapsed_from_sec,
)
from core.taskboard.models import BoardTask
from core.time_utils import now_local


def format_tasks_for_priming(
    board_tasks: list[BoardTask],
    budget_tokens: int = 400,
    *,
    animas_dir: Path | None = None,
) -> str:
    """Format projected tasks in the compact Channel E style."""
    max_chars = budget_tokens * 4
    lines: list[str] = []
    total = 0
    now = now_local()

    active: list[BoardTask] = []
    delegated: list[BoardTask] = []

    for task in board_tasks:
        if task.queue_status == "delegated":
            delegated.append(task)
        else:
            active.append(task)

    active.sort(key=lambda task: (0 if task.source == "human" else 1, task.queue_updated_at or ""))

    for task in active:
        line = _format_active_task(task, now)
        if total + len(line) > max_chars:
            break
        lines.append(line)
        total += len(line) + 1

    if delegated and total < max_chars:
        delegated_section = _format_delegated_tasks(delegated, now, animas_dir, max_chars - total)
        if delegated_section:
            lines.append(delegated_section)

    return "\n".join(lines)


def _format_active_task(task: BoardTask, now: datetime) -> str:
    priority = "🔴 HIGH" if task.source == "human" else "⚪"
    status_icon = "🔄" if task.queue_status == "in_progress" else "📋"
    assignee = task.assignee or task.anima_name
    line = f"- {status_icon} {priority} [{task.task_id[:8]}] {_summary(task)} (assignee: {assignee})"
    if task.queue_status == "in_progress" and task.meta.get("executor") == "taskexec":
        line += f" {t('task_queue.auto_taskexec')}"
    elif task.queue_status and task.queue_status not in {"pending", "in_progress"}:
        line += f" status: {task.queue_status}"
    if task.relay_chain:
        line += f" chain: {' → '.join(task.relay_chain)}"

    elapsed_sec = _elapsed_seconds(task.queue_updated_at or "", now)
    elapsed_str = _format_elapsed_from_sec(elapsed_sec)
    if elapsed_str:
        line += f" {elapsed_str}"
    if elapsed_sec is not None and elapsed_sec >= _STALE_TASK_THRESHOLD_SEC:
        line += " ⚠️ STALE"

    return line


def _format_delegated_tasks(
    delegated: list[BoardTask],
    now: datetime,
    animas_dir: Path | None,
    budget_chars: int,
) -> str:
    lines: list[str] = []
    total = 0
    status_icons = {"done": "✅", "cancelled": "🚫"}
    unknown_label = t("task_queue.delegated_unknown")
    for task in delegated[:5]:
        meta = task.meta or {}
        target = meta.get("delegated_to", unknown_label)
        child_id = meta.get("delegated_task_id", "")
        sub_status = "?"
        if animas_dir is not None and child_id and target != unknown_label:
            sub_status = _resolve_subordinate_display(animas_dir / str(target), str(child_id))
        icon = status_icons.get(sub_status, "⏳")
        elapsed_sec = _elapsed_seconds(task.queue_updated_at or "", now)
        elapsed_str = _format_elapsed_from_sec(elapsed_sec) or ""
        line = f"- 📌 [{task.task_id[:8]}] {_summary(task)} → {target} ({target}: {sub_status} {icon}) {elapsed_str}"
        if total + len(line) > budget_chars:
            break
        lines.append(line)
        total += len(line) + 1
    return "\n".join(lines)


def _resolve_subordinate_display(target_dir: Path, child_id: str) -> str:
    if not target_dir.is_dir():
        return "?"
    try:
        sub_tqm = TaskQueueManager(target_dir)
        sub_task = sub_tqm.get_task_by_id(child_id)
        if sub_task:
            return sub_task.status
        archived = TaskQueueManager._search_archive(target_dir, child_id)
        return archived if archived else t("task_queue.delegated_archived")
    except Exception:
        return "?"


def _summary(task: BoardTask) -> str:
    return task.summary or task.original_instruction or task.task_id
