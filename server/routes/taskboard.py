from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field, field_validator

from core.memory.task_queue import TaskQueueManager
from core.taskboard.models import AttentionVisibility, BoardColumn, BoardTask
from core.taskboard.projector import project_all, project_anima
from core.taskboard.store import TaskBoardStore
from core.time_utils import now_iso

logger = logging.getLogger("animaworks.routes.taskboard")

_ACTIVE_QUEUE_STATUSES = {"pending", "in_progress", "blocked", "delegated"}
_SUPPRESSED_VISIBILITIES = {
    AttentionVisibility.EXPIRED,
    AttentionVisibility.ARCHIVED,
    AttentionVisibility.TOMBSTONED,
}
_CANCEL_QUEUE_VISIBILITIES = _SUPPRESSED_VISIBILITIES
_COLUMN_TITLES = {
    BoardColumn.TODO: "Todo",
    BoardColumn.RUNNING: "Running",
    BoardColumn.BLOCKED: "Blocked",
    BoardColumn.WAITING: "Waiting",
    BoardColumn.REVIEW: "Review",
    BoardColumn.DONE: "Done",
    BoardColumn.SUPPRESSED: "Suppressed",
}


class TaskBoardPatchRequest(BaseModel):
    visibility: AttentionVisibility | None = None
    column: BoardColumn | None = None
    position: float | None = None
    expires_at: str | None = None
    snoozed_until: str | None = None
    notification_key: str | None = None
    reason: str | None = None
    actor: str | None = None

    @field_validator("expires_at", "snoozed_until")
    @classmethod
    def _validate_iso_datetime(cls, value: str | None) -> str | None:
        if value is None:
            return value
        try:
            datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError("must be an ISO 8601 datetime") from exc
        return value


class NotificationAckRequest(BaseModel):
    notification_key: str = Field(min_length=1)
    actor: str | None = None


def create_taskboard_router() -> APIRouter:
    router = APIRouter()

    @router.get("/task-board")
    async def list_task_board(
        request: Request,
        assignee: str | None = None,
        visibility: AttentionVisibility | None = None,
        column: BoardColumn | None = None,
        include_archived: bool = False,
        include_missing: bool = False,
        q: str | None = None,
    ) -> dict[str, Any]:
        """Return the canonical task projection plus TaskBoard presentation metadata."""
        try:
            return await asyncio.to_thread(
                _list_task_board,
                request,
                assignee=assignee,
                visibility=visibility,
                column=column,
                include_archived=include_archived,
                include_missing=include_missing,
                q=q,
            )
        except HTTPException:
            raise
        except Exception as exc:
            logger.exception("TaskBoard list failed")
            raise HTTPException(
                status_code=500,
                detail={"error": "taskboard_unavailable", "message": "TaskBoard API failed"},
            ) from exc

    @router.get("/task-board/summary")
    async def get_task_board_summary(request: Request) -> dict[str, int]:
        """Return TaskBoard summary counts for dashboard use."""
        try:
            paths = _resolve_paths(request)
            return await asyncio.to_thread(
                summarize_task_board,
                paths["animas_dir"],
                paths["shared_dir"],
                paths["anima_names"],
            )
        except Exception as exc:
            logger.exception("TaskBoard summary failed")
            raise HTTPException(
                status_code=500,
                detail={"error": "taskboard_unavailable", "message": "TaskBoard API failed"},
            ) from exc

    @router.patch("/task-board/{anima_name}/{task_id}")
    async def patch_task_board(
        request: Request,
        anima_name: str,
        task_id: str,
        payload: TaskBoardPatchRequest,
    ) -> dict[str, Any]:
        """Update TaskBoard metadata without mutating Board channel data."""
        try:
            return await asyncio.to_thread(_patch_task_board, request, anima_name, task_id, payload)
        except HTTPException:
            raise
        except Exception as exc:
            logger.exception("TaskBoard patch failed")
            raise HTTPException(
                status_code=500,
                detail={"error": "taskboard_unavailable", "message": "TaskBoard API failed"},
            ) from exc

    @router.post("/task-board/{anima_name}/{task_id}/notification-ack")
    async def acknowledge_notification(
        request: Request,
        anima_name: str,
        task_id: str,
        payload: NotificationAckRequest,
    ) -> dict[str, Any]:
        """Record that a runtime notification was acknowledged."""
        try:
            return await asyncio.to_thread(_acknowledge_notification, request, anima_name, task_id, payload)
        except HTTPException:
            raise
        except Exception as exc:
            logger.exception("TaskBoard notification acknowledgement failed")
            raise HTTPException(
                status_code=500,
                detail={"error": "taskboard_unavailable", "message": "TaskBoard API failed"},
            ) from exc

    return router


def summarize_task_board(
    animas_dir: Path,
    shared_dir: Path,
    anima_names: list[str] | tuple[str, ...] | set[str],
) -> dict[str, int]:
    """Summarize projected TaskBoard tasks, respecting TaskBoard visibility."""
    store = _store_for(shared_dir)
    tasks = project_all(
        animas_dir,
        store,
        anima_names=anima_names,
        include_missing=True,
        include_archived=True,
    )
    summary = {
        "pending": 0,
        "in_progress": 0,
        "blocked": 0,
        "delegated": 0,
        "failed_review": 0,
        "snoozed": 0,
        "suppressed": 0,
        "total_active": 0,
    }

    for task in tasks:
        if task.visibility == AttentionVisibility.SNOOZED:
            summary["snoozed"] += 1
            continue
        if task.visibility in _SUPPRESSED_VISIBILITIES:
            summary["suppressed"] += 1
            continue
        if task.visibility != AttentionVisibility.ACTIVE:
            continue

        status = task.queue_status
        if status == "pending":
            summary["pending"] += 1
        elif status == "in_progress":
            summary["in_progress"] += 1
        elif status == "blocked":
            summary["blocked"] += 1
        elif status == "delegated":
            summary["delegated"] += 1
        elif status == "failed" or task.column == BoardColumn.REVIEW:
            summary["failed_review"] += 1

    summary["total_active"] = (
        summary["pending"]
        + summary["in_progress"]
        + summary["blocked"]
        + summary["delegated"]
        + summary["failed_review"]
    )
    return summary


def _list_task_board(
    request: Request,
    *,
    assignee: str | None,
    visibility: AttentionVisibility | None,
    column: BoardColumn | None,
    include_archived: bool,
    include_missing: bool,
    q: str | None,
) -> dict[str, Any]:
    paths = _resolve_paths(request)
    animas_dir = paths["animas_dir"]
    shared_dir = paths["shared_dir"]
    anima_names = paths["anima_names"]
    selected_names = _selected_anima_names(animas_dir, anima_names, assignee)
    store = _store_for(shared_dir)

    tasks = project_all(
        animas_dir,
        store,
        anima_names=selected_names,
        include_missing=include_missing,
        include_archived=True,
    )
    tasks = [
        task
        for task in tasks
        if _matches_filters(
            task,
            visibility=visibility,
            column=column,
            include_archived=include_archived,
            q=q,
        )
    ]

    return {
        "columns": _column_response(tasks),
        "tasks": [_task_to_response(task) for task in tasks],
        "counts": _visibility_counts(tasks),
        "meta": {
            "warnings": {
                "corrupt_task_queue_lines": _count_corrupt_task_queue_lines(animas_dir, selected_names),
            }
        },
    }


def _patch_task_board(
    request: Request,
    anima_name: str,
    task_id: str,
    payload: TaskBoardPatchRequest,
) -> dict[str, Any]:
    paths = _resolve_paths(request)
    animas_dir = paths["animas_dir"]
    shared_dir = paths["shared_dir"]
    _ensure_known_anima(animas_dir, paths["anima_names"], anima_name)

    anima_dir = animas_dir / anima_name
    queue = TaskQueueManager(anima_dir)
    store = _store_for(shared_dir)
    queue_entry = queue.get_task_by_id(task_id)
    metadata = store.get_metadata(anima_name, task_id)
    if queue_entry is None and metadata is None:
        raise HTTPException(status_code=404, detail={"error": "task_not_found", "task_id": task_id})

    fields_set = payload.model_fields_set
    if payload.visibility == AttentionVisibility.SNOOZED and (
        "snoozed_until" not in fields_set or payload.snoozed_until is None
    ):
        raise HTTPException(
            status_code=422,
            detail={"error": "snoozed_until_required", "message": "snoozed visibility requires snoozed_until"},
        )

    actor = _resolve_actor(request, payload.actor, default="human")
    updates = _metadata_updates(payload)
    if updates:
        store.upsert_metadata(
            anima_name=anima_name,
            task_id=task_id,
            actor=actor,
            event_type=_event_type_for_patch(payload),
            **updates,
        )

    if (
        payload.visibility in _CANCEL_QUEUE_VISIBILITIES
        and queue_entry is not None
        and queue_entry.status in _ACTIVE_QUEUE_STATUSES
    ):
        reason = f": {payload.reason}" if payload.reason else ""
        queue.update_status(task_id, "cancelled", summary=f"{payload.visibility.value} by TaskBoard{reason}")

    return {"task": _load_projected_task(animas_dir, store, anima_name, task_id)}


def _acknowledge_notification(
    request: Request,
    anima_name: str,
    task_id: str,
    payload: NotificationAckRequest,
) -> dict[str, Any]:
    paths = _resolve_paths(request)
    animas_dir = paths["animas_dir"]
    shared_dir = paths["shared_dir"]
    _ensure_known_anima(animas_dir, paths["anima_names"], anima_name)

    anima_dir = animas_dir / anima_name
    queue = TaskQueueManager(anima_dir)
    store = _store_for(shared_dir)
    if queue.get_task_by_id(task_id) is None and store.get_metadata(anima_name, task_id) is None:
        raise HTTPException(status_code=404, detail={"error": "task_not_found", "task_id": task_id})

    store.upsert_metadata(
        anima_name=anima_name,
        task_id=task_id,
        actor=_resolve_actor(request, payload.actor, default="runtime"),
        event_type="notification_acknowledged",
        last_notified_at=now_iso(),
        notification_key=payload.notification_key,
    )
    return {"ok": True, "task": _load_projected_task(animas_dir, store, anima_name, task_id)}


def _metadata_updates(payload: TaskBoardPatchRequest) -> dict[str, Any]:
    updates: dict[str, Any] = {}
    for field_name in (
        "visibility",
        "column",
        "position",
        "expires_at",
        "snoozed_until",
        "notification_key",
    ):
        if field_name in payload.model_fields_set:
            value = getattr(payload, field_name)
            updates[field_name] = value.value if hasattr(value, "value") else value
    return updates


def _event_type_for_patch(payload: TaskBoardPatchRequest) -> str | None:
    if "visibility" not in payload.model_fields_set or payload.visibility is None:
        return None
    if payload.visibility == AttentionVisibility.ACTIVE:
        return "visibility_changed"
    return payload.visibility.value


def _resolve_actor(request: Request, requested_actor: str | None, *, default: str) -> str:
    user = getattr(request.state, "user", None)
    username = getattr(user, "username", None)
    if username:
        return str(username)
    return requested_actor or default


def _load_projected_task(
    animas_dir: Path,
    store: TaskBoardStore,
    anima_name: str,
    task_id: str,
) -> dict[str, Any]:
    tasks = project_anima(
        animas_dir / anima_name,
        store,
        anima_name=anima_name,
        include_missing=True,
        include_archived=True,
    )
    for task in tasks:
        if task.task_id == task_id:
            return _task_to_response(task)
    raise HTTPException(status_code=404, detail={"error": "task_not_found", "task_id": task_id})


def _resolve_paths(request: Request) -> dict[str, Any]:
    animas_dir = Path(request.app.state.animas_dir)
    shared_dir = Path(getattr(request.app.state, "shared_dir", animas_dir.parent / "shared"))
    raw_names = getattr(request.app.state, "anima_names", None)
    if raw_names is None:
        anima_names = sorted(path.name for path in animas_dir.iterdir() if path.is_dir()) if animas_dir.exists() else []
    else:
        anima_names = list(raw_names)
    return {"animas_dir": animas_dir, "shared_dir": shared_dir, "anima_names": anima_names}


def _store_for(shared_dir: Path) -> TaskBoardStore:
    return TaskBoardStore(shared_dir / "taskboard.sqlite3")


def _selected_anima_names(animas_dir: Path, anima_names: list[str], assignee: str | None) -> list[str]:
    if assignee is None:
        return sorted(set(anima_names))
    _ensure_known_anima(animas_dir, anima_names, assignee)
    return [assignee]


def _ensure_known_anima(animas_dir: Path, anima_names: list[str], anima_name: str) -> None:
    if anima_name in set(anima_names) or (animas_dir / anima_name).is_dir():
        return
    raise HTTPException(status_code=404, detail={"error": "anima_not_found", "anima_name": anima_name})


def _matches_filters(
    task: BoardTask,
    *,
    visibility: AttentionVisibility | None,
    column: BoardColumn | None,
    include_archived: bool,
    q: str | None,
) -> bool:
    if visibility is not None:
        if task.visibility != visibility:
            return False
    elif not include_archived and task.visibility != AttentionVisibility.ACTIVE:
        return False

    if column is not None and task.column != column:
        return False

    if q:
        needle = q.casefold()
        haystack = " ".join(value for value in (task.summary, task.original_instruction) if value).casefold()
        if needle not in haystack:
            return False

    return True


def _visibility_counts(tasks: list[BoardTask]) -> dict[str, int]:
    counts = {visibility.value: 0 for visibility in AttentionVisibility}
    for task in tasks:
        counts[task.visibility.value] += 1
    return counts


def _column_response(tasks: list[BoardTask]) -> list[dict[str, Any]]:
    counts = {column.value: 0 for column in BoardColumn}
    for task in tasks:
        counts[task.column.value] += 1
    return [
        {"id": column.value, "title": _COLUMN_TITLES[column], "count": counts[column.value]} for column in BoardColumn
    ]


def _task_to_response(task: BoardTask) -> dict[str, Any]:
    data = task.model_dump(mode="json")
    timestamps = [value for value in (task.queue_updated_at, task.board_updated_at) if value]
    data["updated_at"] = (
        max(timestamps, key=lambda value: datetime.fromisoformat(value.replace("Z", "+00:00"))) if timestamps else None
    )
    return data


def _count_corrupt_task_queue_lines(animas_dir: Path, anima_names: list[str]) -> int:
    from core.memory.task_queue import TaskQueueManager

    return sum(
        TaskQueueManager(animas_dir / name).store.maintenance_status(name)["invalid_import_rows"]
        for name in anima_names
    )
