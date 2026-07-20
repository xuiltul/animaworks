from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""External tasks widget API — reads from JSON snapshot store."""

import asyncio
import logging
import uuid
from datetime import datetime
from enum import StrEnum

from fastapi import APIRouter, Query, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from core.external_tasks.store import ExternalTaskStore
from core.paths import get_external_tasks_store_path
from core.time_utils import ensure_aware

logger = logging.getLogger("animaworks.routes.external_tasks")

# ── Constants ────────────────────────────────────


class SourceType(StrEnum):
    GITHUB = "github"
    SLACK = "slack"
    GMAIL = "gmail"
    CHATWORK = "chatwork"
    JIRA = "jira"
    NOTION = "notion"
    OTHER = "other"


class TaskStatus(StrEnum):
    OPEN = "open"
    IN_PROGRESS = "in_progress"
    DONE = "done"
    CANCELLED = "cancelled"


VALID_SORT_KEYS = {"priority", "created_at", "last_updated_at"}
VALID_ORDER = {"asc", "desc"}


# ── Response models ──────────────────────────────


class ExternalTaskItem(BaseModel):
    id: str
    title: str
    status: str
    source_type: str
    source_icon: str
    source_url: str | None = None
    created_at: str
    last_updated_at: str
    priority: int


class SourceHealthMeta(BaseModel):
    status: str
    collected_at: str | None = None
    error: str | None = None


class PaginationMeta(BaseModel):
    total_count: int
    limit: int
    offset: int
    has_more: bool
    last_collected_at: str | None = None
    sources: dict[str, SourceHealthMeta] = Field(default_factory=dict)


class ExternalTasksResponse(BaseModel):
    data: list[ExternalTaskItem]
    meta: PaginationMeta


class ErrorDetail(BaseModel):
    field: str
    value: str
    constraint: str | None = None
    allowed: list[str] | None = None


class ErrorBody(BaseModel):
    code: str
    message: str
    trace_id: str | None = None
    details: list[ErrorDetail] | None = None


class ErrorResponse(BaseModel):
    error: ErrorBody


# ── Helpers ──────────────────────────────────────


def _error_response(status_code: int, code: str, message: str, details: list[dict] | None = None) -> JSONResponse:
    body: dict = {"error": {"code": code, "message": message}}
    if status_code >= 500:
        body["error"]["trace_id"] = str(uuid.uuid4())[:12]
    if details:
        body["error"]["details"] = details
    return JSONResponse(status_code=status_code, content=body)


def _load_snapshot():
    """Blocking load of the external tasks snapshot (for asyncio.to_thread)."""
    return ExternalTaskStore(get_external_tasks_store_path()).load()


def _task_updated_since(task: dict, since_dt: datetime) -> bool:
    """Return True if task.last_updated_at >= since_dt.

    Unparseable / empty timestamps are excluded (skipped) so a single bad
    row cannot 500 the whole endpoint. Naive/aware are aligned via ensure_aware.
    """
    raw = task.get("last_updated_at") or ""
    try:
        text = raw.replace("Z", "+00:00") if isinstance(raw, str) else str(raw)
        updated = ensure_aware(datetime.fromisoformat(text))
    except (TypeError, ValueError):
        return False
    return updated >= since_dt


# ── Router ───────────────────────────────────────


def create_external_tasks_router() -> APIRouter:
    router = APIRouter(tags=["external-tasks"])

    @router.get("/external-tasks", response_model=ExternalTasksResponse)
    async def get_external_tasks(
        request: Request,
        limit: int = Query(default=20, ge=1, le=100),
        offset: int = Query(default=0, ge=0),
        status: str | None = Query(default=None),
        source_type: str | None = Query(default=None),
        since: str | None = Query(default=None),
        sort: str = Query(default="priority"),
        order: str = Query(default="desc"),
    ):
        """Get external tasks for the widget from the snapshot store."""

        # Validate sort key
        if sort not in VALID_SORT_KEYS:
            return _error_response(
                400,
                "INVALID_PARAMETER",
                f"Invalid value for 'sort': must be one of {', '.join(VALID_SORT_KEYS)}.",
                [{"field": "sort", "value": sort, "allowed": list(VALID_SORT_KEYS)}],
            )

        # Validate order
        if order not in VALID_ORDER:
            return _error_response(
                400,
                "INVALID_PARAMETER",
                "Invalid value for 'order': must be 'asc' or 'desc'.",
                [{"field": "order", "value": order, "allowed": list(VALID_ORDER)}],
            )

        # Parse status filter
        status_filter: set[str] | None = None
        if status:
            status_values = {s.strip() for s in status.split(",")}
            valid_statuses = {e.value for e in TaskStatus}
            invalid = status_values - valid_statuses
            if invalid:
                return _error_response(
                    422,
                    "INVALID_FILTER",
                    f"Unknown status: '{', '.join(invalid)}'. Allowed values: {', '.join(sorted(valid_statuses))}.",
                    [{"field": "status", "value": status, "allowed": sorted(valid_statuses)}],
                )
            status_filter = status_values

        # Parse source_type filter
        source_filter: set[str] | None = None
        if source_type:
            source_values = {s.strip() for s in source_type.split(",")}
            valid_sources = {e.value for e in SourceType}
            invalid = source_values - valid_sources
            if invalid:
                return _error_response(
                    422,
                    "INVALID_FILTER",
                    f"Unknown source_type: '{', '.join(invalid)}'. Allowed values: {', '.join(sorted(valid_sources))}.",
                    [{"field": "source_type", "value": source_type, "allowed": sorted(valid_sources)}],
                )
            source_filter = source_values

        # Parse since filter
        since_dt: datetime | None = None
        if since:
            try:
                since_dt = ensure_aware(
                    datetime.fromisoformat(since.replace("Z", "+00:00"))
                )
            except ValueError:
                return _error_response(
                    400,
                    "INVALID_PARAMETER",
                    "Invalid value for 'since': must be ISO 8601 format.",
                    [{"field": "since", "value": since, "constraint": "ISO 8601"}],
                )

        # Load snapshot from store (off event loop)
        snapshot = await asyncio.to_thread(_load_snapshot)
        tasks = [t.model_dump() for t in snapshot.tasks]

        # Apply filters
        if status_filter:
            tasks = [t for t in tasks if t["status"] in status_filter]
        if source_filter:
            tasks = [t for t in tasks if t["source_type"] in source_filter]
        if since_dt:
            tasks = [t for t in tasks if _task_updated_since(t, since_dt)]

        # Sort
        reverse = order == "desc"
        sort_key = sort
        if sort_key == "priority":
            tasks.sort(key=lambda t: (t["priority"], t["created_at"]), reverse=reverse)
        else:
            tasks.sort(key=lambda t: t.get(sort_key, ""), reverse=reverse)

        total_count = len(tasks)

        # Paginate
        paginated = tasks[offset : offset + limit]
        has_more = (offset + limit) < total_count

        sources_meta = {
            name: {
                "status": health.status,
                "collected_at": health.collected_at,
                "error": health.error,
            }
            for name, health in snapshot.sources.items()
        }

        return {
            "data": paginated,
            "meta": {
                "total_count": total_count,
                "limit": limit,
                "offset": offset,
                "has_more": has_more,
                "last_collected_at": snapshot.last_collected_at,
                "sources": sources_meta,
            },
        }

    return router
