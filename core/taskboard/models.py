"""Pydantic models for TaskBoard metadata and projections."""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field

from core.time_utils import now_iso


class AttentionVisibility(StrEnum):
    """Task visibility states used by TaskBoard attention policy."""

    ACTIVE = "active"
    SNOOZED = "snoozed"
    EXPIRED = "expired"
    ARCHIVED = "archived"
    TOMBSTONED = "tombstoned"


class BoardColumn(StrEnum):
    """Display columns for TaskBoard."""

    TODO = "todo"
    RUNNING = "running"
    BLOCKED = "blocked"
    WAITING = "waiting"
    REVIEW = "review"
    DONE = "done"
    SUPPRESSED = "suppressed"


class TaskQueueRef(BaseModel):
    """Stable reference to one task_queue.jsonl entry."""

    anima_name: str
    task_id: str


class TaskBoardMetadata(BaseModel):
    """TaskBoard-only metadata layered over task_queue.jsonl entries."""

    anima_name: str
    task_id: str
    visibility: AttentionVisibility = AttentionVisibility.ACTIVE
    column: BoardColumn | None = None
    position: float | None = None
    expires_at: str | None = None
    snoozed_until: str | None = None
    last_notified_at: str | None = None
    notification_key: str | None = None
    surface_count: int = Field(default=0, ge=0)
    source_ref: str | None = None
    replaced_by: str | None = None
    tombstone_reason: str | None = None
    updated_at: str = Field(default_factory=now_iso)
    updated_by: str = "system"

    @property
    def ref(self) -> TaskQueueRef:
        """Return a compact queue reference for this metadata row."""
        return TaskQueueRef(anima_name=self.anima_name, task_id=self.task_id)


class BoardTask(BaseModel):
    """Projected task for TaskBoard views."""

    anima_name: str
    task_id: str
    queue_missing: bool = False

    source: Literal["human", "anima"] | None = None
    original_instruction: str | None = None
    assignee: str | None = None
    queue_status: str | None = None
    summary: str | None = None
    deadline: str | None = None
    relay_chain: list[str] = Field(default_factory=list)
    meta: dict[str, Any] = Field(default_factory=dict)
    queue_updated_at: str | None = None

    visibility: AttentionVisibility
    column: BoardColumn
    position: float | None = None
    expires_at: str | None = None
    snoozed_until: str | None = None
    last_notified_at: str | None = None
    notification_key: str | None = None
    surface_count: int = Field(default=0, ge=0)
    source_ref: str | None = None
    replaced_by: str | None = None
    tombstone_reason: str | None = None
    board_updated_at: str | None = None
    board_updated_by: str | None = None


class AttentionDecision(BaseModel):
    """Runtime decision derived from TaskBoard visibility and policy."""

    visible_in_prompt: bool = True
    executable: bool = True
    notify_allowed: bool = True
    reason: str = "active"
