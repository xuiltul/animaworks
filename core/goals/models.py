from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Pydantic models for the persistent goal loop."""

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from core.time_utils import now_iso

GoalStatus = Literal["active", "paused", "done", "blocked", "cleared"]
GoalVerdict = Literal["done", "continue", "blocked"]
GoalEventType = Literal[
    "set",
    "pause",
    "resume",
    "clear",
    "judgment",
    "continuation_created",
]


class GoalJudgment(BaseModel):
    """A judge verdict for one completed TaskExec iteration."""

    goal_id: str
    task_id: str = ""
    verdict: GoalVerdict
    reason: str = ""
    continuation_prompt: str = ""
    verification_output: str = ""
    raw_response: str = ""
    failed_open: bool = False
    iteration: int = 0
    created_at: str = Field(default_factory=now_iso)

    @field_validator("reason", "continuation_prompt", "verification_output", "raw_response", mode="before")
    @classmethod
    def _coerce_text(cls, value: object) -> str:
        if value is None:
            return ""
        return str(value)


class GoalState(BaseModel):
    """Current goal state reconstructed from ``state/goal_state.jsonl``."""

    goal_id: str
    title: str = ""
    objective: str
    success_criteria: list[str] = Field(default_factory=list)
    status: GoalStatus = "active"
    max_iterations: int = Field(default=5, ge=1)
    iteration_count: int = 0
    judge_model: str | None = None
    skill_refs: list[str] = Field(default_factory=list)
    skill_rejections: list[dict[str, str]] = Field(default_factory=list)
    related_task_ids: list[str] = Field(default_factory=list)
    last_task_id: str = ""
    last_result_summary: str = ""
    last_judgment: GoalJudgment | None = None
    blocked_reason: str = ""
    created_at: str = Field(default_factory=now_iso)
    updated_at: str = Field(default_factory=now_iso)
    paused_at: str | None = None
    completed_at: str | None = None
    cleared_at: str | None = None

    @field_validator("success_criteria", mode="before")
    @classmethod
    def _coerce_criteria(cls, value: object) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [value] if value.strip() else []
        if isinstance(value, (list, tuple, set)):
            return [str(item).strip() for item in value if str(item).strip()]
        return [str(value).strip()] if str(value).strip() else []


class GoalEvent(BaseModel):
    """Append-only event for goal state replay."""

    event_id: str
    ts: str = Field(default_factory=now_iso)
    event_type: GoalEventType
    goal_id: str
    actor: str = "system"
    payload: dict[str, Any] = Field(default_factory=dict)
