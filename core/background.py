from __future__ import annotations
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.


"""Background task execution manager.

Provides ``BackgroundTaskManager`` which executes long-running tool calls
asynchronously and notifies upon completion.  Results are persisted to
``state/background_tasks/{task_id}.json`` for later retrieval.
"""

import asyncio
import json
import logging
import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from core.exceptions import AnimaWorksError  # noqa: F401

logger = logging.getLogger("animaworks.background")


# ── Data Models ──────────────────────────────────────────────


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class BackgroundTask:
    """Represents a single background task."""

    task_id: str
    anima_name: str
    tool_name: str
    tool_args: dict[str, Any]
    status: TaskStatus = TaskStatus.PENDING
    created_at: float = field(default_factory=time.time)
    completed_at: float | None = None
    result: str | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "anima_name": self.anima_name,
            "tool_name": self.tool_name,
            "tool_args": self.tool_args,
            "status": self.status.value,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
            "result": self.result,
            "error": self.error,
        }

    def summary(self) -> str:
        """Return a short human-readable summary."""
        if self.status == TaskStatus.COMPLETED:
            preview = (self.result or "")[:200]
            return f"[{self.tool_name}] completed: {preview}"
        if self.status == TaskStatus.FAILED:
            return f"[{self.tool_name}] failed: {self.error}"
        return f"[{self.tool_name}] {self.status.value}"


# Callback type for completion notifications
OnTaskCompleteFn = Callable[[BackgroundTask], Awaitable[None]]


# ── Default eligible tools ───────────────────────────────────

_DEFAULT_ELIGIBLE_TOOLS: dict[str, int] = {
    "generate_character_assets": 30,
    "generate_fullbody": 30,
    "generate_bustup": 30,
    "generate_chibi": 30,
    "generate_3d_model": 30,
    "generate_rigged_model": 30,
    "generate_animations": 30,
    "local_llm": 60,
    "run_command": 60,
}


# ── BackgroundTaskManager ────────────────────────────────────


class BackgroundTaskManager:
    """Manages background execution of long-running tool calls.

    Usage::

        mgr = BackgroundTaskManager(anima_dir, anima_name="sakura")
        mgr.on_complete = my_notify_callback

        if mgr.is_eligible("generate_character_assets"):
            task_id = mgr.submit("generate_character_assets", args, execute_fn)
            # Returns immediately; execute_fn runs in background.
    """

    def __init__(
        self,
        anima_dir: Path,
        anima_name: str = "",
        eligible_tools: dict[str, int] | None = None,
    ) -> None:
        self._anima_dir = anima_dir
        self._anima_name = anima_name or anima_dir.name
        self._eligible_tools = eligible_tools or dict(_DEFAULT_ELIGIBLE_TOOLS)
        self._tasks: dict[str, BackgroundTask] = {}
        self._async_tasks: dict[str, asyncio.Task[None]] = {}
        self.on_complete: OnTaskCompleteFn | None = None

        # Ensure storage directory exists
        self._storage_dir.mkdir(parents=True, exist_ok=True)

    @property
    def _storage_dir(self) -> Path:
        return self._anima_dir / "state" / "background_tasks"

    # ── Public API ───────────────────────────────────────────

    @classmethod
    def from_profiles(
        cls,
        anima_dir: Path,
        anima_name: str = "",
        profiles: dict[str, dict[str, dict[str, object]]] | None = None,
        config_eligible: dict[str, int] | None = None,
    ) -> "BackgroundTaskManager":
        """Create a BackgroundTaskManager with eligible tools derived from EXECUTION_PROFILE.

        Merges three layers (later overrides earlier):
        1. ``_DEFAULT_ELIGIBLE_TOOLS`` — hardcoded defaults for A2 compat
        2. ``profiles`` — EXECUTION_PROFILE from tool modules
        3. ``config_eligible`` — explicit config.json overrides
        """
        from core.tools._base import get_eligible_tools_from_profiles

        eligible = dict(_DEFAULT_ELIGIBLE_TOOLS)  # Layer 1
        if profiles:
            profile_eligible = get_eligible_tools_from_profiles(profiles)
            eligible.update(profile_eligible)  # Layer 2
        if config_eligible:
            eligible.update(config_eligible)  # Layer 3 (highest priority)

        return cls(anima_dir, anima_name=anima_name, eligible_tools=eligible)

    def is_eligible(self, tool_name: str) -> bool:
        """Check if a tool is eligible for background execution.

        Accepts both formats:
        - Schema name: ``"generate_3d_model"`` (Mode A2)
        - Profile key: ``"image_gen:3d"`` (Mode A1 submit)
        """
        return tool_name in self._eligible_tools

    def submit(
        self,
        tool_name: str,
        tool_args: dict[str, Any],
        execute_fn: Callable[[str, dict[str, Any]], str],
    ) -> str:
        """Submit a tool call for background execution.

        Args:
            tool_name: Name of the tool to execute.
            tool_args: Arguments to pass to the tool.
            execute_fn: Synchronous callable ``(name, args) -> result_str``.

        Returns:
            The generated task_id.
        """
        task_id = uuid.uuid4().hex[:12]
        task = BackgroundTask(
            task_id=task_id,
            anima_name=self._anima_name,
            tool_name=tool_name,
            tool_args=tool_args,
            status=TaskStatus.RUNNING,
        )
        self._tasks[task_id] = task
        self._save_task(task)

        logger.info(
            "Background task submitted: id=%s tool=%s anima=%s",
            task_id, tool_name, self._anima_name,
        )

        # Schedule the async wrapper
        async_task = asyncio.create_task(
            self._run_task(task, execute_fn),
            name=f"bg-{task_id}",
        )
        self._async_tasks[task_id] = async_task
        return task_id

    async def submit_async(
        self,
        tool_name: str,
        tool_args: dict[str, Any],
        execute_fn: Callable[[str, dict[str, Any]], Awaitable[str]],
    ) -> str:
        """Submit an async tool call for background execution.

        Same as :meth:`submit` but *execute_fn* is an async callable.
        """
        task_id = uuid.uuid4().hex[:12]
        task = BackgroundTask(
            task_id=task_id,
            anima_name=self._anima_name,
            tool_name=tool_name,
            tool_args=tool_args,
            status=TaskStatus.RUNNING,
        )
        self._tasks[task_id] = task
        self._save_task(task)

        logger.info(
            "Background task submitted (async): id=%s tool=%s anima=%s",
            task_id, tool_name, self._anima_name,
        )

        async_task = asyncio.create_task(
            self._run_task_async(task, execute_fn),
            name=f"bg-{task_id}",
        )
        self._async_tasks[task_id] = async_task
        return task_id

    def get_task(self, task_id: str) -> BackgroundTask | None:
        """Get a background task by ID (in-memory first, then disk)."""
        task = self._tasks.get(task_id)
        if task:
            return task
        return self._load_task(task_id)

    def list_tasks(
        self,
        *,
        status: TaskStatus | None = None,
    ) -> list[BackgroundTask]:
        """List all tasks, optionally filtered by status."""
        # Merge in-memory and on-disk tasks
        all_tasks = dict(self._tasks)
        for path in self._storage_dir.glob("*.json"):
            tid = path.stem
            if tid not in all_tasks:
                loaded = self._load_task(tid)
                if loaded:
                    all_tasks[tid] = loaded

        tasks = list(all_tasks.values())
        if status is not None:
            tasks = [t for t in tasks if t.status == status]
        tasks.sort(key=lambda t: t.created_at, reverse=True)
        return tasks

    def active_count(self) -> int:
        """Return the number of currently running background tasks."""
        return sum(
            1 for t in self._tasks.values()
            if t.status == TaskStatus.RUNNING
        )

    # ── Internal execution ───────────────────────────────────

    async def _run_task(
        self,
        task: BackgroundTask,
        execute_fn: Callable[[str, dict[str, Any]], str],
    ) -> None:
        """Run a synchronous tool call in a thread and handle completion."""
        try:
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(
                None,
                execute_fn,
                task.tool_name,
                task.tool_args,
            )
            task.status = TaskStatus.COMPLETED
            task.result = result
            task.completed_at = time.time()
            logger.info(
                "Background task completed: id=%s tool=%s",
                task.task_id, task.tool_name,
            )
        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error = f"{type(e).__name__}: {e}"
            task.completed_at = time.time()
            logger.exception(
                "Background task failed: id=%s tool=%s",
                task.task_id, task.tool_name,
            )
        finally:
            self._save_task(task)
            self._async_tasks.pop(task.task_id, None)
            if self.on_complete:
                try:
                    await self.on_complete(task)
                except Exception:
                    logger.exception(
                        "on_complete callback failed for task %s", task.task_id,
                    )

    async def _run_task_async(
        self,
        task: BackgroundTask,
        execute_fn: Callable[[str, dict[str, Any]], Awaitable[str]],
    ) -> None:
        """Run an async tool call and handle completion."""
        try:
            result = await execute_fn(task.tool_name, task.tool_args)
            task.status = TaskStatus.COMPLETED
            task.result = result
            task.completed_at = time.time()
            logger.info(
                "Background task completed (async): id=%s tool=%s",
                task.task_id, task.tool_name,
            )
        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error = f"{type(e).__name__}: {e}"
            task.completed_at = time.time()
            logger.exception(
                "Background task failed (async): id=%s tool=%s",
                task.task_id, task.tool_name,
            )
        finally:
            self._save_task(task)
            self._async_tasks.pop(task.task_id, None)
            if self.on_complete:
                try:
                    await self.on_complete(task)
                except Exception:
                    logger.exception(
                        "on_complete callback failed for task %s", task.task_id,
                    )

    # ── Persistence ──────────────────────────────────────────

    def _save_task(self, task: BackgroundTask) -> None:
        """Persist task state to disk."""
        path = self._storage_dir / f"{task.task_id}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(task.to_dict(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    def _load_task(self, task_id: str) -> BackgroundTask | None:
        """Load a task from disk."""
        path = self._storage_dir / f"{task_id}.json"
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return BackgroundTask(
                task_id=data["task_id"],
                anima_name=data["anima_name"],
                tool_name=data["tool_name"],
                tool_args=data.get("tool_args", {}),
                status=TaskStatus(data["status"]),
                created_at=data.get("created_at", 0.0),
                completed_at=data.get("completed_at"),
                result=data.get("result"),
                error=data.get("error"),
            )
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.warning("Failed to load background task %s: %s", task_id, e)
            return None

    def cleanup_old_tasks(self, max_age_hours: int = 24) -> int:
        """Remove completed/failed tasks older than max_age_hours.

        Also cleans up stale running tasks (crash orphans) that have been
        in running state for more than 48 hours.

        Returns the number of tasks removed.
        """
        cutoff = time.time() - (max_age_hours * 3600)
        removed = 0
        for path in self._storage_dir.glob("*.json"):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                status = data.get("status", "")
                completed = data.get("completed_at")
                if status in ("completed", "failed") and completed and completed < cutoff:
                    path.unlink()
                    self._tasks.pop(data.get("task_id", ""), None)
                    removed += 1
                # Clean up stale running tasks (crash orphans)
                if status == "running":
                    created = data.get("created_at")
                    stale_cutoff = time.time() - (48 * 3600)  # 48 hours
                    if created and created < stale_cutoff:
                        path.unlink()
                        self._tasks.pop(data.get("task_id", ""), None)
                        removed += 1
                        logger.info(
                            "Cleaned up stale running task: %s",
                            data.get("task_id", path.stem),
                        )
            except (json.JSONDecodeError, OSError):
                continue
        if removed:
            logger.info("Cleaned up %d old background tasks", removed)
        return removed
