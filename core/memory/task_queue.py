from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Task API backed by the runtime SQLite store; JSONL is an interchange format."""

import logging
import sqlite3
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from core.exceptions import TaskPersistenceError as TaskPersistenceError  # noqa: F401
from core.i18n import t
from core.schemas import TaskEntry
from core.time_utils import ensure_aware, now_iso, now_local

logger = logging.getLogger("animaworks.task_queue")

if TYPE_CHECKING:
    from core.taskboard.tasks import TaskStore

# Valid task statuses. "blocked" and "unblock_check" were retired: an anima
# that cannot proceed uses "cancelled" and messages the requester with the
# reason instead of parking itself. "failed" was retired: process-exit
# failures are re-queued to "pending" so the task stays visible to its owner.
_VALID_STATUSES = frozenset({"pending", "in_progress", "delegated", "done", "cancelled"})
# Statuses that can no longer be set, but may still appear in old jsonl rows
# written before this teardown. _load_all() remaps them to "pending" on read.
_RETIRED_STATUSES = frozenset({"blocked", "failed"})
_TERMINAL_STATUSES = frozenset({"done", "cancelled"})
_ARCHIVE_SYNC_STATUSES = frozenset({"done", "cancelled"})
_REACTIVATE_SYNC_STATUSES = frozenset({"pending", "in_progress"})
_ACTIVE_STATUSES = frozenset({"pending", "in_progress", "delegated"})

# Valid task sources
_VALID_SOURCES = frozenset({"human", "anima"})
# Stale task threshold: 30 minutes (one heartbeat cycle)
_STALE_TASK_THRESHOLD_SEC = 1800


def _elapsed_seconds(updated_at: str, now: datetime) -> float | None:
    """Return seconds since updated_at, or None on parse failure."""
    try:
        updated = ensure_aware(datetime.fromisoformat(updated_at))
        return (now - updated).total_seconds()
    except (ValueError, TypeError):
        return None


def _format_elapsed_from_sec(elapsed_sec: float | None) -> str:
    """Format elapsed time as human-readable string (e.g. '⏱️ 47分経過').

    Takes pre-computed elapsed seconds to avoid redundant datetime parsing.
    Returns empty string for None or negative values.
    """
    if elapsed_sec is None or elapsed_sec < 0:
        return ""
    minutes = int(elapsed_sec / 60)
    if minutes < 60:
        return t("task_queue.elapsed_minutes", minutes=minutes)
    hours = minutes // 60
    remaining_min = minutes % 60
    if remaining_min:
        return t("task_queue.elapsed_hours_min", hours=hours, remaining_min=remaining_min)
    return t("task_queue.elapsed_hours", hours=hours)


def _descriptor_ids(anima_dir: Path) -> set[str]:
    """Compatibility name: IDs with saved execution input, never a file scan."""
    try:
        return TaskQueueManager(anima_dir).store.executable_ids(anima_dir.name)
    except (OSError, sqlite3.OperationalError) as exc:
        from core.tasks_dispatch import is_task_permission_error, read_executable_ids_via_server

        if not is_task_permission_error(exc):
            raise
        return read_executable_ids_via_server(anima_dir.name)


def descriptor_exists(anima_dir: Path, task_id: str) -> bool:
    """Compatibility facade for saved execution-input availability."""
    return task_id in _descriptor_ids(anima_dir)


_NOT_EXECUTABLE_NOTE = "This is a backlog task without execution input. Submit it once when ready to execute."


def mark_executability(items: list[dict[str, Any]], anima_dir: Path) -> None:
    """Project input availability for own tasks using one DB/snapshot lookup."""
    descriptor_set = _descriptor_ids(anima_dir)
    for item in items:
        if item.get("status") not in ("pending", "in_progress"):
            continue
        tid = item.get("task_id", "")
        if tid in descriptor_set:
            item["executable"] = True
        else:
            item["executable"] = False
            item["executable_note"] = _NOT_EXECUTABLE_NOTE


def _metadata_expired(expires_at: str | None) -> bool:
    """Return True when a TaskBoard metadata expiry timestamp is in the past."""
    if not expires_at:
        return False
    try:
        return now_local() >= ensure_aware(datetime.fromisoformat(expires_at))
    except (ValueError, TypeError):
        return False


class TaskQueueManager:
    """Stable task API over one durable task/attempt store."""

    def __init__(self, anima_dir: Path) -> None:
        self.anima_dir = anima_dir
        self._queue_path = anima_dir / "state" / "task_queue.jsonl"
        self._store: TaskStore | None = None

    @property
    def store(self) -> TaskStore:
        from core.taskboard.tasks import TaskStore, task_database_path

        if self._store is None:
            from core.taskboard.readiness import require_task_store_ready

            require_task_store_ready(self.anima_dir)
            candidate = TaskStore(task_database_path(self.anima_dir))
            candidate.import_legacy(self.anima_dir)
            self._store = candidate
        return self._store

    def submit(
        self,
        payload: dict[str, Any],
        *,
        source: Literal["human", "anima"] = "anima",
        meta: dict[str, Any] | None = None,
        resume: bool = False,
    ) -> TaskEntry:
        """Publish complete execution input atomically, idempotent by task ID."""
        task_id = str(payload["task_id"])
        entry = self._build_task_entry(
            source=source,
            task_id=task_id,
            original_instruction=str(payload.get("description", "")),
            assignee=self.anima_dir.name,
            summary=str(payload.get("title", task_id)),
            relay_chain=list(payload.get("relay_chain") or []),
            meta={"executor": "taskexec", **(meta or {})},
        )
        self.store.submit(self.anima_dir.name, entry, payload, resume=resume)
        return self.store.read(self.anima_dir.name, archived=True)[task_id]

    @property
    def queue_path(self) -> Path:
        return self._queue_path

    @property
    def archive_path(self) -> Path:
        return self._queue_path.parent / "task_queue_archive.jsonl"

    # ── Write operations ─────────────────────────────────────

    def _build_task_entry(
        self,
        *,
        source: Literal["human", "anima"],
        original_instruction: str,
        assignee: str,
        summary: str,
        relay_chain: list[str] | None = None,
        task_id: str | None = None,
        meta: dict[str, Any] | None = None,
        status: str = "pending",
    ) -> TaskEntry:
        if source not in _VALID_SOURCES:
            raise ValueError(f"Invalid source: {source!r} (must be 'human' or 'anima')")
        if status not in ("pending", "in_progress"):
            raise ValueError(f"Invalid status: {status!r} (must be 'pending' or 'in_progress')")
        now = now_iso()
        return TaskEntry(
            task_id=task_id if task_id else uuid.uuid4().hex[:12],
            ts=now,
            source=source,
            original_instruction=original_instruction,
            assignee=assignee,
            status=status,
            summary=summary,
            relay_chain=relay_chain or [],
            updated_at=now,
            meta=meta or {},
        )

    def add_task(
        self,
        *,
        source: Literal["human", "anima"],
        original_instruction: str,
        assignee: str,
        summary: str,
        relay_chain: list[str] | None = None,
        task_id: str | None = None,
        meta: dict[str, Any] | None = None,
        status: str = "pending",
    ) -> TaskEntry:
        """Add a new task to the queue.

        Returns the created TaskEntry.

        Args:
            source: Origin of the task ('human' or 'anima').
            original_instruction: Full instruction text.
            assignee: Anima name responsible for the task.
            summary: One-line summary.
            relay_chain: Optional delegation path.
            task_id: Optional. Use LLM-specified ID (e.g. from submit_tasks).
                If None, a UUID-based ID is generated.
            meta: Optional metadata (e.g. executor for TaskExec tracking).
            status: Initial status. Default "pending"; "in_progress" for
                submit_tasks tasks picked up by TaskExec.

        Returns:
            The created TaskEntry.

        Raises:
            ValueError: If source is invalid.
        """
        entry = self._build_task_entry(
            source=source,
            original_instruction=original_instruction,
            assignee=assignee,
            summary=summary,
            relay_chain=relay_chain,
            meta=meta,
            task_id=task_id,
            status=status,
        )
        self._append(entry.model_dump())
        logger.info(
            "Task added: id=%s source=%s assignee=%s summary=%s",
            entry.task_id,
            source,
            assignee,
            summary[:50],
        )
        return entry

    def add_task_if_absent(
        self,
        predicate: Callable[[TaskEntry], bool],
        *,
        source: Literal["human", "anima"],
        original_instruction: str,
        assignee: str,
        summary: str,
        relay_chain: list[str] | None = None,
        task_id: str | None = None,
        meta: dict[str, Any] | None = None,
        status: str = "pending",
    ) -> TaskEntry | None:
        """Atomically add a task only when no active task matches ``predicate``."""
        with self._locked_queue():
            for task in self._load_all().values():
                if task.status in _ACTIVE_STATUSES and predicate(task):
                    return None
            entry = self._build_task_entry(
                source=source,
                original_instruction=original_instruction,
                assignee=assignee,
                summary=summary,
                relay_chain=relay_chain,
                task_id=task_id,
                meta=meta,
                status=status,
            )
            self._append_unlocked(entry.model_dump())
        logger.info(
            "Task added: id=%s source=%s assignee=%s summary=%s",
            entry.task_id,
            source,
            assignee,
            summary[:50],
        )
        return entry

    def add_delegated_task(
        self,
        *,
        original_instruction: str,
        assignee: str,
        summary: str,
        relay_chain: list[str] | None = None,
        meta: dict[str, Any] | None = None,
        task_id: str | None = None,
    ) -> TaskEntry:
        """Add a task with 'delegated' status for tracking delegation.

        Used by the delegating supervisor to record that a task was sent
        to a subordinate. The meta field stores delegated_to and delegated_task_id.
        """
        now = now_iso()
        entry = TaskEntry(
            task_id=task_id if task_id else uuid.uuid4().hex[:12],
            ts=now,
            source="anima",
            original_instruction=original_instruction,
            assignee=assignee,
            status="delegated",
            summary=summary,
            relay_chain=relay_chain or [],
            updated_at=now,
            meta=meta or {},
        )
        self._append(entry.model_dump())
        logger.info(
            "Delegated task added: id=%s assignee=%s summary=%s",
            entry.task_id,
            assignee,
            summary[:50],
        )
        return entry

    def update_status(
        self,
        task_id: str,
        status: str,
        *,
        summary: str | None = None,
    ) -> TaskEntry | None:
        """Update the status of an existing task.

        Applies the update atomically to the canonical task record.
        Returns the updated task or None if not found.
        """
        if status in _RETIRED_STATUSES:
            raise ValueError(
                f"Status {status!r} was retired: use 'cancelled' and message the requester with the reason."
            )
        if status not in _VALID_STATUSES:
            logger.warning("Invalid task status: %s", status)
            return None

        tasks = self._load_all()
        task = tasks.get(task_id)
        if task is None:
            logger.warning("Task not found: %s", task_id)
            return None
        # A cancel must stick: a runner that was still executing when the
        # cancel landed must not flip the task to done/blocked afterwards.
        # Only an explicit re-queue (pending) may leave cancelled.
        if task.status == "cancelled" and status not in ("cancelled", "pending"):
            logger.warning("Task %s is cancelled; refusing status=%s", task_id, status)
            return None

        now = now_iso()
        update: dict[str, Any] = {
            "task_id": task_id,
            "status": status,
            "updated_at": now,
            "_event": "update",
        }
        if summary is not None:
            update["summary"] = summary
        self._append(update)

        # Return reconstructed entry
        task = self.get_task_by_id(task_id)
        if task is None or task.status != status:
            return None
        logger.info("Task updated: id=%s status=%s", task_id, status)

        if status in _ARCHIVE_SYNC_STATUSES:
            self.store.after_commit(lambda: self._sync_taskboard_archived(task_id))
        elif status in _REACTIVATE_SYNC_STATUSES:
            self.store.after_commit(lambda: self._sync_taskboard_reactivated(task_id))

        return task

    def _sync_taskboard_archived(self, task_id: str) -> None:
        """Best-effort: close TaskBoard metadata when a task reaches terminal status.

        Only updates an existing metadata row; never creates one. Failures are
        swallowed so the queue update remains authoritative.
        """
        try:
            from core.taskboard.models import AttentionVisibility
            from core.taskboard.store import TaskBoardStore

            anima_name = self.anima_dir.name
            store = TaskBoardStore()
            metadata = store.get_metadata(anima_name, task_id)
            if metadata is None:
                return
            if metadata.visibility in {
                AttentionVisibility.EXPIRED,
                AttentionVisibility.ARCHIVED,
                AttentionVisibility.TOMBSTONED,
            }:
                return
            store.upsert_metadata(
                anima_name=anima_name,
                task_id=task_id,
                actor=anima_name,
                event_type="archived",
                visibility="archived",
                column="done",
            )
        except Exception:
            logger.debug(
                "Failed to archive TaskBoard metadata for task %s",
                task_id,
                exc_info=True,
            )

    def _sync_taskboard_reactivated(self, task_id: str) -> None:
        """Best-effort: revive an archived TaskBoard card when a task re-enters
        an active status, so the pending attention gate does not cancel the
        revived task as "archived by TaskBoard". Tombstoned/expired cards are
        deliberate suppressions and stay untouched.
        """
        try:
            from core.taskboard.models import AttentionVisibility
            from core.taskboard.store import TaskBoardStore

            anima_name = self.anima_dir.name
            store = TaskBoardStore()
            metadata = store.get_metadata(anima_name, task_id)
            if metadata is None or metadata.visibility != AttentionVisibility.ARCHIVED:
                return
            store.upsert_metadata(
                anima_name=anima_name,
                task_id=task_id,
                actor=anima_name,
                event_type="visibility_changed",
                visibility="active",
                column="todo",
            )
        except Exception:
            logger.debug(
                "Failed to reactivate TaskBoard metadata for task %s",
                task_id,
                exc_info=True,
            )

    def update_meta(
        self,
        task_id: str,
        meta_patch: dict[str, Any],
        *,
        summary: str | None = None,
    ) -> TaskEntry | None:
        """Merge task metadata and append a durable update event."""
        tasks = self._load_all()
        task = tasks.get(task_id)
        if task is None:
            logger.warning("Task not found: %s", task_id)
            return None

        now = now_iso()
        merged_meta = dict(task.meta or {})
        merged_meta.update(meta_patch)
        update: dict[str, Any] = {
            "task_id": task_id,
            "meta": meta_patch,
            "updated_at": now,
            "_event": "update",
        }
        if summary is not None:
            update["summary"] = summary
        self._append(update)

        task = self.get_task_by_id(task_id)
        logger.info("Task metadata updated: id=%s keys=%s", task_id, sorted(meta_patch))
        return task

    def load_active_tasks(self) -> dict[str, TaskEntry]:
        """Load all non-terminal tasks from a consistent database snapshot.

        Use this for batch operations to avoid repeated file reads.
        """
        return {tid: t for tid, t in self._load_all().items() if t.status in _ACTIVE_STATUSES}

    # ── Read operations ──────────────────────────────────────

    def _load_all(self, *, include_archived: bool = False) -> dict[str, TaskEntry]:
        """Read canonical records; no replay or filesystem descriptor scan."""
        try:
            return self.store.read(self.anima_dir.name, archived=include_archived)
        except (OSError, sqlite3.OperationalError) as exc:
            from core.tasks_dispatch import is_task_permission_error, read_tasks_via_server

            if not is_task_permission_error(exc):
                raise
            return read_tasks_via_server(self.anima_dir.name, include_archived=include_archived)

    def get_pending(self) -> list[TaskEntry]:
        """Return tasks with status 'pending' or 'in_progress'."""
        tasks = self._load_all()
        return [t for t in tasks.values() if t.status in ("pending", "in_progress")]

    def get_human_tasks(self) -> list[TaskEntry]:
        """Return pending/in_progress tasks with source='human'."""
        return [t for t in self.get_pending() if t.source == "human"]

    def get_all_active(self) -> list[TaskEntry]:
        """Return all non-terminal tasks (pending, in_progress)."""
        tasks = self._load_all()
        return [t for t in tasks.values() if t.status in ("pending", "in_progress")]

    def list_tasks(self, status: str | None = None) -> list[TaskEntry]:
        """List tasks, optionally filtered by status.

        When status is omitted, returns only active tasks
        (pending, in_progress, delegated).
        """
        tasks = self._load_all()
        if status:
            return [t for t in tasks.values() if t.status == status]
        return [t for t in tasks.values() if t.status in _ACTIVE_STATUSES]

    def get_delegated_tasks(self) -> list[TaskEntry]:
        """Return tasks with status 'delegated'."""
        tasks = self._load_all()
        return [t for t in tasks.values() if t.status == "delegated"]

    def get_task_by_id(self, task_id: str) -> TaskEntry | None:
        """Look up a single task by its ID."""
        try:
            return self.store.get(self.anima_dir.name, task_id)
        except (OSError, sqlite3.OperationalError) as exc:
            from core.tasks_dispatch import is_task_permission_error, read_tasks_via_server

            if not is_task_permission_error(exc):
                raise
            return read_tasks_via_server(self.anima_dir.name, include_archived=True, task_id=task_id).get(task_id)

    def get_active_goal_task(self, goal_id: str) -> TaskEntry | None:
        """Return an active task linked to a persistent goal, ignoring suppressed board rows.

        Archived, tombstoned, and expired TaskBoard metadata do not block goal
        continuation; snoozed/active pending work still counts as an existing
        continuation to avoid duplicate tasks.
        """
        if not goal_id:
            return None
        for task in self.load_active_tasks().values():
            if task.meta.get("goal_id") != goal_id:
                continue
            if self._goal_task_suppressed_by_taskboard(task.task_id):
                continue
            return task
        return None

    def list_goal_tasks(self, goal_id: str) -> list[TaskEntry]:
        """Return all queue tasks linked to a persistent goal."""
        if not goal_id:
            return []
        return sorted(
            [task for task in self._load_all().values() if task.meta.get("goal_id") == goal_id],
            key=lambda task: task.updated_at,
            reverse=True,
        )

    def _goal_task_suppressed_by_taskboard(self, task_id: str) -> bool:
        try:
            from core.taskboard.models import AttentionVisibility
            from core.taskboard.store import TaskBoardStore

            metadata = TaskBoardStore().get_metadata(self.anima_dir.name, task_id)
            if metadata is None:
                return False
            if _metadata_expired(metadata.expires_at):
                return True
            return metadata.visibility in {
                AttentionVisibility.ARCHIVED,
                AttentionVisibility.TOMBSTONED,
                AttentionVisibility.EXPIRED,
            }
        except Exception:
            logger.debug("TaskBoard metadata check failed for goal task %s", task_id, exc_info=True)
            return False

    # ── Formatting ───────────────────────────────────────────

    def format_for_priming(self, budget_tokens: int = 400) -> str:
        """Format pending tasks for system prompt injection."""
        tasks = self.get_pending()
        now = now_local()
        chars_per_token = 4
        max_chars = budget_tokens * chars_per_token
        lines: list[str] = []
        total = 0

        if tasks:
            active = sorted(
                tasks,
                key=lambda t: (
                    0 if t.source == "human" else 1,
                    t.updated_at or t.ts,
                ),
                reverse=False,
            )

            for task in active:
                priority = "🔴 HIGH" if task.source == "human" else "⚪"
                status_icon = "🔄" if task.status == "in_progress" else "📋"
                line = f"- {status_icon} {priority} [{task.task_id[:8]}] {task.summary} (assignee: {task.assignee})"
                if task.status == "in_progress" and task.meta.get("executor") == "taskexec":
                    line += f" {t('task_queue.auto_taskexec')}"
                if task.relay_chain:
                    line += f" chain: {' → '.join(task.relay_chain)}"

                elapsed_sec = _elapsed_seconds(task.updated_at, now)
                elapsed_str = _format_elapsed_from_sec(elapsed_sec)
                if elapsed_str:
                    line += f" {elapsed_str}"

                if elapsed_sec is not None and elapsed_sec >= _STALE_TASK_THRESHOLD_SEC:
                    line += " ⚠️ STALE"

                if total + len(line) > max_chars:
                    break
                lines.append(line)
                total += len(line) + 1

        # Delegated tasks (within remaining budget)
        delegated = self.get_delegated_tasks()
        if delegated and total < max_chars:
            try:
                from core.paths import get_animas_dir

                del_section = self.format_delegated_for_priming(get_animas_dir(), budget_chars=max_chars - total)
                if del_section:
                    lines.append(del_section)
                    total += len(del_section) + 1
            except Exception:
                logger.debug("format_for_priming: delegated section failed", exc_info=True)

        return "\n".join(lines) if lines else ""

    def get_stale_tasks(self) -> list[TaskEntry]:
        """Return pending/in_progress tasks not updated for 30+ minutes."""
        now = now_local()
        result: list[TaskEntry] = []
        for task in self.get_pending():
            elapsed = _elapsed_seconds(task.updated_at, now)
            if elapsed is not None and elapsed >= _STALE_TASK_THRESHOLD_SEC:
                result.append(task)
        return result

    # ── Delegation sync ─────────────────────────────────────────

    def sync_delegated(self, animas_dir: Path) -> int:
        """Compatibility facade: aliases read the assignee's canonical record."""
        return 0

    @staticmethod
    def _search_archive(target_dir: Path, child_id: str) -> str | None:
        """Compatibility facade over archived canonical records, not JSONL."""
        entry = TaskQueueManager(target_dir).get_task_by_id(child_id)
        return entry.status if entry and entry.status in _TERMINAL_STATUSES else None

    def format_delegated_for_priming(self, animas_dir: Path, budget_chars: int = 400) -> str:
        """Format delegated tasks with subordinate status for Priming display."""
        delegated = self.get_delegated_tasks()
        if not delegated:
            return ""
        now = now_local()
        lines: list[str] = []
        total = 0
        _status_icons = {"done": "✅", "cancelled": "🚫"}
        unknown_label = t("task_queue.delegated_unknown")
        for task in delegated[:5]:
            meta = task.meta or {}
            target = meta.get("delegated_to", unknown_label)
            sub_status = str(meta.get("delegated_status", "?"))
            icon = _status_icons.get(sub_status, "⏳")
            elapsed_sec = _elapsed_seconds(task.updated_at, now)
            elapsed_str = _format_elapsed_from_sec(elapsed_sec) or ""
            line = f"- 📌 [{task.task_id[:8]}] {task.summary} → {target} ({target}: {sub_status} {icon}) {elapsed_str}"
            if total + len(line) > budget_chars:
                break
            lines.append(line)
            total += len(line) + 1
        return "\n".join(lines)

    def _resolve_subordinate_display(self, target_dir: Path, child_id: str) -> str:
        """Resolve subordinate task status for display (single queue read)."""
        if not target_dir.is_dir():
            return "?"
        try:
            sub_tqm = TaskQueueManager(target_dir)
            sub_task = sub_tqm.get_task_by_id(child_id)
            if sub_task:
                return sub_task.status
            archived = self._search_archive(target_dir, child_id)
            return archived if archived else t("task_queue.delegated_archived")
        except Exception:
            return "?"

    # ── Maintenance ────────────────────────────────────────────

    def compact(self) -> int:
        """Archive terminal records without discarding attempt history."""
        return self.store.compact(self.anima_dir.name)

    # ── Internal ─────────────────────────────────────────────────

    @contextmanager
    def _locked_queue(self) -> Iterator[None]:
        with self.store.transaction():
            yield

    def _append(self, data: dict[str, Any]) -> None:
        """Apply a durable update, preserving the persistence-error contract."""
        try:
            with self._locked_queue():
                self._append_unlocked(data)
        except (OSError, sqlite3.Error) as exc:
            raise TaskPersistenceError(str(exc)) from exc

    def _append_unlocked(self, data: dict[str, Any]) -> None:
        """Apply an update inside the current SQLite transaction."""
        try:
            self.store.apply(self.anima_dir.name, data)
        except OSError as exc:
            logger.exception("Failed to append to task queue")
            raise TaskPersistenceError(str(exc)) from exc
