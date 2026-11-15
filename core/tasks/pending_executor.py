# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
"""Execute claimed TaskStore work in background lanes.

Command tasks submitted via ``animaworks-tool submit`` and LLM tasks are both
claimed from TaskStore; command results remain available through the existing
BackgroundTaskManager result files and notifications.
"""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import logging
import os
import re
import threading
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from core.exceptions import ToolExecutionError
from core.i18n import t
from core.platform.state_writer import get_state_writer, run_writer_sync
from core.platform.tasks import spawn
from core.time_utils import now_iso

if TYPE_CHECKING:
    from core.anima.digital_anima import BackgroundWorkerSlot, DigitalAnima
    from core.runtime.task_runner_supervisor import TaskRunnerSupervisor

logger = logging.getLogger(__name__)


def _cleanup_task_scratch(anima_dir: Path, task_id: str, *, status: str) -> None:
    """Remove scratch only after TaskStore confirms a terminal task state."""
    if status not in {"done", "cancelled"}:
        return

    from core.platform.scratch import remove_scratch_entry, scratch_root, task_scratch_dir

    scratch_dir = task_scratch_dir(anima_dir, task_id)
    try:
        if scratch_root(anima_dir).is_symlink():
            return
        remove_scratch_entry(scratch_dir)
    except FileNotFoundError:
        pass
    except OSError:
        logger.debug("Could not remove task scratch directory %s", scratch_dir, exc_info=True)


def _request_background_review(anima_dir: Path, trigger: str) -> None:
    try:
        from core.memory.maintenance.background_review import request_background_review

        request_background_review(anima_dir, trigger)
    except Exception:
        logger.warning("Could not queue background review (%s)", trigger)


class TaskExecError(RuntimeError):
    """Raised when a TaskExec LLM session encounters a non-recoverable error."""


_PENDING_WATCHER_POLL_INTERVAL = 3.0
_PENDING_TASK_SUBPROCESS_TIMEOUT = 1800
_TASK_RESULT_MAX_CHARS = 2000
_TASK_COMPLETE_NOTIFY_MAX_CHARS = 10_000

_SENTINEL_CANCELLED = "(cancelled)"
_SENTINEL_BUDGET_SKIPPED = "(budget_skipped)"
# The session ended without the anima declaring done or cancelled.
_SENTINEL_UNDECLARED = "(undeclared)"
# Results that mean "this task produced no output a dependent task can use".
_NON_COMPLETING_SENTINELS = {
    _SENTINEL_CANCELLED,
    _SENTINEL_BUDGET_SKIPPED,
    _SENTINEL_UNDECLARED,
}
_DEPENDENCY_UNFINISHED = "a task this one depends on did not complete"

_CANCEL_POLL_SECONDS = 5.0
_MAX_THREAD_ID_LEN = 36
_SAFE_THREAD_ID_RE = re.compile(rf"^[A-Za-z0-9_-]{{1,{_MAX_THREAD_ID_LEN}}}$")
_QUEUE_TERMINAL_STATUSES = {"done", "cancelled"}
# Statuses a runner-side sync must never walk back: the anima declared them, or
# a subordinate now owns the work.
_QUEUE_STICKY_STATUSES = _QUEUE_TERMINAL_STATUSES | {"delegated"}


def _task_thread_id(task_id: str) -> str:
    """Return a session thread id the state writer accepts for *task_id*."""
    if _SAFE_THREAD_ID_RE.fullmatch(task_id):
        return task_id
    # Anima-chosen task ids can exceed 36 chars or contain other characters;
    # the state writer rejects those, so fall back to a stable hash.
    return f"task-{hashlib.sha1(task_id.encode('utf-8')).hexdigest()[:31]}"


def _task_activity_identity(task_desc: dict[str, Any]) -> tuple[str, str, str]:
    """Return stable task id, title, and description for execution events."""
    task_id = str(task_desc.get("task_id") or "unknown")
    description = str(task_desc.get("description") or "")
    title = str(task_desc.get("title") or "").strip()
    if not title:
        title = next((line.strip() for line in description.splitlines() if line.strip()), "")[:100]
    return task_id, title or task_id, description


def _classify_task_result(result: str) -> tuple[str, str]:
    """Map _run_llm_task return value to (queue_status, summary).

    Anything that is not an explicit done or cancelled declaration puts the task
    back to ``pending``: the harness never parks, continues or retries work on
    the anima's behalf.  Terminal engine failures (e.g. AUTH) are surfaced by
    ``_run_llm_task`` raising ``TaskExecError``.
    """
    if result == _SENTINEL_CANCELLED:
        return "cancelled", t("pending_executor.task_cancelled")
    if result == _SENTINEL_BUDGET_SKIPPED:
        return "pending", "execution skipped because token budget is unavailable"
    if result == _SENTINEL_UNDECLARED:
        return "pending", "run ended without a completion declaration"
    return "done", (result or "")[:200]


def _submission_line(submitted_at: str, *, locale: str | None = None) -> str:
    """Build a single-line submission timestamp for the task_exec prompt.

    The TTL was removed; age information is surfaced to the model so it can
    judge staleness itself.  Returns an empty string when no usable timestamp
    is available.  ``locale`` is exposed for deterministic unit tests.
    """
    if not submitted_at:
        return ""
    try:
        dt = datetime.fromisoformat(submitted_at)
    except (ValueError, TypeError):
        return ""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    now = datetime.now(UTC)
    total_seconds = max(0, int((now - dt).total_seconds()))
    hours, rem = divmod(total_seconds, 3600)
    minutes = rem // 60
    from core.time_utils import get_app_timezone

    time_str = dt.astimezone(get_app_timezone()).strftime("%Y-%m-%d %H:%M")
    elapsed = t("pending_executor.elapsed", locale=locale, hours=hours, minutes=minutes)
    return t("pending_executor.submitted_line", locale=locale, time=time_str, elapsed=elapsed)


def _resolve_default_workspace(anima_dir: Path) -> str:
    """Resolve default_workspace from status.json via workspace registry.

    Returns absolute path string, or empty string if not set or resolution fails.
    """
    from core.org.workspace import resolve_default_workspace

    resolved, _alias = resolve_default_workspace(anima_dir)
    return str(resolved) if resolved else ""


# ── DAG helpers ──────────────────────────────────────────────


def _topological_sort(tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return tasks in topological order. Raises ValueError on cycles."""
    task_map = {td["task_id"]: td for td in tasks}
    in_degree: dict[str, int] = {tid: 0 for tid in task_map}
    for td in tasks:
        for dep in td.get("depends_on", []):
            if dep in in_degree:
                in_degree[td["task_id"]] += 1

    queue = [tid for tid, deg in in_degree.items() if deg == 0]
    result: list[dict[str, Any]] = []
    while queue:
        tid = queue.pop(0)
        result.append(task_map[tid])
        for td in tasks:
            if tid in td.get("depends_on", []):
                in_degree[td["task_id"]] -= 1
                if in_degree[td["task_id"]] == 0:
                    queue.append(td["task_id"])

    if len(result) != len(tasks):
        raise ValueError("Cycle detected in task dependencies")
    return result


class PendingTaskExecutor:
    """Claim TaskStore inputs and execute command or LLM tasks."""

    def __init__(
        self,
        anima: DigitalAnima,
        anima_name: str,
        anima_dir: Path,
        shutdown_event: asyncio.Event,
        *,
        task_runner_supervisor: TaskRunnerSupervisor | None = None,
    ) -> None:
        self._anima = anima
        self._anima_name = anima_name
        self._anima_dir = anima_dir
        self._shutdown_event = shutdown_event
        self._accepting_claims = True
        self._claim_gate_lock = threading.Lock()
        self._wake_event = asyncio.Event()
        self._active_dispatch_tasks: set[asyncio.Task[None]] = set()
        self._active_task_ids: set[str] = set()
        self._active_task_ids_lock = threading.Lock()
        self._task_runner_supervisor = task_runner_supervisor
        self._task_isolated = task_runner_supervisor is not None
        self._background_isolated = task_runner_supervisor is not None
        # Fallback attempt numbering for direct calls without a TaskStore claim.
        self._attempt_by_task_id: dict[str, int] = {}
        self._task_queue_manager: Any | None = None

    def _worker_pool_size(self) -> int:
        """Return a validated pool size while tolerating legacy test doubles."""
        value = getattr(self._anima, "_background_worker_pool_size", 1)
        return value if isinstance(value, int) and 1 <= value <= 10 else 1

    def _is_task_active(self, task_id: str) -> bool:
        with self._active_task_ids_lock:
            return task_id in self._active_task_ids

    def _mark_task_active(self, task_id: str) -> None:
        with self._active_task_ids_lock:
            self._active_task_ids.add(task_id)

    def _forget_task_active(self, task_id: str) -> None:
        with self._active_task_ids_lock:
            self._active_task_ids.discard(task_id)

    async def _acquire_worker(self, task_id: str) -> BackgroundWorkerSlot | None:
        acquire = getattr(type(self._anima), "_acquire_background_worker", None)
        if callable(acquire):
            return await self._anima._acquire_background_worker(task_id)
        return None

    async def _release_worker(self, slot: BackgroundWorkerSlot | None) -> None:
        if slot is None:
            return
        release = getattr(type(self._anima), "_release_background_worker", None)
        if callable(release):
            await self._anima._release_background_worker(slot)

    def _track_dispatch_task(self, task: asyncio.Task[None]) -> None:
        """Keep a strong reference to a detached single-task dispatch."""
        self._active_dispatch_tasks.add(task)

        def _done(done: asyncio.Task[None]) -> None:
            self._active_dispatch_tasks.discard(done)
            self.wake()

        task.add_done_callback(_done)

    def _get_task_queue_manager(self) -> Any:
        """Reuse one queue/store wrapper for the lifetime of this watcher."""
        manager = getattr(self, "_task_queue_manager", None)
        if manager is None:
            from core.tasks.queue import TaskQueueManager

            manager = TaskQueueManager(self._anima_dir)
            self._task_queue_manager = manager
        return manager

    def _next_attempt(self, task_id: str) -> int:
        """Allocate a fallback attempt number for direct, non-claimed calls."""
        with self._active_task_ids_lock:
            attempt = self._attempt_by_task_id.get(task_id, 0) + 1
            self._attempt_by_task_id[task_id] = attempt
            return attempt

    def _display_lane_for_task(self, task_id: str, slot_id: int | None) -> str:
        if slot_id is not None and self._worker_pool_size() > 1:
            return f"background-worker:{slot_id}:{task_id}"
        return "background"

    # ── Result save / dependency context ─────────────────────

    async def _asave_task_result(self, task_id: str, summary: str, attempt_token: str | None) -> None:
        """Save under the attempt token through the process StateWriter."""
        await get_state_writer(self._anima_dir).save_task_result(
            task_id,
            attempt_token,
            summary[:_TASK_RESULT_MAX_CHARS],
        )

    def _save_task_result(self, task_id: str, summary: str) -> None:
        """Synchronous compatibility adapter for local callers and tests."""
        from core.tasks.board.tasks import current_attempt_identity

        identity = current_attempt_identity()
        token = str(identity["token"]) if identity else None
        writer = get_state_writer(self._anima_dir)
        run_writer_sync(writer, self._asave_task_result(task_id, summary, token))

    def _schedule_live_fact_extraction(self, session_started_at: str) -> None:
        schedule = getattr(self._anima, "_schedule_live_fact_extraction", None)
        if callable(schedule):
            try:
                schedule("task", session_started_at=session_started_at)
            except Exception as exc:  # noqa: BLE001 - do not change task outcome
                reason = f"{type(exc).__name__}: {exc}".replace("\n", " ")
                logger.warning("[%s] Live fact extraction was not scheduled: %s", self._anima_name, reason[:240])

    def _build_dependency_context(
        self,
        task_desc: dict[str, Any],
        completed: dict[str, str],
    ) -> str:
        """Build context from completed dependency results."""
        parts: list[str] = []
        for dep_id in task_desc.get("depends_on", []):
            result = completed.get(dep_id, "")
            if result:
                parts.append(t("pending_executor.dep_result_header", dep_id=dep_id) + f"\n{result}")
        return "\n\n".join(parts)

    def _record_run_ended(self, task_id: str, stop_kind: str, note: str | None = None) -> None:
        """Stamp when and how the last run of a task ended, for its owner.

        The task's summary (its title) is left alone: an error string in the
        summary made the owner's pending list unreadable.
        """
        if not task_id:
            return
        try:
            patch: dict[str, Any] = {"last_run_ended_at": now_iso(), "last_run_stop_kind": stop_kind}
            if note:
                patch["last_run_note"] = note[:300]
            self._get_task_queue_manager().update_meta(task_id, patch)
        except Exception:
            logger.warning(
                "[%s] Failed to record run end for task %s",
                self._anima_name,
                task_id,
                exc_info=True,
            )

    def _mark_declared_pending(self, task_id: str) -> None:
        """Record that the anima itself handed this run's task back to pending."""
        try:
            self._get_task_queue_manager().update_meta(task_id, {"last_run_declared_pending": True})
        except Exception:
            logger.warning(
                "[%s] Failed to record declared pending for task %s",
                self._anima_name,
                task_id,
                exc_info=True,
            )

    async def _return_task_to_pending(self, task_desc: dict[str, Any], reason: str, *, stop_kind: str) -> None:
        """Put a task whose run ended abnormally back on its owner's pending list.

        There is no failed state and no automatic retry: the task becomes
        visible again in its owner's pending list, the requester is told once,
        and no runnable input is re-enqueued.
        """
        from core.trust import ORIGIN_ANIMA

        task_id, title, _description = _task_activity_identity(task_desc)
        # A queue entry that was already cancelled (e.g. superseded by a newer
        # request) had its runner killed on purpose and nobody is waiting on it.
        entry = self._get_task_queue_entry(task_id)
        if entry is not None and entry.status == "cancelled":
            return
        await asyncio.to_thread(self._save_task_result, task_id, reason)
        self._record_run_ended(task_id, stop_kind, note=reason)
        self._sync_task_queue(task_id, "pending")
        if task_desc.get("_attempt_token"):
            # The canonical attempt wrapper records a durable notification.
            return

        reply_to = task_desc.get("reply_to")
        if isinstance(reply_to, dict):
            reply_to = reply_to.get("name")
        recipient = reply_to if isinstance(reply_to, str) and reply_to else self._anima_name
        try:
            notify_text = t(
                "pending_executor.task_fail_notify",
                task_id=task_id,
                title=title,
                error=reason,
            )
        except Exception:
            logger.warning(
                "[%s] Failed to build task interruption notification for %s",
                self._anima_name,
                recipient,
                exc_info=True,
            )
            return
        for attempt in range(2):
            try:
                self._anima.messenger.send(
                    to=recipient,
                    content=notify_text,
                    origin_chain=[ORIGIN_ANIMA],
                )
                return
            except Exception:
                if attempt == 0:
                    logger.warning(
                        "[%s] Task interruption notification failed, retrying to %s",
                        self._anima_name,
                        recipient,
                    )
                else:
                    logger.error(
                        "[%s] Task interruption notification failed after retry to %s",
                        self._anima_name,
                        recipient,
                        exc_info=True,
                    )

    def _sync_task_queue(
        self,
        task_id: str,
        status: str,
        *,
        summary: str | None = None,
    ) -> None:
        """Update the canonical TaskStore status after an execution outcome.

        Silently skips if the task is not registered in TaskStore.

        A ``pending`` sync (undeclared end, budget skip) keeps the task's
        summary -- its title -- untouched and files the reason under
        ``meta.last_run_note`` instead, so the owner's pending list stays
        readable.
        """
        try:
            manager = self._get_task_queue_manager()
            entry = manager.get_task_by_id(task_id)
            if entry and entry.status == "cancelled":
                # The owner supplied the cancellation reason. A child ending
                # after observing it must not replace it with a generic result.
                return
            if entry and entry.status != status and entry.status in _QUEUE_STICKY_STATUSES:
                # done / cancelled are the anima's own declarations and
                # delegated hands ownership to a subordinate; a runner-side
                # sync must not walk any of them back.
                return
            if status == "pending":
                if summary:
                    manager.update_meta(task_id, {"last_run_note": summary[:300]})
                manager.update_status(task_id, status)
                return
            manager.update_status(task_id, status, summary=summary)
        except Exception:
            logger.warning(
                "[%s] Failed to sync TaskStore task %s status=%s",
                self._anima_name,
                task_id,
                status,
                exc_info=True,
            )

    def _get_task_queue_entry(self, task_id: str) -> Any | None:
        if not task_id:
            return None
        try:
            return self._get_task_queue_manager().get_task_by_id(task_id)
        except Exception:
            logger.debug(
                "Could not check TaskStore for task: %s",
                task_id,
                exc_info=True,
            )
            return None

    def _format_active_sibling_tasks(
        self,
        current_task_id: str,
        *,
        limit: int = 8,
    ) -> str:
        """Format one-line summaries of other in_progress tasks for prompt injection.

        Gives each worker visibility into what sibling workers of the same
        anima are doing, so it can avoid touching the same PR/branch/resource.
        Returns an empty string when there are no siblings.
        """
        try:
            entries = self._get_task_queue_manager().list_tasks(status="in_progress")
        except Exception:
            logger.debug(
                "Could not list sibling tasks for: %s",
                current_task_id,
                exc_info=True,
            )
            return ""
        lines: list[str] = []
        for entry in entries:
            if entry.task_id == current_task_id:
                continue
            summary = (entry.summary or "").strip()
            summary = summary.splitlines()[0] if summary else "-"
            if len(summary) > 160:
                summary = summary[:160] + "..."
            updated = (entry.updated_at or "")[:16]
            lines.append(f"- [{entry.task_id[:8]}] {summary} ({updated})")
            if len(lines) >= limit:
                break
        return "\n".join(lines)

    # ── Watcher loop ─────────────────────────────────────────

    async def quiesce(self) -> None:
        """Stop claiming new TaskStore work while active dispatches continue."""
        self._accepting_claims = False

        def wait_for_active_claim() -> None:
            with self._claim_gate_lock:
                pass

        await asyncio.to_thread(wait_for_active_claim)
        self.wake()

    def _claim_canonical_pending_tasks(self) -> list[dict[str, Any]]:
        """Recover and claim TaskStore inputs with blocking store I/O off-loop."""
        if not self._accepting_claims:
            return []
        try:
            from core.tasks.board.tasks import process_identity

            store = self._get_task_queue_manager().store
            self._recover_task_attempts(store)
            store.satisfy_due_time_waits(self._anima_name)
            self._deliver_task_wakeups(store)
            claims = []
            for payload in store.pending(self._anima_name):
                with self._claim_gate_lock:
                    if not self._accepting_claims:
                        break
                    task_desc = store.claim(
                        self._anima_name,
                        str(payload["task_id"]),
                        process_identity(),
                        max_active=self._worker_pool_size(),
                    )
                if task_desc is not None:
                    claims.append(task_desc)
            return claims
        except Exception:
            logger.exception("Error polling canonical tasks for %s", self._anima_name)
            return []

    def _dispatch_canonical_tasks(self, claims: list[dict[str, Any]]) -> None:
        for task_desc in claims:
            task_id = task_desc["task_id"]
            self._mark_task_active(task_id)
            dispatch = spawn(
                self._execute_canonical_task(task_desc),
                name=f"taskexec-{self._anima_name}-{task_id}",
            )
            self._track_dispatch_task(dispatch)

    async def watcher_loop(self) -> None:
        """Poll TaskStore, recover expired attempts, and dispatch claimed tasks."""
        logger.info("TaskStore watcher started for %s", self._anima_name)

        while not self._shutdown_event.is_set():
            try:
                if not self._accepting_claims:
                    try:
                        await asyncio.wait_for(
                            self._wake_event.wait(),
                            timeout=_PENDING_WATCHER_POLL_INTERVAL,
                        )
                        self._wake_event.clear()
                    except TimeoutError:
                        pass
                    continue
                # One claim transaction owns each task input and its attempt.
                canonical_poll = asyncio.create_task(asyncio.to_thread(self._claim_canonical_pending_tasks))
                try:
                    canonical_claims = await asyncio.shield(canonical_poll)
                except asyncio.CancelledError:
                    self._dispatch_canonical_tasks(await canonical_poll)
                    raise
                self._dispatch_canonical_tasks(canonical_claims)

                try:
                    await asyncio.wait_for(
                        self._wake_event.wait(),
                        timeout=_PENDING_WATCHER_POLL_INTERVAL,
                    )
                    self._wake_event.clear()
                except TimeoutError:
                    pass
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception(
                    "Error in TaskStore watcher for %s",
                    self._anima_name,
                )
                await asyncio.sleep(_PENDING_WATCHER_POLL_INTERVAL)

        active_dispatch_tasks = set(self._active_dispatch_tasks)
        if active_dispatch_tasks:
            try:
                from core.config.models import load_config

                drain_timeout = load_config().background_task.shutdown_drain_seconds
            except Exception:
                drain_timeout = 600.0

            _, pending = await asyncio.wait(
                active_dispatch_tasks,
                timeout=drain_timeout,
            )
            if pending:
                logger.warning(
                    "Background task drain timed out for %s after %.0fs; cancelling %d task(s)",
                    self._anima_name,
                    drain_timeout,
                    len(pending),
                )
            for task in pending:
                task.cancel()
            await asyncio.gather(*active_dispatch_tasks, return_exceptions=True)
            self._active_dispatch_tasks.difference_update(active_dispatch_tasks)
        logger.info("Pending task watcher stopped for %s", self._anima_name)

    def _recover_task_attempts(self, store: Any) -> None:
        """Only proven-dead owners become incomplete; never replay side effects."""
        from core.tasks.board.tasks import identity_liveness

        for attempt in store.active_attempts(self._anima_name):
            if self._is_task_active(attempt["task_id"]):
                continue
            if identity_liveness(json.loads(attempt["identity_json"])) != "dead":
                continue
            entry = store.read(self._anima_name, archived=True).get(attempt["task_id"])
            status = entry.status if entry and entry.status in {"done", "cancelled"} else "pending"
            store.finish(attempt["token"], status=status, stop_kind="owner_exited")

    def _deliver_task_wakeups(self, store: Any) -> None:
        """Retry the durable outbox, independent of periodic heartbeat enablement."""
        for event in store.wakeups(self._anima_name):
            payload = store.get_input(self._anima_name, event["task_id"]) or {}
            reply_to = payload.get("reply_to")
            if isinstance(reply_to, dict):
                reply_to = reply_to.get("name")
            completion = event["reason"] == "completion"
            recipients = set() if completion else {self._anima_name}
            if isinstance(reply_to, str) and reply_to and event["reason"] != "wait_satisfied":
                recipients.add(reply_to)
            try:
                entry = store.read(self._anima_name, archived=True).get(event["task_id"])
                if completion and reply_to == self._anima_name:
                    title = str(payload.get("title") or event["task_id"])
                    result = str(entry.meta.get("result_note") or entry.summary) if entry else ""
                    from core.activity.logger import ActivityLogger

                    ActivityLogger(self._anima_dir).log(
                        "task_completed",
                        summary=t("pending_executor.task_exec_end", title=title, result=result),
                        ctx=f"task:{event['task_id']}",
                        meta={
                            "task_id": event["task_id"],
                            "attempt_token": event["attempt_token"],
                            "result_note": result,
                            "event": "task_completed",
                        },
                    )
                    store.acknowledge_wakeup(self._anima_name, event["attempt_token"])
                    continue

                reported_to = entry.meta.get("reported_to", []) if entry else []
                if (
                    completion
                    and isinstance(reply_to, str)
                    and entry is not None
                    and entry.meta.get("last_attempt_token") == event["attempt_token"]
                    and isinstance(reported_to, list)
                    and reply_to in reported_to
                ):
                    from core.activity.logger import ActivityLogger

                    ActivityLogger(self._anima_dir).log(
                        "task_completion_notify_skipped",
                        summary=t(
                            "pending_executor.completion_notify_skipped",
                            title=str(payload.get("title") or event["task_id"]),
                        ),
                        ctx=f"task:{event['task_id']}",
                        meta={
                            "task_id": event["task_id"],
                            "attempt_token": event["attempt_token"],
                            "recipient": reply_to,
                            "reason": "already_reported",
                        },
                    )
                    store.acknowledge_wakeup(self._anima_name, event["attempt_token"])
                    continue

                if completion:
                    from core.paths import load_prompt

                    result = str(entry.meta.get("result_note") or entry.summary) if entry else ""
                    content = load_prompt(
                        "task_complete_notify",
                        task_id=event["task_id"],
                        title=payload.get("title", event["task_id"]),
                        result_summary=result[:_TASK_COMPLETE_NOTIFY_MAX_CHARS],
                    )
                elif event["reason"] == "normal":
                    # The run itself ended cleanly; only the declaration is missing.
                    content = t(
                        "pending_executor.task_undeclared_notify",
                        task_id=event["task_id"],
                        title=payload.get("title", event["task_id"]),
                    )
                elif event["reason"] == "wait_satisfied":
                    content = t(
                        "pending_executor.task_wait_satisfied_notify",
                        task_id=event["task_id"],
                        title=payload.get("title", event["task_id"]),
                    )
                else:
                    content = t(
                        "pending_executor.task_fail_notify",
                        task_id=event["task_id"],
                        title=payload.get("title", event["task_id"]),
                        error=event["reason"],
                    )
                for recipient in sorted(recipients):
                    self._anima.messenger.send(
                        to=recipient,
                        content=content,
                        msg_type="system_alert",
                        intent="report",
                        meta={
                            "task_id": event["task_id"],
                            "attempt_token": event["attempt_token"],
                            "event": "task_completed" if completion else "task_needs_attention",
                        },
                        delivery_id=f"task-wakeup-{event['attempt_token']}",
                    )
                store.acknowledge_wakeup(self._anima_name, event["attempt_token"])
            except Exception:
                logger.warning("Task wakeup delivery retained for retry: %s", event["task_id"], exc_info=True)

    async def _execute_canonical_task(self, task_desc: dict[str, Any]) -> None:
        from core.tasks.board.tasks import attempt_scope

        task_id = str(task_desc["task_id"])
        token = str(task_desc["_attempt_token"])
        store = self._get_task_queue_manager().store
        identity = {"anima": self._anima_name, "task_id": task_id, "token": token}
        stop_kind = "normal"
        cancel_watch: asyncio.Task[None] | None = None
        try:
            with attempt_scope(identity):
                store.update_attempt_meta(token, {"reported_to": []})
                entries = store.read(self._anima_name, archived=True)
                task_desc = {
                    **task_desc,
                    "_completed_results": {
                        dependency: str(entries[dependency].meta.get("result_note") or entries[dependency].summary)
                        for dependency in task_desc.get("depends_on", [])
                        if dependency in entries
                    },
                }
                execution = asyncio.create_task(self.execute_pending_task(task_desc))
                if not self._task_isolated:
                    cancel_watch = asyncio.create_task(self._watch_canonical_cancel(task_id, execution))
                await execution
        except asyncio.CancelledError:
            stop_kind = "interrupted"
            raise
        except Exception:
            stop_kind = "crash"
            logger.exception("Task attempt failed: %s", task_id)
        finally:
            if cancel_watch is not None:
                cancel_watch.cancel()
                await asyncio.gather(cancel_watch, return_exceptions=True)
            entry = store.read(self._anima_name, archived=True).get(task_id)
            status = entry.status if entry and entry.status in {"done", "cancelled", "delegated"} else "pending"
            if entry and stop_kind == "normal":
                # A caught runner failure/cancellation is newer than the cycle
                # metadata (e.g. shutdown after the model declared completion).
                stop_kind = str(entry.meta.get("last_run_stop_kind") or stop_kind)
            if status == "cancelled" and stop_kind == "normal":
                # Cancellation can terminate an isolated child before it returns
                # a result. The sticky cancelled path must not look like a
                # normally completed run merely because its error was handled.
                stop_kind = "interrupted"
            from core.tasks.board.tasks import identity_liveness

            active = next((item for item in store.active_attempts(self._anima_name) if item["token"] == token), None)
            child_still_live = False
            if active:
                owner = json.loads(active["identity_json"])
                child_pid = owner.get("task_pid") or owner.get("pid")
                child_still_live = child_pid != os.getpid() and identity_liveness(owner) != "dead"
            if not child_still_live:
                result_ref = f"state/task_results/{task_id}/{token}.md"
                # An anima that put its own task back to pending (e.g. waiting
                # on a delegate) already decided what happens next.
                declared_pending = (
                    status == "pending"
                    and stop_kind == "normal"
                    and bool(entry and entry.meta.get("last_run_declared_pending"))
                )
                background_notified = bool(task_desc.get("_background_notification_handled"))
                store.finish(
                    token,
                    status=status,
                    stop_kind=stop_kind,
                    result_ref=result_ref if (self._anima_dir / result_ref).is_file() else "",
                    wakeup=not declared_pending and not background_notified,
                )
            self._forget_task_active(task_id)
            self.wake()

    async def _watch_canonical_cancel(self, task_id: str, execution: asyncio.Task[Any]) -> None:
        """Cancellation must work even when a model/tool yields no stream chunks."""
        while not execution.done():
            await asyncio.sleep(_CANCEL_POLL_SECONDS)
            entry = await asyncio.to_thread(self._get_task_queue_entry, task_id)
            if entry is not None and entry.status == "cancelled":
                execution.cancel()
                return

    async def _run_task_in_worker(
        self,
        task_desc: dict[str, Any],
        completed_results: dict[str, str] | None = None,
        *,
        worker_slot: BackgroundWorkerSlot | None = None,
    ) -> str:
        """Run one LLM task under a worker lease."""
        task_id = task_desc.get("task_id", "unknown")
        leased_here = worker_slot is None
        slot = worker_slot or await self._acquire_worker(task_id)
        try:
            if self._task_isolated and self._task_runner_supervisor is not None:
                # Dependency context is already embedded in task_desc by callers
                # that need it; isolated children re-evaluate from task_desc alone.
                if completed_results:
                    # Stash dep results into a copy so the child can reconstruct context.
                    task_desc = {
                        **task_desc,
                        "_completed_results": completed_results,
                    }
                return await self._run_llm_task_isolated(
                    task_desc,
                    worker_slot=slot,
                )
            return await self._run_llm_task(
                task_desc,
                completed_results,
                worker_slot=slot,
            )
        finally:
            if leased_here:
                await self._release_worker(slot)

    async def _run_llm_task(
        self,
        task_desc: dict[str, Any],
        completed_results: dict[str, str] | None = None,
        *,
        worker_slot: BackgroundWorkerSlot | None = None,
    ) -> str:
        """Run an LLM task and record its complete activity lifecycle."""
        from core.activity.logger import ActivityLogger

        task_id, title, _description = _task_activity_identity(task_desc)
        submitted_by = str(task_desc.get("submitted_by") or "unknown")
        trigger = f"task:{task_id}"
        task_meta = {
            "task_id": task_id,
            "title": title,
            "submitted_by": submitted_by,
            "attempt_token": task_desc.get("_attempt_token", ""),
            "attempt": task_desc.get("_attempt_number"),
        }
        activity = ActivityLogger(self._anima_dir)
        await activity.alog(
            "task_exec_start",
            summary=t("pending_executor.task_exec_start", title=title),
            ctx=trigger,
            meta=task_meta,
        )
        try:
            result = await self._run_llm_task_under_agent_session_context(
                task_desc,
                completed_results,
                worker_slot=worker_slot,
            )
        except asyncio.CancelledError:
            await activity.alog(
                "task_exec_end",
                summary=t("pending_executor.task_exec_end", title=title, result="cancelled"),
                ctx=trigger,
                meta={**task_meta, "status": "cancelled"},
                safe=True,
            )
            _request_background_review(self._anima_dir, "task_end")
            raise
        except Exception as exc:
            error = str(exc).strip()[:200] or type(exc).__name__
            await activity.alog(
                "task_exec_end",
                summary=t("pending_executor.task_exec_end", title=title, result=error),
                ctx=trigger,
                meta={
                    **task_meta,
                    "status": "error",
                    "error": error,
                    "error_type": type(exc).__name__,
                },
                safe=True,
            )
            _request_background_review(self._anima_dir, "task_end")
            raise

        status = {
            _SENTINEL_CANCELLED: "cancelled",
            _SENTINEL_BUDGET_SKIPPED: "budget_skipped",
            _SENTINEL_UNDECLARED: "undeclared",
        }.get(result, "completed")
        await activity.alog(
            "task_exec_end",
            content=result[:2000],
            summary=t("pending_executor.task_exec_end", title=title, result=result[:200]),
            ctx=trigger,
            meta={**task_meta, "status": status, "result": result[:200]},
        )

        # Company timeline: one record per completed task.  This successful
        # ``task_exec_end`` is the single execution-completion point; the
        # ``task_completed`` activity in ``_deliver_task_wakeups`` is emitted
        # per wakeup delivery (and only for some recipients), so recording
        # there would duplicate the same task.
        if status == "completed":
            from core.hub.timeline_activity import record_activity

            record_activity(self._anima_name, via="task", text=result, label=title)

        _request_background_review(self._anima_dir, "task_end")
        return result

    def _task_model_config_override(self, task_desc: dict[str, Any]) -> Any:
        """Build a per-task ModelConfig override when the task specifies a model.

        When ``task_desc["model"]`` is set and valid, an override is built on
        top of the anima's current model config (model / execution_mode
        replaced, credential and fallback_models inherited from the base), so
        the new model can fall back when rate-guarded.  Invalid or unparseable
        values are logged and ignored — the task continues with the anima
        default and is never failed.  Returns ``None`` when no override
        applies.
        """
        from core.activity.logger import ActivityLogger

        requested = task_desc.get("model")
        if not isinstance(requested, str) or not requested.strip():
            return None
        requested = requested.strip()

        anima = getattr(self, "_anima", None)
        base = getattr(anima, "model_config", None)
        if base is None:
            return None

        try:
            from core.config import load_config
            from core.config.model_config import resolve_model_selection

            cfg = load_config()
        except Exception as exc:
            logger.warning(
                "[%s] Could not load config for per-task model override; using default: %s",
                self._anima_name,
                exc,
            )
            return None

        try:
            override = resolve_model_selection(
                base,
                lane="task",
                requested_model=requested,
                config=cfg,
                apply_fallback=False,
            ).effective
        except ValueError as exc:
            logger.warning(
                "[%s] Ignoring invalid per-task model override %r; using anima default: %s",
                self._anima_name,
                requested,
                exc,
            )
            return None
        try:
            ActivityLogger(self._anima_dir).log(
                "model_override",
                summary=t(
                    "pending_executor.model_override",
                    requested=requested,
                    resolved=override.model,
                ),
                ctx=f"task:{task_desc.get('task_id', 'unknown')}",
                meta={
                    "task_id": task_desc.get("task_id", "unknown"),
                    "requested_model": requested,
                    "resolved_model": override.model,
                    "resolved_mode": override.resolved_mode,
                },
            )
        except Exception:
            logger.debug("pending_executor: failed to log model_override activity", exc_info=True)
        return override

    async def _run_llm_task_under_agent_session_context(
        self,
        task_desc: dict[str, Any],
        completed_results: dict[str, str] | None = None,
        *,
        worker_slot: BackgroundWorkerSlot | None = None,
    ) -> str:
        """Core LLM task execution logic shared by parallel and serial paths.

        Returns the result summary string.
        """
        task_id, title, description = _task_activity_identity(task_desc)
        context = task_desc.get("context", "")
        acceptance_criteria = task_desc.get("acceptance_criteria", [])
        constraints = task_desc.get("constraints", [])
        file_paths = task_desc.get("file_paths", [])
        reply_to = task_desc.get("reply_to")
        submitted_by = task_desc.get("submitted_by", "unknown")
        submitted_at = task_desc.get("submitted_at", "")

        # Skip if the task was cancelled in TaskStore (batch path; single path checks in watcher)
        try:
            entry = self._get_task_queue_manager().get_task_by_id(task_id)
            if entry and entry.status == "cancelled":
                logger.info(
                    "[%s] Skipping cancelled LLM task: id=%s",
                    self._anima_name,
                    task_id,
                )
                return _SENTINEL_CANCELLED
        except Exception:
            logger.debug(
                "Could not check TaskStore for cancellation: %s",
                task_id,
                exc_info=True,
            )

        # Mirror the start only after the cancellation gate.
        self._sync_task_queue(task_id, "in_progress")

        # Build dependency context for batch tasks
        dep_context = ""
        if completed_results:
            dep_context = self._build_dependency_context(task_desc, completed_results)

        from core.memory.conversation.streaming_journal import StreamingJournal
        from core.paths import load_prompt

        trigger = f"task:{task_id}"

        _none = t("pending_executor.none_value")
        criteria_text = "\n".join(f"- {c}" for c in acceptance_criteria) if acceptance_criteria else _none
        constraints_text = "\n".join(f"- {c}" for c in constraints) if constraints else _none
        paths_text = "\n".join(f"- {p}" for p in file_paths) if file_paths else _none

        full_context = context or _none
        if dep_context:
            full_context = f"{full_context}\n\n{dep_context}"

        working_directory = task_desc.get("working_directory", "")
        if not working_directory:
            working_directory = _resolve_default_workspace(self._anima_dir)
        prompt = load_prompt(
            "task_exec",
            task_id=task_id,
            title=title,
            submitted_by=submitted_by,
            workspace=working_directory or t("pending_executor.workspace_not_specified"),
            description=description,
            context=full_context,
            acceptance_criteria=criteria_text,
            constraints=constraints_text,
            file_paths=paths_text,
            active_workers=self._format_active_sibling_tasks(task_id) or _none,
            submission_line=_submission_line(submitted_at),
        )

        lane_getter = getattr(type(self._anima), "_agent_for_lane", None)
        if worker_slot is not None:
            agent = worker_slot.agent
        else:
            agent = self._anima._agent_for_lane("background") if callable(lane_getter) else self._anima.agent

        journal = StreamingJournal(self._anima_dir, session_type="task", thread_id=_task_thread_id(task_id))
        await asyncio.to_thread(journal.open, trigger=trigger)

        model_config_override = self._task_model_config_override(task_desc)

        accumulated_text = ""
        result_summary = ""
        tool_call_records: list[dict[str, Any]] = []
        task_failed_reason = ""
        had_error = False
        error_message = ""
        stop_kind = "normal"
        cycle_error_category = ""
        try:
            if worker_slot is not None:
                session_context = worker_slot.session_lock
            else:
                session_context = getattr(self._anima, "_taskexec_session_lock", None)
                if not isinstance(session_context, asyncio.Lock):
                    session_context = asyncio.Lock()
                    self._anima._taskexec_session_lock = session_context
            if session_context is None:
                from contextlib import nullcontext

                session_context = nullcontext()
            async with session_context:
                if worker_slot is not None:
                    interrupt_event = worker_slot.interrupt_event
                elif self._anima and hasattr(self._anima, "_get_interrupt_event"):
                    interrupt_event = self._anima._get_interrupt_event("_taskexec")
                else:
                    interrupt_event = None
                interrupt_events = getattr(self._anima, "_interrupt_events", None)
                if interrupt_event is not None and isinstance(interrupt_events, dict):
                    interrupt_events[task_id] = interrupt_event
                if interrupt_event is not None:
                    interrupt_event.clear()
                    agent.set_interrupt_event(interrupt_event)
                if working_directory:
                    agent.set_task_cwd(Path(working_directory))
                try:
                    agent.reset_reply_tracking(session_type="task")
                    agent.reset_read_paths()
                    next_cancel_poll = time.monotonic() + _CANCEL_POLL_SECONDS
                    async for chunk in agent.run_cycle_streaming(
                        prompt,
                        trigger=trigger,
                        thread_id=_task_thread_id(task_id),
                        model_config_override=model_config_override,
                    ):
                        # A cancel written to TaskStore by another process
                        # (supervisor, TaskBoard, server) only reaches the
                        # running stream through the interrupt event.
                        if interrupt_event is not None and time.monotonic() >= next_cancel_poll:
                            next_cancel_poll = time.monotonic() + _CANCEL_POLL_SECONDS
                            _q = self._get_task_queue_entry(task_id)
                            if _q is not None and _q.status == "cancelled":
                                logger.info(
                                    "[%s] Task %s cancelled in TaskStore; interrupting", self._anima_name, task_id
                                )
                                interrupt_event.set()
                        chunk_type = chunk.get("type")
                        if chunk_type == "text_delta":
                            accumulated_text += chunk.get("text", "")
                            await asyncio.to_thread(journal.write_text, chunk.get("text", ""))
                        elif chunk_type == "error":
                            had_error = True
                            error_message = chunk.get("message", "unknown error")
                            logger.warning(
                                "[%s] Streaming error during task %s: %s",
                                self._anima_name,
                                task_id,
                                error_message,
                            )
                        elif chunk_type == "retry_start":
                            had_error = False
                            error_message = ""
                        elif chunk_type == "cycle_done":
                            cycle_result = chunk.get("cycle_result", {})
                            tool_call_records = cycle_result.get("tool_call_records", [])
                            if not isinstance(tool_call_records, list):
                                tool_call_records = []
                            result_summary = cycle_result.get(
                                "summary",
                                accumulated_text[:500],
                            )
                            stop_kind = str(cycle_result.get("stop_kind") or "normal")
                            cycle_error_category = str(cycle_result.get("error_category") or "")
                            if cycle_result.get("action") == "error" or stop_kind == "stream_error":
                                task_failed_reason = result_summary or "task execution failed"
                            await asyncio.to_thread(journal.finalize, summary=result_summary[:500])
                finally:
                    if task_desc.get("_attempt_token"):
                        self._record_attempt_reported_to(task_desc, agent)
                    agent.set_task_cwd(None)
                    try:
                        terminal_entry = self._get_task_queue_entry(task_id)
                    except Exception:
                        logger.debug("Could not inspect task status for scratch cleanup: %s", task_id, exc_info=True)
                    else:
                        if terminal_entry is not None:
                            _cleanup_task_scratch(
                                self._anima_dir,
                                task_id,
                                status=terminal_entry.status,
                            )
                    if (
                        interrupt_event is not None
                        and isinstance(interrupt_events, dict)
                        and interrupt_events.get(task_id) is interrupt_event
                    ):
                        interrupt_events.pop(task_id, None)
        finally:
            await asyncio.to_thread(journal.close)

        error_suppressed = False
        if had_error or task_failed_reason:
            try:
                _entry = self._get_task_queue_manager().get_task_by_id(task_id)
                if (
                    _entry
                    and _entry.status == "done"
                    and isinstance(_entry.meta, dict)
                    and _entry.meta.get("completed_by") == "agent_declaration"
                ):
                    error_suppressed = True
                    logger.info(
                        "[%s] Task %s stream error suppressed: already marked done in queue",
                        self._anima_name,
                        task_id,
                    )
                    result_summary = str(
                        _entry.meta.get("result_note")
                        or _entry.summary
                        or result_summary
                        or accumulated_text[:500]
                        or t("pending_executor.task_completed")
                    )
            except Exception as e:
                logger.debug("pending_executor: failed to check task queue for task %s: %s", task_id, e)

        if had_error and not error_suppressed:
            raise TaskExecError(f"Task {task_id} encountered streaming error: {error_message}")
        if task_failed_reason and not error_suppressed:
            raise RuntimeError(task_failed_reason)

        if stop_kind == "budget_skipped":
            self._record_run_ended(task_id, stop_kind)
            logger.info("[%s] Task %s skipped without execution: token budget unavailable", self._anima_name, task_id)
            return _SENTINEL_BUDGET_SKIPPED

        if not result_summary:
            result_summary = accumulated_text[:500] or t("pending_executor.task_completed")

        if cycle_error_category == "auth":
            raise TaskExecError("task execution failed due to a terminal authentication error (credential problem)")

        # The ledger is the only record of how a run ended.  A session that did
        # not declare done or cancelled hands the task back to its owner as
        # pending: no continuation, no probe, and no legacy descriptor regeneration.
        entry = self._get_task_queue_entry(task_id)
        if entry is not None:
            # Business completion and execution termination are separate: a
            # declared task can still have an interrupted stream. Persist the
            # current run's outcome for the canonical attempt finalizer too.
            self._record_run_ended(task_id, stop_kind)
            meta = entry.meta if isinstance(entry.meta, dict) else {}
            if entry.status == "cancelled":
                return _SENTINEL_CANCELLED
            if entry.status == "pending" and stop_kind == "normal":
                # The claim set in_progress; pending here means the anima
                # declared it (update_task status=pending) during this run.
                self._mark_declared_pending(task_id)
            if entry.status not in ("done", "delegated"):
                # Keep the actual outcome available to the owner even when
                # completion could not be declared (e.g. a tool failed while
                # updating the ledger). The sentinel controls task state; it
                # must not replace the model's evidence in the result file.
                await asyncio.to_thread(
                    self._save_task_result,
                    task_id,
                    f"{_SENTINEL_UNDECLARED}\n\n{result_summary}",
                )
                logger.info(
                    "[%s] LLM task ended without a completion declaration: id=%s stop_kind=%s",
                    self._anima_name,
                    task_id,
                    stop_kind,
                )
                return _SENTINEL_UNDECLARED
            if meta.get("completed_by") == "agent_declaration":
                result_summary = str(meta.get("result_note") or result_summary)

        # Send completion notification
        if task_desc.get("_attempt_token"):
            # Completion and its retryable outbox event commit together on
            # the root. Notification delivery never changes task outcome.
            reply_to = None
        if reply_to:
            if isinstance(reply_to, dict):
                reply_to = reply_to.get("name")
            elif not isinstance(reply_to, str):
                reply_to = None
        # Skip notification when the result carries no information: the
        # completion is already tracked in the task queue / activity log,
        # and an empty echo only costs the recipient a full LLM cycle.
        if reply_to:
            _summary_body = (result_summary or "").strip()
            if (
                not _summary_body
                or _summary_body == t("pending_executor.task_completed")
                or _summary_body.startswith("[Session interrupted")
            ):
                logger.info(
                    "[%s] Skipping empty task completion notification for %s (task %s)",
                    self._anima_name,
                    reply_to,
                    task_id,
                )
                reply_to = None
        if reply_to:
            try:
                notify_text = load_prompt(
                    "task_complete_notify",
                    task_id=task_id,
                    title=title,
                    result_summary=result_summary[:_TASK_COMPLETE_NOTIFY_MAX_CHARS],
                )
                from core.trust import ORIGIN_ANIMA

                for _attempt in range(2):
                    try:
                        self._anima.messenger.send(
                            to=reply_to,
                            content=notify_text,
                            origin_chain=[ORIGIN_ANIMA],
                        )
                        break
                    except Exception:
                        if _attempt == 0:
                            logger.warning(
                                "[%s] Task completion notification failed, retrying",
                                self._anima_name,
                            )
                        else:
                            logger.error(
                                "[%s] Task completion notification failed after retry to %s",
                                self._anima_name,
                                reply_to,
                                exc_info=True,
                            )
                            if hasattr(self._anima, "_activity"):
                                await self._anima._activity.alog(
                                    "error",
                                    content=f"Task completion notification failed: {task_id} → {reply_to}",
                                )
            except Exception:
                logger.warning(
                    "[%s] Failed to build task completion notification",
                    self._anima_name,
                    exc_info=True,
                )

        logger.info("[%s] LLM task completed: id=%s", self._anima_name, task_id)
        return result_summary

    def _record_attempt_reported_to(self, task_desc: dict[str, Any], agent: Any) -> None:
        """Persist recipients reached by successful send_message calls in this attempt."""
        token = str(task_desc.get("_attempt_token") or "")
        if not token:
            return
        handler = getattr(agent, "_tool_handler", None)
        replied_to_for = getattr(handler, "replied_to_for", None)
        names = replied_to_for("task") if callable(replied_to_for) else set()
        reported_to = sorted(name for name in names if isinstance(name, str) and name)
        try:
            updated = self._get_task_queue_manager().store.update_attempt_meta(token, {"reported_to": reported_to})
            if not updated:
                logger.debug("Task attempt report tracking skipped for inactive attempt %s", token)
        except Exception:
            logger.warning("Could not persist task attempt report tracking for %s", token, exc_info=True)

    def wake(self) -> None:
        """Signal the watcher to check for new tasks immediately."""
        self._wake_event.set()

    async def execute_pending_task(
        self,
        task_desc: dict[str, Any],
        *,
        worker_slot: BackgroundWorkerSlot | None = None,
    ) -> None:
        """Execute a claimed TaskStore item as LLM work or a command task."""
        task_type = task_desc.get("task_type", "command")
        if task_type == "llm":
            await self._execute_llm_task(task_desc, worker_slot=worker_slot)
            return None
        if task_type != "command":
            raise ValueError(f"Unsupported task type: {task_type!r}")
        if not self._anima:
            raise RuntimeError("Cannot execute command task: anima not initialized")

        lane_getter = getattr(type(self._anima), "_agent_for_lane", None)
        agent = self._anima._agent_for_lane("background") if callable(lane_getter) else self._anima.agent
        bg_mgr = agent.background_manager
        if bg_mgr is None:
            raise RuntimeError("BackgroundTaskManager not available for command task")

        tool_name = str(task_desc.get("tool_name", ""))
        subcommand = str(task_desc.get("subcommand", ""))
        task_id = str(task_desc.get("task_id", ""))
        tool_args = {
            "subcommand": subcommand,
            "raw_args": list(task_desc.get("raw_args") or []),
            "anima_dir": task_desc.get("anima_dir", str(self._anima_dir)),
        }
        composite_name = f"{tool_name}:{subcommand}" if subcommand else tool_name
        logger.info(
            "Submitting command task to BackgroundTaskManager: id=%s tool=%s subcmd=%s",
            task_id,
            tool_name,
            subcommand,
        )

        if self._background_isolated and self._task_runner_supervisor is not None:

            async def dispatch_isolated(_name: str, _args: dict[str, Any]) -> str:
                result = await self._execute_command_task_isolated(task_desc)
                return str(result.get("result", ""))

            background_task_id = await bg_mgr.submit_async(
                composite_name,
                tool_args,
                dispatch_isolated,
                task_id=task_id or None,
            )
        else:

            def _dispatch_fn(name: str, args: dict[str, Any]) -> str:
                """Execute the tool via CLI subprocess (same as direct execution)."""
                import subprocess

                module_name = name.split(":")[0] if ":" in name else name
                cmd = ["animaworks-tool", module_name]
                subcmd = args.get("subcommand", "")
                if subcmd:
                    cmd.append(subcmd)
                cmd.extend(args.get("raw_args", []))
                if subcmd and args.get("raw_args") and args["raw_args"][0] == subcmd:
                    cmd = ["animaworks-tool", module_name, *args["raw_args"]]
                cmd.append("-j")

                env = {
                    **os.environ,
                    "ANIMAWORKS_ANIMA_DIR": args.get("anima_dir", ""),
                }
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=_PENDING_TASK_SUBPROCESS_TIMEOUT,
                    env=env,
                    check=False,
                )
                if result.returncode != 0:
                    error_msg = result.stderr.strip() or f"Exit code {result.returncode}"
                    raise ToolExecutionError(f"Tool {name} failed: {error_msg}")
                return result.stdout.strip()

            background_task_id = bg_mgr.submit(
                composite_name,
                tool_args,
                _dispatch_fn,
                task_id=task_id or None,
            )

        active_tasks = getattr(bg_mgr, "_async_tasks", None)
        background_task = active_tasks.get(background_task_id) if isinstance(active_tasks, dict) else None
        if not isinstance(background_task, asyncio.Task):
            if task_desc.get("_attempt_token"):
                raise RuntimeError(f"Background task did not start: {background_task_id}")
            return None
        await background_task
        if task_desc.get("_attempt_token"):
            completed_task = bg_mgr.get_task(background_task_id)
            if completed_task is None:
                raise RuntimeError(f"Background task result is missing: {background_task_id}")
            self._record_command_task_outcome(task_desc, completed_task)
        return None

    def _record_command_task_outcome(self, task_desc: dict[str, Any], background_task: Any) -> None:
        """Persist command output in its TaskStore attempt and legacy result view."""
        task_id = str(task_desc["task_id"])
        task_status = getattr(background_task.status, "value", background_task.status)
        if task_status not in {"completed", "failed"}:
            raise RuntimeError(f"Command task ended in unexpected state: {task_status}")

        result = background_task.result if task_status == "completed" else background_task.error
        summary = str(background_task.summary())
        self._save_task_result(task_id, str(result or summary))

        manager = self._get_task_queue_manager()
        entry = manager.get_task_by_id(task_id)
        if entry is None:
            raise RuntimeError(f"Command task is missing from TaskStore: {task_id}")
        succeeded = task_status == "completed"
        metadata: dict[str, Any] = {
            "result_note": summary[:_TASK_RESULT_MAX_CHARS],
            "command_status": task_status,
            "last_run_ended_at": now_iso(),
            "last_run_stop_kind": "command_completed" if succeeded else "command_failed",
        }
        if not succeeded:
            metadata["last_run_note"] = str(result or summary)[:300]
        manager.update_meta(task_id, metadata)
        if entry.status not in _QUEUE_STICKY_STATUSES:
            manager.update_status(task_id, "done" if succeeded else "pending")
        # BackgroundTaskManager's existing callback delivers the completion or
        # failure notification; do not enqueue a duplicate TaskStore wakeup.
        task_desc["_background_notification_handled"] = True

    async def _execute_command_task_isolated(self, task_desc: dict[str, Any]) -> dict[str, Any]:
        """Run a command-type task inside a background-lane child."""
        from core.runtime.task_runner_supervisor import TaskRunnerError

        assert self._task_runner_supervisor is not None
        task_id = str(task_desc.get("task_id") or "unknown")
        attempt = int(task_desc.get("_attempt_number") or self._next_attempt(task_id))
        payload = {
            "tool_name": task_desc.get("tool_name", ""),
            "subcommand": task_desc.get("subcommand", ""),
            "raw_args": task_desc.get("raw_args", []),
            "anima_dir": task_desc.get("anima_dir", str(self._anima_dir)),
        }

        async def _on_spawned(job: Any) -> None:
            if token := task_desc.get("_attempt_token"):
                self._get_task_queue_manager().store.set_identity(
                    str(token),
                    {
                        "pid": os.getpid(),
                        "task_pid": job.pid,
                        "pgid": job.pgid,
                        "job_id": job.identity.job_id,
                        "root_epoch": job.identity.root_epoch,
                        "process_start_time": job.process_start_time,
                    },
                )

        try:
            return await self._task_runner_supervisor.run_background(
                kind="command",
                payload=payload,
                attempt=attempt,
                display_lane="background",
                on_spawned=_on_spawned,
            )
        except TaskRunnerError as exc:
            logger.warning(
                "[%s] Isolated background command failed: id=%s err=%s",
                self._anima_name,
                task_id,
                exc,
            )
            raise RuntimeError(
                f"INTERRUPTED: background task runner child exited without a result. (cause: {exc})"
            ) from exc

    async def _execute_llm_task(
        self,
        task_desc: dict[str, Any],
        *,
        worker_slot: BackgroundWorkerSlot | None = None,
    ) -> None:
        """Execute an LLM task in an isolated background worker.

        The task is executed as a minimal-context LLM session using
        the task_exec.md template.  Delegates to ``_run_llm_task``
        for the actual execution logic.  In root (supervisor present), the
        LLM session runs in a disposable task-runner child.
        """
        task_id = task_desc.get("task_id", "unknown")
        session_started_at = now_iso()

        logger.info(
            "[%s] Executing LLM task: id=%s title=%s",
            self._anima_name,
            task_id,
            task_desc.get("title", ""),
        )

        preleased = worker_slot is not None
        keepalive_task: asyncio.Future[Any] | None = None
        try:
            pool_capable = callable(getattr(type(self._anima), "_acquire_background_worker", None))
            keepalive = getattr(self._anima, "_keepalive_while_busy", None)
            if callable(keepalive):
                keepalive_result = keepalive()
                if inspect.isawaitable(keepalive_result):
                    keepalive_task = asyncio.ensure_future(keepalive_result)
            if pool_capable or (self._task_isolated and self._task_runner_supervisor is not None):
                # Worker lease also gates concurrent isolated children (pool size).
                result = await self._run_task_in_worker(
                    task_desc,
                    task_desc.get("_completed_results"),
                    worker_slot=worker_slot,
                )
            else:
                result = await self._run_llm_task(task_desc, task_desc.get("_completed_results"))
            if result != _SENTINEL_UNDECLARED:
                await asyncio.to_thread(self._save_task_result, task_id, result)
            status, summary = _classify_task_result(result)
            self._sync_task_queue(task_id, status, summary=summary)
            if status == "done":
                self._schedule_live_fact_extraction(session_started_at)
        except Exception as exc:
            if self._shutdown_event.is_set():
                logger.info(
                    "[%s] Shutdown interrupted LLM task %s; deferring failure to startup recovery",
                    self._anima_name,
                    task_id,
                )
                raise
            logger.exception(
                "[%s] LLM task failed: id=%s",
                self._anima_name,
                task_id,
            )
            await self._return_task_to_pending(
                task_desc,
                f"{type(exc).__name__}: {str(exc)[:200]}",
                stop_kind="crash",
            )
        finally:
            if keepalive_task is not None:
                keepalive_task.cancel()
                await asyncio.gather(keepalive_task, return_exceptions=True)
            if preleased:
                await self._release_worker(worker_slot)
            self._anima._clear_busy_status_sidecar_if_idle()

    async def _run_llm_task_isolated(
        self,
        task_desc: dict[str, Any],
        *,
        worker_slot: BackgroundWorkerSlot | None = None,
    ) -> str:
        """Run TaskExec LLM work in a disposable task-runner child process.

        Caller is responsible for exclusion locks and worker-slot leasing.
        """
        from core.runtime.task_runner_supervisor import TaskRunnerCancelled, TaskRunnerError

        assert self._task_runner_supervisor is not None
        task_id = str(task_desc.get("task_id") or "unknown")
        attempt = int(task_desc.get("_attempt_number") or self._next_attempt(task_id))
        slot_id = worker_slot.slot_id if worker_slot is not None else None
        display_lane = self._display_lane_for_task(task_id, slot_id)

        async def _on_spawned(job: Any) -> None:
            if token := task_desc.get("_attempt_token"):
                self._get_task_queue_manager().store.set_identity(
                    str(token),
                    {
                        "pid": os.getpid(),
                        "task_pid": job.pid,
                        "pgid": job.pgid,
                        "job_id": job.identity.job_id,
                        "root_epoch": job.identity.root_epoch,
                        "process_start_time": job.process_start_time,
                    },
                )

        try:
            isolated = await self._task_runner_supervisor.run_task(
                task_desc,
                attempt=attempt,
                display_lane=display_lane,
                on_spawned=_on_spawned,
            )
        except TaskRunnerCancelled as exc:
            # The queue entry was cancelled on purpose (e.g. superseded); the
            # runner was stopped deliberately, so this is not a crash.
            logger.info("[%s] Isolated TaskExec stopped by queue cancel: id=%s", self._anima_name, task_id)
            raise RuntimeError(f"CANCELLED: the task was cancelled in the queue ({exc}).") from exc
        except TaskRunnerError as exc:
            logger.warning(
                "[%s] Isolated TaskExec child failed: id=%s err=%s",
                self._anima_name,
                task_id,
                exc,
            )
            # Crash semantics: treat as interrupted / retryable for Layer2.
            # Keep the original TaskRunnerError early in the summary (before the
            # 200-char truncation) — flattening it cost a day of log archaeology
            # on 2026-08-12.
            raise RuntimeError(
                f"INTERRUPTED: task runner child exited without a result (cause: {exc}). "
                "May have PARTIALLY EXECUTED; verify actual completion state before re-delegating."
            ) from exc

        result = isolated.get("result")
        if not isinstance(result, str):
            result = str(result) if result is not None else ""
        return result
