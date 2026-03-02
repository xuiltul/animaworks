from __future__ import annotations
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""DigitalAnima -- the public façade.

Implementation is split into Mixin sub-modules for manageability:

- ``_anima_messaging``  -- MessagingMixin (human chat, bootstrap, greet)
- ``_anima_inbox``      -- InboxMixin (Anima-to-Anima inbox processing)
- ``_anima_heartbeat``  -- HeartbeatMixin (heartbeat/cron prompt & cycle)
- ``_anima_lifecycle``  -- LifecycleMixin (heartbeat orchestration, consolidation, cron)
"""

import asyncio
import logging
import re
import threading
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

from core.agent import AgentCore
from core.background import BackgroundTask
from core.memory.activity import ActivityLogger
from core.memory import MemoryManager
from core.messenger import Messenger
from core.i18n import t
from core.exceptions import (  # noqa: F401
    AnimaWorksError,
    ExecutionError,
    LLMAPIError,
    ToolError,
    MemoryIOError,
)
from core.schemas import AnimaStatus, ModelConfig

logger = logging.getLogger("animaworks.anima")

# ── Mixin imports ───────────────────────────────────────────────
from core._anima_messaging import MessagingMixin
from core._anima_inbox import InboxMixin
from core._anima_heartbeat import HeartbeatMixin
from core._anima_lifecycle import LifecycleMixin

# ── Re-exports for backward compatibility ───────────────────────
# Tests and other modules import these symbols from ``core.anima``.
from core._anima_inbox import InboxResult  # noqa: F401
from core._anima_heartbeat import (  # noqa: F401
    _RE_REFLECTION,
    _MIN_REFLECTION_LENGTH,
    _extract_reflection,
)


class DigitalAnima(
    MessagingMixin,
    InboxMixin,
    HeartbeatMixin,
    LifecycleMixin,
):
    """A Digital Anima: encapsulates identity, memory, agent, and communication.

    1 anima = 1 directory.
    """

    _MAX_THREAD_LOCKS = 20
    _THREAD_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_-]{1,36}$")

    @staticmethod
    def _validate_thread_id(thread_id: str) -> None:
        """Validate thread_id to prevent path traversal attacks."""
        if not DigitalAnima._THREAD_ID_PATTERN.match(thread_id):
            raise ValueError(
                f"Invalid thread_id: {thread_id!r}. "
                "Must be 1-36 alphanumeric, underscore, or hyphen characters."
            )

    def __init__(self, anima_dir: Path, shared_dir: Path) -> None:
        self.anima_dir = anima_dir
        self.name = anima_dir.name
        self._activity = ActivityLogger(anima_dir)

        self.memory = MemoryManager(anima_dir)
        self.model_config = self.memory.read_model_config()
        self.messenger = Messenger(shared_dir, self.name)
        self._interrupt_events: dict[str, asyncio.Event] = {}
        self.agent = AgentCore(
            anima_dir, self.memory, self.model_config, self.messenger
        )

        # 3-lock structure: conversation (human chat) / inbox (Anima-to-Anima MSG) / background (HB/cron/TaskExec)
        self._conversation_locks: dict[str, asyncio.Lock] = {}
        self._inbox_lock = asyncio.Lock()
        self._background_lock = asyncio.Lock()
        self._cron_idle = asyncio.Event()
        self._cron_idle.set()  # initially idle (no cron running)
        self._state_file_lock = threading.Lock()  # protects current_task.md / pending.md

        # Parallel task execution (DAG scheduler)
        self._task_semaphore: asyncio.Semaphore | None = None  # lazy init from config
        self._active_parallel_tasks: dict[str, dict[str, Any]] = {}  # task_id -> {title, description, started_at, batch_id, status}
        self.agent._tool_handler.set_state_file_lock(self._state_file_lock)
        self._status_slots: dict[str, str] = {"inbox": "idle", "background": "idle"}
        self._task_slots: dict[str, str] = {"inbox": "", "background": ""}
        self._last_heartbeat: datetime | None = None
        self._last_activity: datetime | None = None
        self._on_lock_released: Callable[[], None] | None = None
        self._pending_executor: Any | None = None  # set by runner after PendingTaskExecutor init

        # Greet cache (1-hour cooldown)
        self._last_greet_at: float | None = None
        self._last_greet_text: str | None = None
        self._last_greet_emotion: str = "neutral"
        self._GREET_COOLDOWN = 3600  # seconds

        # Wire background task completion callback
        self._ws_broadcast: Callable[[dict], Any] | None = None
        if self.agent.background_manager:
            self.agent.background_manager.on_complete = self._on_background_task_complete

        logger.info("DigitalAnima '%s' initialized from %s", self.name, anima_dir)

    # ── Thread lock management ──────────────────────────────────

    def _get_thread_lock(self, thread_id: str) -> asyncio.Lock:
        """Get or create a per-thread conversation lock.

        Implements LRU eviction when max locks reached. Locked (in-use)
        locks are never evicted.
        """
        if thread_id not in self._conversation_locks:
            if len(self._conversation_locks) >= self._MAX_THREAD_LOCKS:
                # Evict oldest idle lock
                for k in list(self._conversation_locks):
                    if not self._conversation_locks[k].locked():
                        del self._conversation_locks[k]
                        break
            self._conversation_locks[thread_id] = asyncio.Lock()
        return self._conversation_locks[thread_id]

    # ── Per-thread interrupt event management ──────────────────

    def _get_interrupt_event(self, thread_id: str = "default") -> asyncio.Event:
        """Get or create a per-thread interrupt event."""
        if thread_id not in self._interrupt_events:
            self._interrupt_events[thread_id] = asyncio.Event()
        return self._interrupt_events[thread_id]

    # ── Config / Callbacks ──────────────────────────────────────

    def set_on_message_sent(
        self, fn: Callable[[str, str, str], None],
    ) -> None:
        """Inject a callback fired after this anima sends a message."""
        self.agent.set_on_message_sent(fn)

    def set_on_schedule_changed(
        self, fn: Callable[[str], Any] | None,
    ) -> None:
        """Inject a callback fired when heartbeat.md or cron.md is modified."""
        self.agent.set_on_schedule_changed(fn)

    def drain_notifications(self) -> list[dict[str, Any]]:
        """Return and clear pending notification events."""
        return self.agent.drain_notifications()

    def drain_background_notifications(self) -> list[str]:
        """Read and remove all pending background task notifications.

        Returns list of notification texts for inclusion in heartbeat context.
        """
        notif_dir = self.agent.anima_dir / "state" / "background_notifications"
        if not notif_dir.is_dir():
            return []

        notifications: list[str] = []
        for path in sorted(notif_dir.glob("*.md")):
            try:
                notifications.append(path.read_text(encoding="utf-8"))
                path.unlink()
            except Exception:
                logger.warning("Failed to read notification: %s", path.name)

        return notifications

    async def interrupt(self, thread_id: str | None = None) -> dict[str, Any]:
        """Interrupt LLM session(s) without killing the process.

        Args:
            thread_id: If provided, only interrupt the specific thread.
                If None, interrupt all active threads (CLI compat).
        """
        if thread_id:
            logger.info("Interrupt requested for anima '%s' thread=%s", self.name, thread_id)
            evt = self._interrupt_events.get(thread_id)
            if evt:
                evt.set()
        else:
            logger.info("Interrupt requested for anima '%s' (all threads)", self.name)
            for evt in self._interrupt_events.values():
                evt.set()
        return {"status": "interrupted", "name": self.name}

    def reload_config(self) -> dict[str, Any]:
        """Hot-reload ModelConfig from status.json without process restart."""
        old = self.model_config
        new = self.memory.read_model_config()
        self.model_config = new
        self.agent.update_model_config(new)
        changes = [
            k for k in ModelConfig.model_fields
            if getattr(old, k) != getattr(new, k)
        ]
        logger.info("reload_config: model=%s, changes=%s", new.model, changes)
        return {"status": "ok", "model": new.model, "changes": changes}

    def set_on_lock_released(self, fn: Callable[[], Any]) -> None:
        """Inject a callback invoked when the anima's lock is released."""
        self._on_lock_released = fn

    def set_ws_broadcast(self, fn: Callable[[dict], Any]) -> None:
        """Inject a WebSocket broadcast function for background task notifications."""
        self._ws_broadcast = fn

    # ── Background task management ──────────────────────────────

    async def _on_background_task_complete(self, task: BackgroundTask) -> None:
        """Callback invoked when a background tool call completes."""
        logger.info(
            "[%s] Background task completed: id=%s tool=%s status=%s",
            self.name, task.task_id, task.tool_name, task.status.value,
        )

        # Broadcast via WebSocket
        if self._ws_broadcast:
            try:
                await self._ws_broadcast({
                    "type": "background_task.done",
                    "data": {
                        "task_id": task.task_id,
                        "anima": self.name,
                        "tool_name": task.tool_name,
                        "status": task.status.value,
                        "result_summary": task.summary(),
                    },
                })
            except Exception:
                logger.exception(
                    "[%s] WebSocket broadcast failed for bg task %s",
                    self.name, task.task_id,
                )

        # Notify human via configured channels
        if self.agent.has_human_notifier:
            try:
                notifier = self.agent.human_notifier
                if notifier:
                    await notifier.notify(
                        subject=t("anima.bg_task_done", tool=task.tool_name),
                        body=task.summary(),
                        priority="normal",
                        anima_name=self.name,
                    )
            except Exception:
                logger.exception(
                    "[%s] Human notification failed for bg task %s",
                    self.name, task.task_id,
                )

        # Send inbox notification so next heartbeat picks up the result
        try:
            summary = task.summary()
            subject = t("anima.bg_task_done", tool=task.tool_name)
            if task.status.value == "failed":
                subject = t("anima.bg_task_failed", tool=task.tool_name)

            notif_dir = self.agent.anima_dir / "state" / "background_notifications"
            notif_dir.mkdir(parents=True, exist_ok=True)
            notif_path = notif_dir / f"{task.task_id}.md"
            notif_content = (
                f"# {subject}\n\n"
                f"{t('anima.bg_notif_task_id', task_id=task.task_id)}\n"
                f"{t('anima.bg_notif_tool', tool=task.tool_name)}\n"
                f"{t('anima.bg_notif_status', status=task.status.value)}\n"
                f"{t('anima.bg_notif_result', summary=summary)}\n"
            )
            notif_path.write_text(notif_content, encoding="utf-8")
            logger.info(
                "[%s] Background task notification written: %s",
                self.name, notif_path.name,
            )
        except Exception:
            logger.exception(
                "[%s] Failed to write bg task notification for %s",
                self.name, task.task_id,
            )

    @property
    def background_tasks(self) -> list[dict[str, Any]]:
        """Return a list of all background tasks as dicts."""
        mgr = self.agent.background_manager
        if not mgr:
            return []
        return [t.to_dict() for t in mgr.list_tasks()]

    def _notify_lock_released(self) -> None:
        if self._on_lock_released:
            try:
                self._on_lock_released()
            except Exception:
                logger.exception("[%s] on_lock_released callback failed", self.name)

    # ── Properties ──────────────────────────────────────────────

    @property
    def needs_bootstrap(self) -> bool:
        """True if this anima has not completed the first-run bootstrap."""
        return (self.anima_dir / "bootstrap.md").exists()

    @property
    def primary_status(self) -> str:
        """Primary status: any conversation:* > inbox > background."""
        for key, val in self._status_slots.items():
            if key.startswith("conversation:") and val != "idle":
                return val
        inbox = self._status_slots.get("inbox", "idle")
        if inbox != "idle":
            return inbox
        return self._status_slots.get("background", "idle")

    @property
    def primary_task(self) -> str:
        """Primary task: any conversation:* > inbox > background."""
        for key, val in self._task_slots.items():
            if key.startswith("conversation:") and val:
                return val
        inbox = self._task_slots.get("inbox", "")
        if inbox:
            return inbox
        return self._task_slots.get("background", "")

    @property
    def status(self) -> AnimaStatus:
        return AnimaStatus(
            name=self.name,
            status=self.primary_status,
            current_task=self.primary_task,
            last_heartbeat=self._last_heartbeat,
            last_activity=self._last_activity,
            pending_messages=self.messenger.unread_count(),
        )
