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
import json
import logging
import os
import re
import threading
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from core.agent import AgentCore
from core.background import BackgroundTask
from core.exceptions import (  # noqa: F401
    AnimaWorksError,
    ExecutionError,
    LLMAPIError,
    MemoryIOError,
    ToolError,
)
from core.i18n import t
from core.memory import MemoryManager
from core.memory.activity import ActivityLogger
from core.messenger import Messenger
from core.schemas import AnimaStatus, ModelConfig
from core.session_compactor import SessionCompactor
from core.time_utils import now_local

logger = logging.getLogger("animaworks.anima")


@dataclass(slots=True)
class BackgroundWorkerSlot:
    """One isolated TaskExec worker and its mutable execution state."""

    slot_id: int
    agent: AgentCore
    session_lock: asyncio.Lock
    interrupt_event: asyncio.Event


# ── Mixin imports ───────────────────────────────────────────────
from core._anima_heartbeat import (  # noqa: F401
    _MIN_REFLECTION_LENGTH,
    _RE_REFLECTION,
    HeartbeatMixin,
    _extract_reflection,
)

# ── Re-exports for backward compatibility ───────────────────────
# Tests and other modules import these symbols from ``core.anima``.
from core._anima_inbox import (
    InboxMixin,
    InboxResult,  # noqa: F401
)
from core._anima_lifecycle import LifecycleMixin
from core._anima_messaging import MessagingMixin


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
    _AGENT_LANES = ("chat", "background", "inbox")

    @staticmethod
    def _validate_thread_id(thread_id: str) -> None:
        """Validate thread_id to prevent path traversal attacks."""
        if not DigitalAnima._THREAD_ID_PATTERN.match(thread_id):
            raise ValueError(
                f"Invalid thread_id: {thread_id!r}. Must be 1-36 alphanumeric, underscore, or hyphen characters."
            )

    def __init__(self, anima_dir: Path, shared_dir: Path, *, busy_status_enabled: bool = True) -> None:
        self.anima_dir = anima_dir
        self.shared_dir = shared_dir
        self.name = anima_dir.name
        self._busy_status_enabled = busy_status_enabled
        self._activity = ActivityLogger(anima_dir)

        self.memory = MemoryManager(anima_dir)
        self.model_config = self.memory.read_model_config()
        self.messenger = Messenger(shared_dir, self.name)
        self._interrupt_events: dict[str, asyncio.Event] = {}
        self._progress_min_interval = 0.5  # seconds
        self._progress_last_mono: float = 0.0

        def _throttled_progress() -> None:
            import time

            mono = time.monotonic()
            if mono - self._progress_last_mono >= self._progress_min_interval:
                self._mark_busy_progress()
                self._progress_last_mono = mono

        self._agent_progress_callback = _throttled_progress

        # 3-lock structure: conversation (human chat) / inbox (Anima-to-Anima MSG) / background (HB/cron/TaskExec)
        self._conversation_locks: dict[str, asyncio.Lock] = {}
        self._active_chat_conversations: dict[str, Any] = {}
        self._inbox_lock = asyncio.Lock()
        self._background_lock = asyncio.Lock()
        # AgentCore carries mutable executor/tool state, so each execution lane
        # gets its own AgentCore and its own preparation/use lock.
        self._agent_session_locks: dict[str, asyncio.Lock] = {lane: asyncio.Lock() for lane in self._AGENT_LANES}
        # Backward-compatible alias for legacy chat-only call sites/tests.
        self._agent_session_lock = self._agent_session_locks["chat"]
        self._cron_idle = asyncio.Event()
        self._cron_idle.set()  # initially idle (no cron running)
        self._state_file_lock = threading.Lock()  # protects current_state.md / pending.md

        # Parallel task execution (DAG scheduler)
        self._task_semaphore: asyncio.Semaphore | None = None  # lazy init from config
        self._active_parallel_tasks: dict[
            str, dict[str, Any]
        ] = {}  # task_id -> {title, description, started_at, batch_id, status}
        self._ws_broadcast: Callable[[dict], Any] | None = None
        self._pending_executor: Any | None = None  # set by runner after PendingTaskExecutor init
        self.agent = self._create_lane_agent("chat")
        self._lane_agents: dict[str, AgentCore] = {
            "chat": self.agent,
            "background": self._create_lane_agent("background"),
            "inbox": self._create_lane_agent("inbox"),
        }
        from core.config.models import resolve_background_worker_pool_size

        self._background_worker_pool_size = resolve_background_worker_pool_size(self.anima_dir)
        self._background_worker_slots: list[BackgroundWorkerSlot] = []
        self._background_worker_queue: asyncio.Queue[BackgroundWorkerSlot] = asyncio.Queue()
        self._active_background_workers: dict[int, str] = {}
        self._background_worker_gate_lock = asyncio.Lock()
        self._background_worker_gate_count = 0
        self._initialize_background_worker_pool()
        self._status_slots: dict[str, str] = {"inbox": "idle", "background": "idle"}
        self._task_slots: dict[str, str] = {"inbox": "", "background": ""}
        self._last_heartbeat: datetime | None = None
        self._last_activity: datetime | None = None
        self._last_progress_at: datetime | None = None
        self._busy_since: datetime | None = None
        self._isolated_busy_jobs_provider: Callable[[], dict[str, Any]] | None = None
        self._on_lock_released: Callable[[], None] | None = None

        # Idle compaction timer (per-thread)
        from core.config.models import load_config

        _idle_min = load_config().heartbeat.idle_compaction_minutes
        self._session_compactor = SessionCompactor(idle_minutes=_idle_min)

        # Greet cache (1-hour cooldown)
        self._last_greet_at: float | None = None
        self._last_greet_text: str | None = None
        self._last_greet_emotion: str = "neutral"
        self._GREET_COOLDOWN = 3600  # seconds

        # Wire background task completion callback
        for agent in self._iter_lane_agents():
            if agent.background_manager:
                agent.background_manager.on_complete = self._on_background_task_complete

        logger.info("DigitalAnima '%s' initialized from %s", self.name, anima_dir)

    # ── Agent lane management ──────────────────────────────────────

    def _create_lane_agent(self, lane: str, *, codex_home: Path | None = None) -> AgentCore:
        """Create an AgentCore for one isolated execution lane."""
        agent = AgentCore(
            self.anima_dir,
            self.memory,
            self.model_config,
            self.messenger,
            codex_home=codex_home,
        )
        agent._progress_callback = self._agent_progress_callback
        agent._tool_handler.set_state_file_lock(self._state_file_lock)
        if agent.background_manager:
            agent.background_manager.on_complete = self._on_background_task_complete
        logger.debug("[%s] Agent lane initialized: %s", self.name, lane)
        return agent

    def _initialize_background_worker_pool(self) -> None:
        """Create TaskExec workers while preserving the legacy single-worker lane."""
        pool_size = self._background_worker_pool_size
        if pool_size == 1:
            slots = [
                BackgroundWorkerSlot(
                    slot_id=0,
                    agent=self._lane_agents["background"],
                    session_lock=self._agent_session_locks["background"],
                    interrupt_event=self._get_interrupt_event("_background"),
                )
            ]
        else:
            slots = []
            for slot_id in range(pool_size):
                codex_home = self.anima_dir / ".codex_home" / "workers" / str(slot_id)
                slots.append(
                    BackgroundWorkerSlot(
                        slot_id=slot_id,
                        agent=self._create_lane_agent(
                            f"background-worker-{slot_id}",
                            codex_home=codex_home,
                        ),
                        session_lock=asyncio.Lock(),
                        interrupt_event=asyncio.Event(),
                    )
                )
        self._background_worker_slots = slots
        for slot in slots:
            self._background_worker_queue.put_nowait(slot)

    async def _acquire_background_worker(self, task_id: str) -> BackgroundWorkerSlot:
        """Lease an isolated TaskExec worker and mark the background lane busy."""
        slot = await self._background_worker_queue.get()
        try:
            first_worker = False
            async with self._background_worker_gate_lock:
                if self._background_worker_gate_count == 0:
                    await self._background_lock.acquire()
                    first_worker = True
                self._background_worker_gate_count += 1
            self._active_background_workers[slot.slot_id] = task_id
            if first_worker:
                self._mark_busy_start()
            else:
                self._mark_busy_progress()
            return slot
        except BaseException:
            self._background_worker_queue.put_nowait(slot)
            raise

    async def _release_background_worker(self, slot: BackgroundWorkerSlot) -> None:
        """Return a TaskExec worker and release the shared background gate if idle."""
        self._active_background_workers.pop(slot.slot_id, None)
        became_idle = False
        async with self._background_worker_gate_lock:
            self._background_worker_gate_count = max(0, self._background_worker_gate_count - 1)
            if self._background_worker_gate_count == 0 and self._background_lock.locked():
                self._background_lock.release()
                became_idle = True
        self._background_worker_queue.put_nowait(slot)
        if became_idle:
            self._notify_lock_released()

    def _iter_lane_agents(self) -> list[AgentCore]:
        """Return unique AgentCore instances for all lanes."""
        agents = getattr(self, "_lane_agents", None)
        if not isinstance(agents, dict):
            return [self.agent] if hasattr(self, "agent") else []
        unique: list[AgentCore] = []
        seen: set[int] = set()
        for agent in agents.values():
            ident = id(agent)
            if ident not in seen:
                seen.add(ident)
                unique.append(agent)
        for slot in getattr(self, "_background_worker_slots", []):
            ident = id(slot.agent)
            if ident not in seen:
                seen.add(ident)
                unique.append(slot.agent)
        return unique

    def _agent_for_lane(self, lane: str) -> AgentCore:
        """Return the AgentCore assigned to *lane*."""
        agents = getattr(self, "_lane_agents", None)
        if isinstance(agents, dict) and lane in agents:
            return agents[lane]
        return self.agent

    def _agent_session_context(self, lane: str = "chat"):
        """Return the session lock context for one execution lane."""
        locks = getattr(self, "_agent_session_locks", None)
        if isinstance(locks, dict):
            lock = locks.get(lane)
            if isinstance(lock, asyncio.Lock):
                return lock
        lock = getattr(self, "_agent_session_lock", None)
        if isinstance(lock, asyncio.Lock):
            return lock
        return nullcontext()

    def _set_pending_executor_wake(self, wake_fn: Callable[[], Any]) -> None:
        """Wire PendingTaskExecutor wake callback into every lane ToolHandler."""
        for agent in self._iter_lane_agents():
            agent._tool_handler.set_pending_executor_wake(wake_fn)

    def _set_active_parallel_tasks_getter(self, getter: Callable[[], dict[str, dict[str, Any]]]) -> None:
        """Wire active parallel task visibility into every lane AgentCore."""
        for agent in self._iter_lane_agents():
            agent._active_parallel_tasks_getter = getter

    # ── Progress tracking ────────────────────────────────────────

    def _set_isolated_busy_jobs_provider(self, provider: Callable[[], dict[str, Any]]) -> None:
        """Include root-owned isolated jobs in the existing busy marker."""
        self._isolated_busy_jobs_provider = provider

    def _isolated_busy_jobs(self) -> dict[str, Any]:
        provider = getattr(self, "_isolated_busy_jobs_provider", None)
        if provider is None:
            return {}
        try:
            return provider()
        except Exception:
            logger.debug("[%s] Failed to collect isolated busy jobs", self.name, exc_info=True)
            return {}

    def _has_active_busy_lock(self) -> bool:
        """Return True while any local execution lane is actively holding work."""
        any_conversation_locked = any(lock.locked() for lock in self._conversation_locks.values())
        active_workers = bool(getattr(self, "_active_background_workers", {}))
        return (
            any_conversation_locked
            or self._background_lock.locked()
            or self._inbox_lock.locked()
            or active_workers
            or bool(self._isolated_busy_jobs())
        )

    def _mark_busy_start(self) -> None:
        """Reset progress timestamp at the start of a new busy period.

        Prevents the health monitor from seeing a stale ``_last_progress_at``
        from the previous task and falsely killing a process that just started.
        """
        now = now_local()
        self._last_progress_at = now
        self._busy_since = now
        self._write_busy_status_sidecar()

    def _mark_busy_progress(self) -> None:
        """Record forward progress for health checks and the busy sidecar."""
        now = now_local()
        self._last_progress_at = now
        if self._busy_since is None:
            self._busy_since = now
        if self._has_active_busy_lock():
            self._write_busy_status_sidecar()

    def _busy_status_sidecar_path(self) -> Path | None:
        """Path used by the supervisor as an IPC-independent busy signal."""
        if not self._busy_status_enabled:
            return None
        shared_dir = getattr(self, "shared_dir", None)
        if shared_dir is None:
            return None
        return Path(shared_dir).parent / "run" / "animas" / f"{self.name}.busy.json"

    def _write_busy_status_sidecar(self) -> None:
        """Write a small progress marker readable when IPC ping is blocked."""
        try:
            now = now_local()
            last_progress = self._last_progress_at or now
            busy_since = self._busy_since or last_progress
            path = self._busy_status_sidecar_path()
            if path is None:
                return
            path.parent.mkdir(parents=True, exist_ok=True)
            lanes: list[str] = []
            subprocesses: list[dict[str, Any]] = []
            try:
                for key, lock in self._conversation_locks.items():
                    if lock.locked():
                        lanes.append(f"conversation:{key}")
                if self._background_lock.locked():
                    lanes.append("background")
                if self._inbox_lock.locked():
                    lanes.append("inbox")
                for slot_id, task_id in getattr(self, "_active_background_workers", {}).items():
                    lanes.append(f"background-worker:{slot_id}:{task_id}")
                for job in self._isolated_busy_jobs().values():
                    identity = getattr(job, "identity", None)
                    lane = getattr(identity, "display_lane", None)
                    if lane and lane not in lanes:
                        lanes.append(str(lane))
                    job_pid = getattr(job, "pid", None)
                    if job_pid is not None:
                        subprocesses.append({"pid": job_pid, "kind": str(getattr(identity, "lane", None) or "task")})
            except Exception:
                logger.debug("[%s] Failed to collect busy status lanes", self.name, exc_info=True)
            payload = {
                "anima": self.name,
                "pid": os.getpid(),
                "is_busy": True,
                "busy_since": busy_since.isoformat(),
                "last_progress_at": last_progress.isoformat(),
                "updated_at": now.isoformat(),
                "lanes": lanes,
                "processes": subprocesses,
            }
            tmp_path = path.with_name(f"{path.name}.{os.getpid()}.tmp")
            tmp_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            tmp_path.replace(path)
        except OSError:
            logger.debug("[%s] Failed to write busy status sidecar", self.name, exc_info=True)

    def _clear_busy_status_sidecar_if_idle(self) -> None:
        """Remove the busy marker once all local execution locks are idle."""
        try:
            if self._has_active_busy_lock():
                return

            path = self._busy_status_sidecar_path()
            if path is None:
                return
            if not path.exists():
                return

            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                data = {}
            if data.get("pid") in (None, os.getpid()) or data.get("anima") == self.name:
                path.unlink(missing_ok=True)
        except OSError:
            logger.debug("[%s] Failed to clear busy status sidecar", self.name, exc_info=True)

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
        self,
        fn: Callable[[str, str, str], None],
    ) -> None:
        """Inject a callback fired after this anima sends a message."""
        for agent in self._iter_lane_agents():
            agent.set_on_message_sent(fn)

    def set_on_schedule_changed(
        self,
        fn: Callable[[str], Any] | None,
    ) -> None:
        """Inject a callback fired when heartbeat.md or cron.md is modified."""
        for agent in self._iter_lane_agents():
            agent.set_on_schedule_changed(fn)

    def drain_notifications(self) -> list[dict[str, Any]]:
        """Return and clear pending notification events."""
        events: list[dict[str, Any]] = []
        for agent in self._iter_lane_agents():
            events.extend(agent.drain_notifications())
        return events

    def drain_background_notifications(self) -> list[str]:
        """Read and remove all pending background notifications.

        Returns list of notification texts for inclusion in heartbeat context.
        """
        return self._drain_background_notifications(
            lambda _path: True,
        )

    def drain_chat_background_notifications(self) -> list[str]:
        """Read task-completion notifications intended for a chat turn.

        Cron-health and token-budget notices are operational context for the
        heartbeat, not background task results requested by the user.  Leave
        those files for the existing heartbeat drain so they are not silently
        consumed by a normal chat turn.
        """
        excluded_prefixes = ("cron_health_", "cron_guard_", "token_budget_")
        return self._drain_background_notifications(
            lambda path: not path.name.startswith(excluded_prefixes),
        )

    def _drain_background_notifications(self, predicate: Callable[[Path], bool]) -> list[str]:
        """Read and remove matching pending background notification files."""
        notif_dir = self.agent.anima_dir / "state" / "background_notifications"
        if not notif_dir.is_dir():
            return []

        notifications: list[str] = []
        for path in sorted(notif_dir.glob("*.md")):
            if not predicate(path):
                continue
            try:
                notifications.append(path.read_text(encoding="utf-8"))
                path.unlink()
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
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
        for agent in self._iter_lane_agents():
            agent.update_model_config(new)
        changes = [k for k in ModelConfig.model_fields if getattr(old, k) != getattr(new, k)]
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
            self.name,
            task.task_id,
            task.tool_name,
            task.status.value,
        )

        # Broadcast via WebSocket
        if self._ws_broadcast:
            try:
                await self._ws_broadcast(
                    {
                        "type": "background_task.done",
                        "data": {
                            "task_id": task.task_id,
                            "anima": self.name,
                            "tool_name": task.tool_name,
                            "status": task.status.value,
                            "result_summary": task.summary(),
                        },
                    }
                )
            except Exception:
                logger.exception(
                    "[%s] WebSocket broadcast failed for bg task %s",
                    self.name,
                    task.task_id,
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
                    self.name,
                    task.task_id,
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
                self.name,
                notif_path.name,
            )
        except Exception:
            logger.exception(
                "[%s] Failed to write bg task notification for %s",
                self.name,
                task.task_id,
            )

    @property
    def background_tasks(self) -> list[dict[str, Any]]:
        """Return a list of all background tasks as dicts."""
        mgr = self.agent.background_manager
        if not mgr:
            return []
        return [t.to_dict() for t in mgr.list_tasks()]

    def _notify_lock_released(self) -> None:
        self._clear_busy_status_sidecar_if_idle()
        if self._on_lock_released:
            try:
                self._on_lock_released()
            except Exception:
                logger.exception("[%s] on_lock_released callback failed", self.name)

    # ── Properties ──────────────────────────────────────────────

    @property
    def needs_bootstrap(self) -> bool:
        """True if this anima has not completed the first-run bootstrap."""
        from core.bootstrap_state import get_bootstrap_status

        return bool(get_bootstrap_status(self.anima_dir).get("needs_bootstrap"))

    @property
    def bootstrap_state(self) -> dict[str, Any]:
        """Return state-aware first-run bootstrap lifecycle information."""
        from core.bootstrap_state import get_bootstrap_status

        return get_bootstrap_status(self.anima_dir)

    @property
    def needs_user_input(self) -> bool:
        """True when interactive bootstrap is waiting for the user's first input."""
        return bool(self.bootstrap_state.get("needs_user_input"))

    @property
    def needs_repair(self) -> bool:
        """True when bootstrap artifacts indicate manual or CLI repair is needed."""
        return bool(self.bootstrap_state.get("needs_repair"))

    @property
    def needs_background_bootstrap(self) -> bool:
        """True when a character-sheet bootstrap can safely run in background."""
        return bool(self.bootstrap_state.get("needs_background_bootstrap"))

    @property
    def primary_status(self) -> str:
        """Primary status: any conversation:* > inbox > background."""
        for key, val in self._status_slots.items():
            if key.startswith("conversation:") and val != "idle":
                return val
        inbox = self._status_slots.get("inbox", "idle")
        if inbox != "idle":
            return inbox
        if self._active_background_workers:
            return "task_exec"
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
        if self._active_background_workers:
            return next(iter(self._active_background_workers.values()))
        return self._task_slots.get("background", "")

    @property
    def status(self) -> AnimaStatus:
        return AnimaStatus(
            name=self.name,
            status=self.primary_status,
            active_label=self.primary_task,
            last_heartbeat=self._last_heartbeat,
            last_activity=self._last_activity,
            pending_messages=self.messenger.unread_count(),
        )
