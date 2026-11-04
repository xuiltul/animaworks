"""Anima-main-side lifecycle manager for disposable task runner processes."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import inspect
import json
import logging
import os
import signal
import sys
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import psutil

from core.i18n import t
from core.platform.env import ANIMAWORKS_ENV_PREFIX, CHROMA_THREAD_ENV_VARS, env_items_with_prefix
from core.platform.process import process_group_exists, signal_tree, snapshot_descendants, subprocess_session_kwargs
from core.platform.subprocess_entries import SubprocessEntry, module_args
from core.runtime.ipc_v2 import (
    IPC_V2_MAX_FRAME_BYTES,
    IPCV2Connection,
    IPCV2ConnectionError,
    IPCV2ConnectionState,
    IPCV2Identity,
    IPCV2ProtocolError,
    ipc_v2_error,
    read_ipc_v2_envelope,
)
from core.runtime.memory_service import MemoryService, MemoryServiceUnavailable
from core.runtime.state_writer import LocalStateWriter, get_state_writer
from core.runtime.transport import cleanup_ipc_endpoint, start_ipc_server
from core.schemas import CronTask
from core.time_utils import now_local

logger = logging.getLogger(__name__)

_TASK_RUNNER_CONNECT_TIMEOUT = 10.0
_RECEIVE_POLL_TIMEOUT_SEC = 15.0
# Post-result exit wait. Under load the child's interpreter shutdown (CLI
# grandchild reaping, thread joins, MCP cleanup) can exceed several seconds;
# a laggy exit must not turn a delivered result into a failure (2026-08-12).
_TASK_RUNNER_EXIT_TIMEOUT = 30.0
# close()-time grace deadline; kept short so a fleet drain is not serialized
# behind stuck children (13 animas x 2 waits would be minutes at 30s).
_TASK_RUNNER_GRACE_TIMEOUT = 5.0
_TASK_RUNNER_TERM_TIMEOUT = 5.0
_HANG_CHECK_INTERVAL_MAX = 5.0
# External (queue) cancel watch cadence; 30s so the queue file is not read on
# every hang-check tick (max 5s).  Patching this to 0 lets tests drive a check
# on the next loop iteration.
_CANCEL_CHECK_INTERVAL = 30.0
_STATE_WRITE_MAX_PAYLOAD_BYTES = 64 * 1024 * 1024

OnSpawned = Callable[["TaskRunnerJob"], Awaitable[None] | None]


class TaskRunnerError(RuntimeError):
    """A task runner could not complete its execution contract."""


class TaskRunnerCancelled(TaskRunnerError):
    """The runner was stopped because its task was cancelled in the queue."""


@dataclass
class _StateWriteAssembly:
    """One validated chunked state-write payload being received."""

    operation: str
    byte_length: int
    sha256: str
    chunks: list[bytes] = field(default_factory=list)
    next_index: int = 0
    received_bytes: int = 0


@dataclass
class TaskRunnerJob:
    """Anima-main registry entry created before a task runner is spawned."""

    identity: IPCV2Identity
    request_id: str
    params: dict[str, Any]
    result: asyncio.Future[dict[str, Any]]
    peer_state: IPCV2ConnectionState
    process: asyncio.subprocess.Process | None = None
    pid: int | None = None
    pgid: int | None = None
    connection: IPCV2Connection | None = None
    last_progress: dict[str, Any] = field(default_factory=dict)
    last_progress_at: float = 0.0
    hang_kill_started: bool = False
    external_cancelled: bool = False
    grace_acked: asyncio.Event = field(default_factory=asyncio.Event)
    process_start_time: float | None = None
    started_at: str = field(default_factory=lambda: now_local().isoformat())
    capabilities: dict[str, bool] = field(default_factory=dict)
    stream_events: list[asyncio.Queue[dict[str, Any] | None]] = field(default_factory=list)
    inject_waiters: dict[str, asyncio.Future[dict[str, Any]]] = field(default_factory=dict)
    interrupt_thread_id: str | None = None
    pending_cancel: bool = False
    state_write_count: int = 0
    state_write_duration_sec: float = 0.0
    state_write_pending: set[str] = field(default_factory=set)
    state_write_results: dict[str, dict[str, Any]] = field(default_factory=dict)


def _now_iso() -> str:
    return now_local().isoformat()


class TaskRunnerSupervisor:
    """Expose one IPC v2 endpoint and run jobs in isolated process groups."""

    def __init__(
        self,
        anima_name: str,
        anima_dir: Path,
        shared_dir: Path,
        *,
        max_concurrent: int | None = None,
        chat_max_concurrency: int = 1,
        runner_liveness_timeout_sec: float = 900.0,
        busy_status_owner: Any | None = None,
        memory_service: MemoryService | None = None,
    ) -> None:
        self.anima_name = anima_name
        self.anima_dir = anima_dir
        self.shared_dir = shared_dir
        self.root_epoch = str(uuid.uuid4())
        self.socket_path = shared_dir.parent / "run" / "sockets" / f"{anima_name}.task-v2.sock"
        self._server: asyncio.Server | None = None
        self._memory_service = memory_service
        self._state_writer = get_state_writer(anima_dir)
        self._start_lock = asyncio.Lock()
        self._jobs: dict[str, TaskRunnerJob] = {}
        self._journal_recovery_lock = asyncio.Lock()
        self._accepting = True
        self._runner_liveness_timeout_sec = max(0.0, float(runner_liveness_timeout_sec))
        self._hang_check_interval = min(
            _HANG_CHECK_INTERVAL_MAX,
            max(0.05, self._runner_liveness_timeout_sec / 2),
        )
        self._busy_status_owner = busy_status_owner
        set_provider = getattr(busy_status_owner, "_set_isolated_busy_jobs_provider", None)
        if callable(set_provider):
            set_provider(lambda: self._jobs)
        # Cap concurrent child processes for task/background lanes (pool size).
        pool = max_concurrent if isinstance(max_concurrent, int) and max_concurrent >= 1 else None
        self._max_concurrent = pool
        self._spawn_semaphore = asyncio.Semaphore(pool) if pool is not None else None
        chat_pool = (
            chat_max_concurrency
            if isinstance(chat_max_concurrency, int)
            and not isinstance(chat_max_concurrency, bool)
            and 1 <= chat_max_concurrency <= 10
            else 1
        )
        self._chat_semaphore = asyncio.Semaphore(chat_pool)
        self._chat_thread_locks: dict[str, asyncio.Lock] = {}
        self._chat_thread_lock_users: dict[str, int] = {}

    @property
    def _chat_lock(self) -> asyncio.Lock:
        """Compatibility accessor for the default-thread lock used by older tests."""
        return self._chat_thread_locks.get("default", asyncio.Lock())

    @asynccontextmanager
    async def _chat_slot(self, thread_id: str) -> AsyncIterator[None]:
        """Serialize one thread, then acquire a process-wide chat slot."""
        key = thread_id or "default"
        lock = self._chat_thread_locks.get(key)
        if lock is None:
            lock = asyncio.Lock()
            self._chat_thread_locks[key] = lock
            self._chat_thread_lock_users[key] = 0
        self._chat_thread_lock_users[key] += 1
        acquired = False
        try:
            await lock.acquire()
            acquired = True
            await self._chat_semaphore.acquire()
            try:
                yield
            finally:
                self._chat_semaphore.release()
        finally:
            if acquired:
                lock.release()
            users = self._chat_thread_lock_users[key] - 1
            if users == 0:
                self._chat_thread_lock_users.pop(key, None)
                if self._chat_thread_locks.get(key) is lock:
                    self._chat_thread_locks.pop(key, None)
            else:
                self._chat_thread_lock_users[key] = users

    def _queue_live_facts(self, trigger: str, started_at: str) -> None:
        """Schedule live fact extraction in this long-lived process.

        Session hooks inside a disposable task-runner child die with the child
        before the debounce fires, so the parent schedules after the job ends.
        """
        try:
            from core.memory.facts.live import schedule_live_fact_extraction

            schedule_live_fact_extraction(self.anima_dir, trigger=trigger, session_started_at=started_at)
        except Exception as exc:  # noqa: BLE001 - never affect the finished job
            logger.warning(
                "[%s] Live fact extraction was not scheduled: %s",
                self.anima_name,
                f"{type(exc).__name__}: {exc}".replace("\n", " ")[:240],
            )

    @property
    def jobs(self) -> dict[str, TaskRunnerJob]:
        """Return the live registry for health diagnostics and tests."""
        return self._jobs

    async def start(self) -> None:
        """Start the Anima main IPC endpoint and its optional memory service."""
        await self._ensure_started()

    async def _ensure_started(self) -> None:
        if self._server is not None:
            return
        async with self._start_lock:
            if self._server is None:
                if self._memory_service is not None:
                    await self._memory_service.start()
                self._server, endpoint = await start_ipc_server(
                    self.socket_path,
                    self._handle_connection,
                    limit=IPC_V2_MAX_FRAME_BYTES + 1,
                )
                logger.info("Task runner IPC v2 endpoint started on %s", endpoint.describe())

    @staticmethod
    def _required_url_environment() -> dict[str, str]:
        """Copy model-service URLs explicitly, failing closed when absent."""
        values = {
            name: value.strip()
            for name, value in env_items_with_prefix(ANIMAWORKS_ENV_PREFIX, suffix="_URL").items()
            if value.strip()
        }
        if "ANIMAWORKS_EMBED_URL" not in values:
            raise TaskRunnerError("required task runner URL is missing: ANIMAWORKS_EMBED_URL")
        for name in ("ANIMAWORKS_VECTOR_URL", "ANIMAWORKS_RERANK_URL"):
            if name not in values:
                raise TaskRunnerError(f"required task runner URL is missing: {name}")
        return values

    def _build_child_environment(
        self,
        url_env: dict[str, str],
        *,
        attempt: int,
        display_lane: str,
    ) -> dict[str, str]:
        """Produce the task runner child's environment.

        Phase3 children reach the vector store owner over HTTP via
        ``ANIMAWORKS_VECTOR_URL`` (propagated through ``url_env``); only the
        the Anima main itself uses its in-process owner transport.
        """
        env = os.environ.copy()
        for name in tuple(env):
            if name.startswith(ANIMAWORKS_ENV_PREFIX) and name.endswith("_URL"):
                env.pop(name)
        # These bound this process's chroma.  Task runners reach the vector store
        # over HTTP, and the tools they run should keep their own defaults.
        for name in CHROMA_THREAD_ENV_VARS:
            env.pop(name, None)
        env.update(url_env)
        env.update(
            {
                "ANIMAWORKS_DATA_DIR": str(self.shared_dir.parent),
                "ANIMAWORKS_ANIMA_NAME": self.anima_name,
                "ANIMAWORKS_ANIMA_DIR": str(self.anima_dir),
                "ANIMAWORKS_TASK_IPC_PATH": str(self.socket_path),
                "ANIMAWORKS_TASK_ROOT_EPOCH": self.root_epoch,
                "ANIMAWORKS_TASK_ATTEMPT": str(attempt),
                "ANIMAWORKS_TASK_DISPLAY_LANE": display_lane,
                "ANIMAWORKS_TASK_ROOT_PID": str(os.getpid()),
            }
        )
        return env

    def build_cron_command_environment(self) -> dict[str, str]:
        """Build the per-Anima environment formerly inherited from a cron runner."""
        url_env = {
            name: value.strip()
            for name, value in env_items_with_prefix(ANIMAWORKS_ENV_PREFIX, suffix="_URL").items()
            if value.strip()
        }
        return self._build_child_environment(url_env, attempt=1, display_lane="background")

    async def run_cron(self, task: CronTask) -> dict[str, Any]:
        """Spawn one cron task runner and return its terminal result."""
        return await self._run_isolated_job(
            lane="cron",
            job_prefix="cron",
            params_builder=lambda url_env: {
                "task": task.model_dump(mode="json"),
                "environment": {"urls": url_env},
            },
            log_context=f"task={task.name}",
        )

    async def run_cron_followup(self, task: CronTask, command_output: str) -> dict[str, Any]:
        """Run only a command cron's LLM follow-up in an isolated runner."""
        return await self._run_isolated_job(
            lane="cron",
            job_prefix="cron-followup",
            params_builder=lambda url_env: {
                "task": task.model_dump(mode="json"),
                "command_output": command_output,
                "environment": {"urls": url_env},
            },
            log_context=f"task={task.name} follow-up",
        )

    async def run_heartbeat(self) -> dict[str, Any]:
        """Spawn one heartbeat task runner and return its terminal result."""
        return await self._run_isolated_job(
            lane="heartbeat",
            job_prefix="heartbeat",
            params_builder=lambda url_env: {"environment": {"urls": url_env}},
            log_context="heartbeat",
        )

    async def run_inbox(self) -> dict[str, Any]:
        """Spawn one inbox (inter-Anima message) task runner and return its terminal result.

        Uses a dedicated ``inbox`` lane rather than reusing the ``cron`` lane so
        that journal recovery (lane name == session type) picks up the inbox
        journal (``session_type="inbox"``).
        """
        started_at = _now_iso()
        result = await self._run_isolated_job(
            lane="inbox",
            job_prefix="inbox",
            params_builder=lambda url_env: {"environment": {"urls": url_env}},
            log_context="inbox",
            display_lane="inbox",
        )
        self._queue_live_facts("inbox", started_at)
        return result

    async def run_task(
        self,
        task_desc: dict[str, Any],
        *,
        attempt: int = 1,
        display_lane: str = "background",
        on_spawned: OnSpawned | None = None,
    ) -> dict[str, Any]:
        """Spawn one TaskExec (lane=task) runner and return its terminal result."""
        task_id = str(task_desc.get("task_id") or "unknown")
        started_at = _now_iso()
        result = await self._run_isolated_job(
            lane="task",
            job_prefix="task",
            params_builder=lambda url_env: {
                "task_desc": task_desc,
                "environment": {"urls": url_env},
            },
            log_context=f"task_id={task_id}",
            attempt=attempt,
            display_lane=display_lane,
            on_spawned=on_spawned,
            use_pool_limit=True,
        )
        self._queue_live_facts("task", started_at)
        return result

    async def run_background(
        self,
        *,
        kind: str,
        payload: dict[str, Any],
        attempt: int = 1,
        display_lane: str = "background",
        on_spawned: OnSpawned | None = None,
    ) -> dict[str, Any]:
        """Spawn one background-lane runner (command tool or consolidation)."""
        return await self._run_isolated_job(
            lane="background",
            job_prefix=f"bg-{kind}",
            params_builder=lambda url_env: {
                "kind": kind,
                "payload": payload,
                "environment": {"urls": url_env},
            },
            log_context=f"kind={kind}",
            attempt=attempt,
            display_lane=display_lane,
            on_spawned=on_spawned,
            use_pool_limit=True,
        )

    async def run_chat(self, *, kind: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Run a chat contract, steering an active Mode S stream when possible."""
        active = None
        if kind == "message":
            found, active, _ = await self._try_inject_active_chat(payload)
            if found and active is not None:
                return self._terminal_result(await asyncio.shield(active.result))
            if found:
                await self._interrupt_active_chat(str(payload.get("thread_id") or "default"))

        thread_id = str(payload.get("thread_id") or "default")
        async with self._chat_slot(thread_id):
            started_at = _now_iso()
            result = await self._run_isolated_job(
                lane="chat",
                job_prefix=f"chat-{kind}",
                params_builder=lambda url_env: {
                    "kind": kind,
                    "payload": payload,
                    "environment": {"urls": url_env},
                },
                log_context=f"kind={kind}",
                display_lane="chat",
            )
            self._queue_live_facts("chat", started_at)
            return result

    async def run_chat_stream(self, payload: dict[str, Any]) -> AsyncIterator[dict[str, Any]]:
        """Run or steer one streaming chat and relay its IPC v2 events."""
        thread_id = str(payload.get("thread_id") or "default")
        found, active, queue = await self._try_inject_active_chat(payload, subscribe=True)
        if found and active is not None and queue is not None:
            try:
                while (event := await queue.get()) is not None:
                    yield event
                yield {"done": True, "result": self._terminal_result(await asyncio.shield(active.result))}
            finally:
                if queue in active.stream_events:
                    active.stream_events.remove(queue)
            return
        if found:
            await self._interrupt_active_chat(thread_id)

        async with self._chat_slot(thread_id):
            queue = asyncio.Queue()

            async def _produce() -> dict[str, Any]:
                try:
                    return await self._run_isolated_job(
                        lane="chat",
                        job_prefix="chat-stream",
                        params_builder=lambda url_env: {
                            "kind": "message",
                            "payload": {**payload, "stream": True},
                            "environment": {"urls": url_env},
                        },
                        log_context=f"thread={thread_id}",
                        display_lane="chat",
                        stream_events=[queue],
                    )
                finally:
                    await queue.put(None)

            started_at = _now_iso()
            producer = asyncio.create_task(_produce())
            try:
                while (event := await queue.get()) is not None:
                    yield event
                result = await producer
                self._queue_live_facts("chat", started_at)
                yield {"done": True, "result": result}
            finally:
                if not producer.done():
                    producer.cancel()
                await asyncio.gather(producer, return_exceptions=True)

    async def _try_inject_active_chat(
        self,
        payload: dict[str, Any],
        *,
        subscribe: bool = False,
    ) -> tuple[bool, TaskRunnerJob | None, asyncio.Queue[dict[str, Any] | None] | None]:
        """Attempt one inject_message event; report whether an active chat existed."""
        thread_id = str(payload.get("thread_id") or "default")
        active = next(
            (
                job
                for job in self._jobs.values()
                if job.identity.lane == "chat"
                and not job.result.done()
                and str((job.params.get("payload") or {}).get("thread_id") or "default") == thread_id
            ),
            None,
        )
        if active is None:
            return False, None, None
        active_payload = active.params.get("payload") or {}
        if (
            not active_payload.get("stream")
            or str(active_payload.get("thread_id") or "default") != thread_id
            or not active.capabilities.get("steer", False)
            or active.connection is None
            or payload.get("images")
            or payload.get("attachment_paths")
        ):
            return True, None, None

        injection_id = f"inject-{uuid.uuid4()}"
        waiter = asyncio.get_running_loop().create_future()
        active.inject_waiters[injection_id] = waiter
        queue: asyncio.Queue[dict[str, Any] | None] | None = asyncio.Queue() if subscribe else None
        if queue is not None:
            active.stream_events.append(queue)
        try:
            await active.connection.send_event(
                "inject_message",
                {
                    "injection_id": injection_id,
                    "message": str(payload.get("message") or ""),
                    "from_person": str(payload.get("from_person") or "human"),
                    "thread_id": thread_id,
                    "attachment_paths": payload.get("attachment_paths") or [],
                },
            )
            result = await asyncio.wait_for(asyncio.shield(waiter), timeout=5.0)
            if result.get("accepted") is True:
                return True, active, queue
            logger.info(
                "Chat steer declined for job=%s: %s", active.identity.job_id, result.get("error") or "unsupported"
            )
        except Exception as exc:
            logger.warning("Chat steer failed for job=%s: %s", active.identity.job_id, exc)
        finally:
            active.inject_waiters.pop(injection_id, None)
        if queue is not None and queue in active.stream_events:
            active.stream_events.remove(queue)
        return True, None, None

    async def _interrupt_active_chat(self, thread_id: str) -> None:
        """Match legacy auto-interrupt-before-lock behavior for a queued turn."""
        for job in list(self._jobs.values()):
            if job.identity.lane != "chat" or job.result.done():
                continue
            active_payload = job.params.get("payload") or {}
            if str(active_payload.get("thread_id") or "default") != (thread_id or "default"):
                continue
            if job.connection is None:
                job.interrupt_thread_id = thread_id
            else:
                await job.connection.send_event("interrupt", {"thread_id": thread_id})

    async def cancel_consolidation(self) -> dict[str, Any]:
        """Best-effort cancel any running consolidation (background) job.

        Sends a ``cancel`` event to the live child; if no job is running (or
        it already finished) this is a no-op.
        """
        for job in list(self._jobs.values()):
            if job.identity.lane != "background":
                continue
            if job.result.done():
                continue
            if job.connection is None:
                job.pending_cancel = True
            else:
                try:
                    await job.connection.send_event("cancel", {})
                except Exception:
                    logger.debug(
                        "Cancel event send failed anima=%s job=%s",
                        self.anima_name,
                        job.identity.job_id,
                        exc_info=True,
                    )
            return {"status": "cancel_requested", "job": job.identity.job_id}
        return {"status": "noop"}

    async def interrupt_chat(self, thread_id: str | None = None) -> dict[str, Any]:
        """Forward an explicit Anima-main interrupt to active chat children."""
        interrupted = False
        for job in list(self._jobs.values()):
            if job.identity.lane != "chat" or job.result.done():
                continue
            if thread_id:
                active_payload = job.params.get("payload") or {}
                if str(active_payload.get("thread_id") or "default") != thread_id:
                    continue
            interrupted = True
            if job.connection is None:
                job.interrupt_thread_id = thread_id or "default"
            else:
                await job.connection.send_event("interrupt", {"thread_id": thread_id or ""})
        return {"status": "interrupted" if interrupted else "idle", "name": self.anima_name}

    @staticmethod
    def _terminal_result(terminal: dict[str, Any]) -> dict[str, Any]:
        if "error" in terminal:
            error = terminal["error"]
            raise TaskRunnerError(f"{error.get('code', 'EXECUTION_ERROR')}: {error.get('message', '')}")
        result = terminal.get("result")
        if not isinstance(result, dict):
            raise TaskRunnerError("task runner returned a malformed result")
        return result

    async def _run_isolated_job(
        self,
        *,
        lane: str,
        job_prefix: str,
        params_builder: Callable[[dict[str, str]], dict[str, Any]],
        log_context: str,
        attempt: int = 1,
        display_lane: str = "background",
        on_spawned: OnSpawned | None = None,
        use_pool_limit: bool = False,
        stream_events: list[asyncio.Queue[dict[str, Any] | None]] | None = None,
    ) -> dict[str, Any]:
        """Spawn one task runner process and return its terminal result."""
        if not self._accepting:
            raise TaskRunnerError("task runner supervisor is shutting down")
        await self._ensure_started()
        url_env = self._required_url_environment()

        if attempt < 1:
            raise TaskRunnerError("attempt must be >= 1")

        sem = self._spawn_semaphore if use_pool_limit else None
        if sem is not None:
            await sem.acquire()
        try:
            return await self._spawn_and_await(
                lane=lane,
                job_prefix=job_prefix,
                params_builder=params_builder,
                log_context=log_context,
                attempt=attempt,
                display_lane=display_lane,
                on_spawned=on_spawned,
                url_env=url_env,
                stream_events=stream_events,
            )
        finally:
            if sem is not None:
                sem.release()

    async def _spawn_and_await(
        self,
        *,
        lane: str,
        job_prefix: str,
        params_builder: Callable[[dict[str, str]], dict[str, Any]],
        log_context: str,
        attempt: int,
        display_lane: str,
        on_spawned: OnSpawned | None,
        url_env: dict[str, str],
        stream_events: list[asyncio.Queue[dict[str, Any] | None]] | None = None,
    ) -> dict[str, Any]:
        job_id = f"{job_prefix}-{uuid.uuid4()}"
        request_id = f"run-{uuid.uuid4()}"
        identity = IPCV2Identity(
            job_id=job_id,
            root_epoch=self.root_epoch,
            attempt=attempt,
            lane=lane,
            display_lane=display_lane,
        )
        loop = asyncio.get_running_loop()
        job = TaskRunnerJob(
            identity=identity,
            request_id=request_id,
            params=params_builder(url_env),
            result=loop.create_future(),
            peer_state=IPCV2ConnectionState(identity),
            last_progress_at=loop.time(),
            stream_events=list(stream_events or []),
        )
        # A recovery worker must finish unlinking old journals before a new
        # child can open a journal in the same runtime.
        async with self._journal_recovery_lock:
            self._jobs[job_id] = job

        env = self._build_child_environment(
            url_env,
            attempt=attempt,
            display_lane=display_lane,
        )

        hang_watch: asyncio.Task[None] | None = None
        stderr_file = None
        try:
            stderr_path = self.shared_dir.parent / "logs" / "animas" / self.anima_name / "task-runner.stderr.log"
            stderr_path.parent.mkdir(parents=True, exist_ok=True)
            stderr_file = stderr_path.open("ab")
            process = await asyncio.create_subprocess_exec(
                sys.executable,
                *module_args(SubprocessEntry.TASK_RUNNER),
                "--anima",
                self.anima_name,
                "--lane",
                lane,
                "--job",
                job_id,
                env=env,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=stderr_file,
                **subprocess_session_kwargs(),
            )
            job.process = process
            job.pid = process.pid
            job.pgid = process.pid
            self._mark_busy_start()
            try:
                import psutil

                job.process_start_time = float(psutil.Process(process.pid).create_time())
            except Exception:
                job.process_start_time = None
            logger.info(
                "Spawned %s task runner anima=%s %s job=%s pid=%s pgid=%s attempt=%s",
                lane,
                self.anima_name,
                log_context,
                job_id,
                job.pid,
                job.pgid,
                attempt,
            )
            if on_spawned is not None:
                maybe = on_spawned(job)
                if inspect.isawaitable(maybe):
                    await maybe

            process_wait = asyncio.create_task(process.wait())
            hang_watch = asyncio.create_task(self._watch_job(job))
            done, _ = await asyncio.wait(
                {job.result, process_wait},
                return_when=asyncio.FIRST_COMPLETED,
            )
            if process_wait in done and not job.result.done():
                try:
                    await asyncio.wait_for(asyncio.shield(job.result), timeout=0.25)
                except TimeoutError as exc:
                    if job.external_cancelled:
                        raise TaskRunnerCancelled(
                            f"task runner stopped because the task was cancelled (exit={process.returncode})"
                        ) from exc
                    raise TaskRunnerError(
                        f"task runner exited before returning a result (exit={process.returncode})"
                    ) from exc
            terminal = await job.result
            slow_exit = False
            if not process_wait.done():
                try:
                    await asyncio.wait_for(process_wait, timeout=_TASK_RUNNER_EXIT_TIMEOUT)
                except TimeoutError:
                    # The child already delivered its terminal result; a laggy exit
                    # must not fail a completed task. Kill the group, keep the result.
                    self._terminate_job_group(job)
                    try:
                        await asyncio.wait_for(process.wait(), timeout=2.0)
                    except TimeoutError:
                        self._kill_job_group(job)
                        await process.wait()
                    slow_exit = True
                    logger.warning(
                        "Task runner delivered a result but did not exit within %.0fs; "
                        "killed group and kept the result anima=%s job=%s pid=%s",
                        _TASK_RUNNER_EXIT_TIMEOUT,
                        self.anima_name,
                        job_id,
                        job.pid,
                    )
            result = self._terminal_result(terminal)
            if not slow_exit and process.returncode != 0:
                # The result was already delivered before the child exited; a
                # nonzero exit (e.g. a laggy EPIPE during shutdown) must not
                # discard an executed task's result.
                logger.warning(
                    "result kept despite nonzero exit anima=%s job=%s task_id=%s returncode=%s",
                    self.anima_name,
                    job_id,
                    (job.params.get("task_desc") or {}).get("task_id"),
                    process.returncode,
                )
            return result
        except BaseException:
            # Failure in IPC/on_spawned is as capable of leaving a live child
            # as cancellation. Reap the exact recorded group before releasing
            # ownership; escalate after the existing bounded TERM grace.
            if job.process is not None and job.process.returncode is None:
                await self._terminate_hung_job(job)
            raise
        finally:
            if stderr_file is not None:
                stderr_file.close()
            if hang_watch is not None:
                if not job.hang_kill_started:
                    hang_watch.cancel()
                await asyncio.gather(hang_watch, return_exceptions=True)
            self._jobs.pop(job_id, None)
            if job.state_write_count:
                logger.info(
                    "Task runner state_write summary: anima=%s lane=%s job=%s operations=%d write_ms=%.1f",
                    self.anima_name,
                    job.identity.lane,
                    job.identity.job_id,
                    job.state_write_count,
                    job.state_write_duration_sec * 1000,
                )
            for queue in job.stream_events:
                queue.put_nowait(None)
            for waiter in job.inject_waiters.values():
                if not waiter.done():
                    waiter.set_result({"accepted": False, "error": "chat job ended"})
            self._clear_busy_if_idle()
            # A-07: recover orphan streaming journals after every task ends.
            await self._recover_task_journals((lane,))

    async def _watch_job(self, job: TaskRunnerJob) -> None:
        """Kill only this task-runner group after progress stops.

        Besides hang detection, a ``task``-lane job whose queue entry was
        externally cancelled (e.g. superseded by a newer PR exact) is terminated
        here.  The queue cancel check is rate-limited to ``_CANCEL_CHECK_INTERVAL``
        seconds to avoid reading the queue file on every hang-check tick.
        """
        last_cancel_check = 0.0
        while job.identity.job_id in self._jobs:
            await asyncio.sleep(self._hang_check_interval)
            process = job.process
            if process is None or process.returncode is not None:
                return
            now = asyncio.get_running_loop().time()
            if job.identity.lane == "task" and now - last_cancel_check >= _CANCEL_CHECK_INTERVAL:
                last_cancel_check = now
                if await self._task_is_externally_cancelled(job):
                    job.hang_kill_started = True
                    job.external_cancelled = True
                    logger.info(
                        "Task runner external cancel: anima=%s job=%s task_id=%s",
                        self.anima_name,
                        job.identity.job_id,
                        (job.params.get("task_desc") or {}).get("task_id"),
                    )
                    await self._terminate_hung_job(job)
                    return
            idle_sec = now - job.last_progress_at
            if idle_sec <= self._runner_liveness_timeout_sec:
                continue
            job.hang_kill_started = True
            logger.error(
                "Task runner hang: anima=%s job=%s lane=%s pid=%s pgid=%s idle=%.1fs threshold=%.1fs",
                self.anima_name,
                job.identity.job_id,
                job.identity.lane,
                job.pid,
                job.pgid,
                idle_sec,
                self._runner_liveness_timeout_sec,
            )
            await self._terminate_hung_job(job)
            return

    async def _task_is_externally_cancelled(self, job: TaskRunnerJob) -> bool:
        """Return True when the task's queue entry was externally cancelled.

        Reads the live task queue (via a thread so the event loop is not blocked)
        and returns True only for a ``cancelled`` status.  Any queue read error
        is swallowed and treated as not-cancelled so the watcher keeps going.
        """
        try:
            task_desc = job.params.get("task_desc") or {}
            task_id = task_desc.get("task_id")
            if not task_id:
                return False
            from core.tasks.queue import TaskQueueManager

            entry = await asyncio.to_thread(
                TaskQueueManager(self.anima_dir, read_only=True).get_task_by_id,
                task_id,
            )
            return entry is not None and entry.status == "cancelled"
        except Exception:
            logger.debug(
                "Task runner external-cancel check failed for job=%s",
                job.identity.job_id,
                exc_info=True,
            )
            return False

    async def _terminate_hung_job(self, job: TaskRunnerJob) -> None:
        """Terminate a hung task group, escalating to SIGKILL after grace."""
        process = job.process
        if process is None:
            return
        self._terminate_job_group(job)
        try:
            await asyncio.wait_for(asyncio.shield(process.wait()), timeout=_TASK_RUNNER_TERM_TIMEOUT)
        except TimeoutError:
            pass
        if self._job_group_exists(job):
            self._kill_job_group(job)
        if process.returncode is None:
            await process.wait()

    @staticmethod
    def _job_group_exists(job: TaskRunnerJob) -> bool:
        return process_group_exists(
            job.pgid,
            fallback_alive=job.process is not None and job.process.returncode is None,
        )

    def _mark_busy_start(self) -> None:
        owner = self._busy_status_owner
        if owner is None:
            return
        callback = getattr(owner, "_mark_busy_start" if len(self._jobs) == 1 else "_mark_busy_progress", None)
        if callable(callback):
            callback()

    def _mark_busy_progress(self) -> None:
        callback = getattr(self._busy_status_owner, "_mark_busy_progress", None)
        if callable(callback):
            callback()

    def _record_progress(self, job: TaskRunnerJob, data: dict[str, Any]) -> None:
        job.last_progress = data
        job.last_progress_at = asyncio.get_running_loop().time()
        self._mark_busy_progress()

    def _clear_busy_if_idle(self) -> None:
        callback = getattr(self._busy_status_owner, "_clear_busy_status_sidecar_if_idle", None)
        if callable(callback):
            callback()

    async def _recover_task_journals(
        self,
        session_types: tuple[str, ...] = ("task", "heartbeat", "chat", "cron", "inbox"),
    ) -> None:
        """Best-effort orphan StreamingJournal recovery after a child exits.

        All disk work runs in a thread so a saturated disk can never stall the
        Anima main event loop (which would delay every child's ack).
        """
        # Snapshot event-loop-owned inputs before handing disk work to a thread.
        owner = self._busy_status_owner
        model_config = getattr(owner, "model_config", None)

        def _disk_recovery() -> list[dict[str, Any]]:
            outcomes: list[dict[str, Any]] = []
            try:
                from core.memory.conversation.memory import ConversationMemory
                from core.memory.conversation.streaming_journal import StreamingJournal
            except Exception:
                return outcomes
            for session_type in session_types:
                try:
                    if not StreamingJournal.has_orphan(self.anima_dir, session_type=session_type):
                        continue
                    thread_ids = StreamingJournal.list_orphan_thread_ids(self.anima_dir, session_type)
                    for thread_id in thread_ids or ("default",):
                        recovery = StreamingJournal.recover(
                            self.anima_dir,
                            session_type,
                            thread_id=thread_id,
                        )
                        if recovery is None:
                            continue
                        if session_type == "chat" and recovery.recovered_text:
                            if model_config is None:
                                logger.error("Cannot persist recovered chat journal without Anima main model config")
                            else:
                                conversation = ConversationMemory(
                                    self.anima_dir,
                                    model_config,
                                    thread_id=thread_id,
                                )
                                marker = t("anima.response_interrupted")
                                saved_text = recovery.recovered_text + "\n" + marker
                                last_assistant = next(
                                    (turn for turn in reversed(conversation.load().turns) if turn.role == "assistant"),
                                    None,
                                )
                                already_saved = last_assistant is not None and (
                                    last_assistant.content == saved_text
                                    or recovery.recovered_text.strip()
                                    in last_assistant.content.replace(marker, "").strip()
                                )
                                if not already_saved:
                                    conversation.append_turn("assistant", saved_text)
                                    conversation.save()
                        StreamingJournal.confirm_recovery(
                            self.anima_dir,
                            session_type,
                            thread_id=thread_id,
                        )
                        logger.info(
                            "Recovered orphan journal after task exit: anima=%s session=%s thread=%s",
                            self.anima_name,
                            session_type,
                            thread_id,
                        )
                        outcomes.append(
                            {
                                "session_type": session_type,
                                "thread_id": thread_id,
                                "recovered_text": recovery.recovered_text,
                                "tool_calls": recovery.tool_calls,
                            }
                        )
                except Exception:
                    logger.debug(
                        "Orphan journal recovery failed for %s/%s",
                        self.anima_name,
                        session_type,
                        exc_info=True,
                    )
            return outcomes

        async with self._journal_recovery_lock:
            # File existence does not mean orphaned: a sibling child may still
            # be writing. Defer the entire lane until its final child exits.
            active_lanes = {job.identity.lane for job in self._jobs.values()}
            session_types = tuple(lane for lane in session_types if lane not in active_lanes)
            if not session_types:
                return
            recovery_task = asyncio.create_task(asyncio.to_thread(_disk_recovery))
            try:
                await asyncio.shield(recovery_task)
            except asyncio.CancelledError:
                # Cancelling to_thread does not stop its disk work. Retain the
                # registration lock until that work has really finished.
                await recovery_task
                raise

    async def _state_write_worker(
        self,
        connection: IPCV2Connection,
        job: TaskRunnerJob,
        queue: asyncio.Queue[tuple[str, str, dict[str, Any]] | None],
    ) -> None:
        """Apply queued writes in order without blocking the frame receiver."""
        while (item := await queue.get()) is not None:
            request_id, operation, payload = item
            started = asyncio.get_running_loop().time()
            try:
                writer = self._state_writer
                if not isinstance(writer, LocalStateWriter):
                    raise TaskRunnerError("Anima main state writer is not local")
                result = await writer.execute_operation(operation, payload)
                response = {"result": result}
            except Exception as exc:
                logger.exception(
                    "Task state write failed: anima=%s job=%s operation=%s",
                    self.anima_name,
                    job.identity.job_id,
                    operation,
                )
                response = {
                    "error": ipc_v2_error("STATE_WRITE_ERROR", str(exc), retryable=False),
                }
            finally:
                job.state_write_duration_sec += asyncio.get_running_loop().time() - started
                job.state_write_pending.discard(request_id)
            job.state_write_results[request_id] = response
            reply_connection = job.connection or connection
            try:
                if "error" in response:
                    await reply_connection.send_response(request_id, error=response["error"])
                else:
                    await reply_connection.send_response(request_id, result=response["result"])
            except Exception:
                # The disk operation is already durable. A task runner that
                # disappeared while it was in flight cannot undo the write.
                logger.debug(
                    "Could not return state-write response: anima=%s job=%s request=%s",
                    self.anima_name,
                    job.identity.job_id,
                    request_id,
                    exc_info=True,
                )

    async def _accept_state_write(
        self,
        connection: IPCV2Connection,
        job: TaskRunnerJob,
        queue: asyncio.Queue[tuple[str, str, dict[str, Any]] | None],
        assemblies: dict[str, _StateWriteAssembly],
        request: Any,
    ) -> None:
        """Validate a state_write frame and enqueue only complete operations."""
        request_id = request.body["request_id"]
        if request_id in job.state_write_results:
            response = job.state_write_results[request_id]
            if "error" in response:
                await connection.send_response(request_id, error=response["error"])
            else:
                await connection.send_response(request_id, result=response["result"])
            return
        if request_id in job.state_write_pending:
            return
        params = request.body.get("params") or {}
        phase = params.get("phase", "write")
        try:
            if phase == "write":
                operation = params.get("operation")
                payload = params.get("payload")
                if not isinstance(operation, str) or not operation or not isinstance(payload, dict):
                    raise ValueError("state_write requires operation and payload object")
                self._queue_state_write(job, queue, request_id, operation, payload)
                return

            transaction_id = params.get("transaction_id")
            if not isinstance(transaction_id, str) or not transaction_id or len(transaction_id) > 80:
                raise ValueError("state_write transaction_id is invalid")
            if phase == "begin":
                operation = params.get("operation")
                byte_length = params.get("byte_length")
                digest = params.get("sha256")
                if (
                    not isinstance(operation, str)
                    or not operation
                    or not isinstance(byte_length, int)
                    or isinstance(byte_length, bool)
                    or byte_length < 1
                    or byte_length > _STATE_WRITE_MAX_PAYLOAD_BYTES
                    or not isinstance(digest, str)
                    or len(digest) != 64
                ):
                    raise ValueError("state_write begin metadata is invalid")
                if transaction_id in assemblies:
                    raise ValueError("duplicate state_write transaction_id")
                assemblies[transaction_id] = _StateWriteAssembly(operation, byte_length, digest)
                return

            if phase == "chunk":
                assembly = assemblies.get(transaction_id)
                index = params.get("index")
                encoded = params.get("data")
                if assembly is None or not isinstance(index, int) or isinstance(index, bool):
                    raise ValueError("state_write chunk has no matching begin")
                if index != assembly.next_index or not isinstance(encoded, str):
                    raise ValueError("state_write chunk index is out of order")
                chunk = base64.b64decode(encoded, validate=True)
                if assembly.received_bytes + len(chunk) > assembly.byte_length:
                    raise ValueError("state_write chunks exceed declared byte length")
                assembly.chunks.append(chunk)
                assembly.received_bytes += len(chunk)
                assembly.next_index += 1
                return

            if phase == "commit":
                assembly = assemblies.pop(transaction_id, None)
                if assembly is None or assembly.received_bytes != assembly.byte_length:
                    raise ValueError("state_write commit has an incomplete payload")
                encoded_payload = b"".join(assembly.chunks)
                if hashlib.sha256(encoded_payload).hexdigest() != assembly.sha256:
                    raise ValueError("state_write payload checksum mismatch")
                decoded = json.loads(encoded_payload.decode("utf-8"))
                if not isinstance(decoded, dict) or decoded.get("operation") != assembly.operation:
                    raise ValueError("state_write payload does not match begin metadata")
                payload = decoded.get("payload")
                if not isinstance(payload, dict):
                    raise ValueError("state_write payload must be an object")
                self._queue_state_write(job, queue, request_id, assembly.operation, payload)
                return

            raise ValueError(f"unsupported state_write phase: {phase!r}")
        except Exception as exc:
            await connection.send_response(
                request_id,
                error=ipc_v2_error("STATE_WRITE_PROTOCOL_ERROR", str(exc), retryable=False),
            )

    @staticmethod
    def _queue_state_write(
        job: TaskRunnerJob,
        queue: asyncio.Queue[tuple[str, str, dict[str, Any]] | None],
        request_id: str,
        operation: str,
        payload: dict[str, Any],
    ) -> None:
        """Deduplicate application request IDs before ordered disk dispatch."""
        if request_id in job.state_write_results or request_id in job.state_write_pending:
            return
        job.state_write_count += 1
        job.state_write_pending.add(request_id)
        queue.put_nowait((request_id, operation, payload))

    async def _handle_connection(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        connection: IPCV2Connection | None = None
        job: TaskRunnerJob | None = None
        state_write_queue: asyncio.Queue[tuple[str, str, dict[str, Any]] | None] | None = None
        state_write_worker: asyncio.Task[None] | None = None
        assemblies: dict[str, _StateWriteAssembly] = {}
        try:
            first = await asyncio.wait_for(
                read_ipc_v2_envelope(reader),
                timeout=_TASK_RUNNER_CONNECT_TIMEOUT,
            )
            job = self._jobs.get(first.identity.job_id)
            if job is None:
                temporary = IPCV2ConnectionState(first.identity)
                connection = IPCV2Connection(reader, writer, temporary)
                await connection.send_event(
                    "protocol_error",
                    ipc_v2_error("UNKNOWN_JOB", "job is not registered", retryable=False),
                )
                return
            connection = IPCV2Connection(reader, writer, job.peer_state)
            accepted = await connection.accept_first(first)
            if accepted is None or accepted.kind != "event" or accepted.body.get("event") != "hello":
                raise IPCV2ProtocolError("the first task runner frame must be hello")
            job.connection = connection
            hello_data = accepted.body["data"]
            capabilities = hello_data.get("capabilities") or {}
            if not isinstance(capabilities, dict):
                raise IPCV2ProtocolError("hello.capabilities must be an object")
            job.capabilities = {
                "reconnect": capabilities.get("reconnect") is True,
                "steer": capabilities.get("steer") is True,
            }
            last_received = hello_data.get("last_received_seq", 0)
            if not isinstance(last_received, int) or isinstance(last_received, bool) or last_received < 0:
                raise IPCV2ProtocolError("hello.last_received_seq must be a non-negative integer")
            replay_through = job.peer_state.send_seq
            await connection.send_event(
                "hello_ack",
                {
                    "capabilities": {"reconnect": True, "steer": True},
                    "last_received_seq": job.peer_state.received_seq,
                    "last_acked_seq": job.peer_state.last_acked_seq,
                },
            )
            await connection.replay_after(last_received, through_seq=replay_through)
            state_write_queue = asyncio.Queue()
            state_write_worker = asyncio.create_task(
                self._state_write_worker(connection, job, state_write_queue),
                name=f"state-write-{job.identity.job_id}",
            )
            await connection.send_request(job.request_id, "run", job.params)
            if job.interrupt_thread_id is not None:
                await connection.send_event("interrupt", {"thread_id": job.interrupt_thread_id})
            if job.pending_cancel:
                await connection.send_event("cancel", {})
                job.pending_cancel = False

            while True:
                try:
                    envelope = await asyncio.wait_for(connection.receive(), timeout=_RECEIVE_POLL_TIMEOUT_SEC)
                except TimeoutError:
                    # A quiet interval is not an error: runner liveness is judged
                    # by the liveness watchdog (runner_liveness_timeout), not this socket.
                    # Severing here used to permanently mute healthy runners and
                    # produce mass false hang-kills after Anima-main event-loop stalls.
                    current = self._jobs.get(job.identity.job_id)
                    if current is None or current.connection is not connection:
                        break  # superseded by a reconnect, or the job is gone
                    continue
                if envelope.kind == "request":
                    if envelope.body["method"] == "state_write" and state_write_queue is not None:
                        await self._accept_state_write(connection, job, state_write_queue, assemblies, envelope)
                    else:
                        # The state-write operation is the only task-runner
                        # request accepted on this channel.
                        await connection.send_response(
                            envelope.body["request_id"],
                            error=ipc_v2_error("PROTOCOL_ERROR", "unsupported task request", retryable=False),
                        )
                    continue
                if envelope.kind == "response":
                    if envelope.body["request_id"] != job.request_id:
                        raise IPCV2ProtocolError("response request_id does not match the run contract")
                    terminal = (
                        {"error": envelope.body["error"]}
                        if envelope.body.get("error") is not None
                        else {"result": envelope.body["result"]}
                    )
                    if not job.result.done():
                        job.result.set_result(terminal)
                    continue
                if envelope.kind == "event" and envelope.body["event"] == "progress":
                    if envelope.identity.display_lane != job.identity.display_lane:
                        raise IPCV2ProtocolError("progress display_lane does not match registry")
                    self._record_progress(job, envelope.body["data"])
                    continue
                if envelope.kind == "event" and envelope.body["event"] == "grace_ack":
                    job.grace_acked.set()
                    continue
                if envelope.kind == "event" and envelope.body["event"] == "capabilities":
                    job.capabilities["steer"] = envelope.body["data"].get("steer") is True
                    continue
                if envelope.kind == "event" and envelope.body["event"] == "tasks_submitted":
                    # A child task runner published tasks; wake the Anima main's
                    # PendingTaskExecutor so it does not wait a poll interval.
                    from core.tasks.wake import request_wake

                    request_wake(self.anima_name)
                    continue
                if envelope.kind == "event" and envelope.body["event"] == "stream":
                    if job.identity.lane != "chat" or not job.stream_events:
                        raise IPCV2ProtocolError("stream event is only valid for streaming chat jobs")
                    for queue in tuple(job.stream_events):
                        await queue.put(envelope.body["data"])
                    continue
                if envelope.kind == "event" and envelope.body["event"] == "inject_result":
                    data = envelope.body["data"]
                    waiter = job.inject_waiters.get(str(data.get("injection_id") or ""))
                    if waiter is not None and not waiter.done():
                        waiter.set_result(data)
                    continue
        except (TimeoutError, IPCV2ConnectionError):
            logger.debug("Task runner IPC connection ended", exc_info=True)
        except Exception as exc:
            logger.warning("Task runner IPC protocol failure: %s", exc, exc_info=True)
            if connection is not None and connection.state.identity.job_id in self._jobs:
                job = self._jobs.get(connection.state.identity.job_id)
                if job is not None and not job.result.done():
                    job.result.set_result({"error": ipc_v2_error("PROTOCOL_ERROR", str(exc), retryable=False)})
                    self._terminate_job_group(job)
        finally:
            if state_write_queue is not None and state_write_worker is not None:
                state_write_queue.put_nowait(None)
                try:
                    await asyncio.shield(state_write_worker)
                except asyncio.CancelledError:
                    await state_write_worker
                    raise
            if connection is not None:
                current_job = self._jobs.get(connection.state.identity.job_id)
                if current_job is not None and current_job.connection is connection:
                    current_job.connection = None
                await connection.close()
            else:
                writer.close()
                await writer.wait_closed()

    async def handle_memory(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        """Serve Anima-main-local and IPC callers through the same memory queue."""
        if self._memory_service is None:
            raise MemoryServiceUnavailable("Anima main memory service is disabled")
        return await self._memory_service.handle(method, params)

    async def repair_memory(self, *, include_shared: bool) -> dict[str, Any]:
        """Run phase3 repair inside the Anima-main-owned memory service."""
        if self._memory_service is None:
            raise MemoryServiceUnavailable("Anima main memory service is disabled")
        return await self._memory_service.repair(include_shared=include_shared)

    async def close(self) -> None:
        """Stop accepting jobs, grace active runners, then reap process groups."""
        self._accepting = False
        for job in list(self._jobs.values()):
            if job.connection is not None:
                try:
                    await job.connection.send_event(
                        "grace",
                        {"deadline_seconds": _TASK_RUNNER_GRACE_TIMEOUT},
                    )
                except Exception:
                    logger.debug(
                        "Could not send grace to task runner %s",
                        job.identity.job_id,
                        exc_info=True,
                    )

        # Wait briefly for grace_ack from each child before escalating.
        if self._jobs:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*(job.grace_acked.wait() for job in self._jobs.values())),
                    timeout=_TASK_RUNNER_GRACE_TIMEOUT,
                )
            except TimeoutError:
                logger.debug("Timed out waiting for grace_ack from some task runners")

        processes = [job.process for job in self._jobs.values() if job.process is not None]
        if processes:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*(process.wait() for process in processes)),
                    timeout=_TASK_RUNNER_GRACE_TIMEOUT,
                )
            except TimeoutError:
                for job in list(self._jobs.values()):
                    self._terminate_job_group(job)
                try:
                    await asyncio.wait_for(
                        asyncio.gather(*(process.wait() for process in processes)),
                        timeout=2.0,
                    )
                except TimeoutError:
                    for job in list(self._jobs.values()):
                        self._kill_job_group(job)
                    await asyncio.gather(*(process.wait() for process in processes), return_exceptions=True)
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None
        if self._memory_service is not None:
            await self._memory_service.close()
        cleanup_ipc_endpoint(self.socket_path)
        await self._recover_task_journals()

    @staticmethod
    def _descendant_processes(job: TaskRunnerJob) -> list[psutil.Process]:
        """Every descendant of the runner, including ones outside its process group.

        Codex's ``codex-linux-sandbox`` starts each shell tool in its own session,
        so ``killpg`` alone leaves its ``grep -R``/``find`` children alive as
        orphans (2026-08-30: 84 orphan sessions saturated disk IO and hang-killed
        every TaskExec in a loop).  Collect them *before* the parent dies, because
        orphans are reparented and become untraceable.
        """
        if not job.pid:
            return []
        return snapshot_descendants(job.pid)

    @staticmethod
    def _signal_job_group(job: TaskRunnerJob, sig: int) -> None:
        process = job.process
        if process is None:
            return
        # Without a process group the pid is the only handle; once the child
        # has been reaped its pid may be reused, so never signal it.
        if process.returncode is not None and (os.name == "nt" or job.pgid is None):
            return
        descendants = TaskRunnerSupervisor._descendant_processes(job)
        signal_tree(job.pid, sig, pgid=job.pgid, descendants=descendants)

    @staticmethod
    def _terminate_job_group(job: TaskRunnerJob) -> None:
        TaskRunnerSupervisor._signal_job_group(job, signal.SIGTERM)

    @staticmethod
    def _kill_job_group(job: TaskRunnerJob) -> None:
        TaskRunnerSupervisor._signal_job_group(job, signal.SIGKILL)


__all__ = ["TaskRunnerError", "TaskRunnerJob", "TaskRunnerSupervisor"]
