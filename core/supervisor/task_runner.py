"""Disposable task runner entry point.

Usage: ``python -m core.supervisor.task_runner --anima X --lane cron|heartbeat|task|background --job ID``
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import re
import sys
import threading
import traceback
import uuid
from pathlib import Path
from typing import Any

from core.anima import DigitalAnima
from core.i18n import t
from core.paths import get_animas_dir, get_data_dir, get_shared_dir
from core.schemas import CronTask
from core.supervisor.ipc import IPCRequest
from core.supervisor.ipc_v2 import (
    IPC_V2_MAX_FRAME_BYTES,
    IPCV2BackpressureTimeout,
    IPCV2Connection,
    IPCV2ConnectionError,
    IPCV2ConnectionState,
    IPCV2Envelope,
    IPCV2Identity,
    IPCV2PayloadTooLarge,
    ipc_v2_error,
)
from core.supervisor.streaming_handler import StreamingIPCHandler
from core.supervisor.transport import open_ipc_connection

logger = logging.getLogger(__name__)

_CONNECT_DEADLINE_SECONDS = 10.0
_PROGRESS_INTERVAL_SECONDS = 5.0
_RECONNECT_RETRY_SECONDS = 5.0
_SUPPORTED_LANES = frozenset({"chat", "cron", "heartbeat", "task", "background"})

# Module-level registry of open StreamingJournal instances for grace flush.
_ACTIVE_JOURNALS: list[Any] = []


class _MemoryRpcClient:
    """Bridge synchronous VectorStore operations to the task runner's async IPC."""

    def __init__(self, connection: IPCV2Connection) -> None:
        self.connection = connection
        self.loop = asyncio.get_running_loop()
        self.loop_thread = threading.get_ident()
        self.pending: dict[str, asyncio.Future[dict[str, Any]]] = {}

    def request(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        if threading.get_ident() == self.loop_thread:
            raise RuntimeError("synchronous memory read attempted on the task runner event loop")
        future = asyncio.run_coroutine_threadsafe(self._request(method, params), self.loop)
        return future.result(timeout=120.0)

    async def _request(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        request_id = f"memory-{uuid.uuid4()}"
        response = self.loop.create_future()
        self.pending[request_id] = response
        try:
            await self.connection.send_request(request_id, method, params)
            return await asyncio.wait_for(response, timeout=120.0)
        finally:
            self.pending.pop(request_id, None)

    def accept_response(self, envelope: IPCV2Envelope) -> bool:
        request_id = envelope.body["request_id"]
        response = self.pending.get(request_id)
        if response is None:
            return False
        error = envelope.body.get("error")
        if error is not None:
            response.set_exception(RuntimeError(f"{error['code']}: {error['message']}"))
        else:
            response.set_result(envelope.body["result"])
        return True

    def close(self) -> None:
        for response in self.pending.values():
            if not response.done():
                response.set_exception(RuntimeError("root memory IPC closed"))
        self.pending.clear()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one isolated Anima task")
    parser.add_argument("--anima", required=True)
    parser.add_argument("--lane", required=True, choices=("chat", "heartbeat", "cron", "task", "background"))
    parser.add_argument("--job", required=True)
    return parser.parse_args(argv)


def _required_environment(args: argparse.Namespace) -> tuple[Path, IPCV2Identity]:
    embed_url = os.environ.get("ANIMAWORKS_EMBED_URL", "").strip()
    if not embed_url:
        raise RuntimeError("ANIMAWORKS_EMBED_URL is required in a task runner")
    socket_value = os.environ.get("ANIMAWORKS_TASK_IPC_PATH", "").strip()
    root_epoch = os.environ.get("ANIMAWORKS_TASK_ROOT_EPOCH", "").strip()
    attempt_value = os.environ.get("ANIMAWORKS_TASK_ATTEMPT", "").strip()
    display_lane = os.environ.get("ANIMAWORKS_TASK_DISPLAY_LANE", "").strip()
    if not socket_value or not root_epoch or not attempt_value or not display_lane:
        raise RuntimeError("task runner IPC environment is incomplete")
    identity = IPCV2Identity(
        job_id=args.job,
        root_epoch=root_epoch,
        attempt=int(attempt_value),
        lane=args.lane,
        display_lane=display_lane,
    )
    identity.validate()
    return Path(socket_value), identity


async def execute_cron_contract(anima: DigitalAnima, task: CronTask) -> dict[str, Any]:
    """Execute the complete legacy cron contract inside the child process."""
    if task.type == "llm":
        result = await anima.run_cron_task(
            task.name,
            task.description,
            **({"skills": task.skills} if task.skills else {}),
        )
        result_dict = result.model_dump(mode="json")
        return {
            "task_type": "llm",
            "result": result_dict,
            "success": result.action not in {"error", "cancelled", "failed"},
            "usage": result.usage,
        }
    if task.type != "command":
        raise ValueError(f"unknown cron type: {task.type!r}")

    result = await anima.run_cron_command(
        task.name,
        command=task.command,
        tool=task.tool,
        args=task.args,
    )
    success = result.get("exit_code", 1) == 0
    usage: dict[str, int] | None = None
    stdout = str(result.get("stdout", "")).strip()
    should_follow_up = bool(stdout and success and task.trigger_heartbeat)
    if should_follow_up and task.skip_pattern:
        try:
            should_follow_up = re.search(task.skip_pattern, stdout) is None
        except re.error as exc:
            logger.warning(
                "Invalid skip_pattern %r for task %r: %s; continuing without skip",
                task.skip_pattern,
                task.name,
                exc,
            )
    followup: dict[str, Any] | None = None
    if should_follow_up:
        followup_result = await anima.run_cron_task(
            task.name,
            task.description or t("scheduler.cron_fallback_description", task_name=task.name),
            command_output=stdout,
            **({"skills": task.skills} if task.skills else {}),
        )
        followup = followup_result.model_dump(mode="json")
        success = success and followup_result.action not in {"error", "cancelled", "failed"}
        usage = followup_result.usage
    return {
        "task_type": "command",
        "result": result,
        "followup_result": followup,
        "success": success,
        "usage": usage,
    }


async def execute_heartbeat_contract(
    anima: DigitalAnima,
    *,
    cascade_suppressed_senders: list[str] | None = None,
) -> dict[str, Any]:
    """Execute the legacy heartbeat contract inside the child process."""
    senders = set(cascade_suppressed_senders) if cascade_suppressed_senders else None
    result = await anima.run_heartbeat(cascade_suppressed_senders=senders)
    result_dict = result.model_dump(mode="json")
    return {
        "task_type": "heartbeat",
        "result": result_dict,
        "success": result.action not in {"error", "cancelled", "failed"},
        "usage": result.usage,
    }


async def execute_task_contract(anima: DigitalAnima, task_desc: dict[str, Any]) -> dict[str, Any]:
    """Execute a TaskExec LLM task inside the child process.

    Claim / lease / queue sync stay on the anima root; this contract only
    produces the result string (and related metadata) for the root to apply.
    """
    from core.supervisor.pending_executor import (
        _SENTINEL_CANCELLED,
        _SENTINEL_CONTINUED,
        _SENTINEL_DEFERRED,
        _SENTINEL_EXPIRED,
        PendingTaskExecutor,
    )

    shutdown = asyncio.Event()
    executor = PendingTaskExecutor(
        anima=anima,
        anima_name=anima.name,
        anima_dir=anima.anima_dir,
        shutdown_event=shutdown,
    )
    completed_results = task_desc.get("_completed_results")
    if not isinstance(completed_results, dict):
        completed_results = None
    result = await executor._run_llm_task(task_desc, completed_results)
    return {
        "task_type": "llm",
        "result": result,
        "success": result not in {_SENTINEL_CANCELLED, _SENTINEL_CONTINUED, _SENTINEL_EXPIRED, _SENTINEL_DEFERRED},
    }


async def execute_background_contract(
    anima: DigitalAnima,
    *,
    kind: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    """Execute a background-lane contract (consolidation or command tool)."""
    if kind == "consolidation":
        consolidation_type = str(payload.get("consolidation_type") or "daily")
        project = payload.get("project")
        kwargs = {"consolidation_type": consolidation_type}
        if project is not None:
            kwargs["project"] = project
        result = await anima.run_consolidation(**kwargs)
        result_dict = result.model_dump(mode="json") if result is not None else {}
        return {
            "task_type": "consolidation",
            "result": result_dict,
            "success": bool(result is not None and getattr(result, "action", "completed") not in {"error", "failed"}),
            "summary": (result.summary[:500] if result and result.summary else ""),
            "duration_ms": result.duration_ms if result else 0,
        }
    if kind == "command":
        # Run a single pending command tool synchronously in the child.
        import subprocess

        from core.exceptions import ToolExecutionError

        tool_name = str(payload.get("tool_name") or "")
        subcommand = str(payload.get("subcommand") or "")
        raw_args = list(payload.get("raw_args") or [])
        anima_dir = str(payload.get("anima_dir") or anima.anima_dir)
        module_name = tool_name.split(":")[0] if ":" in tool_name else tool_name
        cmd = ["animaworks-tool", module_name]
        if subcommand:
            cmd.append(subcommand)
        cmd.extend(raw_args)
        if subcommand and raw_args and raw_args[0] == subcommand:
            cmd = ["animaworks-tool", module_name, *raw_args]
        cmd.append("-j")
        env = {**os.environ, "ANIMAWORKS_ANIMA_DIR": anima_dir}
        completed = await asyncio.to_thread(
            subprocess.run,
            cmd,
            capture_output=True,
            text=True,
            timeout=1800,
            env=env,
            check=False,
        )
        if completed.returncode != 0:
            error_msg = completed.stderr.strip() or f"Exit code {completed.returncode}"
            raise ToolExecutionError(f"Tool {tool_name} failed: {error_msg}")
        return {
            "task_type": "command",
            "result": completed.stdout.strip(),
            "success": True,
        }
    raise ValueError(f"unknown background kind: {kind!r}")


async def execute_chat_contract(
    anima: DigitalAnima,
    *,
    kind: str,
    payload: dict[str, Any],
    send_stream_event: Any,
) -> dict[str, Any]:
    """Execute one legacy chat contract inside a disposable child."""
    try:
        if kind == "greet":
            return await anima.process_greet()
        if kind == "bootstrap":
            result = await anima.run_bootstrap()
            return {"status": "completed", "summary": result.summary, "duration_ms": result.duration_ms}
        if kind != "message":
            raise ValueError(f"unknown chat contract kind: {kind!r}")

        if payload.get("stream"):
            handler = StreamingIPCHandler(anima, anima.name, anima.anima_dir)
            request = IPCRequest(id="chat", method="process_message", params=payload)
            async for response in handler.handle_stream(request):
                if response.error:
                    raise RuntimeError(response.error.get("message") or "chat stream failed")
                if response.chunk is not None:
                    await send_stream_event("stream", {"stream": True, "chunk": response.chunk})
                if response.done:
                    return response.result or {"response": "", "replied_to": []}
            return {"response": "", "replied_to": []}

        result = await anima.process_message(
            str(payload.get("message") or ""),
            from_person=str(payload.get("from_person") or "human"),
            intent=str(payload.get("intent") or ""),
            images=payload.get("images") or None,
            attachment_paths=payload.get("attachment_paths") or None,
            thread_id=str(payload.get("thread_id") or "default"),
            include_cycle_result=True,
            source=str(payload.get("source") or ""),
            voice_mode=payload.get("voice_mode") is True,
            meeting_room_id=str(payload.get("meeting_room_id") or ""),
            meeting_participants=payload.get("meeting_participants") or None,
            model=payload.get("model") or None,
        )
        cycle_result = result if isinstance(result, dict) else {}
        return {
            "response": cycle_result.get("summary", "") if cycle_result else result,
            "cycle_result": cycle_result,
            "images": cycle_result.get("images") or [],
            "replied_to": [],
            "notifications": anima.drain_notifications(),
        }
    finally:
        _fsync_chat_state(anima.anima_dir)


def _fsync_chat_state(anima_dir: Path) -> None:
    """Durably flush chat conversation, resume, and journal files before exit."""
    paths = [
        anima_dir / "state" / "conversation.json",
        anima_dir / "shortterm" / "streaming_journal_chat.jsonl",
        anima_dir / "shortterm" / "streaming_journal.jsonl",
    ]
    paths.extend((anima_dir / "state").glob("current_session_chat*.json"))
    paths.extend((anima_dir / "state" / "conversations").glob("*.json"))
    chat_shortterm = anima_dir / "shortterm" / "chat"
    if chat_shortterm.is_dir():
        paths.extend(path for path in chat_shortterm.rglob("*") if path.is_file())
    for path in paths:
        if not path.is_file():
            continue
        try:
            fd = os.open(path, os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
        except OSError:
            logger.warning("Failed to fsync chat state file %s", path, exc_info=True)


async def _connect(
    socket_path: Path,
    state: IPCV2ConnectionState,
) -> tuple[IPCV2Connection, IPCV2Envelope]:
    deadline = asyncio.get_running_loop().time() + _CONNECT_DEADLINE_SECONDS
    last_error: Exception | None = None
    while asyncio.get_running_loop().time() < deadline:
        try:
            reader, writer = await open_ipc_connection(
                socket_path,
                limit=IPC_V2_MAX_FRAME_BYTES + 1,
            )
            connection = IPCV2Connection(reader, writer, state)
            await connection.send_event(
                "hello",
                {
                    "capabilities": {"reconnect": True, "steer": False},
                    "last_received_seq": state.received_seq,
                    "last_acked_seq": state.last_acked_seq,
                },
            )
            while True:
                envelope = await connection.receive()
                if envelope.kind == "event" and envelope.body["event"] == "hello_ack":
                    continue
                if envelope.kind == "request" and envelope.body["method"] == "run":
                    return connection, envelope
        except (OSError, IPCV2ConnectionError, IPCV2BackpressureTimeout) as exc:
            # BackpressureTimeout: hello ack >5s under root load spikes — retry
            # within the connect deadline instead of failing the whole startup.
            last_error = exc
            await asyncio.sleep(0.1)
    raise IPCV2ConnectionError(f"could not connect to anima root: {last_error}")


class _RootLink:
    """Owns the connection to the anima root and re-dials it when it breaks.

    The hang watchdog kills any runner whose progress stops for
    busy_hang_threshold seconds, so a broken control socket must never
    permanently silence a healthy runner: every consumer goes through this
    link and any of them may heal it (single-flight via the lock).
    """

    def __init__(
        self,
        connection: IPCV2Connection,
        socket_path: Path,
        state: IPCV2ConnectionState,
        request_id: str,
        memory_client: _MemoryRpcClient | None = None,
    ) -> None:
        self.connection = connection
        self._socket_path = socket_path
        self._state = state
        self._request_id = request_id
        self.memory_client = memory_client
        self._lock = asyncio.Lock()

    async def send_event(self, event: str, data: dict[str, Any] | None = None) -> int:
        return await self.connection.send_event(event, data)

    async def reconnect(self, broken: IPCV2Connection) -> IPCV2Connection:
        """Re-dial the anima root after *broken* died; returns the live connection."""
        async with self._lock:
            if self.connection is not broken:
                return self.connection  # another consumer already healed it
            try:
                await broken.close()
            except Exception:
                logger.debug("Failed to close broken root connection", exc_info=True)
            connection, replayed_run = await _connect(self._socket_path, self._state)
            if replayed_run.body["request_id"] != self._request_id:
                await connection.close()
                raise IPCV2ConnectionError("reconnect returned a different run contract")
            self.connection = connection
            if self.memory_client is not None:
                self.memory_client.connection = connection
            logger.info("Task runner IPC reconnected to anima root")
            return connection


async def _progress_loop(link: _RootLink, identity: IPCV2Identity) -> None:
    """Send progress heartbeats forever; survive and heal IPC failures.

    This loop must never die while execution is alive — a silent runner is
    hang-killed after busy_hang_threshold even when it is working fine.
    """
    while True:
        connection = link.connection
        try:
            await connection.send_event(
                "progress",
                {
                    "pid": os.getpid(),
                    "pgid": os.getpgrp() if hasattr(os, "getpgrp") else os.getpid(),
                    "lane": identity.lane,
                    "job_id": identity.job_id,
                    "display_lane": identity.display_lane,
                    "progress_at": asyncio.get_running_loop().time(),
                },
            )
        except Exception as exc:
            logger.warning("Task runner progress send failed; reconnecting: %s", exc)
            try:
                await link.reconnect(connection)
            except Exception as reconnect_exc:
                logger.warning(
                    "Task runner reconnect to anima root failed; will retry: %s", reconnect_exc
                )
                await asyncio.sleep(_RECONNECT_RETRY_SECONDS)
            continue
        await asyncio.sleep(_PROGRESS_INTERVAL_SECONDS)


async def _receive_after_reconnect(link: _RootLink, broken: IPCV2Connection) -> IPCV2Envelope:
    """Heal the root connection, then behave like ``connection.receive()``."""
    while True:
        try:
            connection = await link.reconnect(broken)
        except Exception as exc:
            logger.warning("Task runner reconnect to anima root failed; retrying: %s", exc)
            broken = link.connection
            await asyncio.sleep(_RECONNECT_RETRY_SECONDS)
            continue
        return await connection.receive()


async def _parent_monitor(expected_parent_pid: int) -> None:
    """Return when the spawning anima root is no longer our parent."""
    while os.getppid() == expected_parent_pid:
        await asyncio.sleep(1.0)


def _flush_active_journals() -> dict[str, Any]:
    """Flush every tracked StreamingJournal buffer (grace contract A-07)."""
    flushed = 0
    last_seq = 0
    for journal in list(_ACTIVE_JOURNALS):
        try:
            if hasattr(journal, "close"):
                journal.close()
                flushed += 1
        except Exception:
            logger.debug("Failed to flush journal during grace", exc_info=True)
    return {"flushed_journals": flushed, "last_journal_seq": last_seq}


async def _send_grace_ack(connection: IPCV2Connection, *, grace_seq: int = 0) -> None:
    flush_info = _flush_active_journals()
    try:
        await connection.send_event(
            "grace_ack",
            {
                "grace_seq": grace_seq,
                "flush": flush_info,
                "last_journal_seq": flush_info.get("last_journal_seq", 0),
            },
        )
    except Exception:
        logger.debug("Failed to send grace_ack", exc_info=True)


async def _send_terminal(
    connection: IPCV2Connection,
    socket_path: Path,
    state: IPCV2ConnectionState,
    request_id: str,
    receiver: asyncio.Task[IPCV2Envelope] | None,
    *,
    result: dict[str, Any] | None = None,
    error: dict[str, Any] | None = None,
) -> IPCV2Connection:
    """Send a terminal response, reconnecting once if the socket was lost."""
    try:
        terminal_seq = await connection.send_response(request_id, result=result, error=error)
    except IPCV2PayloadTooLarge:
        terminal_seq = await connection.send_response(
            request_id,
            error=ipc_v2_error(
                "PAYLOAD_TOO_LARGE",
                "task result exceeded the 4 MiB IPC frame limit",
                retryable=False,
            ),
        )
    except IPCV2ConnectionError:
        await connection.close()
        connection, replayed_run = await _connect(socket_path, state)
        if replayed_run.body["request_id"] != request_id:
            await connection.close()
            raise
        terminal_seq = await connection.send_response(request_id, result=result, error=error)

    if receiver is not None:
        receiver.cancel()
        await asyncio.gather(receiver, return_exceptions=True)
    ack_receiver = asyncio.create_task(connection.receive())
    try:
        await connection.wait_for_ack(terminal_seq)
    finally:
        ack_receiver.cancel()
        await asyncio.gather(ack_receiver, return_exceptions=True)
    return connection


async def _prepare_execution(
    args: argparse.Namespace,
    identity: IPCV2Identity,
    params: dict[str, Any],
    *,
    send_stream_event: Any | None = None,
    control: dict[str, Any] | None = None,
) -> asyncio.Task[dict[str, Any]]:
    """Validate the run contract and return the lane execution task."""
    if identity.lane not in _SUPPORTED_LANES:
        raise ValueError(f"unsupported task runner lane: {identity.lane!r}")

    anima = DigitalAnima(
        anima_dir=get_animas_dir() / args.anima,
        shared_dir=get_shared_dir(),
        busy_status_enabled=False,
    )
    if control is not None:
        control["anima"] = anima
    if identity.lane == "chat":
        kind = params.get("kind")
        payload = params.get("payload")
        if not isinstance(kind, str) or not isinstance(payload, dict) or send_stream_event is None:
            raise ValueError("run contract requires chat kind, payload, and event sender")
        return asyncio.create_task(
            execute_chat_contract(
                anima,
                kind=kind,
                payload=payload,
                send_stream_event=send_stream_event,
            )
        )
    if identity.lane == "heartbeat":
        raw_senders = params.get("cascade_suppressed_senders")
        senders: list[str] | None
        if raw_senders is None:
            senders = None
        elif isinstance(raw_senders, list) and all(isinstance(item, str) for item in raw_senders):
            senders = list(raw_senders)
        else:
            raise ValueError("cascade_suppressed_senders must be a list of strings or null")
        return asyncio.create_task(execute_heartbeat_contract(anima, cascade_suppressed_senders=senders))

    if identity.lane == "task":
        task_desc = params.get("task_desc")
        if not isinstance(task_desc, dict):
            raise ValueError("run contract requires task_desc")
        return asyncio.create_task(execute_task_contract(anima, task_desc))

    if identity.lane == "background":
        kind = params.get("kind")
        payload = params.get("payload")
        if not isinstance(kind, str) or not kind:
            raise ValueError("run contract requires background kind")
        if not isinstance(payload, dict):
            raise ValueError("run contract requires background payload object")
        return asyncio.create_task(execute_background_contract(anima, kind=kind, payload=payload))

    task_data = params.get("task")
    if not isinstance(task_data, dict):
        raise ValueError("run contract requires task")
    task = CronTask.model_validate(task_data)
    return asyncio.create_task(execute_cron_contract(anima, task))


async def run_task(args: argparse.Namespace, socket_path: Path, identity: IPCV2Identity) -> int:
    state = IPCV2ConnectionState(identity)
    connection, run_envelope = await _connect(socket_path, state)
    request_id = run_envelope.body["request_id"]
    params = run_envelope.body["params"]
    contract_urls = (params.get("environment") or {}).get("urls")
    if not isinstance(contract_urls, dict) or contract_urls.get("ANIMAWORKS_EMBED_URL") != os.environ.get(
        "ANIMAWORKS_EMBED_URL"
    ):
        await connection.send_response(
            request_id,
            error=ipc_v2_error(
                "PROTOCOL_ERROR",
                "run contract URL environment is missing or inconsistent",
                retryable=False,
            ),
        )
        await connection.close()
        return 2

    memory_client: _MemoryRpcClient | None = None
    if os.environ.get("ANIMAWORKS_MEMORY_VIA_ROOT") == "1":
        from core.memory.rag.singleton import configure_ipc_vector_requester

        memory_client = _MemoryRpcClient(connection)
        configure_ipc_vector_requester(memory_client.request)

    link = _RootLink(connection, socket_path, state, request_id, memory_client)

    try:
        execution_control: dict[str, Any] = {}
        execution = await _prepare_execution(
            args,
            identity,
            params,
            send_stream_event=link.send_event,
            control=execution_control,
        )
        anima = execution_control.get("anima")
        await connection.send_event(
            "capabilities",
            {
                "steer": bool(
                    identity.lane == "chat" and anima is not None and anima.agent._executor.supports_message_injection
                )
            },
        )
    except ValueError as exc:
        await connection.send_response(
            request_id,
            error=ipc_v2_error("PROTOCOL_ERROR", str(exc), retryable=False),
        )
        await connection.close()
        return 2
    except Exception as exc:
        await connection.send_response(
            request_id,
            error=ipc_v2_error("EXECUTION_ERROR", str(exc), retryable=False),
        )
        await connection.close()
        return 1
    progress = asyncio.create_task(_progress_loop(link, identity))
    expected_parent_pid = int(os.environ.get("ANIMAWORKS_TASK_ROOT_PID", os.getppid()))
    parent_monitor = asyncio.create_task(_parent_monitor(expected_parent_pid))
    receiver: asyncio.Task[IPCV2Envelope] | None = asyncio.create_task(connection.receive())
    cancelled = False
    graced = False
    root_lost = False
    try:
        while not execution.done():
            waiters: set[asyncio.Task[Any]] = {execution, parent_monitor}
            if receiver is not None:
                waiters.add(receiver)
            done, _ = await asyncio.wait(waiters, return_when=asyncio.FIRST_COMPLETED)
            if execution in done:
                break
            if parent_monitor in done:
                execution.cancel()
                root_lost = True
                break
            if receiver is not None and receiver in done:
                # The link may have been healed by another consumer while we
                # were waiting; always act on the current connection.
                connection = link.connection
                try:
                    control = receiver.result()
                except IPCV2ConnectionError:
                    # Do NOT silence progress: heal the link and keep both the
                    # control channel and the heartbeat alive. A permanently
                    # mute runner gets hang-killed even while working fine.
                    receiver = asyncio.create_task(_receive_after_reconnect(link, connection))
                    continue
                if control.kind == "response":
                    if memory_client is None or not memory_client.accept_response(control):
                        execution.cancel()
                        raise IPCV2ConnectionError("unexpected response from anima root")
                    receiver = asyncio.create_task(connection.receive())
                    continue
                if control.kind == "event" and control.body["event"] == "grace":
                    # A-07: stop work (finally flushes journals) → grace_ack → exit.
                    grace_seq = 0
                    data = control.body.get("data") or {}
                    if isinstance(data, dict) and isinstance(data.get("deadline_seconds"), (int, float)):
                        grace_seq = int(data.get("deadline_seconds") or 0)
                    execution.cancel()
                    try:
                        await execution
                    except asyncio.CancelledError:
                        pass
                    await _send_grace_ack(connection, grace_seq=grace_seq)
                    graced = True
                    cancelled = True
                    break
                if control.kind == "event" and control.body["event"] == "cancel":
                    execution.cancel()
                    cancelled = True
                    break
                if control.kind == "event" and control.body["event"] == "interrupt":
                    anima = execution_control.get("anima")
                    data = control.body.get("data") or {}
                    if anima is not None:
                        await anima.interrupt(thread_id=data.get("thread_id") or None)
                    receiver = asyncio.create_task(connection.receive())
                    continue
                if control.kind == "event" and control.body["event"] == "inject_message":
                    data = control.body.get("data") or {}
                    injection_id = str(data.get("injection_id") or "")
                    accepted = False
                    error = ""
                    anima = execution_control.get("anima")
                    try:
                        if anima is not None and injection_id:
                            accepted = await anima.inject_message(
                                str(data.get("message") or ""),
                                from_person=str(data.get("from_person") or "human"),
                                thread_id=str(data.get("thread_id") or "default"),
                                attachment_paths=data.get("attachment_paths") or None,
                            )
                    except Exception as exc:
                        logger.warning("Chat message injection failed: %s", exc, exc_info=True)
                        error = str(exc)
                    await connection.send_event(
                        "inject_result",
                        {"injection_id": injection_id, "accepted": accepted, "error": error},
                    )
                    receiver = asyncio.create_task(connection.receive())
                    continue
                receiver = asyncio.create_task(connection.receive())

        connection = link.connection
        if cancelled or root_lost:
            try:
                await execution
            except asyncio.CancelledError:
                pass
            if not root_lost and not graced:
                connection = await _send_terminal(
                    connection,
                    socket_path,
                    state,
                    request_id,
                    receiver,
                    error=ipc_v2_error("CANCELLED", "task runner was cancelled", retryable=True),
                )
            elif graced and not root_lost:
                connection = await _send_terminal(
                    connection,
                    socket_path,
                    state,
                    request_id,
                    receiver,
                    error=ipc_v2_error("CANCELLED", "task runner shut down under grace", retryable=True),
                )
            return 1

        try:
            result = await execution
        except Exception as exc:
            logger.exception("Task runner execution failed")
            connection = await _send_terminal(
                connection,
                socket_path,
                state,
                request_id,
                receiver,
                error=ipc_v2_error("EXECUTION_ERROR", str(exc), retryable=False),
            )
            return 1
        connection = await _send_terminal(
            connection,
            socket_path,
            state,
            request_id,
            receiver,
            result=result,
        )
        return 0
    finally:
        if memory_client is not None:
            from core.memory.rag.singleton import configure_ipc_vector_requester

            memory_client.close()
            configure_ipc_vector_requester(None)
        progress.cancel()
        parent_monitor.cancel()
        if receiver is not None:
            receiver.cancel()
        await asyncio.gather(
            progress,
            parent_monitor,
            *(tuple([receiver]) if receiver is not None else ()),
            return_exceptions=True,
        )
        # _send_terminal may have re-dialed on its own; close every connection
        # this runner still holds (close() is idempotent).
        for conn in {connection, link.connection}:
            await conn.close()


def _setup_logging(anima_name: str) -> None:
    from core.config import load_config
    from core.logging_config import setup_anima_logging

    try:
        redaction_enabled = load_config().logging.redaction_enabled
    except Exception:
        redaction_enabled = True
    setup_anima_logging(
        anima_name=anima_name,
        log_dir=get_data_dir() / "logs",
        level="INFO",
        also_to_console=False,
        redaction_enabled=redaction_enabled,
    )


async def main() -> int:
    args = parse_args()
    socket_path, identity = _required_environment(args)
    _setup_logging(args.anima)
    try:
        from core.config.global_permissions import GlobalPermissionsCache
        from core.paths import get_global_permissions_path

        GlobalPermissionsCache.get().load(get_global_permissions_path(), interactive=False)
    except FileNotFoundError:
        logger.warning("permissions.global.json not found; global command checks disabled")
    return await run_task(args, socket_path, identity)


if __name__ == "__main__":
    try:
        raise SystemExit(asyncio.run(main()))
    except Exception as exc:
        # Full traceback: the bare message ("[Errno 2] ... current.log") has
        # proven undiagnosable without the raising frame.
        traceback.print_exc(file=sys.stderr)
        print(f"task runner startup failed: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
