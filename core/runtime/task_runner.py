"""Disposable task runner entry point.

Usage: ``python -m core.runtime.task_runner --anima X --lane cron|heartbeat|task|background --job ID``
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
import time
import traceback
import uuid
from pathlib import Path
from typing import Any

from core.anima.digital_anima import DigitalAnima
from core.i18n import t
from core.paths import get_animas_dir, get_data_dir, get_shared_dir
from core.platform.env import get_env
from core.platform.process import snapshot_descendants, terminate_tree
from core.platform.tasks import spawn
from core.runtime.ipc import IPCRequest
from core.runtime.ipc_v2 import (
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
from core.runtime.state_writer import IpcStateWriter, configure_state_writer, get_state_writer
from core.runtime.streaming_handler import StreamingIPCHandler
from core.runtime.transport import open_ipc_connection
from core.schemas import CronTask

logger = logging.getLogger(__name__)

_CONNECT_DEADLINE_SECONDS = 10.0
_PROGRESS_INTERVAL_SECONDS = 5.0
_RECONNECT_RETRY_SECONDS = 5.0
_SUPPORTED_LANES = frozenset({"chat", "cron", "heartbeat", "task", "background", "inbox"})

# Total budget and per-try wait for retrying a backpressured terminal ack.
_TERMINAL_ACK_TIMEOUT = 60.0
_TERMINAL_ACK_RETRY_SLEEP_SECONDS = 2.0

# Module-level registry of open StreamingJournal instances for grace flush.
_ACTIVE_JOURNALS: list[Any] = []


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one isolated Anima task")
    parser.add_argument("--anima", required=True)
    parser.add_argument("--lane", required=True, choices=("chat", "heartbeat", "cron", "task", "background", "inbox"))
    parser.add_argument("--job", required=True)
    return parser.parse_args(argv)


def _required_environment(args: argparse.Namespace) -> tuple[Path, IPCV2Identity]:
    embed_url = get_env("ANIMAWORKS_EMBED_URL", "").strip()
    if not embed_url:
        raise RuntimeError("ANIMAWORKS_EMBED_URL is required in a task runner")
    socket_value = get_env("ANIMAWORKS_TASK_IPC_PATH", "").strip()
    root_epoch = get_env("ANIMAWORKS_TASK_ROOT_EPOCH", "").strip()
    attempt_value = get_env("ANIMAWORKS_TASK_ATTEMPT", "").strip()
    display_lane = get_env("ANIMAWORKS_TASK_DISPLAY_LANE", "").strip()
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
    from core.runtime.cron_followup import command_followup_output, should_suppress_cron_followup

    command_output = command_followup_output(task, result)
    anima_dir = getattr(anima, "anima_dir", None)
    if (
        command_output is not None
        and isinstance(anima_dir, Path)
        and should_suppress_cron_followup(anima_dir, task.name, command_output)
    ):
        command_output = None
    followup: dict[str, Any] | None = None
    if command_output is not None:
        followup_result = await anima.run_cron_task(
            task.name,
            task.description or t("scheduler.cron_fallback_description", task_name=task.name),
            command_output=command_output,
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


async def execute_cron_followup_contract(
    anima: DigitalAnima,
    task: CronTask,
    command_output: str,
) -> dict[str, Any]:
    """Run only a command cron's LLM follow-up after Anima-main-side execution."""
    if task.type != "command":
        raise ValueError("command follow-up requires a command cron task")
    result = await anima.run_cron_task(
        task.name,
        task.description or t("scheduler.cron_fallback_description", task_name=task.name),
        command_output=command_output,
        **({"skills": task.skills} if task.skills else {}),
    )
    return {
        "task_type": "command_followup",
        "result": result.model_dump(mode="json"),
        "success": result.action not in {"error", "cancelled", "failed"},
        "usage": result.usage,
    }


async def execute_heartbeat_contract(anima: DigitalAnima) -> dict[str, Any]:
    """Execute the legacy heartbeat contract inside the child process."""
    result = await anima.run_heartbeat()
    result_dict = result.model_dump(mode="json")
    return {
        "task_type": "heartbeat",
        "result": result_dict,
        "success": result.action not in {"error", "cancelled", "failed"},
        "usage": result.usage,
    }


async def execute_inbox_contract(anima: DigitalAnima) -> dict[str, Any]:
    """Execute the inbox (inter-Anima message) contract inside the child process.

    Mirrors :func:`execute_heartbeat_contract` but runs
    ``DigitalAnima.process_inbox_message`` which is the same function the
    the Anima main previously invoked inline.
    """
    result = await anima.process_inbox_message()
    result_dict = result.model_dump(mode="json")
    success = result.action not in {"error", "cancelled", "failed"} and not result.reason
    return {
        "task_type": "inbox",
        "result": result_dict,
        "success": success,
        "usage": getattr(result, "usage", None),
    }


async def execute_task_contract(anima: DigitalAnima, task_desc: dict[str, Any]) -> dict[str, Any]:
    """Execute a TaskExec LLM task inside the child process.

    Claim / lease / queue sync stay on the Anima main; this contract only
    produces the result string (and related metadata) for the Anima main to apply.
    """
    from core.tasks.pending_executor import (
        _NON_COMPLETING_SENTINELS,
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
    from core.tasks.board.tasks import attempt_scope, process_identity
    from core.tasks.queue import TaskQueueManager

    token = task_desc.get("_attempt_token")
    identity = None
    if token:
        identity = {"anima": anima.name, "task_id": str(task_desc["task_id"]), "token": str(token)}
        if not TaskQueueManager(anima.anima_dir).store.set_identity(str(token), process_identity()):
            raise RuntimeError("Task claim expired before child execution")
    with attempt_scope(identity):
        result = await executor._run_llm_task(task_desc, completed_results)
    return {
        "task_type": "llm",
        "result": result,
        "success": result not in _NON_COMPLETING_SENTINELS,
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
        deadline_s = payload.get("deadline_s")
        kwargs = {"consolidation_type": consolidation_type}
        if project is not None:
            kwargs["project"] = project
        if isinstance(deadline_s, (int, float)):
            kwargs["deadline_s"] = float(deadline_s)
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
        from core.tasks.pending_executor import _PENDING_TASK_SUBPROCESS_TIMEOUT

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
        env = {**os.environ, "ANIMAWORKS_ANIMA_DIR": anima_dir}

        def _run(argv: list[str]) -> subprocess.CompletedProcess[str]:
            return subprocess.run(
                argv,
                capture_output=True,
                text=True,
                timeout=_PENDING_TASK_SUBPROCESS_TIMEOUT,
                env=env,
                check=False,
            )

        # Prefer JSON output, but many tool subcommands do not define -j.
        completed = await asyncio.to_thread(_run, [*cmd, "-j"])
        if completed.returncode == 2 and "unrecognized arguments: -j" in completed.stderr:
            completed = await asyncio.to_thread(_run, cmd)
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
            return await anima.process_greet(
                mode=payload.get("mode", "visit"),
                user_name=payload.get("user_name", ""),
                user_id=payload.get("user_id", ""),
            )
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

        attachment_kwargs: dict[str, Any] = {}
        if payload.get("display_message") is not None:
            attachment_kwargs["display_message"] = payload["display_message"]
        if payload.get("files"):
            attachment_kwargs["files"] = payload["files"]
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
            model=payload.get("model") or None,
            **attachment_kwargs,
        )
        cycle_result = result if isinstance(result, dict) else {}
        return {
            "response": cycle_result.get("summary", "") if cycle_result else result,
            "cycle_result": cycle_result,
            "images": cycle_result.get("images") or [],
            "files": cycle_result.get("files") or [],
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
    *,
    on_response: Any | None = None,
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
                if envelope.kind == "response":
                    if callable(on_response):
                        on_response(envelope)
                    continue
                if envelope.kind == "request" and envelope.body["method"] == "run":
                    return connection, envelope
        except (OSError, IPCV2ConnectionError, IPCV2BackpressureTimeout) as exc:
            # BackpressureTimeout: hello ack >5s under Anima-main load spikes — retry
            # within the connect deadline instead of failing the whole startup.
            last_error = exc
            await asyncio.sleep(0.1)
    raise IPCV2ConnectionError(f"could not connect to Anima main: {last_error}")


class _RootLink:
    """Owns the connection to the Anima main and re-dials it when it breaks.

    The hang watchdog kills any runner whose progress stops for
    runner_liveness_timeout seconds, so a broken control socket must never
    permanently silence a healthy runner: every consumer goes through this
    link and any of them may heal it (single-flight via the lock).
    """

    def __init__(
        self,
        connection: IPCV2Connection,
        socket_path: Path,
        state: IPCV2ConnectionState,
        request_id: str,
    ) -> None:
        self.connection = connection
        self._socket_path = socket_path
        self._state = state
        self._request_id = request_id
        self._lock = asyncio.Lock()
        self._pending_responses: dict[str, asyncio.Future[dict[str, Any]]] = {}

    async def send_event(self, event: str, data: dict[str, Any] | None = None) -> int:
        return await self.connection.send_event(event, data)

    async def request(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        """Send one request and await its multiplexed response frame."""
        request_id = f"state-{uuid.uuid4().hex}"
        future: asyncio.Future[dict[str, Any]] = asyncio.get_running_loop().create_future()
        self._pending_responses[request_id] = future
        try:
            connection = self.connection
            try:
                await connection.send_request(request_id, method, params)
            except IPCV2ConnectionError:
                connection = await self.reconnect(connection)
                await connection.send_request(request_id, method, params)
            response = await asyncio.wait_for(future, timeout=300.0)
            if "error" in response:
                error = response["error"]
                raise RuntimeError(f"{error.get('code', 'STATE_WRITE_ERROR')}: {error.get('message', '')}")
            result = response.get("result")
            if not isinstance(result, dict):
                raise RuntimeError("Anima main returned a malformed state-write response")
            return result
        finally:
            self._pending_responses.pop(request_id, None)
            if not future.done():
                future.cancel()

    async def send_unanswered_request(self, method: str, params: dict[str, Any]) -> None:
        """Send one chunk-protocol frame; the peer ACKs at the transport layer."""
        request_id = f"state-{uuid.uuid4().hex}"
        connection = self.connection
        try:
            await connection.send_request(request_id, method, params)
        except IPCV2ConnectionError:
            connection = await self.reconnect(connection)
            await connection.send_request(request_id, method, params)

    def receive_response(self, envelope: IPCV2Envelope) -> bool:
        """Resolve the waiter for one state-write response, if present."""
        request_id = envelope.body["request_id"]
        future = self._pending_responses.get(request_id)
        if future is None or future.done():
            return False
        if envelope.body.get("error") is not None:
            future.set_result({"error": envelope.body["error"]})
        else:
            future.set_result({"result": envelope.body.get("result")})
        return True

    async def reconnect(self, broken: IPCV2Connection) -> IPCV2Connection:
        """Re-dial the Anima main after *broken* died; returns the live connection."""
        async with self._lock:
            if self.connection is not broken:
                return self.connection  # another consumer already healed it
            try:
                await broken.close()
            except Exception:
                logger.debug("Failed to close broken Anima main connection", exc_info=True)
            connection, replayed_run = await _connect(
                self._socket_path,
                self._state,
                on_response=self.receive_response,
            )
            if replayed_run.body["request_id"] != self._request_id:
                await connection.close()
                raise IPCV2ConnectionError("reconnect returned a different run contract")
            self.connection = connection
            logger.info("Task runner IPC reconnected to Anima main")
            return connection


async def _progress_loop(link: _RootLink, identity: IPCV2Identity) -> None:
    """Send progress heartbeats forever; survive and heal IPC failures.

    This loop must never die while execution is alive — a silent runner is
    hang-killed after runner_liveness_timeout even when it is working fine.
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
                logger.warning("Task runner reconnect to Anima main failed; will retry: %s", reconnect_exc)
                await asyncio.sleep(_RECONNECT_RETRY_SECONDS)
            continue
        await asyncio.sleep(_PROGRESS_INTERVAL_SECONDS)


async def _receive_after_reconnect(link: _RootLink, broken: IPCV2Connection) -> IPCV2Envelope:
    """Heal the Anima main connection, then behave like ``connection.receive()``."""
    while True:
        try:
            connection = await link.reconnect(broken)
        except Exception as exc:
            logger.warning("Task runner reconnect to Anima main failed; retrying: %s", exc)
            broken = link.connection
            await asyncio.sleep(_RECONNECT_RETRY_SECONDS)
            continue
        return await connection.receive()


async def _parent_monitor(expected_parent_pid: int) -> None:
    """Return when the spawning Anima main is no longer our parent."""
    while os.getppid() == expected_parent_pid:  # noqa: ASYNC110 — polls the OS parent PID, which has no asyncio.Event
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

    # Wait for the ack with backpressure retries.  A stalled Anima main event loop
    # (e.g. disk saturation) can push ack delivery past the fixed backpressure
    # timeout; retrying instead of dying lets the Anima main catch up.  Ack progress
    # is tracked on the shared connection state, so re-waiting is safe.  A
    # lost socket is recovered the same way as the main send path above.
    deadline = asyncio.get_running_loop().time() + _TERMINAL_ACK_TIMEOUT
    while True:
        ack_receiver = asyncio.create_task(connection.receive())
        try:
            await connection.wait_for_ack(terminal_seq)
            break
        except IPCV2BackpressureTimeout:
            now = asyncio.get_running_loop().time()
            if now >= deadline:
                raise
            logger.warning(
                "terminal ack backpressured, retrying seq=%s",
                terminal_seq,
            )
            await asyncio.sleep(min(_TERMINAL_ACK_RETRY_SLEEP_SECONDS, deadline - now))
        except IPCV2ConnectionError:
            # Recover like the main send path: reconnect and resend.
            await connection.close()
            connection, replayed_run = await _connect(socket_path, state)
            if replayed_run.body["request_id"] != request_id:
                await connection.close()
                raise
            terminal_seq = await connection.send_response(request_id, result=result, error=error)
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
        return asyncio.create_task(execute_heartbeat_contract(anima))

    if identity.lane == "inbox":
        return asyncio.create_task(execute_inbox_contract(anima))

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
    if "command_output" in params:
        command_output = params.get("command_output")
        if task.type != "command" or not isinstance(command_output, str):
            raise ValueError("command follow-up requires a command task and string output")
        return asyncio.create_task(execute_cron_followup_contract(anima, task, command_output))
    return asyncio.create_task(execute_cron_contract(anima, task))


async def run_task(args: argparse.Namespace, socket_path: Path, identity: IPCV2Identity) -> int:
    state = IPCV2ConnectionState(identity)
    connection, run_envelope = await _connect(socket_path, state)
    request_id = run_envelope.body["request_id"]
    params = run_envelope.body["params"]
    contract_urls = (params.get("environment") or {}).get("urls")
    if not isinstance(contract_urls, dict) or contract_urls.get("ANIMAWORKS_EMBED_URL") != get_env(
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

    link = _RootLink(connection, socket_path, state, request_id)
    configure_state_writer(IpcStateWriter(link))

    # Route child-process task submissions back to the Anima main so its
    # PendingTaskExecutor does not wait a full poll interval after a submit.
    _loop = asyncio.get_running_loop()

    def _schedule_tasks_submitted_event(link: _RootLink) -> None:
        async def _send() -> None:
            try:
                await link.send_event("tasks_submitted", {})
            except Exception:
                logger.debug("tasks_submitted wake event failed", exc_info=True)

        spawn(_send(), name="tasks-submitted-wakeup")

    from core.tasks.wake import register_wake

    register_wake(args.anima, lambda: _loop.call_soon_threadsafe(_schedule_tasks_submitted_event, link))

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
        configure_state_writer(None)
        return 2
    except Exception as exc:
        await connection.send_response(
            request_id,
            error=ipc_v2_error("EXECUTION_ERROR", str(exc), retryable=False),
        )
        await connection.close()
        configure_state_writer(None)
        return 1
    progress = asyncio.create_task(_progress_loop(link, identity))
    expected_parent_pid = int(get_env("ANIMAWORKS_TASK_ROOT_PID", str(os.getppid())))
    parent_monitor = asyncio.create_task(_parent_monitor(expected_parent_pid))
    receiver: asyncio.Task[IPCV2Envelope] | None = asyncio.create_task(connection.receive())
    cancelled = False
    graced = False
    root_lost = False

    # Steer/inject handlers run as their own tasks.  Awaiting them directly in
    # this loop deadlocks: their state writes (`link.request`) need response
    # frames that only this loop's `connection.receive()` can deliver, and a
    # blocked loop also stops ACK processing (progress sends then time out).
    # Strong references are kept until the loop drains so tasks are never
    # garbage collected mid-flight (ruff RUF006).
    control_tasks: set[asyncio.Task[Any]] = set()
    # Injections leave the control loop but must still run one at a time in
    # arrival order: each one persists the conversation, and two concurrent
    # persists of the same thread could interleave.
    inject_lock = asyncio.Lock()

    async def _inject_message_event(data: dict[str, Any]) -> None:
        injection_id = str(data.get("injection_id") or "")
        accepted = False
        error = ""
        expires_at = data.get("expires_at")
        async with inject_lock:
            # Checked after waiting for the lock: time spent behind an earlier
            # injection counts against this steer's deadline too.
            expired = (
                isinstance(expires_at, (int, float)) and not isinstance(expires_at, bool) and time.time() >= expires_at
            )
            if expired:
                # The supervisor already gave up and queued the message for the
                # lock; injecting it here too would process it twice.
                error = "expired"
            else:
                anima_for_inject = execution_control.get("anima")
                try:
                    if anima_for_inject is not None and injection_id:
                        accepted = await anima_for_inject.inject_message(
                            str(data.get("message") or ""),
                            from_person=str(data.get("from_person") or "human"),
                            thread_id=str(data.get("thread_id") or "default"),
                            attachment_paths=data.get("attachment_paths") or None,
                        )
                except Exception as exc:
                    logger.warning("Chat message injection failed: %s", exc, exc_info=True)
                    error = str(exc)
        try:
            # Send through the link so the result still lands after a reconnect
            # that happened while the injection was in flight.
            await link.send_event(
                "inject_result",
                {"injection_id": injection_id, "accepted": accepted, "error": error},
            )
        except Exception as exc:
            logger.warning("Chat inject_result send failed: %s", exc, exc_info=True)

    async def _interrupt_event(data: dict[str, Any]) -> None:
        anima_for_interrupt = execution_control.get("anima")
        if anima_for_interrupt is not None:
            await anima_for_interrupt.interrupt(thread_id=data.get("thread_id") or None)

    def _start_control_task(coro: Any) -> None:
        task = asyncio.create_task(coro)
        control_tasks.add(task)

    async def _drain_control_tasks() -> None:
        tasks = tuple(control_tasks)
        control_tasks.clear()
        for task in tasks:
            if not task.done():
                task.cancel()
        if not tasks:
            return
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for result in results:
            if isinstance(result, Exception):
                logger.warning("Chat control task failed: %s", result)

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
                    if control.body["request_id"] == request_id:
                        execution.cancel()
                        raise IPCV2ConnectionError("unexpected run response from Anima main")
                    if not link.receive_response(control):
                        logger.debug("Ignoring unsolicited Anima-main response %s", control.body["request_id"])
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
                    data = control.body.get("data") or {}
                    _start_control_task(_interrupt_event(data))
                    receiver = asyncio.create_task(connection.receive())
                    continue
                if control.kind == "event" and control.body["event"] == "inject_message":
                    data = control.body.get("data") or {}
                    _start_control_task(_inject_message_event(data))
                    receiver = asyncio.create_task(connection.receive())
                    continue
                receiver = asyncio.create_task(connection.receive())

        await _drain_control_tasks()
        connection = link.connection
        if cancelled or root_lost:
            try:
                await execution
            except asyncio.CancelledError:
                pass
            if not root_lost and anima is not None:
                writer_for_task = get_state_writer()
                if isinstance(writer_for_task, IpcStateWriter):
                    await writer_for_task.drain()
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

        writer_for_task = get_state_writer() if anima is not None else None
        try:
            result = await execution
            if isinstance(writer_for_task, IpcStateWriter):
                await writer_for_task.drain()
        except Exception as exc:
            if isinstance(writer_for_task, IpcStateWriter):
                await writer_for_task.drain()
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
        # Control tasks may still be in flight when the loop exits early
        # (cancel/grace/root_lost/execution error); never leave them dangling.
        await _drain_control_tasks()
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
        from core.tasks.wake import unregister_wake

        unregister_wake(args.anima)
        configure_state_writer(None)


def _setup_logging(anima_name: str) -> None:
    from core.config import load_config
    from core.infra.logging_config import setup_anima_logging

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


def _cleanup_descendants() -> None:
    """Reap leaked descendant processes that a dying runner would otherwise orphan.

    External CLI tools (e.g. codex-linux-sandbox's per-tool sessions) may
    outlive the runner that launched them, so on the way out we politely TERM
    every descendant (letting it flush) and then KILL the stubborn ones.
    """
    children = snapshot_descendants(os.getpid())
    if not children:
        return
    pid_count = len(children)
    terminate_tree(
        os.getpid(),
        descendants=children,
        include_root=False,
        grace_sec=2.0,
    )
    logger.warning(
        "cleaned up leaked child processes: pid_count=%s",
        pid_count,
    )


async def main() -> int:
    from core.runtime.process_role import set_process_role

    set_process_role("task_runner")
    args = parse_args()
    socket_path, identity = _required_environment(args)
    _setup_logging(args.anima)
    from core.memory.maintenance.background_review import enable_deferred_requests

    enable_deferred_requests()
    try:
        from core.config.global_permissions import GlobalPermissionsCache
        from core.paths import get_global_permissions_path

        GlobalPermissionsCache.get().load(get_global_permissions_path(), interactive=False)
    except FileNotFoundError:
        logger.warning("permissions.global.json not found; global command checks disabled")
    try:
        return await run_task(args, socket_path, identity)
    finally:
        _cleanup_descendants()


if __name__ == "__main__":
    try:
        raise SystemExit(asyncio.run(main()))
    except Exception as exc:
        # A runner that died mid-execution must not leave external tools orphaned.
        _cleanup_descendants()
        # Full traceback: the bare message ("[Errno 2] ... current.log") has
        # proven undiagnosable without the raising frame.
        traceback.print_exc(file=sys.stderr)
        print(f"task runner startup failed: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
