"""
Process handle for managing child Anima processes.
"""

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import logging
import os
import subprocess
import sys
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

import psutil

from core.exceptions import AnimaNotRunningError, IPCConnectionError, ProcessError
from core.platform.process import subprocess_session_kwargs, terminate_subprocess
from core.supervisor.ipc import IPCClient, IPCRequest, IPCResponse
from core.time_utils import ensure_aware, now_local

logger = logging.getLogger(__name__)


# Default upper bound on how long stop() waits for an in-flight interactive
# stream (a user-facing chat response) to finish before shutting the process
# down. A rolling restart / recovery sweep must not cut a chat off
# mid-generation — the graceful grace period inside stop() is only ~5s.
DEFAULT_STREAM_DRAIN_TIMEOUT_SEC = 120.0


# ── Process State ──────────────────────────────────────────────────


class ProcessState(Enum):
    """State of a child process."""

    STARTING = "starting"  # Process spawned, waiting for socket
    RUNNING = "running"  # Process running, socket connected
    STOPPING = "stopping"  # Shutdown requested
    STOPPED = "stopped"  # Process exited normally
    FAILED = "failed"  # Process crashed or killed
    RESTARTING = "restarting"  # Auto-restart in progress


@dataclass
class ProcessStats:
    """Process statistics."""

    started_at: datetime
    stopped_at: datetime | None = None
    restart_count: int = 0
    last_ping_at: datetime | None = None
    missed_pings: int = 0
    last_busy_since: datetime | None = None
    exit_code: int | None = None


# ── Process Handle ──────────────────────────────────────────────────


class ProcessHandle:
    """
    Handle for a child Anima process.

    Manages process lifecycle (spawn, monitor, kill, restart) and
    IPC communication.
    """

    def __init__(
        self,
        anima_name: str,
        socket_path: Path,
        animas_dir: Path,
        shared_dir: Path,
        log_dir: Path | None = None,
        child_env_urls: dict[str, str] | None = None,
        startup_ready_timeout: float = 120.0,
    ):
        self.anima_name = anima_name
        self.socket_path = socket_path
        self.animas_dir = animas_dir
        self.shared_dir = shared_dir
        self.log_dir = log_dir
        self._child_env_urls = child_env_urls or {}
        self.startup_ready_timeout = startup_ready_timeout

        self.state = ProcessState.STOPPED
        self.process: subprocess.Popen | None = None
        self.ipc_client: IPCClient | None = None
        self.stats = ProcessStats(started_at=now_local())
        self._streaming_lock = asyncio.Lock()
        self._streaming = False
        self._streaming_started_at: datetime | None = None
        self.stopping_since: datetime | None = None
        self._stderr_file: Any | None = None

    @property
    def is_streaming(self) -> bool:
        """Whether the process is currently streaming (read-safe snapshot)."""
        return self._streaming

    @property
    def streaming_started_at(self) -> datetime | None:
        """When the current streaming session started (read-safe snapshot)."""
        return self._streaming_started_at

    async def start(self) -> None:
        """
        Start the child process.

        Spawns the subprocess and waits for socket connection.
        """
        if self.state not in (ProcessState.STOPPED, ProcessState.FAILED):
            raise ProcessError(f"Cannot start process in state {self.state}")

        self.state = ProcessState.STARTING
        self.stats = ProcessStats(started_at=now_local())

        # Spawn child process
        cmd = [
            sys.executable,
            "-m",
            "core.supervisor.runner",
            "--anima-name",
            self.anima_name,
            "--socket-path",
            str(self.socket_path),
            "--animas-dir",
            str(self.animas_dir),
            "--shared-dir",
            str(self.shared_dir),
            "--log-dir",
            str(self.log_dir) if self.log_dir else "/tmp",
        ]

        logger.info("Starting process: %s", self.anima_name)
        logger.debug("Command: %s", " ".join(cmd))

        stderr_path: Path | None = None
        try:
            # Redirect stderr to a log file for post-mortem debugging;
            # stdout is discarded because the child writes its own log files.
            if self.log_dir:
                stderr_dir = self.log_dir / "animas" / self.anima_name
                stderr_dir.mkdir(parents=True, exist_ok=True)
                stderr_path = stderr_dir / "stderr.log"
                # Rotate if exceeds 5 MB (keep one backup)
                if stderr_path.exists():
                    try:
                        if stderr_path.stat().st_size > 5 * 1024 * 1024:
                            backup = stderr_dir / "stderr.log.1"
                            if backup.exists():
                                backup.unlink()
                            stderr_path.rename(backup)
                            logger.info(
                                "Rotated stderr.log for %s (>5MB)",
                                self.anima_name,
                            )
                    except OSError:
                        logger.debug("stderr.log rotation failed", exc_info=True)

            self._stderr_file = (
                open(stderr_path, "a") if stderr_path else None  # noqa: SIM115
            )

            child_env = os.environ.copy()
            child_env.update(self._child_env_urls)

            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=self._stderr_file if self._stderr_file else subprocess.DEVNULL,
                env=child_env,
                **subprocess_session_kwargs(),
            )
            logger.info("Process started: %s (PID %s)", self.anima_name, self.process.pid)

            # Wait for socket to be created (IPC server starts before
            # heavy DigitalAnima init, so this should be quick)
            await self._wait_for_socket(timeout=15.0)

            # Connect IPC client
            self.ipc_client = IPCClient(self.socket_path)
            await self.ipc_client.connect(timeout=5.0)

            # Wait for Anima to finish initialization (RAG model loading
            # etc.) by polling the ping endpoint until status is "ok"
            await self._wait_for_ready(timeout=self.startup_ready_timeout)

            self.state = ProcessState.RUNNING
            logger.info("Process running: %s", self.anima_name)

        except BaseException as e:
            logger.error("Failed to start process %s: %s", self.anima_name, e)

            # Log stderr file location for debugging
            if stderr_path and stderr_path.exists():
                logger.error("Subprocess stderr log: %s", stderr_path)

            self.state = ProcessState.FAILED
            await self._cleanup()
            raise

    async def _wait_for_socket(self, timeout: float) -> None:
        """Wait for socket file to be created."""
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        while loop.time() < deadline:
            if self.socket_path.exists():
                logger.debug("Socket file created: %s", self.socket_path)
                return
            # Check if the subprocess exited early (crash before socket)
            if self.process and self.process.poll() is not None:
                raise ProcessError(
                    f"Process '{self.anima_name}' exited with code {self.process.returncode} before creating socket"
                )
            await asyncio.sleep(0.1)

        raise TimeoutError(f"Socket file not created: {self.socket_path}")

    async def _wait_for_ready(self, timeout: float) -> None:
        """Wait for the Anima to finish initialization.

        The child process creates the IPC socket immediately, then loads
        the heavy DigitalAnima (RAG models, etc.).  This method polls
        the ``ping`` endpoint until the response reports ``status: "ok"``.
        """
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout

        while loop.time() < deadline:
            # Check if the subprocess exited unexpectedly
            if self.process and self.process.poll() is not None:
                raise ProcessError(
                    f"Process '{self.anima_name}' exited with code {self.process.returncode} during initialization"
                )

            try:
                request = IPCRequest(id=f"ping_{uuid.uuid4().hex[:8]}", method="ping", params={})
                if not self.ipc_client:
                    raise IPCConnectionError(f"IPC client not initialized for {self.anima_name}")
                response = await self.ipc_client.send_request(request, timeout=5.0)
                if response.result and response.result.get("status") == "ok":
                    logger.info(
                        "Anima ready: %s (init took %.1fs)",
                        self.anima_name,
                        loop.time() - (deadline - timeout),
                    )
                    return
                # status == "initializing" — keep polling
                logger.debug(
                    "Anima initializing: %s (status=%s)",
                    self.anima_name,
                    response.result.get("status") if response.result else "unknown",
                )
            except Exception as e:
                logger.debug("Ready check failed for %s: %s", self.anima_name, e)

            await asyncio.sleep(1.0)

        raise TimeoutError(f"Anima '{self.anima_name}' not ready within {timeout}s")

    async def send_request(self, method: str, params: dict, timeout: float = 60.0) -> IPCResponse:
        """
        Send IPC request to child process.

        Args:
            method: The method name
            params: Request parameters
            timeout: Timeout in seconds

        Returns:
            The response

        Raises:
            RuntimeError: If process is not running
            asyncio.TimeoutError: If timeout exceeded
        """
        if self.state == ProcessState.RESTARTING:
            raise ProcessError(f"Process restarting: {self.anima_name}")
        if self.state != ProcessState.RUNNING:
            raise AnimaNotRunningError(f"Process not running: {self.state}")

        if not self.ipc_client:
            raise IPCConnectionError("IPC client not connected")

        request = IPCRequest(id=f"req_{uuid.uuid4().hex[:8]}", method=method, params=params)

        try:
            return await self.ipc_client.send_request(request, timeout=timeout)
        except IPCConnectionError:
            if self.process and self.process.poll() is not None:
                self.state = ProcessState.FAILED
            raise

    async def send_request_stream(
        self,
        method: str,
        params: dict,
        timeout: float | None = None,
    ) -> AsyncIterator[IPCResponse]:
        """
        Send IPC request to child process and yield streaming responses.

        Args:
            method: The method name
            params: Request parameters (should include stream=True)
            timeout: Per-chunk timeout in seconds. Resets on each received
                chunk. If None, resolved from config (default 60s).

        Yields:
            IPCResponse objects (chunks and final result)

        Raises:
            RuntimeError: If process is not running
            asyncio.TimeoutError: If timeout exceeded
        """
        if self.state == ProcessState.RESTARTING:
            raise ProcessError(f"Process restarting: {self.anima_name}")
        if self.state != ProcessState.RUNNING:
            raise AnimaNotRunningError(f"Process not running: {self.state}")

        if not self.ipc_client:
            raise IPCConnectionError("IPC client not connected")

        request = IPCRequest(id=f"req_{uuid.uuid4().hex[:8]}", method=method, params=params)

        async with self._streaming_lock:
            self._streaming = True
            self._streaming_started_at = now_local()
        logger.info(
            "[PH-STREAM] start anima=%s method=%s req_id=%s state=%s pid=%s",
            self.anima_name,
            method,
            request.id,
            self.state.value,
            self.process.pid if self.process else "N/A",
        )
        chunk_count = 0
        try:
            async for response in self.ipc_client.send_request_stream(request, timeout=timeout):
                chunk_count += 1
                yield response
        except IPCConnectionError as e:
            if self.process and self.process.poll() is not None:
                self.state = ProcessState.FAILED
                logger.error(
                    "[PH-STREAM] FAILED (process dead) anima=%s method=%s chunks=%d error=%s",
                    self.anima_name,
                    method,
                    chunk_count,
                    e,
                )
            else:
                logger.warning(
                    "[PH-STREAM] IPC error (process alive) anima=%s method=%s chunks=%d error=%s",
                    self.anima_name,
                    method,
                    chunk_count,
                    e,
                )
            raise
        finally:
            async with self._streaming_lock:
                elapsed = (
                    (now_local() - ensure_aware(self._streaming_started_at)).total_seconds()
                    if self._streaming_started_at
                    else 0
                )
                self._streaming = False
                self._streaming_started_at = None
            logger.info(
                "[PH-STREAM] end anima=%s method=%s chunks=%d elapsed=%.1fs",
                self.anima_name,
                method,
                chunk_count,
                elapsed,
            )

    async def ping(
        self,
        timeout: float = 5.0,
        *,
        return_details: bool = False,
    ) -> bool | dict[str, Any]:
        """
        Send ping to check if process is alive.

        Returns:
            True if pong received, False otherwise
        """
        if self.state != ProcessState.RUNNING:
            self.stats.missed_pings += 1
            return {"success": False, "is_busy": False} if return_details else False

        try:
            response = await self.send_request("ping", {}, timeout=timeout)
            if response.error:
                logger.warning("Ping failed for %s: %s", self.anima_name, response.error)
                self.stats.missed_pings += 1
                return {"success": False, "is_busy": False} if return_details else False

            result = response.result or {}
            self.stats.last_ping_at = now_local()
            self.stats.missed_pings = 0
            is_busy = bool(result.get("is_busy", False))
            if return_details:
                return {
                    "success": True,
                    "is_busy": is_busy,
                    "last_progress_at": result.get("last_progress_at"),
                }
            return True

        except TimeoutError:
            logger.warning("Ping timeout for %s", self.anima_name)
            self.stats.missed_pings += 1
            return {"success": False, "is_busy": False} if return_details else False
        except Exception as e:
            is_transport_error = "Unix IPC transport is unavailable" in str(e)
            if not is_transport_error:
                logger.error("Ping error for %s: %s", self.anima_name, e)
            else:
                logger.debug("Ping transport error for %s: %s", self.anima_name, e)
            self.stats.missed_pings += 1
            if return_details:
                return {"success": False, "is_busy": False, "transport_error": is_transport_error}
            return False

    async def _drain_active_stream(self, timeout: float) -> None:
        """Wait for an in-flight interactive stream to finish before stopping.

        Protects a user-facing chat response from being cut off mid-generation
        by a graceful stop (rolling restart, RAG repair, reconcile). Bounded by
        ``timeout``; if the stream does not finish in time (e.g. a genuinely
        stuck turn) the stop proceeds anyway. Hang recovery uses ``kill()``
        directly and never reaches this path.
        """
        if timeout <= 0 or not self.is_streaming:
            return
        started = self.streaming_started_at
        logger.info(
            "Draining in-flight stream before stop: anima=%s started_at=%s timeout=%.0fs",
            self.anima_name,
            started.isoformat() if started else None,
            timeout,
        )
        try:
            async with asyncio.timeout(timeout):
                while self.is_streaming and self.process and self.process.poll() is None:
                    await asyncio.sleep(0.2)
        except TimeoutError:
            logger.warning(
                "Stream drain timed out for %s after %.0fs; proceeding with stop",
                self.anima_name,
                timeout,
            )
            return
        logger.info("In-flight stream drained for %s; proceeding with stop", self.anima_name)

    async def stop(
        self,
        timeout: float = 10.0,
        *,
        drain_streams: bool = True,
        drain_timeout: float | None = None,
    ) -> None:
        """
        Stop the child process gracefully.

        Shutdown flow:
        0. Drain any in-flight interactive stream (bounded by ``drain_timeout``)
        1. Send IPC shutdown command (wait 5s)
        2. If not exited, send SIGTERM (wait timeout/2)
        3. If still not exited, send SIGKILL

        Args:
            timeout: Total timeout in seconds for graceful shutdown
            drain_streams: When True (default), wait for an in-flight
                user-facing chat stream to finish before stopping so a rolling
                restart / RAG repair / reconcile does not cut a response off
                mid-generation. Hang recovery uses ``kill()`` directly and
                never reaches this path.
            drain_timeout: Upper bound for the stream drain (defaults to
                :data:`DEFAULT_STREAM_DRAIN_TIMEOUT_SEC`).
        """
        if self.state in (ProcessState.STOPPED, ProcessState.FAILED):
            if self.process and self.process.poll() is None:
                logger.warning(
                    "Process %s in %s state but still alive (PID %s), forcing stop",
                    self.anima_name,
                    self.state.value,
                    self.process.pid,
                )
            else:
                logger.debug("Process already stopped: %s", self.anima_name)
                await self._cleanup()
                return

        logger.info("Stopping process: %s", self.anima_name)

        if drain_streams:
            await self._drain_active_stream(
                drain_timeout if drain_timeout is not None else DEFAULT_STREAM_DRAIN_TIMEOUT_SEC
            )

        if not self.process:
            self.state = ProcessState.STOPPED
            await self._cleanup()
            return

        # Step 1: Snapshot descendant PIDs BEFORE sending shutdown.
        # If the runner exits quickly, we can still find and kill its
        # children (CLI subprocesses like claude, codex, cursor-agent).
        child_pids: list[int] = []
        try:
            parent = psutil.Process(self.process.pid)
            child_pids = [c.pid for c in parent.children(recursive=True)]
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

        # Step 2: Send IPC shutdown request BEFORE changing state
        # (send_request requires state == RUNNING)
        if self.state == ProcessState.RUNNING:
            try:
                logger.debug("Sending shutdown request to %s", self.anima_name)
                await self.send_request("shutdown", {}, timeout=5.0)
            except Exception as e:
                logger.warning("Shutdown request failed for %s: %s", self.anima_name, e)

        self.state = ProcessState.STOPPING
        self.stopping_since = now_local()

        # Step 3: Wait for graceful exit
        # NOTE: self.process may become None if the process was cleared by a
        # concurrent coroutine (e.g. during the await in send_request above).
        if self.process is None:
            self.state = ProcessState.STOPPED
            self.stats.stopped_at = now_local()
            await self._cleanup()
            return

        try:
            grace_period = min(timeout / 2, 5.0)
            async with asyncio.timeout(grace_period):
                while self.process and self.process.poll() is None:
                    await asyncio.sleep(0.1)

            self.stats.exit_code = self.process.returncode if self.process else None
            logger.info("Process exited gracefully: %s (code=%s)", self.anima_name, self.stats.exit_code)

            # Step 3b: Runner exited but children may still be alive.
            # Kill any orphaned descendants (CLI subprocesses).
            self._kill_orphaned_children(child_pids)

        except TimeoutError:
            # Step 4: Send SIGTERM to process session group
            logger.warning("Process did not exit gracefully, sending SIGTERM: %s", self.anima_name)
            try:
                if self.process is None:
                    raise TimeoutError  # skip to STOPPED
                terminate_subprocess(self.process, force=False)
                async with asyncio.timeout(timeout / 2):
                    while self.process and self.process.poll() is None:
                        await asyncio.sleep(0.1)

                self.stats.exit_code = self.process.returncode
                logger.info("Process terminated: %s (code=%s)", self.anima_name, self.stats.exit_code)

            except TimeoutError:
                # Step 5: Force SIGKILL
                logger.error("Process did not respond to SIGTERM, sending SIGKILL: %s", self.anima_name)
                await self.kill()

        self.state = ProcessState.STOPPED
        self.stats.stopped_at = now_local()
        await self._cleanup()

    def _kill_orphaned_children(self, child_pids: list[int]) -> None:
        """Kill previously-snapshotted children that are still alive.

        After the runner process exits, its CLI subprocesses (claude, codex,
        cursor-agent, gemini) may still be running because:
        - On Windows ``CREATE_NEW_PROCESS_GROUP`` does not auto-kill children
        - The runner's asyncio task cancellation may not reach the executor
          subprocess in time

        This method ensures those orphans are cleaned up.
        """
        if not child_pids:
            return

        for pid in child_pids:
            try:
                proc = psutil.Process(pid)
                if not proc.is_running() or proc.status() == psutil.STATUS_ZOMBIE:
                    continue
                # Also kill any grandchildren this process spawned
                for grandchild in proc.children(recursive=True):
                    try:
                        logger.info(
                            "Killing orphaned grandchild of %s: PID %d (%s)",
                            self.anima_name,
                            grandchild.pid,
                            grandchild.name(),
                        )
                        grandchild.kill()
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
                logger.info(
                    "Killing orphaned child of %s: PID %d (%s)",
                    self.anima_name,
                    pid,
                    proc.name(),
                )
                proc.kill()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass

    async def kill(self) -> None:
        """Force kill the process with SIGKILL."""
        if not self.process:
            return

        logger.warning("Killing process: %s (PID %s)", self.anima_name, self.process.pid)
        terminate_subprocess(self.process, force=True)
        await asyncio.get_running_loop().run_in_executor(None, self.process.wait)
        self.stats.exit_code = self.process.returncode
        self.state = ProcessState.FAILED
        await self._cleanup()

    async def _cleanup(self) -> None:
        """Clean up resources (including killing orphaned subprocesses)."""
        if self.process:
            if self.process.poll() is None:
                # Still alive — kill and wait
                logger.warning(
                    "Killing orphaned subprocess: %s (PID %s)",
                    self.anima_name,
                    self.process.pid,
                )
                terminate_subprocess(self.process, force=False)
                try:
                    self.process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    terminate_subprocess(self.process, force=True)
                    self.process.wait()
            else:
                # Already exited — explicitly reap to prevent zombie
                try:
                    self.process.wait(timeout=1)
                except (subprocess.TimeoutExpired, ChildProcessError):
                    pass
            self.process = None

        if self.ipc_client:
            try:
                await self.ipc_client.close()
            except OSError:
                logger.debug("IPC close error during cleanup", exc_info=True)
            self.ipc_client = None

        if self._stderr_file:
            try:
                self._stderr_file.close()
            except OSError:
                logger.debug("Failed to close stderr file", exc_info=True)
            self._stderr_file = None

        if self.socket_path.exists():
            self.socket_path.unlink()
            logger.debug("Socket file removed: %s", self.socket_path)

    def is_alive(self) -> bool:
        """Check if process is alive."""
        if not self.process:
            return False
        if self.process.poll() is not None:  # noqa: SIM103
            return False
        return True

    def get_pid(self) -> int | None:
        """Get process PID."""
        return self.process.pid if self.process else None
