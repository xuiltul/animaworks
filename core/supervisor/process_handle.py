"""
Process handle for managing child Anima processes.
"""

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import logging
import subprocess
import sys
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from core.time_utils import ensure_aware, now_jst
from io import TextIOWrapper
from pathlib import Path
from typing import Any

from core.exceptions import IPCConnectionError as IPCConnectionErr  # noqa: F401
from core.supervisor.ipc import IPCClient, IPCRequest, IPCResponse

logger = logging.getLogger(__name__)


# ── Process State ──────────────────────────────────────────────────

class ProcessState(Enum):
    """State of a child process."""
    STARTING = "starting"       # Process spawned, waiting for socket
    RUNNING = "running"          # Process running, socket connected
    STOPPING = "stopping"        # Shutdown requested
    STOPPED = "stopped"          # Process exited normally
    FAILED = "failed"            # Process crashed or killed
    RESTARTING = "restarting"    # Auto-restart in progress


@dataclass
class ProcessStats:
    """Process statistics."""
    started_at: datetime
    stopped_at: datetime | None = None
    restart_count: int = 0
    last_ping_at: datetime | None = None
    missed_pings: int = 0
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
        log_dir: Path | None = None
    ):
        self.anima_name = anima_name
        self.socket_path = socket_path
        self.animas_dir = animas_dir
        self.shared_dir = shared_dir
        self.log_dir = log_dir

        self.state = ProcessState.STOPPED
        self.process: subprocess.Popen | None = None
        self.ipc_client: IPCClient | None = None
        self.stats = ProcessStats(started_at=now_jst())
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
            raise RuntimeError(f"Cannot start process in state {self.state}")

        self.state = ProcessState.STARTING
        self.stats = ProcessStats(started_at=now_jst())

        # Spawn child process
        cmd = [
            sys.executable,
            "-m", "core.supervisor.runner",
            "--anima-name", self.anima_name,
            "--socket-path", str(self.socket_path),
            "--animas-dir", str(self.animas_dir),
            "--shared-dir", str(self.shared_dir),
            "--log-dir", str(self.log_dir) if self.log_dir else "/tmp"
        ]

        logger.info("Starting process: %s", self.anima_name)
        logger.debug("Command: %s", " ".join(cmd))

        try:
            # Redirect stderr to a log file for post-mortem debugging;
            # stdout is discarded because the child writes its own log files.
            stderr_path: Path | None = None
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

            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=self._stderr_file if self._stderr_file else subprocess.DEVNULL,
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
            await self._wait_for_ready(timeout=120.0)

            self.state = ProcessState.RUNNING
            logger.info("Process running: %s", self.anima_name)

        except Exception as e:
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
                raise RuntimeError(
                    f"Process '{self.anima_name}' exited with code "
                    f"{self.process.returncode} before creating socket"
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
                raise RuntimeError(
                    f"Process '{self.anima_name}' exited with code "
                    f"{self.process.returncode} during initialization"
                )

            try:
                request = IPCRequest(
                    id=f"ping_{uuid.uuid4().hex[:8]}",
                    method="ping",
                    params={}
                )
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

        raise TimeoutError(
            f"Anima '{self.anima_name}' not ready within {timeout}s"
        )

    async def send_request(
        self,
        method: str,
        params: dict,
        timeout: float = 60.0
    ) -> IPCResponse:
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
            raise RuntimeError(f"Process restarting: {self.anima_name}")
        if self.state != ProcessState.RUNNING:
            raise RuntimeError(f"Process not running: {self.state}")

        if not self.ipc_client:
            raise RuntimeError("IPC client not connected")

        request = IPCRequest(
            id=f"req_{uuid.uuid4().hex[:8]}",
            method=method,
            params=params
        )

        try:
            return await self.ipc_client.send_request(request, timeout=timeout)
        except RuntimeError:
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
            raise RuntimeError(f"Process restarting: {self.anima_name}")
        if self.state != ProcessState.RUNNING:
            raise RuntimeError(f"Process not running: {self.state}")

        if not self.ipc_client:
            raise RuntimeError("IPC client not connected")

        request = IPCRequest(
            id=f"req_{uuid.uuid4().hex[:8]}",
            method=method,
            params=params
        )

        async with self._streaming_lock:
            self._streaming = True
            self._streaming_started_at = now_jst()
        logger.info(
            "[PH-STREAM] start anima=%s method=%s req_id=%s state=%s pid=%s",
            self.anima_name, method, request.id, self.state.value,
            self.process.pid if self.process else "N/A",
        )
        chunk_count = 0
        try:
            async for response in self.ipc_client.send_request_stream(
                request, timeout=timeout
            ):
                chunk_count += 1
                yield response
        except RuntimeError as e:
            logger.info(
                "[PH-STREAM] FAILED anima=%s method=%s chunks=%d error=%s",
                self.anima_name, method, chunk_count, e,
            )
            self.state = ProcessState.FAILED
            raise
        finally:
            async with self._streaming_lock:
                elapsed = (now_jst() - ensure_aware(self._streaming_started_at)).total_seconds() if self._streaming_started_at else 0
                self._streaming = False
                self._streaming_started_at = None
            logger.info(
                "[PH-STREAM] end anima=%s method=%s chunks=%d elapsed=%.1fs",
                self.anima_name, method, chunk_count, elapsed,
            )

    async def ping(self, timeout: float = 5.0) -> bool:
        """
        Send ping to check if process is alive.

        Returns:
            True if pong received, False otherwise
        """
        if self.state != ProcessState.RUNNING:
            self.stats.missed_pings += 1
            return False

        try:
            response = await self.send_request("ping", {}, timeout=timeout)
            if response.error:
                logger.warning("Ping failed for %s: %s", self.anima_name, response.error)
                self.stats.missed_pings += 1
                return False

            self.stats.last_ping_at = now_jst()
            self.stats.missed_pings = 0
            return True

        except asyncio.TimeoutError:
            logger.warning("Ping timeout for %s", self.anima_name)
            self.stats.missed_pings += 1
            return False
        except Exception as e:
            logger.error("Ping error for %s: %s", self.anima_name, e)
            self.stats.missed_pings += 1
            return False

    async def stop(self, timeout: float = 10.0) -> None:
        """
        Stop the child process gracefully.

        Shutdown flow:
        1. Send IPC shutdown command (wait 5s)
        2. If not exited, send SIGTERM (wait timeout/2)
        3. If still not exited, send SIGKILL

        Args:
            timeout: Total timeout in seconds for graceful shutdown
        """
        if self.state in (ProcessState.STOPPED, ProcessState.FAILED):
            if self.process and self.process.poll() is None:
                logger.warning(
                    "Process %s in %s state but still alive (PID %s), forcing stop",
                    self.anima_name, self.state.value, self.process.pid,
                )
            else:
                logger.debug("Process already stopped: %s", self.anima_name)
                await self._cleanup()
                return

        logger.info("Stopping process: %s", self.anima_name)

        if not self.process:
            self.state = ProcessState.STOPPED
            await self._cleanup()
            return

        # Step 1: Send IPC shutdown request BEFORE changing state
        # (send_request requires state == RUNNING)
        if self.state == ProcessState.RUNNING:
            try:
                logger.debug("Sending shutdown request to %s", self.anima_name)
                await self.send_request("shutdown", {}, timeout=5.0)
            except Exception as e:
                logger.warning("Shutdown request failed for %s: %s", self.anima_name, e)

        self.state = ProcessState.STOPPING
        self.stopping_since = now_jst()

        # Step 2: Wait for graceful exit
        try:
            grace_period = min(timeout / 2, 5.0)
            async with asyncio.timeout(grace_period):
                while self.process.poll() is None:
                    await asyncio.sleep(0.1)

            self.stats.exit_code = self.process.returncode
            logger.info("Process exited gracefully: %s (code=%s)", self.anima_name, self.stats.exit_code)

        except asyncio.TimeoutError:
            # Step 3: Send SIGTERM
            logger.warning("Process did not exit gracefully, sending SIGTERM: %s", self.anima_name)
            try:
                self.process.terminate()
                async with asyncio.timeout(timeout / 2):
                    while self.process.poll() is None:
                        await asyncio.sleep(0.1)

                self.stats.exit_code = self.process.returncode
                logger.info("Process terminated: %s (code=%s)", self.anima_name, self.stats.exit_code)

            except asyncio.TimeoutError:
                # Step 4: Force SIGKILL
                logger.error("Process did not respond to SIGTERM, sending SIGKILL: %s", self.anima_name)
                await self.kill()

        self.state = ProcessState.STOPPED
        self.stats.stopped_at = now_jst()
        await self._cleanup()

    async def kill(self) -> None:
        """Force kill the process with SIGKILL."""
        if not self.process:
            return

        logger.warning("Killing process: %s (PID %s)", self.anima_name, self.process.pid)
        self.process.kill()
        await asyncio.get_running_loop().run_in_executor(None, self.process.wait)
        self.stats.exit_code = self.process.returncode
        self.state = ProcessState.FAILED
        await self._cleanup()

    async def _cleanup(self) -> None:
        """Clean up resources (including killing orphaned subprocesses)."""
        # Kill subprocess if still alive
        if self.process and self.process.poll() is None:
            logger.warning(
                "Killing orphaned subprocess: %s (PID %s)",
                self.anima_name, self.process.pid,
            )
            self.process.terminate()
            try:
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()

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
        if self.process.poll() is not None:
            return False
        return True

    def get_pid(self) -> int | None:
        """Get process PID."""
        return self.process.pid if self.process else None
