from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Client-side lifecycle manager for the isolated RAG vector worker."""

import asyncio
import logging
import os
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from core.platform.process import subprocess_session_kwargs

logger = logging.getLogger("animaworks.rag.vector_worker")


@dataclass(frozen=True)
class VectorWorkerResponse:
    status_code: int
    data: dict[str, Any]


class VectorWorkerUnavailable(RuntimeError):
    """Raised when the vector worker cannot serve a request."""


@dataclass
class TemporaryVectorWorker:
    """Synchronous handle for one-shot CLI worker isolation."""

    manager: VectorWorkerManager
    previous_vector_url: str | None
    had_previous_vector_url: bool

    def stop(self) -> None:
        """Stop the temporary worker and restore the caller's vector URL."""
        try:
            asyncio.run(self.manager.stop())
        finally:
            if self.had_previous_vector_url and self.previous_vector_url is not None:
                os.environ["ANIMAWORKS_VECTOR_URL"] = self.previous_vector_url
            else:
                os.environ.pop("ANIMAWORKS_VECTOR_URL", None)


class VectorWorkerManager:
    """Starts, stops, and proxies requests to the isolated vector worker."""

    def __init__(
        self,
        *,
        enabled: bool,
        host: str,
        port: int,
        log_dir: Path,
        startup_timeout: float = 10.0,
        request_timeout: float = 30.0,
        restart_backoff: float = 2.0,
        shutdown_timeout: float = 30.0,
        fallback_direct: bool = True,
    ) -> None:
        self.enabled = enabled
        self.host = host
        self.port = port
        self.log_dir = log_dir
        self.startup_timeout = startup_timeout
        self.request_timeout = request_timeout
        self.restart_backoff = restart_backoff
        self.shutdown_timeout = shutdown_timeout
        self.fallback_direct = fallback_direct
        self.process: subprocess.Popen[bytes] | None = None
        self.base_url: str | None = None
        self.native_crash_detected = False
        self._lock = asyncio.Lock()
        self._last_start_attempt = 0.0

    @classmethod
    def from_config(cls, config: Any, *, log_dir: Path) -> VectorWorkerManager:
        rag = config.rag
        return cls(
            enabled=bool(getattr(rag, "vector_worker_enabled", True)),
            host=str(getattr(rag, "vector_worker_host", "127.0.0.1")),
            port=int(getattr(rag, "vector_worker_port", 0)),
            log_dir=log_dir,
            startup_timeout=float(getattr(rag, "vector_worker_startup_timeout_seconds", 10.0)),
            request_timeout=float(getattr(rag, "vector_worker_request_timeout_seconds", 30.0)),
            restart_backoff=float(getattr(rag, "vector_worker_restart_backoff_seconds", 2.0)),
            shutdown_timeout=float(getattr(rag, "vector_worker_shutdown_timeout_seconds", 30.0)),
            fallback_direct=bool(getattr(rag, "vector_worker_fallback_direct", False)),
        )

    async def start(self) -> None:
        if not self.enabled:
            return
        try:
            await self._ensure_running()
        except Exception:
            logger.exception("Vector worker failed to start")

    async def stop(self) -> None:
        proc = self.process
        self.process = None
        self.base_url = None
        if proc is None or proc.poll() is not None:
            return
        proc.terminate()
        try:
            await asyncio.to_thread(proc.wait, timeout=self.shutdown_timeout)
        except subprocess.TimeoutExpired:
            logger.warning(
                "Vector worker did not exit after %.1fs; sending SIGKILL",
                self.shutdown_timeout,
            )
            proc.kill()
            await asyncio.to_thread(proc.wait, timeout=5)

    async def post(self, path: str, payload: dict[str, Any]) -> VectorWorkerResponse:
        if not self.enabled:
            raise VectorWorkerUnavailable("vector worker disabled")
        await self._ensure_running(payload=payload)
        assert self.base_url is not None
        path = "/" + path.lstrip("/")
        try:
            async with httpx.AsyncClient(base_url=self.base_url, timeout=self.request_timeout) as client:
                resp = await client.post(path, json=payload)
        except httpx.RequestError as exc:
            self._record_crash_if_exited(payload)
            raise VectorWorkerUnavailable(str(exc)) from exc
        try:
            data = resp.json()
        except ValueError:
            data = {"detail": resp.text}
        return VectorWorkerResponse(
            status_code=resp.status_code, data=data if isinstance(data, dict) else {"data": data}
        )

    async def _ensure_running(self, *, payload: dict[str, Any] | None = None) -> None:
        if self._is_running():
            return
        async with self._lock:
            if self._is_running():
                return
            self._record_crash_if_exited(payload or {})
            now = time.monotonic()
            if now - self._last_start_attempt < self.restart_backoff:
                raise VectorWorkerUnavailable("vector worker restart backoff active")
            self._last_start_attempt = now
            await self._start_process()

    def _is_running(self) -> bool:
        return self.process is not None and self.process.poll() is None and self.base_url is not None

    async def _start_process(self) -> None:
        from core.paths import PROJECT_DIR

        port = self.port or _choose_free_port(self.host)
        self.base_url = f"http://{self.host}:{port}"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        log_path = self.log_dir / "vector-worker.log"
        log_file = open(log_path, "a", encoding="utf-8")  # noqa: SIM115
        env = os.environ.copy()
        env.pop("ANIMAWORKS_VECTOR_URL", None)
        env["ANIMAWORKS_ALLOW_DIRECT_CHROMA"] = "1"
        cmd = [
            sys.executable,
            "-m",
            "core.memory.rag.vector_worker",
            "--host",
            self.host,
            "--port",
            str(port),
        ]
        try:
            self.process = subprocess.Popen(
                cmd,
                cwd=PROJECT_DIR,
                env=env,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                **subprocess_session_kwargs(),
            )
        finally:
            log_file.close()
        try:
            await self._wait_until_healthy()
        except Exception:
            await self.stop()
            raise
        self.native_crash_detected = False
        logger.info("Vector worker started: pid=%s url=%s log=%s", self.process.pid, self.base_url, log_path)

    async def _wait_until_healthy(self) -> None:
        assert self.base_url is not None
        deadline = time.monotonic() + self.startup_timeout
        last_error: Exception | None = None
        while time.monotonic() < deadline:
            if self.process is not None and self.process.poll() is not None:
                raise VectorWorkerUnavailable(f"vector worker exited early: {self.process.returncode}")
            try:
                async with httpx.AsyncClient(timeout=1.0) as client:
                    resp = await client.get(f"{self.base_url}/health")
                if resp.status_code == 200:
                    return
            except httpx.RequestError as exc:
                last_error = exc
            await asyncio.sleep(0.2)
        raise VectorWorkerUnavailable(f"vector worker did not become healthy: {last_error}")

    def _record_crash_if_exited(self, payload: dict[str, Any]) -> None:
        proc = self.process
        if proc is None or proc.poll() is None:
            return
        returncode = proc.returncode
        self.process = None
        self.base_url = None
        self.native_crash_detected = returncode in {-11, 139}
        if not self.native_crash_detected:
            return
        try:
            from core.memory.rag.repair import record_chroma_error

            record_chroma_error(
                anima_name=payload.get("anima_name"),
                collection=str(payload.get("collection") or "<vector_worker>"),
                error=-11,
                source="vector_worker",
            )
        except Exception:
            logger.debug("Failed to record vector worker crash", exc_info=True)


def _choose_free_port(host: str) -> int:
    bind_host = "127.0.0.1" if host == "0.0.0.0" else host
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((bind_host, 0))
        return int(sock.getsockname()[1])


def start_temporary_vector_worker(
    *,
    config: Any | None = None,
    log_dir: Path | None = None,
) -> TemporaryVectorWorker:
    """Start a short-lived vector worker and point this process at it."""
    if config is None:
        from core.config import load_config

        config = load_config()
    if log_dir is None:
        from core.paths import get_data_dir

        log_dir = get_data_dir() / "logs"

    manager = VectorWorkerManager.from_config(config, log_dir=log_dir)
    if not manager.enabled:
        raise VectorWorkerUnavailable("vector worker disabled")

    asyncio.run(manager._ensure_running(payload={}))  # noqa: SLF001
    if manager.base_url is None:
        raise VectorWorkerUnavailable("vector worker did not provide a base URL")

    had_previous = "ANIMAWORKS_VECTOR_URL" in os.environ
    previous = os.environ.get("ANIMAWORKS_VECTOR_URL")
    os.environ["ANIMAWORKS_VECTOR_URL"] = manager.base_url
    logger.info("Using temporary vector worker: pid=%s url=%s", getattr(manager.process, "pid", None), manager.base_url)
    return TemporaryVectorWorker(
        manager=manager,
        previous_vector_url=previous,
        had_previous_vector_url=had_previous,
    )
