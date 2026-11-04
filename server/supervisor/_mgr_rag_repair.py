"""
Supervised RAG repair mixin for ProcessSupervisor.
"""

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any, Protocol

from core.memory.rag.repair import state as repair_state
from core.platform.tasks import spawn

logger = logging.getLogger(__name__)


def _list_anima_dir_names(animas_dir: Path) -> list[str]:
    """List Anima directory names, sorted. Blocking: call off the loop."""
    if not animas_dir.exists():
        return []
    return sorted(path.name for path in animas_dir.iterdir() if path.is_dir())


class _RAGRepairHost(Protocol):
    """Structural supervisor members used by RAGRepairMixin."""

    _broadcast_event: Any
    _broadcast_rag_repair_event: Any
    _handle_failed_rag_repair: Any
    _last_rag_repair_poll_at: Any
    _rag_repair_max_concurrent: Any
    _rag_repair_poll_interval_seconds: Any
    _rag_repair_timeout_seconds: Any
    _rag_repairs_in_progress: Any
    _read_rag_repair_state: Any
    _restarting: Any
    _run_rag_repair_step: Any
    _run_supervised_rag_repair: Any
    _run_uninterrupted_rag_repair: Any
    _write_rag_repair_state: Any
    animas_dir: Any
    processes: Any
    send_request: Any


class RAGRepairMixin:
    """Supervisor-owned RAG repair lifecycle helpers."""

    async def _poll_requested_rag_repairs(self: _RAGRepairHost) -> None:
        """Start supervised RAG repairs requested by anima processes."""
        now = asyncio.get_running_loop().time()
        interval = self._rag_repair_poll_interval_seconds()
        last = getattr(self, "_last_rag_repair_poll_at", 0.0)
        if now - last < interval:
            return
        self._last_rag_repair_poll_at = now

        in_progress: set[str] = getattr(self, "_rag_repairs_in_progress", set())
        max_concurrent = self._rag_repair_max_concurrent()
        for anima_name in await asyncio.to_thread(_list_anima_dir_names, self.animas_dir):
            # Staging rebuilds are CPU/IO-heavy, so cap concurrent repair work.
            if len(in_progress) >= max_concurrent:
                break
            if anima_name in in_progress or anima_name in self._restarting:
                continue
            process = self.processes.get(anima_name)
            if process is None or not process.is_alive():
                continue
            state = await asyncio.to_thread(self._read_rag_repair_state, anima_name)
            # The await above can let a concurrent repair start for this Anima.
            if state.get("status") == "requested" and anima_name not in in_progress:
                in_progress.add(anima_name)
                self._rag_repairs_in_progress = in_progress
                spawn(
                    self._run_supervised_rag_repair(anima_name, state),
                    name=f"rag-repair-{anima_name}",
                )

    async def _run_supervised_rag_repair(self: _RAGRepairHost, anima_name: str, state: dict[str, object]) -> None:
        """Repair one anima's RAG DB through its root memory owner.

        The caller (``_poll_requested_rag_repairs``) has already reserved this
        anima in ``_rag_repairs_in_progress`` for concurrency accounting; this
        method only clears it again in ``finally``.
        """
        in_progress: set[str] = getattr(self, "_rag_repairs_in_progress", set())
        in_progress.add(anima_name)
        self._rag_repairs_in_progress = in_progress

        reason = str(state.get("reason") or "requested_rag_repair")
        include_shared = bool(state.get("include_shared", True))
        try:
            await self._run_uninterrupted_rag_repair(anima_name, reason, include_shared)
        finally:
            in_progress.discard(anima_name)

    async def _run_uninterrupted_rag_repair(
        self: _RAGRepairHost,
        anima_name: str,
        reason: str,
        include_shared: bool,
    ) -> None:
        """Fence RAG access while repairing without stopping the anima process."""
        repair_succeeded = False
        await asyncio.to_thread(
            self._write_rag_repair_state,
            anima_name,
            {
                "status": "repairing",
                "stage": repair_state.STAGE_FENCE_ACCESS,
                "pid": None,
                "reason": reason,
                "include_shared": include_shared,
                "last_error": None,
            },
        )
        try:
            result = await self._run_rag_repair_step(
                anima_name,
                reason,
                include_shared,
                stage=repair_state.STAGE_REPAIR,
            )
            if not result["ok"]:
                await self._handle_failed_rag_repair(
                    anima_name,
                    reason,
                    include_shared,
                    result,
                )
                return
            repair_succeeded = True
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.exception("Uninterrupted RAG repair failed: %s", anima_name)
            await self._handle_failed_rag_repair(
                anima_name,
                reason,
                include_shared,
                {
                    "ok": False,
                    "status": "failed",
                    "error": str(exc),
                },
            )
        finally:
            current = await asyncio.to_thread(self._read_rag_repair_state, anima_name)
            status = "healthy" if repair_succeeded else str(current.get("status") or "failed")
            if status in repair_state.ACTIVE_REPAIR_STATUSES:
                status = "failed"
            await asyncio.to_thread(
                self._write_rag_repair_state,
                anima_name,
                {
                    "status": status,
                    "stage": repair_state.STAGE_UNFENCE,
                    "pid": None,
                    "reason": reason,
                    "include_shared": include_shared,
                    "repair_nonce": None,
                    "last_error": None if repair_succeeded else current.get("last_error"),
                },
            )
            if repair_succeeded:
                await self._broadcast_rag_repair_event(anima_name, "healthy", reason, None)

    async def _run_rag_repair_step(
        self: _RAGRepairHost,
        anima_name: str,
        reason: str,
        include_shared: bool,
        *,
        stage: str = "repair_process",
    ) -> dict[str, object]:
        await asyncio.to_thread(
            self._write_rag_repair_state,
            anima_name,
            {
                "status": "repairing",
                "stage": stage,
                "pid": None,
                "reason": reason,
                "include_shared": include_shared,
                "last_error": None,
            },
        )
        try:
            return await self.send_request(
                anima_name,
                "repair_memory",
                {"reason": reason, "include_shared": include_shared},
                timeout=float(self._rag_repair_timeout_seconds() + 30),
            )
        except Exception as exc:
            return {"ok": False, "status": "failed", "error": str(exc)}

    async def _handle_failed_rag_repair(
        self: _RAGRepairHost,
        anima_name: str,
        reason: str,
        include_shared: bool,
        result: dict[str, object],
    ) -> None:
        error = str(result["error"])
        await asyncio.to_thread(
            self._write_rag_repair_state,
            anima_name,
            {
                "status": "failed",
                "stage": "failed",
                "pid": None,
                "reason": reason,
                "include_shared": include_shared,
                "repair_nonce": None,
                "last_error": error,
            },
        )
        await self._broadcast_rag_repair_event(anima_name, "failed", reason, error)

    async def _broadcast_rag_repair_event(
        self: _RAGRepairHost,
        anima_name: str,
        status: str,
        reason: str,
        error: str | None,
    ) -> None:
        try:
            await self._broadcast_event(
                "system.rag_repair",
                {
                    "anima": anima_name,
                    "status": status,
                    "reason": reason,
                    "error": error,
                },
            )
        except Exception:
            logger.debug("Failed to broadcast rag_repair event", exc_info=True)

    def _rag_repair_timeout_seconds(self: _RAGRepairHost) -> int:
        try:
            from core.config import load_config

            return int(getattr(load_config().rag, "repair_timeout_seconds", 1800))
        except Exception:
            return 1800

    def _rag_repair_poll_interval_seconds(self: _RAGRepairHost) -> float:
        try:
            from core.config import load_config

            return float(getattr(load_config().rag, "repair_poll_interval_seconds", 5))
        except Exception:
            return 5.0

    def _rag_repair_max_concurrent(self: _RAGRepairHost) -> int:
        try:
            from core.config import load_config

            return max(1, int(getattr(load_config().rag, "repair_max_concurrent", 1)))
        except Exception:
            return 1

    def _read_rag_repair_state(self: _RAGRepairHost, anima_name: str) -> dict[str, object]:
        return repair_state.read_state(anima_name, animas_dir=self.animas_dir)

    def _write_rag_repair_state(self: _RAGRepairHost, anima_name: str, updates: dict[str, object]) -> None:
        repair_state.update_repair_state(
            anima_name,
            animas_dir=self.animas_dir,
            **updates,
        )
