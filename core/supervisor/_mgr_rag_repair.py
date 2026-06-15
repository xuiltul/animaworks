"""
Supervised RAG repair mixin for ProcessSupervisor.
"""

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

from core.memory.rag import repair_state

logger = logging.getLogger(__name__)


class RAGRepairMixin:
    """Supervisor-owned RAG repair lifecycle helpers."""

    async def _maybe_repair_rag_before_restart(
        self,
        anima_name: str,
        handle,
    ) -> bool:
        """Run RAG repair before restart when failure correlates with RAG corruption."""
        try:
            from core.memory.rag.repair import classify_corruption_error, get_repair_service

            service = get_repair_service()
            reason = classify_corruption_error(handle.stats.exit_code)
            if reason is None and service.has_recent_corruption(anima_name):
                reason = "recent_rag_corruption"
            if reason is None:
                return False

            # Evidence-based suspicion ("recent_rag_corruption") can be a
            # transient cache-poisoning false positive: the DB raised sqlite-level
            # corruption errors ("Failed to get segments", "file is not a
            # database") while the on-disk store stayed intact. Quarantining and
            # rebuilding a healthy DB on that soft evidence is wasted work that
            # loads the vector worker and feeds the repair churn. If quick_check
            # confirms the SQLite store is intact, skip the repair and just
            # restart. Hard signals (native_segfault from the exit code) are NOT
            # gated — a crash in native chroma code warrants a rebuild even when
            # SQLite looks fine (the fault may be in an hnsw segment).
            if reason == "recent_rag_corruption":
                from core.memory.rag.sqlite_health import quick_check_chroma_sqlite

                persist_dir = self.animas_dir / anima_name / "vectordb"
                if quick_check_chroma_sqlite(persist_dir).status == "ok":
                    logger.info(
                        "Skipping pre-restart RAG repair; on-disk SQLite passed quick_check "
                        "(transient cache poisoning, no rebuild needed): anima=%s",
                        anima_name,
                    )
                    return False

            logger.warning(
                "RAG corruption suspected before restart: anima=%s reason=%s exit_code=%s",
                anima_name,
                reason,
                handle.stats.exit_code,
            )
            self._write_rag_repair_state(
                anima_name,
                {
                    "status": "repairing",
                    "stage": "repair_process",
                    "reason": reason,
                    "source": "supervisor",
                    "include_shared": True,
                    "last_error": None,
                },
            )
            result = await self._run_rag_repair_cli_process(
                anima_name,
                reason=reason,
                include_shared=True,
            )
            if result["ok"]:
                self._restart_counts[anima_name] = 0
                self._write_rag_repair_state(
                    anima_name,
                    {
                        "status": "success",
                        "stage": "restart_anima",
                        "pid": None,
                        "reason": reason,
                        "source": "supervisor",
                        "include_shared": True,
                        "last_error": None,
                    },
                )
                try:
                    await self._broadcast_event(
                        "system.rag_repair",
                        {
                            "anima": anima_name,
                            "status": "success",
                            "reason": reason,
                        },
                    )
                except Exception:
                    logger.debug("Failed to broadcast rag_repair event", exc_info=True)
                return True

            self._write_rag_repair_state(
                anima_name,
                {
                    "status": "failed",
                    "stage": "failed",
                    "pid": None,
                    "reason": reason,
                    "source": "supervisor",
                    "include_shared": True,
                    "last_error": result["error"],
                },
            )
            logger.warning(
                "RAG repair did not complete before restart: anima=%s status=%s error=%s",
                anima_name,
                result["status"],
                result["error"],
            )
            return False
        except ImportError:
            return False
        except Exception:
            logger.exception("RAG repair check failed before restart: %s", anima_name)
            return False

    async def _poll_requested_rag_repairs(self) -> None:
        """Start supervised RAG repairs requested by anima processes."""
        now = asyncio.get_running_loop().time()
        interval = self._rag_repair_poll_interval_seconds()
        last = getattr(self, "_last_rag_repair_poll_at", 0.0)
        if now - last < interval:
            return
        self._last_rag_repair_poll_at = now

        in_progress: set[str] = getattr(self, "_rag_repairs_in_progress", set())
        max_concurrent = self._rag_repair_max_concurrent()
        for anima_dir in sorted(self.animas_dir.iterdir() if self.animas_dir.exists() else []):
            if not anima_dir.is_dir():
                continue
            # Cap concurrent repairs: the vector worker is single-threaded, so many
            # rebuilds at once saturate it, make reindex upserts fail, and leave
            # stub DBs that re-trigger repair. Remaining "requested" animas are
            # picked up on later polls, one (or few) at a time.
            if len(in_progress) >= max_concurrent:
                break
            anima_name = anima_dir.name
            if anima_name in in_progress or anima_name in self._restarting:
                continue
            state = self._read_rag_repair_state(anima_name)
            if state.get("status") == "requested":
                in_progress.add(anima_name)
                self._rag_repairs_in_progress = in_progress
                asyncio.create_task(self._run_supervised_rag_repair(anima_name, state))

    async def _run_supervised_rag_repair(self, anima_name: str, state: dict[str, object]) -> None:
        """Stop one anima, repair its RAG DB in a CLI subprocess, then restart it.

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
            if not await self._stop_anima_for_rag_repair(anima_name, reason, include_shared):
                return

            result = await self._run_rag_repair_step(anima_name, reason, include_shared)
            if not result["ok"]:
                await self._handle_failed_rag_repair(anima_name, reason, include_shared, result)
                return

            await self._restart_after_rag_repair(anima_name, reason, include_shared)
        finally:
            in_progress.discard(anima_name)

    async def _stop_anima_for_rag_repair(self, anima_name: str, reason: str, include_shared: bool) -> bool:
        try:
            self._write_rag_repair_state(
                anima_name,
                {
                    "status": "stopping",
                    "stage": "stop_anima",
                    "pid": None,
                    "reason": reason,
                    "include_shared": include_shared,
                    "last_error": None,
                },
            )
            if anima_name in self.processes:
                await self.stop_anima(anima_name)
            return True
        except Exception as exc:
            self._write_rag_repair_state(
                anima_name,
                {
                    "status": "failed",
                    "stage": "stop_anima",
                    "pid": None,
                    "reason": reason,
                    "include_shared": include_shared,
                    "last_error": str(exc),
                },
            )
            await self._broadcast_rag_repair_event(anima_name, "failed", reason, str(exc))
            return False

    async def _run_rag_repair_step(self, anima_name: str, reason: str, include_shared: bool) -> dict[str, object]:
        self._write_rag_repair_state(
            anima_name,
            {
                "status": "repairing",
                "stage": "repair_process",
                "pid": None,
                "reason": reason,
                "include_shared": include_shared,
                "last_error": None,
            },
        )
        return await self._run_rag_repair_cli_process(
            anima_name,
            reason=reason,
            include_shared=include_shared,
        )

    async def _handle_failed_rag_repair(
        self,
        anima_name: str,
        reason: str,
        include_shared: bool,
        result: dict[str, object],
    ) -> None:
        error = str(result["error"])
        self._write_rag_repair_state(
            anima_name,
            {
                "status": "failed",
                "stage": "failed",
                "pid": None,
                "reason": reason,
                "include_shared": include_shared,
                "last_error": error,
            },
        )
        await self._broadcast_rag_repair_event(anima_name, "failed", reason, error)
        if (self.animas_dir / anima_name / "vectordb").exists():
            try:
                await self.start_anima(anima_name)
            except Exception:
                logger.debug("Failed to restart %s after failed RAG repair", anima_name, exc_info=True)

    async def _restart_after_rag_repair(self, anima_name: str, reason: str, include_shared: bool) -> None:
        self._write_rag_repair_state(
            anima_name,
            {
                "status": "success",
                "stage": "restart_anima",
                "pid": None,
                "reason": reason,
                "include_shared": include_shared,
                "last_error": None,
            },
        )
        try:
            await self.start_anima(anima_name)
        except Exception as exc:
            self._write_rag_repair_state(
                anima_name,
                {
                    "status": "repair_success_restart_failed",
                    "stage": "restart_anima",
                    "pid": None,
                    "reason": reason,
                    "include_shared": include_shared,
                    "last_error": str(exc),
                },
            )
            await self._broadcast_rag_repair_event(
                anima_name,
                "repair_success_restart_failed",
                reason,
                str(exc),
            )
            return

        self._write_rag_repair_state(
            anima_name,
            {
                "status": "healthy",
                "stage": "complete",
                "pid": None,
                "reason": reason,
                "include_shared": include_shared,
                "last_error": None,
            },
        )
        await self._broadcast_rag_repair_event(anima_name, "healthy", reason, None)

    async def _run_rag_repair_cli_process(
        self,
        anima_name: str,
        *,
        reason: str,
        include_shared: bool,
    ) -> dict[str, object]:
        cmd = [
            sys.executable,
            "-m",
            "cli",
            "repair-rag",
            "--anima",
            anima_name,
            "--full",
            "--reason",
            reason,
        ]
        if include_shared:
            cmd.append("--shared")
        timeout = self._rag_repair_timeout_seconds()
        repo_root = Path(__file__).resolve().parents[2]
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=repo_root,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        self._write_rag_repair_state(
            anima_name,
            {
                "status": "repairing",
                "stage": "repair_process",
                "pid": proc.pid,
                "started_at": repair_state.now_iso(),
                "reason": reason,
                "include_shared": include_shared,
                "last_error": None,
            },
        )
        try:
            stdout_b, stderr_b = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except TimeoutError:
            proc.kill()
            await proc.communicate()
            return {
                "ok": False,
                "status": "timeout",
                "error": f"repair-rag timed out after {timeout}s",
            }
        stdout = stdout_b.decode(errors="replace").strip()
        stderr = stderr_b.decode(errors="replace").strip()
        if proc.returncode == 0:
            return {"ok": True, "status": "success", "stdout": stdout, "stderr": stderr}
        return {
            "ok": False,
            "status": "failed",
            "stdout": stdout,
            "stderr": stderr,
            "error": stderr or stdout or f"repair-rag exited with code {proc.returncode}",
        }

    async def _broadcast_rag_repair_event(
        self,
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

    def _rag_repair_timeout_seconds(self) -> int:
        try:
            from core.config import load_config

            return int(getattr(load_config().rag, "repair_timeout_seconds", 1800))
        except Exception:
            return 1800

    def _rag_repair_poll_interval_seconds(self) -> float:
        try:
            from core.config import load_config

            return float(getattr(load_config().rag, "repair_poll_interval_seconds", 5))
        except Exception:
            return 5.0

    def _rag_repair_max_concurrent(self) -> int:
        try:
            from core.config import load_config

            return max(1, int(getattr(load_config().rag, "repair_max_concurrent", 1)))
        except Exception:
            return 1

    def _rag_repair_state_path(self, anima_name: str) -> Path:
        return repair_state.state_path(anima_name, animas_dir=self.animas_dir)

    def _read_rag_repair_state(self, anima_name: str) -> dict[str, object]:
        return repair_state.read_state(anima_name, animas_dir=self.animas_dir)

    def _write_rag_repair_state(self, anima_name: str, updates: dict[str, object]) -> None:
        repair_state.update_repair_state(
            anima_name,
            animas_dir=self.animas_dir,
            **updates,
        )
