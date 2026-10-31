"""
Reconciliation mixin for ProcessSupervisor.
"""

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import json
import logging
import time

from core.anima.workspace_files import IDENTITY_FILE
from server.supervisor._manager_protocols import _ReconcileMixinHost

logger = logging.getLogger(__name__)


class ReconcileMixin:
    """Reconciliation loop: syncs desired state (disk) with actual processes."""

    async def _reconciliation_loop(self: _ReconcileMixinHost) -> None:
        """Periodically reconcile desired state (disk) with actual state (processes)."""
        logger.info("Reconciliation loop started (interval=%.0fs)", self.reconciliation_config.interval_sec)

        while not self._shutdown:
            try:
                await asyncio.sleep(self.reconciliation_config.interval_sec)
                await self._reconcile()
                await self._flush_task_notices()
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("Reconciliation failed")

        logger.info("Reconciliation loop stopped")

    async def _flush_task_notices(self: _ReconcileMixinHost) -> None:
        """Send batched task board notices whose actor has gone quiet."""
        from core.tasks.board.notices import flush_task_notices

        try:
            await asyncio.to_thread(flush_task_notices)
        except Exception:
            logger.exception("Task notice flush failed")

    async def _reconcile(self: _ReconcileMixinHost) -> None:
        """Scan animas_dir and sync desired state with actual process state."""
        if self._shutdown:
            return
        self._check_config_freshness()

        if not self.animas_dir.exists():
            if self._restart_ctl is not None:
                for name in self._restart_ctl.names():
                    self._restart_ctl.forget(name)
            return

        running = set(self.processes.keys())

        # Build desired state from disk
        on_disk: dict[str, bool] = {}  # name -> enabled
        # Anima dirs that have IDENTITY.md but no status.json.
        # These are either legacy animas or factory-in-progress.
        # They must NOT be auto-started but must NOT be killed if running.
        on_disk_incomplete: set[str] = set()
        for anima_dir in sorted(self.animas_dir.iterdir()):
            if not anima_dir.is_dir():
                continue
            if not (anima_dir / IDENTITY_FILE).exists():
                continue
            # status.json is created as the final step of anima_factory.
            # Its absence means creation may still be in progress.
            if not (anima_dir / "status.json").exists():
                on_disk_incomplete.add(anima_dir.name)
                continue
            on_disk[anima_dir.name] = self.read_anima_enabled(anima_dir)

        # Drop restart state for animas that are disabled or have been removed
        # from disk, including those with no process to enter the stop loops.
        if self._restart_ctl is not None:
            for name in self._restart_ctl.names():
                if name in on_disk and on_disk[name]:
                    continue
                if name in on_disk_incomplete:
                    continue
                self._restart_ctl.forget(name)

        # Safety net: ensure a restart worker exists for every known restart
        # record whose anima is enabled but not running (covers server
        # restart and any lost worker). Recovery is driven by the unified
        # RestartController, not by a separate cooldown here.
        if self._restart_ctl is not None:
            for name in self._restart_ctl.names():
                if self._shutdown:
                    break
                if name in self.processes:
                    continue
                if name in self._restarting:
                    continue
                if name not in on_disk or not on_disk[name]:
                    continue
                if name in self._starting or name in self._bootstrapping:
                    continue
                self._ensure_restart_worker(name)

        # Update running set after recovery attempts
        running = set(self.processes.keys())

        # restart_requested フラグチェック
        rag_repairs_in_progress: set[str] = getattr(self, "_rag_repairs_in_progress", set())
        for name in list(on_disk.keys()):
            anima_dir = self.animas_dir / name
            from core.anima.settings_store import update_status
            from core.platform.status_store import read_status

            if not read_status(anima_dir).get("restart_requested"):
                continue
            if name in rag_repairs_in_progress:
                logger.info("Reconciliation: deferring restart for %s (RAG repair in progress)", name)
                continue
            # Clear the flag under the same lock as all status writers.
            restart_requested = False

            def clear_restart_request(status: dict[str, object]) -> None:
                nonlocal restart_requested
                restart_requested = bool(status.pop("restart_requested", None))

            try:
                update_status(anima_dir, clear_restart_request)
            except (json.JSONDecodeError, OSError, ValueError):
                logger.debug("Failed to clear restart_requested for %s", name, exc_info=True)
                continue
            if not restart_requested:
                continue
            logger.info("Reconciliation: restart_requested for %s, restarting", name)
            try:
                await self.restart_anima(name)
            except Exception:
                logger.exception("Reconciliation: failed to restart %s (restart_requested)", name)
            continue

        # Update running set after restart_requested handling
        running = set(self.processes.keys())

        # Evict stale entries from _recently_stopped (older than 30s)
        _now = time.monotonic()
        for _rs_name in list(getattr(self, "_recently_stopped", {})):
            if _now - self._recently_stopped[_rs_name] > 30.0:
                del self._recently_stopped[_rs_name]

        # enabled + not running → start
        for name, enabled in on_disk.items():
            if self._shutdown:
                break
            if enabled and name not in running:
                if name in self._restarting:
                    logger.debug("Reconciliation: skipping %s (restart in progress)", name)
                    continue
                if name in self._starting:
                    logger.debug("Reconciliation: skipping %s (start in progress)", name)
                    continue
                if name in self._bootstrapping:
                    logger.debug("Reconciliation: skipping %s (bootstrap in progress)", name)
                    continue
                if name in rag_repairs_in_progress:
                    logger.debug("Reconciliation: skipping %s (RAG repair in progress)", name)
                    continue
                # Safety margin: avoid spawning a process right after it was
                # stopped — the health-check restart path may already be
                # starting a new instance, and a race here causes DUPLICATE
                # PROCESS errors.
                _stopped_at = getattr(self, "_recently_stopped", {}).get(name)
                if _stopped_at is not None and (time.monotonic() - _stopped_at) < 5.0:
                    logger.debug("Reconciliation: skipping %s (recently stopped, safety margin)", name)
                    continue
                # Respect the restart state machine's backoff window instead of
                # a separate start-failure cooldown.
                if (
                    self._restart_ctl is not None
                    and self._restart_ctl.get(name) is not None
                    and not self._restart_ctl.is_due(name)
                ):
                    continue
                logger.info("Reconciliation: starting anima %s", name)
                try:
                    await self.start_anima(name)
                    if self.on_anima_added:
                        self.on_anima_added(name)
                except Exception as exc:
                    logger.exception(
                        "Reconciliation: failed to start %s",
                        name,
                    )
                    if self._restart_ctl is not None:
                        self._restart_ctl.record_failure(name, f"{type(exc).__name__}: {exc}")
                        self._ensure_restart_worker(name)

        # disabled + running → stop
        for name, enabled in on_disk.items():
            if not enabled and name in running:
                if name in self._bootstrapping:
                    logger.info("Reconciliation: deferring stop for %s (bootstrap in progress)", name)
                    continue
                if name in rag_repairs_in_progress:
                    logger.info("Reconciliation: deferring stop for %s (RAG repair in progress)", name)
                    continue
                logger.info(
                    "Reconciliation: stopping anima %s (disabled)",
                    name,
                )
                try:
                    await self.stop_anima(name)
                    if self._restart_ctl is not None:
                        self._restart_ctl.forget(name)
                    if self.on_anima_removed:
                        self.on_anima_removed(name)
                except Exception:
                    logger.exception(
                        "Reconciliation: failed to stop %s",
                        name,
                    )

        # removed from disk + running → stop
        # Protect running animas whose directory exists (IDENTITY.md present)
        # even if status.json is missing (legacy or factory-in-progress).
        for name in list(running):
            if name not in on_disk and name not in on_disk_incomplete:
                if name in self._bootstrapping:
                    logger.info("Reconciliation: deferring stop for %s (bootstrap in progress)", name)
                    continue
                if name in rag_repairs_in_progress:
                    logger.info("Reconciliation: deferring stop for %s (RAG repair in progress)", name)
                    continue
                logger.info(
                    "Reconciliation: stopping anima %s (removed from disk)",
                    name,
                )
                try:
                    await self.stop_anima(name)
                    if self._restart_ctl is not None:
                        self._restart_ctl.forget(name)
                    if self.on_anima_removed:
                        self.on_anima_removed(name)
                except Exception:
                    logger.exception(
                        "Reconciliation: failed to stop %s",
                        name,
                    )

    async def _reconcile_assets(self: _ReconcileMixinHost) -> None:
        """Explicitly generate missing assets for animas when requested."""
        try:
            from core.anima.asset_reconciler import find_animas_with_missing_assets, reconcile_anima_assets
            from core.config.models import load_config

            enable_3d = True
            image_style: str = "realistic"
            try:
                cfg = load_config()
                enable_3d = cfg.image_gen.enable_3d
                image_style = cfg.image_gen.image_style
            except Exception:
                logger.debug(
                    "Failed to read image_gen config, using defaults",
                    exc_info=True,
                )

            incomplete = find_animas_with_missing_assets(
                self.animas_dir,
                enable_3d=enable_3d,
                image_style=image_style,  # type: ignore[arg-type]
            )
            if not incomplete:
                return

            logger.info(
                "Asset reconciliation: %d anima(s) with missing %s assets",
                len(incomplete),
                image_style,
            )
            for anima_name, _check in incomplete:
                anima_dir = self.animas_dir / anima_name
                result = await reconcile_anima_assets(
                    anima_dir,
                    enable_3d=enable_3d,
                    image_style=image_style,  # type: ignore[arg-type]
                )
                if not result.get("skipped"):
                    await self._broadcast_event(
                        "anima.assets_updated",
                        {"name": anima_name, "source": "reconciliation"},
                    )
        except Exception:
            logger.exception("Asset reconciliation failed")

    def _check_config_freshness(self: _ReconcileMixinHost) -> None:
        """Detect config.json changes and refresh the singleton cache.

        This is a supplementary auto-detection mechanism.  The primary
        trigger is the ``POST /api/system/reload`` API endpoint.
        """
        try:
            import hashlib

            from core.config.models import get_config_path, load_config

            config_path = get_config_path()
            content_hash = hashlib.sha256(config_path.read_bytes()).hexdigest()
            if not hasattr(self, "_last_config_hash"):
                self._last_config_hash = content_hash
                return
            if content_hash != self._last_config_hash:
                self._last_config_hash = content_hash
                load_config()
                logger.info("Reconciliation: config.json changed, cache refreshed")
        except Exception:
            logger.debug("Config freshness check failed", exc_info=True)
