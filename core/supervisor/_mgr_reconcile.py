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

from core.supervisor.process_handle import ProcessState

logger = logging.getLogger(__name__)


class ReconcileMixin:
    """Reconciliation loop: syncs desired state (disk) with actual processes."""

    async def _reconciliation_loop(self) -> None:
        """Periodically reconcile desired state (disk) with actual state (processes)."""
        logger.info("Reconciliation loop started (interval=%.0fs)",
                     self.reconciliation_config.interval_sec)

        while not self._shutdown:
            try:
                await asyncio.sleep(self.reconciliation_config.interval_sec)
                await self._reconcile()
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("Reconciliation failed")

        logger.info("Reconciliation loop stopped")

    async def _reconcile(self) -> None:
        """Scan animas_dir and sync desired state with actual process state."""
        if not self.animas_dir.exists():
            return

        running = set(self.processes.keys())

        # Build desired state from disk
        on_disk: dict[str, bool] = {}  # name -> enabled
        # Anima dirs that have identity.md but no status.json.
        # These are either legacy animas or factory-in-progress.
        # They must NOT be auto-started but must NOT be killed if running.
        on_disk_incomplete: set[str] = set()
        for anima_dir in sorted(self.animas_dir.iterdir()):
            if not anima_dir.is_dir():
                continue
            if not (anima_dir / "identity.md").exists():
                continue
            # status.json is created as the final step of anima_factory.
            # Its absence means creation may still be in progress.
            if not (anima_dir / "status.json").exists():
                on_disk_incomplete.add(anima_dir.name)
                continue
            on_disk[anima_dir.name] = self.read_anima_enabled(anima_dir)

        # permanently failed + enabled → recover after 1-minute cooldown
        for name in list(self._permanently_failed):
            handle = self.processes.get(name)
            if handle is None:
                self._permanently_failed.discard(name)
                continue
            if name not in on_disk or not on_disk[name]:
                continue
            now = asyncio.get_running_loop().time()
            failed_since = self._failed_log_times.get(name, 0)
            if now - failed_since < 60:
                continue
            logger.info(
                "Reconciliation: recovering permanently failed process %s "
                "(cooldown elapsed, resetting retries)",
                name,
            )
            del self.processes[name]
            self._restart_counts.pop(name, None)
            self._permanently_failed.discard(name)
            self._failed_log_times.pop(name, None)
            try:
                await self.start_anima(name)
                if self.on_anima_added:
                    self.on_anima_added(name)
            except Exception:
                logger.exception(
                    "Reconciliation: failed to recover %s", name,
                )

        # Update running set after recovery attempts
        running = set(self.processes.keys())

        # restart_requested フラグチェック
        for name in list(on_disk.keys()):
            anima_dir = self.animas_dir / name
            status_file = anima_dir / "status.json"
            try:
                status = json.loads(status_file.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            if not status.get("restart_requested"):
                continue
            # Clear flag to prevent re-trigger
            status.pop("restart_requested", None)
            status_file.write_text(
                json.dumps(status, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            logger.info("Reconciliation: restart_requested for %s, restarting", name)
            try:
                await self.restart_anima(name)
            except Exception:
                logger.exception("Reconciliation: failed to restart %s (restart_requested)", name)
            continue

        # Update running set after restart_requested handling
        running = set(self.processes.keys())

        # enabled + not running → start
        for name, enabled in on_disk.items():
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
                logger.info("Reconciliation: starting anima %s", name)
                try:
                    await self.start_anima(name)
                    if self.on_anima_added:
                        self.on_anima_added(name)
                except Exception:
                    logger.exception(
                        "Reconciliation: failed to start %s", name,
                    )

        # disabled + running → stop
        for name, enabled in on_disk.items():
            if not enabled and name in running:
                if name in self._bootstrapping:
                    logger.info("Reconciliation: deferring stop for %s (bootstrap in progress)", name)
                    continue
                logger.info(
                    "Reconciliation: stopping anima %s (disabled)", name,
                )
                try:
                    await self.stop_anima(name)
                    if self.on_anima_removed:
                        self.on_anima_removed(name)
                except Exception:
                    logger.exception(
                        "Reconciliation: failed to stop %s", name,
                    )

        # removed from disk + running → stop
        # Protect running animas whose directory exists (identity.md present)
        # even if status.json is missing (legacy or factory-in-progress).
        for name in list(running):
            if name not in on_disk and name not in on_disk_incomplete:
                if name in self._bootstrapping:
                    logger.info("Reconciliation: deferring stop for %s (bootstrap in progress)", name)
                    continue
                logger.info(
                    "Reconciliation: stopping anima %s (removed from disk)",
                    name,
                )
                try:
                    await self.stop_anima(name)
                    if self.on_anima_removed:
                        self.on_anima_removed(name)
                except Exception:
                    logger.exception(
                        "Reconciliation: failed to stop %s", name,
                    )

        # Check for missing anima assets (fallback generation)
        await self._reconcile_assets()

    async def _reconcile_assets(self) -> None:
        """Check for and generate missing anima assets during reconciliation."""
        try:
            from core.asset_reconciler import find_animas_with_missing_assets, reconcile_anima_assets
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
