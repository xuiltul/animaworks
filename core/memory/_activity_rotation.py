from __future__ import annotations
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.

"""Rotation mixin for ActivityLogger.

Internal module — import from :mod:`core.memory.activity` instead.
"""

import logging
from datetime import date, timedelta
from pathlib import Path
from typing import Any

logger = logging.getLogger("animaworks.activity")


class RotationMixin:
    """Mixin providing log rotation methods for ActivityLogger."""

    def rotate(
        self,
        *,
        mode: str = "size",
        max_size_mb: int = 1024,
        max_age_days: int = 7,
    ) -> dict[str, Any]:
        """Rotate activity log files by deleting old entries.

        Args:
            mode: ``"size"`` (total size cap), ``"time"`` (age cap),
                or ``"both"`` (time then size).
            max_size_mb: Maximum total size in MB (per-anima).
            max_age_days: Maximum age in days for ``"time"``/``"both"``.

        Returns:
            Dict with ``deleted_files`` count and ``freed_bytes``.
        """
        if not self._log_dir.exists():  # type: ignore[attr-defined]
            return {"deleted_files": 0, "freed_bytes": 0}

        today_str = date.today().isoformat()
        files = sorted(self._log_dir.glob("*.jsonl"))  # type: ignore[attr-defined]

        deleted_count = 0
        freed_bytes = 0

        # Phase 1: time-based deletion
        if mode in ("time", "both"):
            cutoff = date.today() - timedelta(days=max_age_days)
            remaining: list[Path] = []
            for f in files:
                file_date_str = f.stem
                if file_date_str == today_str:
                    remaining.append(f)
                    continue
                try:
                    file_date = date.fromisoformat(file_date_str)
                except ValueError:
                    remaining.append(f)
                    continue
                if file_date < cutoff:
                    size = f.stat().st_size
                    f.unlink()
                    deleted_count += 1
                    freed_bytes += size
                    logger.debug("Rotation (time): deleted %s (%d bytes)", f.name, size)
                else:
                    remaining.append(f)
            files = remaining

        # Phase 2: size-based deletion
        if mode in ("size", "both"):
            max_bytes = max_size_mb * 1024 * 1024
            file_sizes = {f: f.stat().st_size for f in files}
            total_size = sum(file_sizes.values())
            for f in files:
                if total_size <= max_bytes:
                    break
                if f.stem == today_str:
                    continue
                size = file_sizes[f]
                f.unlink()
                total_size -= size
                deleted_count += 1
                freed_bytes += size
                logger.debug("Rotation (size): deleted %s (%d bytes)", f.name, size)

        if deleted_count:
            logger.info(
                "Rotation completed for %s: deleted=%d freed=%d bytes",
                self.anima_dir.name, deleted_count, freed_bytes,  # type: ignore[attr-defined]
            )

        return {"deleted_files": deleted_count, "freed_bytes": freed_bytes}

    @staticmethod
    def rotate_all(
        animas_dir: Path,
        *,
        mode: str = "size",
        max_size_mb: int = 1024,
        max_age_days: int = 7,
    ) -> dict[str, dict[str, Any]]:
        """Run rotation for all Animas under *animas_dir*.

        Returns:
            Dict mapping anima name to rotation result.
        """
        from core.memory.activity import ActivityLogger

        results: dict[str, dict[str, Any]] = {}
        if not animas_dir.exists():
            return results
        for anima_dir in sorted(animas_dir.iterdir()):
            if not anima_dir.is_dir():
                continue
            log_dir = anima_dir / "activity_log"
            if not log_dir.exists():
                continue
            try:
                al = ActivityLogger(anima_dir)
                result = al.rotate(
                    mode=mode,
                    max_size_mb=max_size_mb,
                    max_age_days=max_age_days,
                )
                if result["deleted_files"] > 0:
                    results[anima_dir.name] = result
            except Exception:
                logger.exception("Rotation failed for %s", anima_dir.name)
        return results
