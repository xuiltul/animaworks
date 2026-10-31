from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.

"""Version tracking for AnimaWorks runtime migrations."""

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

from core.platform.atomic_io import atomic_write_json
from core.time_utils import now_local

logger = logging.getLogger(__name__)

_STATE_FILE = "migration_state.json"
_MIN_SUPPORTED_VERSION = (0, 14, 0)


class UnsupportedRuntimeVersionError(RuntimeError):
    """Raised when runtime data predates the minimum supported migration base."""


def _get_package_version() -> str:
    """Resolve the current AnimaWorks package version."""
    try:
        from importlib.metadata import version

        return version("animaworks")
    except Exception:
        logger.debug("Best-effort operation failed", exc_info=True)
    try:
        from core.paths import PROJECT_DIR

        toml = PROJECT_DIR / "pyproject.toml"
        if toml.exists():
            for line in toml.read_text(encoding="utf-8").splitlines():
                if line.strip().startswith("version"):
                    return line.split("=", 1)[1].strip().strip('"').strip("'")
    except Exception:
        logger.debug("Best-effort operation failed", exc_info=True)
    return "0.0.0"


# ── Data model ──────────────────────────────────────────────


@dataclass
class MigrationState:
    """Persistent state of applied migrations."""

    applied_version: str = ""
    steps_applied: dict[str, str] = field(default_factory=dict)
    last_migrated_at: str = ""


# ── Tracker ─────────────────────────────────────────────────


class MigrationTracker:
    """Read/write ``migration_state.json`` in the runtime data directory."""

    def __init__(self, data_dir: Path) -> None:
        self._path = data_dir / _STATE_FILE
        self._state: MigrationState | None = None

    def load(self) -> MigrationState:
        """Load state from disk, returning empty state if missing."""
        if self._state is not None:
            return self._state
        if self._path.exists():
            try:
                raw = json.loads(self._path.read_text(encoding="utf-8"))
                self._state = MigrationState(
                    applied_version=raw.get("applied_version", ""),
                    steps_applied=raw.get("steps_applied", {}),
                    last_migrated_at=raw.get("last_migrated_at", ""),
                )
            except Exception:
                logger.warning("Corrupt %s — starting fresh", self._path.name)
                self._state = MigrationState()
        else:
            self._state = MigrationState()
        return self._state

    def save(self, state: MigrationState) -> None:
        """Persist state to disk."""
        self._state = state
        payload = {
            "applied_version": state.applied_version,
            "steps_applied": state.steps_applied,
            "last_migrated_at": state.last_migrated_at,
        }
        atomic_write_json(self._path, payload, indent=2, ensure_ascii=False)

    def is_step_applied(self, step_id: str) -> bool:
        return step_id in self.load().steps_applied

    def mark_applied(self, step_id: str) -> None:
        state = self.load()
        state.steps_applied[step_id] = now_local().isoformat()
        state.applied_version = _get_package_version()
        state.last_migrated_at = now_local().isoformat()
        self.save(state)

    def unmark_applied(self, step_id: str) -> bool:
        """Remove a step from the applied set so it can run again.

        Returns True when the step was present. Keeps ``applied_version`` so a
        rollback can be followed by a restart on the same version.
        """
        state = self.load()
        if step_id not in state.steps_applied:
            return False
        del state.steps_applied[step_id]
        state.last_migrated_at = now_local().isoformat()
        self.save(state)
        return True

    def get_current_version(self) -> str:
        return _get_package_version()


def assert_supported_runtime_version(data_dir: Path) -> None:
    """Reject direct upgrades from runtime versions older than 0.14.0.

    A missing migration state is expected for a fresh installation. Older
    runtime versions that predate version tracking also have no reliable
    version to compare, so only a recorded ``applied_version`` is enforced.
    """
    tracker = MigrationTracker(data_dir)
    state = tracker.load()
    applied_version = state.applied_version.strip()
    if not applied_version:
        return

    match = re.match(r"^v?(\d+)\.(\d+)(?:\.(\d+))?", applied_version)
    if not match:
        from core.i18n import t

        message = t("migrate.unsupported_runtime_version", version=applied_version)
        raise UnsupportedRuntimeVersionError(message)
    version = tuple(int(part or 0) for part in match.groups())
    if version >= _MIN_SUPPORTED_VERSION:
        return

    from core.i18n import t

    message = t("migrate.unsupported_runtime_version", version=applied_version)
    raise UnsupportedRuntimeVersionError(message)
