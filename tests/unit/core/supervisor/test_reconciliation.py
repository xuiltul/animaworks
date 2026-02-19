# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for ProcessSupervisor._reconcile() — on_disk_incomplete protection.

Verifies that the reconciliation loop correctly handles anima directories
that have identity.md but are missing status.json (incomplete / factory-in-progress).
These animas must NOT be auto-started, but if already running they must NOT
be killed either.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.supervisor.manager import ProcessSupervisor


# ── Helpers ──────────────────────────────────────────────────────────


def _make_supervisor(tmp_path: Path) -> ProcessSupervisor:
    """Create a minimal ProcessSupervisor rooted under *tmp_path*."""
    animas_dir = tmp_path / "animas"
    animas_dir.mkdir(parents=True, exist_ok=True)
    shared_dir = tmp_path / "shared"
    shared_dir.mkdir(parents=True, exist_ok=True)
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    return ProcessSupervisor(
        animas_dir=animas_dir,
        shared_dir=shared_dir,
        run_dir=run_dir,
    )


def _create_anima_dir(
    animas_dir: Path,
    name: str,
    *,
    has_identity: bool = True,
    has_status: bool = True,
    enabled: bool = True,
) -> Path:
    """Create a mock anima directory on disk with optional files."""
    d = animas_dir / name
    d.mkdir(parents=True, exist_ok=True)
    if has_identity:
        (d / "identity.md").write_text(f"# {name}", encoding="utf-8")
    if has_status:
        (d / "status.json").write_text(
            json.dumps({"enabled": enabled}), encoding="utf-8"
        )
    return d


# ── Tests ────────────────────────────────────────────────────────────


class TestReconcileOnDiskIncomplete:
    """Tests for the on_disk_incomplete protection in _reconcile()."""

    async def test_running_anima_without_status_json_not_killed(self, tmp_path: Path):
        """A running anima with identity.md but no status.json must NOT be stopped."""
        sup = _make_supervisor(tmp_path)
        _create_anima_dir(sup.animas_dir, "sakura", has_identity=True, has_status=False)

        # Simulate a running process
        sup.processes["sakura"] = MagicMock()
        sup.stop_anima = AsyncMock()

        with patch.object(sup, "_reconcile_assets", new_callable=AsyncMock):
            await sup._reconcile()

        sup.stop_anima.assert_not_called()

    async def test_running_anima_with_status_json_enabled_not_killed(self, tmp_path: Path):
        """A running anima with status.json enabled=true must NOT be stopped."""
        sup = _make_supervisor(tmp_path)
        _create_anima_dir(
            sup.animas_dir, "sakura", has_identity=True, has_status=True, enabled=True
        )

        sup.processes["sakura"] = MagicMock()
        sup.stop_anima = AsyncMock()
        sup.start_anima = AsyncMock()

        with patch.object(sup, "_reconcile_assets", new_callable=AsyncMock):
            await sup._reconcile()

        sup.stop_anima.assert_not_called()

    async def test_running_anima_truly_removed_is_killed(self, tmp_path: Path):
        """A running anima with NO directory on disk MUST be stopped."""
        sup = _make_supervisor(tmp_path)
        # No anima directory created at all

        sup.processes["ghost"] = MagicMock()
        sup.stop_anima = AsyncMock()

        with patch.object(sup, "_reconcile_assets", new_callable=AsyncMock):
            await sup._reconcile()

        sup.stop_anima.assert_called_once_with("ghost")

    async def test_running_anima_disabled_is_stopped(self, tmp_path: Path):
        """A running anima with status.json enabled=false MUST be stopped."""
        sup = _make_supervisor(tmp_path)
        _create_anima_dir(
            sup.animas_dir, "sakura", has_identity=True, has_status=True, enabled=False
        )

        sup.processes["sakura"] = MagicMock()
        sup.stop_anima = AsyncMock()

        with patch.object(sup, "_reconcile_assets", new_callable=AsyncMock):
            await sup._reconcile()

        sup.stop_anima.assert_called_once_with("sakura")

    async def test_incomplete_anima_not_auto_started(self, tmp_path: Path):
        """An anima with identity.md but no status.json must NOT be auto-started."""
        sup = _make_supervisor(tmp_path)
        _create_anima_dir(sup.animas_dir, "sakura", has_identity=True, has_status=False)

        # No running processes
        sup.start_anima = AsyncMock()
        sup.stop_anima = AsyncMock()

        with patch.object(sup, "_reconcile_assets", new_callable=AsyncMock):
            await sup._reconcile()

        sup.start_anima.assert_not_called()
        sup.stop_anima.assert_not_called()

    async def test_dir_without_identity_md_ignored(self, tmp_path: Path):
        """A directory without identity.md is completely ignored.

        It won't be in on_disk or on_disk_incomplete, so a running process
        with that name will be treated as "removed from disk" and stopped.
        """
        sup = _make_supervisor(tmp_path)
        # Create dir with status.json but NO identity.md
        d = sup.animas_dir / "noidentity"
        d.mkdir()
        (d / "status.json").write_text('{"enabled": true}', encoding="utf-8")

        # Simulate a running process with that name
        sup.processes["noidentity"] = MagicMock()
        sup.stop_anima = AsyncMock()

        with patch.object(sup, "_reconcile_assets", new_callable=AsyncMock):
            await sup._reconcile()

        # Should be killed: not in on_disk, not in on_disk_incomplete
        sup.stop_anima.assert_called_once_with("noidentity")

    async def test_mixed_animas_correct_behavior(self, tmp_path: Path):
        """Test with multiple animas in different states simultaneously.

        - sakura: running, identity only (no status.json) -> PROTECTED
        - kotoha: running, identity + status enabled     -> KEPT
        - ghost:  running, no directory at all            -> KILLED
        - hinata: not running, identity + status enabled  -> STARTED
        """
        sup = _make_supervisor(tmp_path)

        # sakura: running, has identity but NO status.json -> protected
        _create_anima_dir(
            sup.animas_dir, "sakura", has_identity=True, has_status=False
        )

        # kotoha: running, has identity + status enabled -> kept
        _create_anima_dir(
            sup.animas_dir, "kotoha", has_identity=True, has_status=True, enabled=True
        )

        # ghost: running, NO directory at all -> killed
        # (no directory created)

        # hinata: NOT running, has identity + status enabled -> started
        _create_anima_dir(
            sup.animas_dir, "hinata", has_identity=True, has_status=True, enabled=True
        )

        sup.processes["sakura"] = MagicMock()
        sup.processes["kotoha"] = MagicMock()
        sup.processes["ghost"] = MagicMock()

        sup.start_anima = AsyncMock()
        sup.stop_anima = AsyncMock()

        with patch.object(sup, "_reconcile_assets", new_callable=AsyncMock):
            await sup._reconcile()

        # ghost should be stopped (removed from disk)
        sup.stop_anima.assert_called_once_with("ghost")
        # hinata should be started (enabled + not running)
        sup.start_anima.assert_called_once_with("hinata")
