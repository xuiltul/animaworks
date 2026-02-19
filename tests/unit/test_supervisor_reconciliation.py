"""
Unit tests for ProcessSupervisor reconciliation feature.
"""

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import json
import pytest
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import AsyncMock, MagicMock, patch

from core.supervisor.manager import (
    ProcessSupervisor,
    RestartPolicy,
    HealthConfig,
    ReconciliationConfig,
)
from core.supervisor.process_handle import ProcessHandle, ProcessState


@pytest.fixture
def temp_dirs():
    """Create temporary directories for testing."""
    with TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        yield {
            "animas_dir": tmp / "animas",
            "shared_dir": tmp / "shared",
            "run_dir": tmp / "run",
        }


@pytest.fixture
def supervisor(temp_dirs):
    """Create a ProcessSupervisor instance."""
    return ProcessSupervisor(
        animas_dir=temp_dirs["animas_dir"],
        shared_dir=temp_dirs["shared_dir"],
        run_dir=temp_dirs["run_dir"],
        restart_policy=RestartPolicy(
            max_retries=3,
            backoff_base_sec=0.1,
            backoff_max_sec=1.0,
        ),
        health_config=HealthConfig(
            ping_interval_sec=0.5,
            ping_timeout_sec=0.2,
            max_missed_pings=2,
            startup_grace_sec=0.5,
        ),
    )


# ── read_anima_enabled ──────────────────────────────────────────


@pytest.mark.asyncio
async def test_read_anima_enabled_no_status_file(temp_dirs):
    """No status.json → returns True (backward compatibility)."""
    anima_dir = temp_dirs["animas_dir"] / "alice"
    anima_dir.mkdir(parents=True)

    result = ProcessSupervisor.read_anima_enabled(anima_dir)

    assert result is True


@pytest.mark.asyncio
async def test_read_anima_enabled_true(temp_dirs):
    """status.json with enabled: true → returns True."""
    anima_dir = temp_dirs["animas_dir"] / "alice"
    anima_dir.mkdir(parents=True)
    (anima_dir / "status.json").write_text(
        json.dumps({"enabled": True}), encoding="utf-8"
    )

    result = ProcessSupervisor.read_anima_enabled(anima_dir)

    assert result is True


@pytest.mark.asyncio
async def test_read_anima_enabled_false(temp_dirs):
    """status.json with enabled: false → returns False."""
    anima_dir = temp_dirs["animas_dir"] / "alice"
    anima_dir.mkdir(parents=True)
    (anima_dir / "status.json").write_text(
        json.dumps({"enabled": False}), encoding="utf-8"
    )

    result = ProcessSupervisor.read_anima_enabled(anima_dir)

    assert result is False


@pytest.mark.asyncio
async def test_read_anima_enabled_missing_key(temp_dirs):
    """status.json with empty object → returns True (default)."""
    anima_dir = temp_dirs["animas_dir"] / "alice"
    anima_dir.mkdir(parents=True)
    (anima_dir / "status.json").write_text(
        json.dumps({}), encoding="utf-8"
    )

    result = ProcessSupervisor.read_anima_enabled(anima_dir)

    assert result is True


@pytest.mark.asyncio
async def test_read_anima_enabled_invalid_json(temp_dirs):
    """status.json with invalid JSON → returns True (safe fallback)."""
    anima_dir = temp_dirs["animas_dir"] / "alice"
    anima_dir.mkdir(parents=True)
    (anima_dir / "status.json").write_text(
        "not valid json!!!", encoding="utf-8"
    )

    result = ProcessSupervisor.read_anima_enabled(anima_dir)

    assert result is True


# ── _reconcile ───────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_reconcile_starts_enabled_anima(supervisor, temp_dirs):
    """Enabled anima on disk, not running → start_anima() + on_anima_added called."""
    animas_dir = temp_dirs["animas_dir"]
    animas_dir.mkdir(parents=True)
    alice_dir = animas_dir / "alice"
    alice_dir.mkdir()
    (alice_dir / "identity.md").write_text("Alice identity", encoding="utf-8")
    (alice_dir / "status.json").write_text(
        json.dumps({"enabled": True}), encoding="utf-8"
    )

    supervisor.start_anima = AsyncMock()
    supervisor.stop_anima = AsyncMock()
    callback = MagicMock()
    supervisor.on_anima_added = callback

    await supervisor._reconcile()

    supervisor.start_anima.assert_called_once_with("alice")
    callback.assert_called_once_with("alice")
    supervisor.stop_anima.assert_not_called()


@pytest.mark.asyncio
async def test_reconcile_stops_disabled_anima(supervisor, temp_dirs):
    """Disabled anima on disk, running → stop_anima() + on_anima_removed called."""
    animas_dir = temp_dirs["animas_dir"]
    animas_dir.mkdir(parents=True)
    alice_dir = animas_dir / "alice"
    alice_dir.mkdir()
    (alice_dir / "identity.md").write_text("Alice identity", encoding="utf-8")
    (alice_dir / "status.json").write_text(
        json.dumps({"enabled": False}), encoding="utf-8"
    )

    # Simulate running process
    handle = MagicMock(spec=ProcessHandle)
    handle.anima_name = "alice"
    supervisor.processes["alice"] = handle

    supervisor.start_anima = AsyncMock()
    supervisor.stop_anima = AsyncMock()
    callback = MagicMock()
    supervisor.on_anima_removed = callback

    await supervisor._reconcile()

    supervisor.stop_anima.assert_called_once_with("alice")
    callback.assert_called_once_with("alice")
    supervisor.start_anima.assert_not_called()


@pytest.mark.asyncio
async def test_reconcile_stops_removed_anima(supervisor, temp_dirs):
    """Anima in processes but not on disk → stop_anima() + on_anima_removed called."""
    animas_dir = temp_dirs["animas_dir"]
    animas_dir.mkdir(parents=True)

    # Process running but no directory on disk
    handle = MagicMock(spec=ProcessHandle)
    handle.anima_name = "alice"
    supervisor.processes["alice"] = handle

    supervisor.start_anima = AsyncMock()
    supervisor.stop_anima = AsyncMock()
    callback = MagicMock()
    supervisor.on_anima_removed = callback

    await supervisor._reconcile()

    supervisor.stop_anima.assert_called_once_with("alice")
    callback.assert_called_once_with("alice")
    supervisor.start_anima.assert_not_called()


@pytest.mark.asyncio
async def test_reconcile_no_status_file_incomplete_protection(supervisor, temp_dirs):
    """Anima with identity.md but no status.json → on_disk_incomplete, not started, not killed."""
    animas_dir = temp_dirs["animas_dir"]
    animas_dir.mkdir(parents=True)
    alice_dir = animas_dir / "alice"
    alice_dir.mkdir()
    (alice_dir / "identity.md").write_text("Alice identity", encoding="utf-8")
    # No status.json — treated as incomplete (legacy or factory-in-progress)

    supervisor.start_anima = AsyncMock()
    supervisor.stop_anima = AsyncMock()
    callback = MagicMock()
    supervisor.on_anima_added = callback

    await supervisor._reconcile()

    supervisor.start_anima.assert_not_called()
    supervisor.stop_anima.assert_not_called()
    callback.assert_not_called()


@pytest.mark.asyncio
async def test_reconcile_skips_dir_without_identity(supervisor, temp_dirs):
    """Directory exists but no identity.md → skipped."""
    animas_dir = temp_dirs["animas_dir"]
    animas_dir.mkdir(parents=True)
    alice_dir = animas_dir / "alice"
    alice_dir.mkdir()
    # No identity.md

    supervisor.start_anima = AsyncMock()
    supervisor.stop_anima = AsyncMock()

    await supervisor._reconcile()

    supervisor.start_anima.assert_not_called()
    supervisor.stop_anima.assert_not_called()


@pytest.mark.asyncio
async def test_reconcile_skips_already_running(supervisor, temp_dirs):
    """Enabled anima already running → no action."""
    animas_dir = temp_dirs["animas_dir"]
    animas_dir.mkdir(parents=True)
    alice_dir = animas_dir / "alice"
    alice_dir.mkdir()
    (alice_dir / "identity.md").write_text("Alice identity", encoding="utf-8")
    (alice_dir / "status.json").write_text(
        json.dumps({"enabled": True}), encoding="utf-8"
    )

    # Already running
    handle = MagicMock(spec=ProcessHandle)
    handle.anima_name = "alice"
    supervisor.processes["alice"] = handle

    supervisor.start_anima = AsyncMock()
    supervisor.stop_anima = AsyncMock()

    await supervisor._reconcile()

    supervisor.start_anima.assert_not_called()
    supervisor.stop_anima.assert_not_called()


@pytest.mark.asyncio
async def test_reconcile_start_anima_failure(supervisor, temp_dirs):
    """start_anima raises → exception logged, other animas still processed."""
    animas_dir = temp_dirs["animas_dir"]
    animas_dir.mkdir(parents=True)
    alice_dir = animas_dir / "alice"
    alice_dir.mkdir()
    (alice_dir / "identity.md").write_text("Alice identity", encoding="utf-8")
    (alice_dir / "status.json").write_text(
        json.dumps({"enabled": True}), encoding="utf-8"
    )
    bob_dir = animas_dir / "bob"
    bob_dir.mkdir()
    (bob_dir / "identity.md").write_text("Bob identity", encoding="utf-8")
    (bob_dir / "status.json").write_text(
        json.dumps({"enabled": True}), encoding="utf-8"
    )

    # start_anima fails for alice but succeeds for bob
    async def start_side_effect(name: str) -> None:
        if name == "alice":
            raise RuntimeError("spawn failed")

    supervisor.start_anima = AsyncMock(side_effect=start_side_effect)
    supervisor.stop_anima = AsyncMock()
    added_callback = MagicMock()
    supervisor.on_anima_added = added_callback

    # Should not raise — failure is caught internally
    await supervisor._reconcile()

    # alice attempted but failed; bob attempted and succeeded
    assert supervisor.start_anima.call_count == 2
    # Callback only invoked for successful starts (bob)
    added_callback.assert_called_once_with("bob")


@pytest.mark.asyncio
async def test_reconcile_callback_not_set(supervisor, temp_dirs):
    """Callbacks are None → no error when reconciliation triggers actions."""
    animas_dir = temp_dirs["animas_dir"]
    animas_dir.mkdir(parents=True)
    alice_dir = animas_dir / "alice"
    alice_dir.mkdir()
    (alice_dir / "identity.md").write_text("Alice identity", encoding="utf-8")
    (alice_dir / "status.json").write_text(
        json.dumps({"enabled": True}), encoding="utf-8"
    )

    supervisor.start_anima = AsyncMock()
    supervisor.stop_anima = AsyncMock()
    supervisor.on_anima_added = None
    supervisor.on_anima_removed = None

    # Should not raise
    await supervisor._reconcile()

    supervisor.start_anima.assert_called_once_with("alice")


# ── ReconciliationConfig ─────────────────────────────────────────


@pytest.mark.asyncio
async def test_reconciliation_config_defaults():
    """Verify ReconciliationConfig default values."""
    config = ReconciliationConfig()

    assert config.interval_sec == 30.0


# ── Supervisor attribute ─────────────────────────────────────────


@pytest.mark.asyncio
async def test_supervisor_has_reconciliation_task_attr(supervisor):
    """Supervisor init has _reconciliation_task attribute."""
    assert hasattr(supervisor, "_reconciliation_task")
    assert supervisor._reconciliation_task is None
