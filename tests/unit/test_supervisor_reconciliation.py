"""
Unit tests for ProcessSupervisor reconciliation feature.
"""

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
            "persons_dir": tmp / "persons",
            "shared_dir": tmp / "shared",
            "run_dir": tmp / "run",
        }


@pytest.fixture
def supervisor(temp_dirs):
    """Create a ProcessSupervisor instance."""
    return ProcessSupervisor(
        persons_dir=temp_dirs["persons_dir"],
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


# ── read_person_enabled ──────────────────────────────────────────


@pytest.mark.asyncio
async def test_read_person_enabled_no_status_file(temp_dirs):
    """No status.json → returns True (backward compatibility)."""
    person_dir = temp_dirs["persons_dir"] / "alice"
    person_dir.mkdir(parents=True)

    result = ProcessSupervisor.read_person_enabled(person_dir)

    assert result is True


@pytest.mark.asyncio
async def test_read_person_enabled_true(temp_dirs):
    """status.json with enabled: true → returns True."""
    person_dir = temp_dirs["persons_dir"] / "alice"
    person_dir.mkdir(parents=True)
    (person_dir / "status.json").write_text(
        json.dumps({"enabled": True}), encoding="utf-8"
    )

    result = ProcessSupervisor.read_person_enabled(person_dir)

    assert result is True


@pytest.mark.asyncio
async def test_read_person_enabled_false(temp_dirs):
    """status.json with enabled: false → returns False."""
    person_dir = temp_dirs["persons_dir"] / "alice"
    person_dir.mkdir(parents=True)
    (person_dir / "status.json").write_text(
        json.dumps({"enabled": False}), encoding="utf-8"
    )

    result = ProcessSupervisor.read_person_enabled(person_dir)

    assert result is False


@pytest.mark.asyncio
async def test_read_person_enabled_missing_key(temp_dirs):
    """status.json with empty object → returns True (default)."""
    person_dir = temp_dirs["persons_dir"] / "alice"
    person_dir.mkdir(parents=True)
    (person_dir / "status.json").write_text(
        json.dumps({}), encoding="utf-8"
    )

    result = ProcessSupervisor.read_person_enabled(person_dir)

    assert result is True


@pytest.mark.asyncio
async def test_read_person_enabled_invalid_json(temp_dirs):
    """status.json with invalid JSON → returns True (safe fallback)."""
    person_dir = temp_dirs["persons_dir"] / "alice"
    person_dir.mkdir(parents=True)
    (person_dir / "status.json").write_text(
        "not valid json!!!", encoding="utf-8"
    )

    result = ProcessSupervisor.read_person_enabled(person_dir)

    assert result is True


# ── _reconcile ───────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_reconcile_starts_enabled_person(supervisor, temp_dirs):
    """Enabled person on disk, not running → start_person() + on_person_added called."""
    persons_dir = temp_dirs["persons_dir"]
    persons_dir.mkdir(parents=True)
    alice_dir = persons_dir / "alice"
    alice_dir.mkdir()
    (alice_dir / "identity.md").write_text("Alice identity", encoding="utf-8")
    (alice_dir / "status.json").write_text(
        json.dumps({"enabled": True}), encoding="utf-8"
    )

    supervisor.start_person = AsyncMock()
    supervisor.stop_person = AsyncMock()
    callback = MagicMock()
    supervisor.on_person_added = callback

    await supervisor._reconcile()

    supervisor.start_person.assert_called_once_with("alice")
    callback.assert_called_once_with("alice")
    supervisor.stop_person.assert_not_called()


@pytest.mark.asyncio
async def test_reconcile_stops_disabled_person(supervisor, temp_dirs):
    """Disabled person on disk, running → stop_person() + on_person_removed called."""
    persons_dir = temp_dirs["persons_dir"]
    persons_dir.mkdir(parents=True)
    alice_dir = persons_dir / "alice"
    alice_dir.mkdir()
    (alice_dir / "identity.md").write_text("Alice identity", encoding="utf-8")
    (alice_dir / "status.json").write_text(
        json.dumps({"enabled": False}), encoding="utf-8"
    )

    # Simulate running process
    handle = MagicMock(spec=ProcessHandle)
    handle.person_name = "alice"
    supervisor.processes["alice"] = handle

    supervisor.start_person = AsyncMock()
    supervisor.stop_person = AsyncMock()
    callback = MagicMock()
    supervisor.on_person_removed = callback

    await supervisor._reconcile()

    supervisor.stop_person.assert_called_once_with("alice")
    callback.assert_called_once_with("alice")
    supervisor.start_person.assert_not_called()


@pytest.mark.asyncio
async def test_reconcile_stops_removed_person(supervisor, temp_dirs):
    """Person in processes but not on disk → stop_person() + on_person_removed called."""
    persons_dir = temp_dirs["persons_dir"]
    persons_dir.mkdir(parents=True)

    # Process running but no directory on disk
    handle = MagicMock(spec=ProcessHandle)
    handle.person_name = "alice"
    supervisor.processes["alice"] = handle

    supervisor.start_person = AsyncMock()
    supervisor.stop_person = AsyncMock()
    callback = MagicMock()
    supervisor.on_person_removed = callback

    await supervisor._reconcile()

    supervisor.stop_person.assert_called_once_with("alice")
    callback.assert_called_once_with("alice")
    supervisor.start_person.assert_not_called()


@pytest.mark.asyncio
async def test_reconcile_no_status_file_backward_compat(supervisor, temp_dirs):
    """Person with identity.md but no status.json → treated as enabled, started."""
    persons_dir = temp_dirs["persons_dir"]
    persons_dir.mkdir(parents=True)
    alice_dir = persons_dir / "alice"
    alice_dir.mkdir()
    (alice_dir / "identity.md").write_text("Alice identity", encoding="utf-8")
    # No status.json

    supervisor.start_person = AsyncMock()
    supervisor.stop_person = AsyncMock()
    callback = MagicMock()
    supervisor.on_person_added = callback

    await supervisor._reconcile()

    supervisor.start_person.assert_called_once_with("alice")
    callback.assert_called_once_with("alice")


@pytest.mark.asyncio
async def test_reconcile_skips_dir_without_identity(supervisor, temp_dirs):
    """Directory exists but no identity.md → skipped."""
    persons_dir = temp_dirs["persons_dir"]
    persons_dir.mkdir(parents=True)
    alice_dir = persons_dir / "alice"
    alice_dir.mkdir()
    # No identity.md

    supervisor.start_person = AsyncMock()
    supervisor.stop_person = AsyncMock()

    await supervisor._reconcile()

    supervisor.start_person.assert_not_called()
    supervisor.stop_person.assert_not_called()


@pytest.mark.asyncio
async def test_reconcile_skips_already_running(supervisor, temp_dirs):
    """Enabled person already running → no action."""
    persons_dir = temp_dirs["persons_dir"]
    persons_dir.mkdir(parents=True)
    alice_dir = persons_dir / "alice"
    alice_dir.mkdir()
    (alice_dir / "identity.md").write_text("Alice identity", encoding="utf-8")
    (alice_dir / "status.json").write_text(
        json.dumps({"enabled": True}), encoding="utf-8"
    )

    # Already running
    handle = MagicMock(spec=ProcessHandle)
    handle.person_name = "alice"
    supervisor.processes["alice"] = handle

    supervisor.start_person = AsyncMock()
    supervisor.stop_person = AsyncMock()

    await supervisor._reconcile()

    supervisor.start_person.assert_not_called()
    supervisor.stop_person.assert_not_called()


@pytest.mark.asyncio
async def test_reconcile_callback_not_set(supervisor, temp_dirs):
    """Callbacks are None → no error when reconciliation triggers actions."""
    persons_dir = temp_dirs["persons_dir"]
    persons_dir.mkdir(parents=True)
    alice_dir = persons_dir / "alice"
    alice_dir.mkdir()
    (alice_dir / "identity.md").write_text("Alice identity", encoding="utf-8")
    (alice_dir / "status.json").write_text(
        json.dumps({"enabled": True}), encoding="utf-8"
    )

    supervisor.start_person = AsyncMock()
    supervisor.stop_person = AsyncMock()
    supervisor.on_person_added = None
    supervisor.on_person_removed = None

    # Should not raise
    await supervisor._reconcile()

    supervisor.start_person.assert_called_once_with("alice")


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
