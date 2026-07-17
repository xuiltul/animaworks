# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for per-anima lifecycle lock (start/stop serialization).

Covers TOCTOU start-vs-disable, handle-identity-safe pop on stop, and
respawn clean-skip when anima is disabled.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.supervisor.manager import ProcessSupervisor


@pytest.fixture
def supervisor(tmp_path: Path) -> ProcessSupervisor:
    """Create a ProcessSupervisor with test paths."""
    animas_dir = tmp_path / "animas"
    animas_dir.mkdir()
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    shared_dir = tmp_path / "shared"
    shared_dir.mkdir()

    return ProcessSupervisor(
        animas_dir=animas_dir,
        shared_dir=shared_dir,
        run_dir=run_dir,
    )


def _write_status(animas_dir: Path, name: str, *, enabled: bool) -> Path:
    anima_dir = animas_dir / name
    anima_dir.mkdir(parents=True, exist_ok=True)
    (anima_dir / "status.json").write_text(
        json.dumps({"enabled": enabled}),
        encoding="utf-8",
    )
    return anima_dir


class TestStartStopLifecycleLock:
    """start_anima and stop_anima share per-anima lifecycle lock."""

    @pytest.mark.asyncio
    async def test_start_disabled_mid_start_does_not_register(
        self,
        supervisor: ProcessSupervisor,
    ) -> None:
        """TOCTOU: disable during slow handle.start → process not left running.

        start_anima holds lifecycle lock during start. disable writes
        enabled=false then stop_anima waits for the lock, then no-ops if
        start already refused — or stops if start registered after re-check.

        Scenario: start begins while enabled; under lock we re-check and
        if disabled mid-flight before processes[name]=handle, refuse.
        Here we disable *before* start so start refuses under lock.
        Concurrent case: slow start, disable stop waits, then cleans up.
        """
        name = "test-anima"
        _write_status(supervisor.animas_dir, name, enabled=True)

        start_entered = asyncio.Event()
        allow_start_finish = asyncio.Event()

        async def slow_start(*_args, **_kwargs):
            start_entered.set()
            await allow_start_finish.wait()

        with patch("core.supervisor.manager.ProcessHandle") as MockHandle:
            mock_handle = AsyncMock()
            mock_handle.start = AsyncMock(side_effect=slow_start)
            mock_handle.stop = AsyncMock()
            mock_handle.get_pid = MagicMock(return_value=111)
            mock_handle.send_request = AsyncMock(
                return_value=MagicMock(error=None, result={"needs_bootstrap": False}),
            )
            MockHandle.return_value = mock_handle

            start_task = asyncio.create_task(supervisor.start_anima(name))
            await asyncio.wait_for(start_entered.wait(), timeout=2.0)

            # While start holds lifecycle lock (inside handle.start), flip
            # disabled and request stop (queues on same lock).
            _write_status(supervisor.animas_dir, name, enabled=False)
            stop_task = asyncio.create_task(supervisor.stop_anima(name))
            await asyncio.sleep(0.05)

            allow_start_finish.set()
            results = await asyncio.gather(start_task, stop_task, return_exceptions=True)

        assert all(r is None for r in results), f"unexpected: {results}"
        # Final state: no live process (stop after start registration, or
        # start never registered — both acceptable; must not leave orphan).
        assert name not in supervisor.processes


class TestStopHandleIdentityGuard:
    """stop_anima must not pop a replaced handle."""

    @pytest.mark.asyncio
    async def test_stop_does_not_pop_replaced_handle(
        self,
        supervisor: ProcessSupervisor,
    ) -> None:
        """During old handle drain, processes[name] is swapped → new handle remains."""
        name = "test-anima"
        drain_started = asyncio.Event()
        allow_drain_finish = asyncio.Event()

        old_handle = MagicMock()
        new_handle = MagicMock()
        new_handle.stop = AsyncMock()

        async def slow_stop(*_args, **_kwargs):
            drain_started.set()
            # Simulate concurrent start replacing the handle mid-drain
            supervisor.processes[name] = new_handle
            await allow_drain_finish.wait()

        old_handle.stop = AsyncMock(side_effect=slow_stop)
        supervisor.processes[name] = old_handle

        stop_task = asyncio.create_task(supervisor.stop_anima(name))
        await asyncio.wait_for(drain_started.wait(), timeout=2.0)
        allow_drain_finish.set()
        await stop_task

        assert supervisor.processes.get(name) is new_handle
        old_handle.stop.assert_awaited_once()
        new_handle.stop.assert_not_awaited()


class TestRespawnDisabledCleanSkip:
    """Disabled respawn must not pollute failure state."""

    @pytest.mark.asyncio
    async def test_respawn_disabled_returns_none_without_pollution(
        self,
        supervisor: ProcessSupervisor,
    ) -> None:
        name = "test-anima"
        _write_status(supervisor.animas_dir, name, enabled=False)
        supervisor.restart_policy.max_retries = 2

        # Pre-seed counters to prove they are not written by disabled path
        assert name not in supervisor._start_fail_counts
        assert name not in supervisor._failure_reasons
        assert name not in supervisor._permanently_failed

        with patch("core.supervisor.manager.ProcessHandle") as MockHandle:
            result = await supervisor._respawn_anima_transaction(name)

        assert result is None
        MockHandle.assert_not_called()
        assert name not in supervisor._start_fail_counts
        assert name not in supervisor._failure_reasons
        assert name not in supervisor._permanently_failed
        assert name not in supervisor.processes

    @pytest.mark.asyncio
    async def test_handle_process_failure_disabled_no_restart_count(
        self,
        supervisor: ProcessSupervisor,
    ) -> None:
        from core.supervisor.process_handle import ProcessState

        name = "test-anima"
        _write_status(supervisor.animas_dir, name, enabled=False)
        supervisor.restart_policy.max_retries = 3

        old_handle = MagicMock()
        old_handle.state = ProcessState.FAILED
        supervisor.processes[name] = old_handle

        async def mock_stop(anima_name: str, **_kwargs) -> None:
            supervisor.processes.pop(anima_name, None)

        supervisor.stop_anima = AsyncMock(side_effect=mock_stop)

        with patch("core.supervisor.manager.asyncio.sleep", new_callable=AsyncMock):
            await supervisor._handle_process_failure(name, old_handle)

        assert name not in supervisor.processes
        assert name not in supervisor._restarting
        assert name not in supervisor._restart_counts
        assert name not in supervisor._permanently_failed
        assert name not in supervisor._start_fail_counts
        assert name not in supervisor._failure_reasons
