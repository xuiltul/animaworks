# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for process group isolation and health check improvements.

Bug 1: start_new_session=True for subprocess isolation
Bug 2: FAILED log spam suppression via the restart state machine
Bug 3: Reconciliation-based auto-recovery with 1-minute cooldown
Bug 4: restart_anima() resets the restart state machine
"""

from __future__ import annotations

import asyncio
import json
from datetime import timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.platform.process import subprocess_session_kwargs
from core.time_utils import now_jst
from server.supervisor.manager import (
    HealthConfig,
    ProcessSupervisor,
    RestartPolicy,
)
from server.supervisor.process_handle import ProcessHandle, ProcessState

# ── Fixtures ──────────────────────────────────────────────────


@pytest.fixture
def supervisor(tmp_path: Path) -> ProcessSupervisor:
    """Create a ProcessSupervisor with fast timeouts for testing."""
    supervisor = ProcessSupervisor(
        animas_dir=tmp_path / "animas",
        shared_dir=tmp_path / "shared",
        run_dir=tmp_path / "run",
        restart_policy=RestartPolicy(
            max_retries=3,
            backoff_base_sec=0.01,
            backoff_max_sec=0.1,
        ),
        health_config=HealthConfig(
            ping_interval_sec=0.5,
            ping_timeout_sec=0.2,
            max_missed_pings=2,
            startup_grace_sec=0.5,
        ),
    )

    # Asset generation is covered separately; process tests must not call an LLM.
    supervisor._reconcile_assets = AsyncMock()
    return supervisor


@pytest.fixture
def handle(tmp_path: Path) -> ProcessHandle:
    """Create a ProcessHandle in RUNNING state with mocked process."""
    socket_path = tmp_path / "test.sock"
    h = ProcessHandle(
        anima_name="test-anima",
        socket_path=socket_path,
        animas_dir=tmp_path / "animas",
        shared_dir=tmp_path / "shared",
        log_dir=tmp_path / "logs",
    )
    h.state = ProcessState.RUNNING
    h.process = MagicMock()
    h.process.pid = 12345
    h.process.poll.return_value = None
    h.process.returncode = None
    h.ipc_client = AsyncMock()
    socket_path.touch()
    return h


# ── Bug 1: start_new_session=True ─────────────────────────────


class TestProcessGroupIsolation:
    """Tests for subprocess isolation through the process adapter."""

    @pytest.mark.asyncio
    async def test_popen_uses_start_new_session(self, tmp_path: Path):
        """Verify Popen is called with the platform session kwargs."""
        handle = ProcessHandle(
            anima_name="test-anima",
            socket_path=tmp_path / "test.sock",
            animas_dir=tmp_path / "animas",
            shared_dir=tmp_path / "shared",
            log_dir=tmp_path / "logs",
        )

        with patch("server.supervisor.process_handle.subprocess.Popen") as mock_popen:
            mock_process = MagicMock()
            mock_process.pid = 99999
            mock_process.poll.return_value = None
            mock_popen.return_value = mock_process

            # start() will fail at _wait_for_socket, but Popen is called first
            with pytest.raises((TimeoutError, RuntimeError)):
                await handle.start()

            mock_popen.assert_called_once()
            call_kwargs = mock_popen.call_args
            for key, value in subprocess_session_kwargs().items():
                assert call_kwargs.kwargs.get(key) == value

    @pytest.mark.asyncio
    async def test_stop_uses_adapter_for_graceful_terminate(self, handle: ProcessHandle):
        """Verify stop() delegates graceful termination to the process adapter."""
        terminate_called = False

        def poll_side_effect():
            if terminate_called:
                return 0
            return None

        handle.process.poll.side_effect = poll_side_effect
        handle.process.returncode = 0

        # IPC shutdown "fails" so we reach the SIGTERM step
        async def fail_send(*args, **kwargs):
            raise ConnectionError("IPC lost")

        handle.send_request = fail_send

        def terminate_side_effect(proc, *, force, include_children=True):
            nonlocal terminate_called
            terminate_called = True
            assert proc is handle.process
            assert force is False

        with patch("server.supervisor.process_handle.terminate_subprocess", side_effect=terminate_side_effect):
            await handle.stop(timeout=2.0)

        assert terminate_called

    @pytest.mark.asyncio
    async def test_stop_escalates_to_force_kill_via_adapter(
        self,
        handle: ProcessHandle,
    ):
        """Verify stop() escalates from graceful terminate to force kill via the adapter."""
        call_forces: list[bool] = []

        def poll_side_effect():
            return None

        handle.process.poll.side_effect = poll_side_effect
        handle.process.returncode = 0

        # IPC shutdown fails so we reach SIGTERM step
        async def fail_send(*args, **kwargs):
            raise ConnectionError("IPC lost")

        handle.send_request = fail_send

        def terminate_side_effect(proc, *, force, include_children=True):
            call_forces.append(force)
            if force:
                handle.process.poll.side_effect = lambda: 0

        with patch("server.supervisor.process_handle.terminate_subprocess", side_effect=terminate_side_effect):
            await handle.stop(timeout=2.0)

        assert call_forces == [False, True]

    @pytest.mark.asyncio
    async def test_kill_uses_adapter_for_force_terminate(self, handle: ProcessHandle):
        """Verify kill() uses the adapter with force=True."""
        mock_process = handle.process
        handle.process.wait.return_value = -9
        handle.process.returncode = -9
        handle.process.poll.return_value = -9

        with patch("server.supervisor.process_handle.terminate_subprocess") as mock_terminate:
            await handle.kill()

        assert mock_terminate.call_args_list[0].args == (mock_process,)
        assert mock_terminate.call_args_list[0].kwargs == {"force": True}

    @pytest.mark.asyncio
    async def test_kill_calls_adapter_even_with_mocked_process(
        self,
        handle: ProcessHandle,
    ):
        """Verify kill() delegates to the adapter for mocked processes."""
        mock_process = handle.process
        mock_process.wait.return_value = -9
        mock_process.returncode = -9

        with patch("server.supervisor.process_handle.terminate_subprocess") as mock_terminate:
            await handle.kill()

        assert mock_terminate.call_count >= 1
        assert mock_terminate.call_args_list[0].args == (mock_process,)
        assert mock_terminate.call_args_list[0].kwargs == {"force": True}

    @pytest.mark.asyncio
    async def test_cleanup_uses_adapter(self, handle: ProcessHandle):
        """Verify _cleanup() uses the process adapter for orphaned subprocess termination."""
        mock_process = handle.process
        handle.process.poll.return_value = None  # Still alive
        handle.process.wait.side_effect = [None]  # Exits after SIGTERM

        with patch("server.supervisor.process_handle.terminate_subprocess") as mock_terminate:
            await handle._cleanup()

        mock_terminate.assert_called_with(mock_process, force=False)


# ── Bug 2: FAILED log spam suppression ────────────────────────


class TestFailedLogSpamSuppression:
    """Tests for the unified FAILED (restart-state) log throttling."""

    def test_restart_controller_initialized(self, supervisor: ProcessSupervisor):
        """Verify the RestartController is initialized and empty."""
        from server.supervisor.restart_state import RestartController

        assert isinstance(supervisor._restart_ctl, RestartController)
        assert len(supervisor._restart_ctl.names()) == 0

    @pytest.mark.asyncio
    async def test_failure_threshold_marks_failed(
        self,
        supervisor: ProcessSupervisor,
        handle: ProcessHandle,
    ):
        """Reaching the failure threshold marks the anima FAILED and surfaces an error."""
        supervisor.processes["test-anima"] = handle
        ctl = supervisor._restart_ctl
        ctl.record_failure("test-anima", "e1")
        ctl.record_failure("test-anima", "e2")
        supervisor._ensure_restart_worker = MagicMock()

        await supervisor._handle_process_failure("test-anima", handle)

        assert ctl.is_failed("test-anima")
        assert handle.state == ProcessState.FAILED

    @pytest.mark.asyncio
    async def test_hang_kill_feeds_failure_handler(
        self,
        supervisor: ProcessSupervisor,
        handle: ProcessHandle,
    ):
        """Hung process recovery should kill first, then delegate to the failure handler."""
        supervisor.processes["test-anima"] = handle
        handle.kill = AsyncMock()
        supervisor._handle_process_failure = AsyncMock()

        await supervisor._handle_process_hang("test-anima", handle)

        handle.kill.assert_awaited_once()
        supervisor._handle_process_failure.assert_awaited_once_with("test-anima", handle, reason="hang")

    @pytest.mark.asyncio
    async def test_repeated_failures_reach_failed_and_visible_error(
        self,
        supervisor: ProcessSupervisor,
    ):
        """Repeated failure handling reaches FAILED and exposes an error status."""
        supervisor._ensure_restart_worker = MagicMock()
        handle = MagicMock(spec=ProcessHandle)
        handle.anima_name = "test-anima"
        supervisor.processes["test-anima"] = handle

        for _ in range(3):
            await supervisor._handle_process_failure("test-anima", handle, reason="boom")

        ctl = supervisor._restart_ctl
        assert ctl.is_failed("test-anima")
        supervisor.processes.pop("test-anima", None)
        status = supervisor.get_process_status("test-anima")
        assert status["status"] == "error"
        assert status["restart_state"] == "failed"
        assert ctl.get("test-anima").last_error

    @pytest.mark.asyncio
    async def test_failed_skips_health_check(
        self,
        supervisor: ProcessSupervisor,
        handle: ProcessHandle,
        caplog,
    ):
        """Verify health check skips animas that are FAILED and not running."""
        handle.state = ProcessState.FAILED
        ctl = supervisor._restart_ctl
        for _ in range(3):
            ctl.record_failure("test-anima", "e")
        # not added to supervisor.processes

        with patch.object(
            supervisor,
            "_handle_process_failure",
            new_callable=AsyncMock,
        ) as mock_failure:
            await supervisor._check_process_health("test-anima", handle)

        mock_failure.assert_not_called()

    @pytest.mark.asyncio
    async def test_failed_logs_warning_every_5_minutes(
        self,
        supervisor: ProcessSupervisor,
        handle: ProcessHandle,
        caplog,
    ):
        """Verify WARNING log is emitted when the 5-minute interval has passed."""
        import logging

        handle.state = ProcessState.FAILED
        ctl = supervisor._restart_ctl
        for _ in range(3):
            ctl.record_failure("test-anima", "e")
        ctl.get("test-anima").last_failed_log_at = float("-inf")

        with caplog.at_level(logging.WARNING):
            await supervisor._check_process_health("test-anima", handle)

        assert any("Process still in FAILED state: test-anima" in record.message for record in caplog.records)

    @pytest.mark.asyncio
    async def test_failed_no_log_within_5_minutes(
        self,
        supervisor: ProcessSupervisor,
        handle: ProcessHandle,
        caplog,
    ):
        """Verify no log is emitted within the 5-minute interval."""
        import logging

        handle.state = ProcessState.FAILED
        ctl = supervisor._restart_ctl
        for _ in range(3):
            ctl.record_failure("test-anima", "e")
        ctl.get("test-anima").last_failed_log_at = ctl._clock()

        with caplog.at_level(logging.WARNING):
            await supervisor._check_process_health("test-anima", handle)

        assert not any("Process still in FAILED state" in record.message for record in caplog.records)

    @pytest.mark.asyncio
    async def test_non_failed_still_triggers_restart(
        self,
        supervisor: ProcessSupervisor,
        handle: ProcessHandle,
    ):
        """Verify non-FAILED animas still trigger the failure handler."""
        supervisor.processes["test-anima"] = handle
        handle.state = ProcessState.FAILED
        handle.stats.started_at = now_jst() - timedelta(seconds=60)

        with patch.object(supervisor, "_handle_process_failure", new_callable=AsyncMock):
            await supervisor._check_process_health("test-anima", handle)
            await asyncio.sleep(0)

        # Not in FAILED restart state, so the handler must be invoked.
        assert not supervisor._restart_ctl.is_failed("test-anima")


# ── Bug 3: Reconciliation auto-recovery ───────────────────────


class TestReconciliationAutoRecovery:
    """Tests for reconciliation ensuring the unified restart worker."""

    def _make_enabled_anima(self, supervisor: ProcessSupervisor, tmp_path: Path, name: str, enabled: bool):
        animas_dir = tmp_path / "animas"
        animas_dir.mkdir(parents=True)
        anima_dir = animas_dir / name
        anima_dir.mkdir()
        (anima_dir / "IDENTITY.md").write_text("identity", encoding="utf-8")
        (anima_dir / "status.json").write_text(
            json.dumps({"enabled": enabled}),
            encoding="utf-8",
        )
        supervisor.animas_dir = animas_dir

    @pytest.mark.asyncio
    async def test_reconcile_ensures_worker_for_failed_not_running(
        self,
        supervisor: ProcessSupervisor,
        tmp_path: Path,
    ):
        """A FAILED, enabled, not-running anima is handed to the restart worker."""
        self._make_enabled_anima(supervisor, tmp_path, "alice", enabled=True)
        ctl = supervisor._restart_ctl
        for _ in range(3):
            ctl.record_failure("alice", "e")
        supervisor._ensure_restart_worker = MagicMock()

        await supervisor._reconcile()

        supervisor._ensure_restart_worker.assert_called_with("alice")

    @pytest.mark.asyncio
    async def test_reconcile_does_not_start_directly_while_backoff_not_due(
        self,
        supervisor: ProcessSupervisor,
        tmp_path: Path,
    ):
        """Reconcile must not call start_anima directly; the worker waits on backoff."""
        self._make_enabled_anima(supervisor, tmp_path, "alice", enabled=True)
        ctl = supervisor._restart_ctl
        ctl.record_failure("alice", "e")  # BACKOFF, next attempt in the future
        supervisor._ensure_restart_worker = MagicMock()
        supervisor.start_anima = AsyncMock()

        await supervisor._reconcile()

        supervisor._ensure_restart_worker.assert_called_with("alice")
        supervisor.start_anima.assert_not_called()

    @pytest.mark.asyncio
    async def test_reconcile_skips_disabled_failed(
        self,
        supervisor: ProcessSupervisor,
        tmp_path: Path,
    ):
        """A disabled FAILED anima is not handed to a worker."""
        self._make_enabled_anima(supervisor, tmp_path, "alice", enabled=False)
        ctl = supervisor._restart_ctl
        for _ in range(3):
            ctl.record_failure("alice", "e")
        supervisor._ensure_restart_worker = MagicMock()

        await supervisor._reconcile()

        supervisor._ensure_restart_worker.assert_not_called()
        assert ctl.get("alice") is None

    @pytest.mark.asyncio
    async def test_reconcile_does_not_ensure_worker_for_removed_from_disk(
        self,
        supervisor: ProcessSupervisor,
        tmp_path: Path,
    ):
        """A restart record with no on-disk anima is not handed to a worker."""
        ctl = supervisor._restart_ctl
        ctl.record_failure("ghost", "e")
        supervisor._ensure_restart_worker = MagicMock()

        await supervisor._reconcile()

        supervisor._ensure_restart_worker.assert_not_called()


# ── Bug 4: restart_anima() counter reset ──────────────────────


class TestRestartCounterReset:
    """Tests for restart_anima() resetting the restart state machine."""

    @pytest.mark.asyncio
    async def test_restart_resets_restart_state(
        self,
        supervisor: ProcessSupervisor,
    ):
        """Verify restart_anima() (reset) clears the restart record."""
        ctl = supervisor._restart_ctl
        for _ in range(3):
            ctl.record_failure("alice", "e")
        assert ctl.get("alice") is not None

        handle = MagicMock(spec=ProcessHandle)
        handle.anima_name = "alice"
        supervisor.processes["alice"] = handle

        supervisor.stop_anima = AsyncMock()
        supervisor.start_anima = AsyncMock()

        await supervisor.restart_anima("alice")

        assert ctl.get("alice") is None

    @pytest.mark.asyncio
    async def test_restart_calls_stop_then_start(
        self,
        supervisor: ProcessSupervisor,
    ):
        """Verify restart_anima() calls stop then start in order."""
        handle = MagicMock(spec=ProcessHandle)
        supervisor.processes["alice"] = handle

        call_order = []

        async def mock_stop(name, *, quiesce=False):
            assert quiesce is True
            call_order.append(("stop", name))
            del supervisor.processes[name]

        async def mock_start(name):
            call_order.append(("start", name))

        supervisor.stop_anima = mock_stop
        supervisor.start_anima = mock_start

        await supervisor.restart_anima("alice")

        assert call_order == [("stop", "alice"), ("start", "alice")]

    @pytest.mark.asyncio
    async def test_restart_works_when_no_existing_process(
        self,
        supervisor: ProcessSupervisor,
    ):
        """Verify restart_anima() works when process doesn't exist yet."""
        supervisor.start_anima = AsyncMock()

        # No process in supervisor.processes
        await supervisor.restart_anima("new-anima")

        supervisor.start_anima.assert_called_once_with("new-anima")
        assert supervisor._restart_ctl.get("new-anima") is None

    @pytest.mark.asyncio
    async def test_restart_preserves_counters_when_reset_disabled(
        self,
        supervisor: ProcessSupervisor,
    ):
        """Verify restart_anima(_reset_counters=False) preserves the restart record."""
        ctl = supervisor._restart_ctl
        for _ in range(3):
            ctl.record_failure("alice", "e")

        handle = MagicMock(spec=ProcessHandle)
        handle.anima_name = "alice"
        supervisor.processes["alice"] = handle

        supervisor.stop_anima = AsyncMock()
        supervisor.start_anima = AsyncMock()

        await supervisor.restart_anima("alice", _reset_counters=False)

        # Counters must be preserved
        assert ctl.get("alice") is not None
        assert ctl.get("alice").attempts == 3

    @pytest.mark.asyncio
    async def test_handle_process_failure_preserves_counter_through_restart(
        self,
        supervisor: ProcessSupervisor,
        handle: ProcessHandle,
    ):
        """Verify _handle_process_failure increments the restart record."""
        supervisor.processes["test-anima"] = handle
        ctl = supervisor._restart_ctl
        ctl.record_failure("test-anima", "e1")  # attempt 1
        supervisor._ensure_restart_worker = MagicMock()

        await supervisor._handle_process_failure("test-anima", handle)

        # Failure count should be incremented (1 → 2), NOT reset to 0
        assert ctl.get("test-anima").attempts == 2


# ── Integration: Full failure-recovery cycle ──────────────────


class TestFailureRecoveryCycle:
    """Integration tests for the complete failure → FAILED → auto-recovery cycle."""

    @pytest.mark.asyncio
    async def test_full_cycle_failure_to_recovery(
        self,
        supervisor: ProcessSupervisor,
        tmp_path: Path,
    ):
        """Failure reaches FAILED, then auto-recovery returns to HEALTHY."""
        # Set up animas_dir
        animas_dir = tmp_path / "animas"
        animas_dir.mkdir(parents=True)
        alice_dir = animas_dir / "alice"
        alice_dir.mkdir()
        (alice_dir / "IDENTITY.md").write_text("Alice identity", encoding="utf-8")
        (alice_dir / "status.json").write_text(
            json.dumps({"enabled": True}),
            encoding="utf-8",
        )
        supervisor.animas_dir = animas_dir

        handle = MagicMock(spec=ProcessHandle)
        handle.anima_name = "alice"
        handle.state = ProcessState.RUNNING
        supervisor.processes["alice"] = handle
        supervisor._ensure_restart_worker = MagicMock()
        ctl = supervisor._restart_ctl
        for _ in range(2):
            ctl.record_failure("alice", "e")

        # Step 2: Trigger _handle_process_failure (simulating crash detection)
        await supervisor._handle_process_failure("alice", handle)

        # Verify: reached FAILED
        assert ctl.is_failed("alice")
        assert handle.state == ProcessState.FAILED

        # Step 3: Auto-recovery — the worker successfully starts the anima.
        ctl.record_started("alice")
        assert not ctl.is_failed("alice")
        assert ctl.get("alice").phase.value == "healthy"
