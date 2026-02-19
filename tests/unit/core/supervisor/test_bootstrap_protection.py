# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for _bootstrapping guard in _reconcile() and bootstrap retry limits.

Verifies:
- _reconcile() skips bootstrapping animas in all 3 decision points
- _run_bootstrap() logs warnings when process is not running after bootstrap
- _run_bootstrap() tracks retry counts and renames bootstrap.md on max retries
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch

import pytest

from core.supervisor.manager import ProcessSupervisor
from core.supervisor.process_handle import ProcessState


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
    has_bootstrap: bool = False,
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
    if has_bootstrap:
        (d / "bootstrap.md").write_text("# Bootstrap instructions", encoding="utf-8")
    return d


def _mock_handle(state: ProcessState = ProcessState.RUNNING) -> MagicMock:
    """Create a mock ProcessHandle with given state."""
    handle = MagicMock()
    handle.state = state
    return handle


# ── Tests: _reconcile() bootstrapping guard ──────────────────────────


class TestReconcileBootstrappingGuard:
    """Tests for _bootstrapping guard in _reconcile()."""

    async def test_enabled_not_running_skipped_when_bootstrapping(self, tmp_path: Path):
        """enabled + not running: bootstrapping anima must NOT be started."""
        sup = _make_supervisor(tmp_path)
        _create_anima_dir(sup.animas_dir, "kotoha", enabled=True)

        # kotoha is not in processes (not running) but is bootstrapping
        sup._bootstrapping.add("kotoha")
        sup.start_anima = AsyncMock()

        with patch.object(sup, "_reconcile_assets", new_callable=AsyncMock):
            await sup._reconcile()

        sup.start_anima.assert_not_called()

    async def test_disabled_running_deferred_when_bootstrapping(self, tmp_path: Path):
        """disabled + running: bootstrapping anima must NOT be stopped."""
        sup = _make_supervisor(tmp_path)
        _create_anima_dir(sup.animas_dir, "kotoha", enabled=False)

        sup.processes["kotoha"] = _mock_handle()
        sup._bootstrapping.add("kotoha")
        sup.stop_anima = AsyncMock()

        with patch.object(sup, "_reconcile_assets", new_callable=AsyncMock):
            await sup._reconcile()

        sup.stop_anima.assert_not_called()

    async def test_removed_from_disk_deferred_when_bootstrapping(self, tmp_path: Path):
        """removed from disk + running: bootstrapping anima must NOT be stopped."""
        sup = _make_supervisor(tmp_path)
        # No directory for kotoha on disk

        sup.processes["kotoha"] = _mock_handle()
        sup._bootstrapping.add("kotoha")
        sup.stop_anima = AsyncMock()

        with patch.object(sup, "_reconcile_assets", new_callable=AsyncMock):
            await sup._reconcile()

        sup.stop_anima.assert_not_called()

    async def test_enabled_not_running_started_after_bootstrap_completes(self, tmp_path: Path):
        """After bootstrap completes, reconciliation should start the anima normally."""
        sup = _make_supervisor(tmp_path)
        _create_anima_dir(sup.animas_dir, "kotoha", enabled=True)

        # Not bootstrapping, not running -> should start
        sup.start_anima = AsyncMock()

        with patch.object(sup, "_reconcile_assets", new_callable=AsyncMock):
            await sup._reconcile()

        sup.start_anima.assert_called_once_with("kotoha")

    async def test_disabled_running_stopped_after_bootstrap_completes(self, tmp_path: Path):
        """After bootstrap completes, disabled anima should be stopped."""
        sup = _make_supervisor(tmp_path)
        _create_anima_dir(sup.animas_dir, "kotoha", enabled=False)

        sup.processes["kotoha"] = _mock_handle()
        # Not in _bootstrapping -> should stop
        sup.stop_anima = AsyncMock()

        with patch.object(sup, "_reconcile_assets", new_callable=AsyncMock):
            await sup._reconcile()

        sup.stop_anima.assert_called_once_with("kotoha")

    async def test_removed_running_stopped_after_bootstrap_completes(self, tmp_path: Path):
        """After bootstrap completes, removed anima should be stopped."""
        sup = _make_supervisor(tmp_path)
        # No directory on disk

        sup.processes["kotoha"] = _mock_handle()
        # Not in _bootstrapping -> should stop
        sup.stop_anima = AsyncMock()

        with patch.object(sup, "_reconcile_assets", new_callable=AsyncMock):
            await sup._reconcile()

        sup.stop_anima.assert_called_once_with("kotoha")

    async def test_restarting_guard_still_works(self, tmp_path: Path):
        """_restarting guard should still function alongside _bootstrapping."""
        sup = _make_supervisor(tmp_path)
        _create_anima_dir(sup.animas_dir, "kotoha", enabled=True)

        sup._restarting.add("kotoha")
        sup.start_anima = AsyncMock()

        with patch.object(sup, "_reconcile_assets", new_callable=AsyncMock):
            await sup._reconcile()

        sup.start_anima.assert_not_called()


# ── Tests: _run_bootstrap() process state warning ───────────────────


class TestBootstrapProcessStateWarning:
    """Tests for process state check in _run_bootstrap() finally block."""

    async def test_warning_when_process_not_found_after_bootstrap(self, tmp_path: Path):
        """Warning should be logged when process is gone after bootstrap."""
        sup = _make_supervisor(tmp_path)
        _create_anima_dir(sup.animas_dir, "kotoha", has_bootstrap=True)
        sup.ws_manager = None

        # Process exists at start but will be removed before finally block
        mock_response = MagicMock()
        mock_response.error = {"message": "killed"}

        async def _send_request_and_remove(*args, **kwargs):
            """Simulate reconciliation killing the process mid-bootstrap."""
            del sup.processes["kotoha"]
            return mock_response

        mock_handle = _mock_handle()
        mock_handle.send_request = AsyncMock(side_effect=_send_request_and_remove)
        sup.processes["kotoha"] = mock_handle

        with patch("core.supervisor.manager.logger") as mock_logger:
            await sup._run_bootstrap("kotoha")

            # Check that warning was logged
            warning_calls = [
                call for call in mock_logger.warning.call_args_list
                if "process not running" in str(call)
            ]
            assert len(warning_calls) == 1

    async def test_no_warning_when_process_running_after_success(self, tmp_path: Path):
        """No warning when bootstrap succeeds and process is still running."""
        sup = _make_supervisor(tmp_path)
        _create_anima_dir(sup.animas_dir, "kotoha", has_bootstrap=True)
        sup.ws_manager = None

        mock_response = MagicMock()
        mock_response.error = None
        mock_response.result = {"duration_ms": 5000}
        mock_handle = _mock_handle(ProcessState.RUNNING)
        mock_handle.send_request = AsyncMock(return_value=mock_response)
        sup.processes["kotoha"] = mock_handle

        with patch("core.supervisor.manager.logger") as mock_logger:
            await sup._run_bootstrap("kotoha")

            warning_calls = [
                call for call in mock_logger.warning.call_args_list
                if "process not running" in str(call)
            ]
            assert len(warning_calls) == 0


# ── Tests: Bootstrap retry limit ─────────────────────────────────────


class TestBootstrapRetryLimit:
    """Tests for bootstrap retry count tracking and max retry behavior."""

    async def test_retry_count_incremented_on_failure(self, tmp_path: Path):
        """Retry counter should increment after failed bootstrap."""
        sup = _make_supervisor(tmp_path)
        _create_anima_dir(sup.animas_dir, "kotoha", has_bootstrap=True)
        sup.ws_manager = None

        mock_response = MagicMock()
        mock_response.error = {"message": "failed"}
        mock_handle = _mock_handle()
        mock_handle.send_request = AsyncMock(return_value=mock_response)
        sup.processes["kotoha"] = mock_handle

        assert sup._bootstrap_retry_counts.get("kotoha", 0) == 0

        await sup._run_bootstrap("kotoha")

        assert sup._bootstrap_retry_counts["kotoha"] == 1

    async def test_retry_count_reset_on_success(self, tmp_path: Path):
        """Retry counter should be cleared after successful bootstrap."""
        sup = _make_supervisor(tmp_path)
        _create_anima_dir(sup.animas_dir, "kotoha", has_bootstrap=True)
        sup.ws_manager = None
        sup._bootstrap_retry_counts["kotoha"] = 2

        mock_response = MagicMock()
        mock_response.error = None
        mock_response.result = {"duration_ms": 5000}
        mock_handle = _mock_handle()
        mock_handle.send_request = AsyncMock(return_value=mock_response)
        sup.processes["kotoha"] = mock_handle

        await sup._run_bootstrap("kotoha")

        assert "kotoha" not in sup._bootstrap_retry_counts

    async def test_bootstrap_md_renamed_on_max_retries(self, tmp_path: Path):
        """bootstrap.md should be renamed to .failed after max retries."""
        sup = _make_supervisor(tmp_path)
        anima_dir = _create_anima_dir(
            sup.animas_dir, "kotoha", has_bootstrap=True
        )
        sup.ws_manager = None
        sup._bootstrap_retry_counts["kotoha"] = 3  # Already at max (default=3)

        await sup._run_bootstrap("kotoha")

        assert not (anima_dir / "bootstrap.md").exists()
        assert (anima_dir / "bootstrap.md.failed").exists()

    async def test_bootstrap_not_started_on_max_retries(self, tmp_path: Path):
        """Bootstrap should not add to _bootstrapping set when max retries reached."""
        sup = _make_supervisor(tmp_path)
        _create_anima_dir(sup.animas_dir, "kotoha", has_bootstrap=True)
        sup.ws_manager = None
        sup._bootstrap_retry_counts["kotoha"] = 3

        await sup._run_bootstrap("kotoha")

        assert "kotoha" not in sup._bootstrapping

    async def test_retry_count_incremented_on_timeout(self, tmp_path: Path):
        """Retry counter should increment on timeout."""
        sup = _make_supervisor(tmp_path)
        _create_anima_dir(sup.animas_dir, "kotoha", has_bootstrap=True)
        sup.ws_manager = None

        import asyncio

        mock_handle = _mock_handle()
        mock_handle.send_request = AsyncMock(side_effect=asyncio.TimeoutError)
        sup.processes["kotoha"] = mock_handle

        await sup._run_bootstrap("kotoha")

        assert sup._bootstrap_retry_counts["kotoha"] == 1

    async def test_retry_count_incremented_on_exception(self, tmp_path: Path):
        """Retry counter should increment on unexpected exception."""
        sup = _make_supervisor(tmp_path)
        _create_anima_dir(sup.animas_dir, "kotoha", has_bootstrap=True)
        sup.ws_manager = None

        mock_handle = _mock_handle()
        mock_handle.send_request = AsyncMock(side_effect=RuntimeError("IPC broken"))
        sup.processes["kotoha"] = mock_handle

        await sup._run_bootstrap("kotoha")

        assert sup._bootstrap_retry_counts["kotoha"] == 1

    async def test_progressive_retry_counting(self, tmp_path: Path):
        """Multiple failures should progressively increase the count."""
        sup = _make_supervisor(tmp_path)
        _create_anima_dir(sup.animas_dir, "kotoha", has_bootstrap=True)
        sup.ws_manager = None

        mock_response = MagicMock()
        mock_response.error = {"message": "failed"}
        mock_handle = _mock_handle()
        mock_handle.send_request = AsyncMock(return_value=mock_response)
        sup.processes["kotoha"] = mock_handle

        # Run 3 failed bootstraps
        for expected_count in range(1, 4):
            await sup._run_bootstrap("kotoha")
            assert sup._bootstrap_retry_counts["kotoha"] == expected_count

        # 4th attempt should hit max retries and rename bootstrap.md
        await sup._run_bootstrap("kotoha")
        assert not (sup.animas_dir / "kotoha" / "bootstrap.md").exists()
        assert (sup.animas_dir / "kotoha" / "bootstrap.md.failed").exists()

    async def test_max_retries_broadcast_event(self, tmp_path: Path):
        """Max retries should broadcast a max_retries_exceeded event."""
        sup = _make_supervisor(tmp_path)
        _create_anima_dir(sup.animas_dir, "kotoha", has_bootstrap=True)
        sup._bootstrap_retry_counts["kotoha"] = 3
        sup._broadcast_event = AsyncMock()

        await sup._run_bootstrap("kotoha")

        sup._broadcast_event.assert_called_once_with(
            "anima.bootstrap",
            {"name": "kotoha", "status": "max_retries_exceeded"},
        )

    async def test_bootstrapping_set_cleared_after_failure(self, tmp_path: Path):
        """_bootstrapping set must be cleared even after failure."""
        sup = _make_supervisor(tmp_path)
        _create_anima_dir(sup.animas_dir, "kotoha", has_bootstrap=True)
        sup.ws_manager = None

        mock_handle = _mock_handle()
        mock_handle.send_request = AsyncMock(side_effect=RuntimeError("fail"))
        sup.processes["kotoha"] = mock_handle

        await sup._run_bootstrap("kotoha")

        assert "kotoha" not in sup._bootstrapping
