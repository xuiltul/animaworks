# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
"""E2E tests for bootstrap protection during reconciliation.

Tests the complete flow: bootstrapping state + reconciliation cycles
to ensure bootstrapping animas survive reconciliation without being killed.
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.supervisor.manager import ProcessSupervisor, ReconciliationConfig
from core.supervisor.process_handle import ProcessState


# ── Helpers ──────────────────────────────────────────────────────────


def _make_supervisor(tmp_path: Path) -> ProcessSupervisor:
    """Create a minimal ProcessSupervisor with fast reconciliation."""
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
    enabled: bool = True,
    has_bootstrap: bool = False,
) -> Path:
    """Create a complete anima directory on disk."""
    d = animas_dir / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "identity.md").write_text(f"# {name}", encoding="utf-8")
    (d / "status.json").write_text(
        json.dumps({"enabled": enabled}), encoding="utf-8"
    )
    if has_bootstrap:
        (d / "bootstrap.md").write_text("# Bootstrap", encoding="utf-8")
    return d


def _mock_handle(state: ProcessState = ProcessState.RUNNING) -> MagicMock:
    """Create a mock ProcessHandle."""
    handle = MagicMock()
    handle.state = state
    return handle


# ── E2E Tests ────────────────────────────────────────────────────────


class TestBootstrapReconciliationE2E:
    """End-to-end tests for bootstrap + reconciliation interaction."""

    async def test_bootstrapping_anima_survives_full_reconciliation_cycle(
        self, tmp_path: Path
    ):
        """Simulate a full scenario: anima starts, bootstrapping begins,
        reconciliation runs, anima should NOT be killed.

        This tests the complete flow described in the issue:
        1. Anima starts and bootstrap begins
        2. Reconciliation cycle runs
        3. Anima should still be present (not stopped)
        """
        sup = _make_supervisor(tmp_path)
        anima_dir = _create_anima_dir(
            sup.animas_dir, "kotoha", enabled=True, has_bootstrap=True
        )

        # Simulate: kotoha is running and bootstrapping
        sup.processes["kotoha"] = _mock_handle()
        sup._bootstrapping.add("kotoha")
        sup.stop_anima = AsyncMock()
        sup.start_anima = AsyncMock()

        # Run reconciliation multiple times (simulating 30s intervals)
        with patch.object(sup, "_reconcile_assets", new_callable=AsyncMock):
            for _ in range(5):
                await sup._reconcile()

        # kotoha should never be stopped
        sup.stop_anima.assert_not_called()
        # kotoha is running, so start should not be called either
        sup.start_anima.assert_not_called()

    async def test_bootstrap_complete_then_reconciliation_respects_state(
        self, tmp_path: Path
    ):
        """After bootstrap completes, reconciliation should work normally.

        1. Bootstrap is in progress -> reconciliation skips
        2. Bootstrap completes -> reconciliation acts normally
        """
        sup = _make_supervisor(tmp_path)
        _create_anima_dir(
            sup.animas_dir, "kotoha", enabled=False, has_bootstrap=True
        )

        sup.processes["kotoha"] = _mock_handle()
        sup._bootstrapping.add("kotoha")
        sup.stop_anima = AsyncMock()

        with patch.object(sup, "_reconcile_assets", new_callable=AsyncMock):
            # Phase 1: bootstrapping -> stop deferred
            await sup._reconcile()
            sup.stop_anima.assert_not_called()

            # Phase 2: bootstrap completes -> disabled anima stopped
            sup._bootstrapping.discard("kotoha")
            await sup._reconcile()
            sup.stop_anima.assert_called_once_with("kotoha")

    async def test_disabled_during_bootstrap_then_stopped_after(
        self, tmp_path: Path
    ):
        """An anima disabled during bootstrap should be stopped after
        bootstrap completes, not during.
        """
        sup = _make_supervisor(tmp_path)

        # Start as enabled
        anima_dir = _create_anima_dir(
            sup.animas_dir, "kotoha", enabled=True, has_bootstrap=True
        )
        sup.processes["kotoha"] = _mock_handle()
        sup._bootstrapping.add("kotoha")
        sup.stop_anima = AsyncMock()
        sup.start_anima = AsyncMock()

        with patch.object(sup, "_reconcile_assets", new_callable=AsyncMock):
            # Reconciliation while enabled + bootstrapping -> no action
            await sup._reconcile()
            sup.stop_anima.assert_not_called()

            # User disables the anima while bootstrap is running
            (anima_dir / "status.json").write_text(
                json.dumps({"enabled": False}), encoding="utf-8"
            )

            # Reconciliation while disabled + bootstrapping -> deferred
            await sup._reconcile()
            sup.stop_anima.assert_not_called()

            # Bootstrap completes
            sup._bootstrapping.discard("kotoha")

            # Reconciliation while disabled + not bootstrapping -> stopped
            await sup._reconcile()
            sup.stop_anima.assert_called_once_with("kotoha")

    async def test_retry_limit_prevents_infinite_loop(self, tmp_path: Path):
        """Simulate the infinite loop scenario: repeated bootstrap failures
        should eventually disable further attempts by renaming bootstrap.md.
        """
        sup = _make_supervisor(tmp_path)
        anima_dir = _create_anima_dir(
            sup.animas_dir, "kotoha", enabled=True, has_bootstrap=True
        )
        sup.ws_manager = None

        # Simulate 3 failures (process not found each time)
        for i in range(3):
            # No process handle -> immediate failure
            sup.processes.pop("kotoha", None)
            await sup._run_bootstrap("kotoha")
            assert sup._bootstrap_retry_counts["kotoha"] == i + 1
            assert (anima_dir / "bootstrap.md").exists()

        # 4th attempt: max retries reached, bootstrap.md renamed
        await sup._run_bootstrap("kotoha")
        assert not (anima_dir / "bootstrap.md").exists()
        assert (anima_dir / "bootstrap.md.failed").exists()

        # needs_bootstrap would now return False since bootstrap.md is gone
        # This breaks the infinite loop

    async def test_successful_bootstrap_resets_retry_count(self, tmp_path: Path):
        """A successful bootstrap after failures should reset the retry counter,
        allowing future bootstraps to try again from zero.
        """
        sup = _make_supervisor(tmp_path)
        _create_anima_dir(
            sup.animas_dir, "kotoha", enabled=True, has_bootstrap=True
        )
        sup.ws_manager = None

        # 2 failures
        for _ in range(2):
            sup.processes.pop("kotoha", None)
            await sup._run_bootstrap("kotoha")

        assert sup._bootstrap_retry_counts["kotoha"] == 2

        # Successful bootstrap
        mock_response = MagicMock()
        mock_response.error = None
        mock_response.result = {"duration_ms": 5000}
        mock_handle = _mock_handle()
        mock_handle.send_request = AsyncMock(return_value=mock_response)
        sup.processes["kotoha"] = mock_handle

        await sup._run_bootstrap("kotoha")

        # Counter should be reset
        assert "kotoha" not in sup._bootstrap_retry_counts

    async def test_mixed_animas_bootstrap_isolation(self, tmp_path: Path):
        """Bootstrap protection for one anima should not affect others.

        - kotoha: bootstrapping, enabled -> protected
        - sakura: not bootstrapping, disabled, running -> stopped
        - hinata: not bootstrapping, enabled, not running -> started
        """
        sup = _make_supervisor(tmp_path)
        _create_anima_dir(sup.animas_dir, "kotoha", enabled=True)
        _create_anima_dir(sup.animas_dir, "sakura", enabled=False)
        _create_anima_dir(sup.animas_dir, "hinata", enabled=True)

        sup.processes["kotoha"] = _mock_handle()
        sup.processes["sakura"] = _mock_handle()
        sup._bootstrapping.add("kotoha")
        # hinata: not in processes (not running)

        sup.start_anima = AsyncMock()
        sup.stop_anima = AsyncMock()

        with patch.object(sup, "_reconcile_assets", new_callable=AsyncMock):
            await sup._reconcile()

        # sakura should be stopped (disabled, not bootstrapping)
        sup.stop_anima.assert_called_once_with("sakura")
        # hinata should be started (enabled, not running, not bootstrapping)
        sup.start_anima.assert_called_once_with("hinata")
