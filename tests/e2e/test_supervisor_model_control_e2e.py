# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
"""E2E tests for supervisor model-control tools and reconciliation restart_requested.

Tests cover:
- set_subordinate_model updates config.json with new model
- restart_subordinate writes restart_requested: true to status.json
- set_subordinate_model + restart_subordinate in sequence
- Reconciliation detects restart_requested and calls restart_anima
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.config.models import AnimaModelConfig, load_config, save_config
from core.supervisor.manager import ProcessSupervisor
from core.tooling.handler import ToolHandler


# ── Helpers ───────────────────────────────────────────────────────────────────


def _write_status(anima_dir: Path, **fields) -> None:
    """Write (or overwrite) status.json in *anima_dir* with *fields*."""
    status_file = anima_dir / "status.json"
    status_file.write_text(
        json.dumps(fields, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _read_status(anima_dir: Path) -> dict:
    """Read status.json from *anima_dir* and return as dict."""
    return json.loads((anima_dir / "status.json").read_text(encoding="utf-8"))


def _make_tool_handler(supervisor_dir: Path, supervisor_name: str) -> ToolHandler:
    """Build a minimal ToolHandler for the supervisor anima."""
    memory = MagicMock()
    memory.read_permissions.return_value = ""

    return ToolHandler(
        anima_dir=supervisor_dir,
        memory=memory,
        messenger=None,
        tool_registry=[],
    )


def _mock_config_for_check(animas: dict[str, dict]) -> MagicMock:
    """Build a MagicMock config with AnimaModelConfig entries for _check_subordinate."""
    config = MagicMock()
    config.animas = {
        name: AnimaModelConfig(**fields)
        for name, fields in animas.items()
    }
    return config


# ── E2E Tests ─────────────────────────────────────────────────────────────────


@pytest.mark.e2e
class TestSetSubordinateModelE2E:
    """E2E1: set_subordinate_model updates config.json."""

    def test_set_subordinate_model_updates_config(
        self, data_dir: Path, tmp_path: Path
    ) -> None:
        """set_subordinate_model should write new model to config.json animas.worker."""
        from core.config import invalidate_cache

        animas_dir = data_dir / "animas"

        # Create supervisor anima directory
        supervisor_dir = animas_dir / "supervisor"
        supervisor_dir.mkdir(parents=True, exist_ok=True)

        # Create worker anima directory
        worker_dir = animas_dir / "worker"
        worker_dir.mkdir(parents=True, exist_ok=True)
        _write_status(worker_dir, enabled=True, supervisor="supervisor", model="claude-sonnet-4-20250514")

        # Register both animas in config.json via load/save
        config = load_config()
        config.animas["supervisor"] = AnimaModelConfig()
        config.animas["worker"] = AnimaModelConfig(
            model="claude-sonnet-4-20250514",
            supervisor="supervisor",
        )
        save_config(config)
        invalidate_cache()

        handler = _make_tool_handler(supervisor_dir, "supervisor")

        # _check_subordinate calls load_config() — use real config via data_dir env
        result = handler.handle(
            "set_subordinate_model",
            {"name": "worker", "model": "claude-opus-4-6", "reason": "e2e test"},
        )

        assert "claude-opus-4-6" in result, f"Unexpected result: {result}"

        # Reload config from disk and verify
        invalidate_cache()
        updated_config = load_config()
        assert updated_config.animas["worker"].model == "claude-opus-4-6"


@pytest.mark.e2e
class TestRestartSubordinateE2E:
    """E2E2: restart_subordinate sets restart_requested: true in status.json."""

    def test_restart_subordinate_sets_flag(
        self, data_dir: Path, tmp_path: Path
    ) -> None:
        """restart_subordinate should write restart_requested: true to status.json."""
        from core.config import invalidate_cache

        animas_dir = data_dir / "animas"

        # Create supervisor anima directory
        supervisor_dir = animas_dir / "supervisor"
        supervisor_dir.mkdir(parents=True, exist_ok=True)

        # Create worker anima directory with existing status.json (enabled: true)
        worker_dir = animas_dir / "worker"
        worker_dir.mkdir(parents=True, exist_ok=True)
        _write_status(worker_dir, enabled=True, supervisor="supervisor", model="claude-sonnet-4-20250514")

        # Register in config.json
        config = load_config()
        config.animas["supervisor"] = AnimaModelConfig()
        config.animas["worker"] = AnimaModelConfig(
            model="claude-sonnet-4-20250514",
            supervisor="supervisor",
        )
        save_config(config)
        invalidate_cache()

        handler = _make_tool_handler(supervisor_dir, "supervisor")

        result = handler.handle(
            "restart_subordinate",
            {"name": "worker", "reason": "e2e test restart"},
        )

        assert "再起動" in result, f"Unexpected result: {result}"

        status = _read_status(worker_dir)
        assert status.get("restart_requested") is True, (
            f"restart_requested should be True in status.json; got: {status}"
        )
        # Original fields must be preserved
        assert status.get("enabled") is True
        assert status.get("supervisor") == "supervisor"


@pytest.mark.e2e
class TestSetModelAndRestartSequenceE2E:
    """E2E3: set_subordinate_model + restart_subordinate in sequence."""

    def test_model_change_then_restart(
        self, data_dir: Path, tmp_path: Path
    ) -> None:
        """Sequential set_subordinate_model + restart_subordinate should update
        both config.json model and status.json restart_requested."""
        from core.config import invalidate_cache

        animas_dir = data_dir / "animas"

        supervisor_dir = animas_dir / "supervisor"
        supervisor_dir.mkdir(parents=True, exist_ok=True)

        worker_dir = animas_dir / "worker"
        worker_dir.mkdir(parents=True, exist_ok=True)
        _write_status(worker_dir, enabled=True, supervisor="supervisor", model="claude-sonnet-4-20250514")

        config = load_config()
        config.animas["supervisor"] = AnimaModelConfig()
        config.animas["worker"] = AnimaModelConfig(
            model="claude-sonnet-4-20250514",
            supervisor="supervisor",
        )
        save_config(config)
        invalidate_cache()

        handler = _make_tool_handler(supervisor_dir, "supervisor")

        # Step 1: change model
        result1 = handler.handle(
            "set_subordinate_model",
            {"name": "worker", "model": "claude-opus-4-6"},
        )
        assert "claude-opus-4-6" in result1, f"Unexpected result1: {result1}"

        # Step 2: request restart
        result2 = handler.handle(
            "restart_subordinate",
            {"name": "worker"},
        )
        assert "再起動" in result2, f"Unexpected result2: {result2}"

        # Verify config.json model update
        invalidate_cache()
        updated_config = load_config()
        assert updated_config.animas["worker"].model == "claude-opus-4-6", (
            f"Expected model 'claude-opus-4-6' in config.json; "
            f"got: {updated_config.animas['worker'].model}"
        )

        # Verify status.json restart flag
        status = _read_status(worker_dir)
        assert status.get("restart_requested") is True, (
            f"restart_requested should be True; got: {status}"
        )


@pytest.mark.e2e
class TestReconciliationRestartRequestedE2E:
    """E2E4: Reconciliation detects restart_requested and calls restart_anima."""

    @pytest.mark.asyncio
    async def test_reconcile_detects_restart_requested(
        self, data_dir: Path
    ) -> None:
        """When status.json has restart_requested: true, _reconcile() should call
        restart_anima and clear the flag from status.json."""
        animas_dir = data_dir / "animas"
        shared_dir = data_dir / "shared"
        run_dir = data_dir / "run"
        run_dir.mkdir(parents=True, exist_ok=True)

        # Create worker with both identity.md and status.json (restart_requested)
        worker_dir = animas_dir / "worker"
        worker_dir.mkdir(parents=True, exist_ok=True)
        (worker_dir / "identity.md").write_text(
            "# Worker\nTest anima.", encoding="utf-8"
        )
        _write_status(
            worker_dir,
            enabled=True,
            supervisor="supervisor",
            model="claude-sonnet-4-20250514",
            restart_requested=True,
        )

        supervisor = ProcessSupervisor(
            animas_dir=animas_dir,
            shared_dir=shared_dir,
            run_dir=run_dir,
        )

        with (
            patch.object(supervisor, "restart_anima", new_callable=AsyncMock) as mock_restart,
            patch.object(supervisor, "start_anima", new_callable=AsyncMock),
            patch.object(supervisor, "_reconcile_assets", new_callable=AsyncMock),
        ):
            await supervisor._reconcile()

        # restart_anima should have been called for "worker"
        restarted_names = [call.args[0] for call in mock_restart.call_args_list]
        assert "worker" in restarted_names, (
            f"restart_anima should have been called for 'worker'; "
            f"calls: {mock_restart.call_args_list}"
        )

        # restart_requested flag must be cleared from status.json
        status = _read_status(worker_dir)
        assert "restart_requested" not in status, (
            f"restart_requested should be removed after reconciliation; got: {status}"
        )
