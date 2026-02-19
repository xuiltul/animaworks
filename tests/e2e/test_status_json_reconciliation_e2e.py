# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
"""E2E tests for status.json creation and reconciliation protection.

Verifies the complete flow:
1. create_blank() produces status.json with enabled=true
2. Reconciliation recognises blank-created animas and does NOT kill them
3. Animas with identity.md but no status.json are protected (on_disk_incomplete)
4. Animas truly removed from disk are still killed
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.anima_factory import create_blank
from core.supervisor.manager import (
    ProcessSupervisor,
    RestartPolicy,
    HealthConfig,
)


# ── Helpers ──────────────────────────────────────────────────


def _make_blank_template(tmp_path: Path) -> Path:
    """Create a minimal _blank template with identity.md (mirrors real template)."""
    blank_dir = tmp_path / "_blank_template"
    blank_dir.mkdir()
    (blank_dir / "identity.md").write_text(
        "# {name}\n\nBlank identity for {name}.\n", encoding="utf-8",
    )
    return blank_dir


def _make_supervisor(animas_dir: Path, tmp_path: Path) -> ProcessSupervisor:
    """Create a minimal ProcessSupervisor wired to *animas_dir*."""
    shared_dir = tmp_path / "shared"
    shared_dir.mkdir(parents=True, exist_ok=True)
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    return ProcessSupervisor(
        animas_dir=animas_dir,
        shared_dir=shared_dir,
        run_dir=run_dir,
        restart_policy=RestartPolicy(
            max_retries=3, backoff_base_sec=0.1, backoff_max_sec=1.0,
        ),
        health_config=HealthConfig(
            ping_interval_sec=0.5, ping_timeout_sec=0.2,
            max_missed_pings=2, startup_grace_sec=0.5,
        ),
    )


# ── E2E Tests ────────────────────────────────────────────────


class TestCreateBlankReconciliationE2E:
    """End-to-end: create_blank → reconciliation correctly handles the anima."""

    async def test_create_blank_produces_status_json(self, tmp_path):
        """create_blank() must produce a status.json so reconciliation sees it."""
        animas_dir = tmp_path / "animas"
        animas_dir.mkdir()
        blank_dir = _make_blank_template(tmp_path)
        with patch("core.anima_factory.BLANK_TEMPLATE_DIR", blank_dir), \
             patch("core.anima_factory.BOOTSTRAP_TEMPLATE", tmp_path / "no"):
            anima_dir = create_blank(animas_dir, "sakura")

        status_path = anima_dir / "status.json"
        assert status_path.exists(), "create_blank must create status.json"
        data = json.loads(status_path.read_text(encoding="utf-8"))
        assert data["enabled"] is True

    async def test_blank_anima_survives_reconciliation(self, tmp_path):
        """A blank-created anima (with status.json) must survive reconciliation."""
        animas_dir = tmp_path / "animas"
        animas_dir.mkdir()
        blank_dir = _make_blank_template(tmp_path)
        with patch("core.anima_factory.BLANK_TEMPLATE_DIR", blank_dir), \
             patch("core.anima_factory.BOOTSTRAP_TEMPLATE", tmp_path / "no"):
            anima_dir = create_blank(animas_dir, "sakura")

        sup = _make_supervisor(animas_dir, tmp_path)
        sup.processes["sakura"] = MagicMock()
        sup.start_anima = AsyncMock()
        sup.stop_anima = AsyncMock()

        with patch.object(sup, "_reconcile_assets", new_callable=AsyncMock):
            await sup._reconcile()

        sup.stop_anima.assert_not_called()

    async def test_blank_anima_started_by_reconciliation(self, tmp_path):
        """An enabled blank-created anima NOT running should be started by reconciliation."""
        animas_dir = tmp_path / "animas"
        animas_dir.mkdir()
        blank_dir = _make_blank_template(tmp_path)
        with patch("core.anima_factory.BLANK_TEMPLATE_DIR", blank_dir), \
             patch("core.anima_factory.BOOTSTRAP_TEMPLATE", tmp_path / "no"):
            create_blank(animas_dir, "sakura")

        sup = _make_supervisor(animas_dir, tmp_path)
        # sakura NOT in sup.processes → should be started
        sup.start_anima = AsyncMock()
        sup.stop_anima = AsyncMock()

        with patch.object(sup, "_reconcile_assets", new_callable=AsyncMock):
            await sup._reconcile()

        sup.start_anima.assert_called_once_with("sakura")


class TestIncompleteAnimaReconciliationE2E:
    """End-to-end: anima with identity.md but no status.json is protected."""

    async def test_legacy_anima_not_killed_when_running(self, tmp_path):
        """A legacy/incomplete anima (identity.md, no status.json) must NOT be killed."""
        animas_dir = tmp_path / "animas"
        animas_dir.mkdir()
        sakura_dir = animas_dir / "sakura"
        sakura_dir.mkdir()
        (sakura_dir / "identity.md").write_text("# sakura\n", encoding="utf-8")
        # No status.json — simulating legacy or factory-in-progress

        sup = _make_supervisor(animas_dir, tmp_path)
        sup.processes["sakura"] = MagicMock()
        sup.start_anima = AsyncMock()
        sup.stop_anima = AsyncMock()

        with patch.object(sup, "_reconcile_assets", new_callable=AsyncMock):
            await sup._reconcile()

        sup.stop_anima.assert_not_called()
        sup.start_anima.assert_not_called()

    async def test_truly_removed_anima_still_killed(self, tmp_path):
        """An anima whose directory is completely gone MUST be killed."""
        animas_dir = tmp_path / "animas"
        animas_dir.mkdir()
        # No directory for "ghost" at all

        sup = _make_supervisor(animas_dir, tmp_path)
        sup.processes["ghost"] = MagicMock()
        sup.start_anima = AsyncMock()
        sup.stop_anima = AsyncMock()

        with patch.object(sup, "_reconcile_assets", new_callable=AsyncMock):
            await sup._reconcile()

        sup.stop_anima.assert_called_once_with("ghost")

    async def test_mixed_scenario_full_lifecycle(self, tmp_path):
        """Mixed scenario: blank-created + legacy + ghost animas in one reconciliation."""
        animas_dir = tmp_path / "animas"
        animas_dir.mkdir()

        # 1. sakura: created via create_blank → has status.json, running → KEEP
        blank_dir = _make_blank_template(tmp_path)
        with patch("core.anima_factory.BLANK_TEMPLATE_DIR", blank_dir), \
             patch("core.anima_factory.BOOTSTRAP_TEMPLATE", tmp_path / "no"):
            create_blank(animas_dir, "sakura")

        # 2. kotoha: legacy, identity.md only, running → PROTECT (on_disk_incomplete)
        kotoha_dir = animas_dir / "kotoha"
        kotoha_dir.mkdir()
        (kotoha_dir / "identity.md").write_text("# kotoha\n", encoding="utf-8")

        # 3. hinata: has status.json, enabled, NOT running → START
        hinata_dir = animas_dir / "hinata"
        hinata_dir.mkdir()
        (hinata_dir / "identity.md").write_text("# hinata\n", encoding="utf-8")
        (hinata_dir / "status.json").write_text(
            json.dumps({"enabled": True}), encoding="utf-8",
        )

        # 4. ghost: NO directory, running → KILL
        # (don't create any directory)

        sup = _make_supervisor(animas_dir, tmp_path)
        sup.processes["sakura"] = MagicMock()
        sup.processes["kotoha"] = MagicMock()
        sup.processes["ghost"] = MagicMock()
        # hinata is NOT in processes

        sup.start_anima = AsyncMock()
        sup.stop_anima = AsyncMock()

        with patch.object(sup, "_reconcile_assets", new_callable=AsyncMock):
            await sup._reconcile()

        # ghost should be stopped (removed from disk)
        sup.stop_anima.assert_called_once_with("ghost")
        # hinata should be started (enabled, not running)
        sup.start_anima.assert_called_once_with("hinata")
