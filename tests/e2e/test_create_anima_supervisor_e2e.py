# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
"""E2E tests for the --supervisor flag in create-anima and reconciliation status.json guard.

Tests cover:
- CLI create-anima with --supervisor flag setting supervisor in status.json
- --supervisor overriding the character sheet's supervisor value
- Without --supervisor, supervisor is read from the character sheet
- Reconciliation skipping anima directories without status.json
- Reconciliation starting anima directories with both identity.md and status.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from core.anima_factory import create_from_md
from core.supervisor.manager import ProcessSupervisor

# ── Sample character sheets ──────────────────────────────────

SUPERVISOR_TEST_SHEET = """\
# Character: worker1

## 基本情報

| 項目 | 設定 |
|------|------|
| 英名 | worker1 |
| 日本語名 | ワーカー1 |
| 役職/専門 | テスト担当 |
| 上司 | sakura |
| 役割 | worker |
| 実行モード | autonomous |
| モデル | claude-sonnet-4-20250514 |
| credential | anthropic |

## 人格

テスト用の人格設定です。

## 役割・行動方針

テスト業務を担当します。
"""

SUPERVISOR_TEST_SHEET_BOSS = """\
# Character: worker2

## 基本情報

| 項目 | 設定 |
|------|------|
| 英名 | worker2 |
| 日本語名 | ワーカー2 |
| 役職/専門 | テスト担当 |
| 上司 | sakura |
| 役割 | worker |
| 実行モード | autonomous |
| モデル | claude-sonnet-4-20250514 |
| credential | anthropic |

## 人格

テスト用の人格設定です。

## 役割・行動方針

テスト業務を担当します。
"""

NO_SUPERVISOR_SHEET = """\
# Character: worker3

## 基本情報

| 項目 | 設定 |
|------|------|
| 英名 | worker3 |
| 日本語名 | ワーカー3 |
| 役職/専門 | テスト担当 |
| 上司 | (なし) |
| 役割 | worker |
| 実行モード | autonomous |
| モデル | claude-sonnet-4-20250514 |
| credential | anthropic |

## 人格

テスト用の人格設定です。

## 役割・行動方針

テスト業務を担当します。
"""


# ── Helpers ──────────────────────────────────────────────────


def _write_character_sheet(directory: Path, content: str, filename: str = "sheet.md") -> Path:
    """Write a character sheet file and return its path."""
    path = directory / filename
    path.write_text(content, encoding="utf-8")
    return path


def _build_create_anima_namespace(
    from_md: str,
    name: str | None = None,
    supervisor: str | None = None,
) -> argparse.Namespace:
    """Build an argparse.Namespace matching the create-anima subcommand."""
    return argparse.Namespace(
        from_md=from_md,
        name=name,
        template=None,
        supervisor=supervisor,
    )


# ── Tests: CLI create-anima with --supervisor ────────────────


@pytest.mark.e2e
class TestCreateAnimaSupervisorE2E:
    """Test that --supervisor flag correctly sets supervisor in status.json."""

    def test_create_anima_with_supervisor_sets_status(
        self, data_dir: Path, tmp_path: Path,
    ):
        """animaworks create-anima --from-md --supervisor should set supervisor in status.json."""
        sheet_path = _write_character_sheet(tmp_path, SUPERVISOR_TEST_SHEET)

        args = _build_create_anima_namespace(
            from_md=str(sheet_path),
            supervisor="boss",
        )

        with patch("cli.commands.init_cmd._register_anima_in_config"):
            from cli.commands.anima import cmd_create_anima
            cmd_create_anima(args)

        anima_dir = data_dir / "animas" / "worker1"
        assert anima_dir.exists(), "Anima directory should be created"

        status = json.loads(
            (anima_dir / "status.json").read_text(encoding="utf-8")
        )
        assert status["supervisor"] == "boss"

    def test_create_anima_supervisor_overrides_sheet_value(
        self, data_dir: Path, tmp_path: Path,
    ):
        """--supervisor flag should override the supervisor in character sheet."""
        # The sheet has 上司 = sakura, but we pass --supervisor=boss
        sheet_path = _write_character_sheet(tmp_path, SUPERVISOR_TEST_SHEET_BOSS)

        args = _build_create_anima_namespace(
            from_md=str(sheet_path),
            supervisor="boss",
        )

        with patch("cli.commands.init_cmd._register_anima_in_config"):
            from cli.commands.anima import cmd_create_anima
            cmd_create_anima(args)

        anima_dir = data_dir / "animas" / "worker2"
        status = json.loads(
            (anima_dir / "status.json").read_text(encoding="utf-8")
        )
        # --supervisor=boss should override the sheet value "sakura"
        assert status["supervisor"] == "boss"

    def test_create_anima_without_supervisor_uses_sheet(
        self, data_dir: Path, tmp_path: Path,
    ):
        """Without --supervisor, supervisor comes from character sheet."""
        sheet_path = _write_character_sheet(tmp_path, SUPERVISOR_TEST_SHEET)

        args = _build_create_anima_namespace(
            from_md=str(sheet_path),
            supervisor=None,
        )

        with patch("cli.commands.init_cmd._register_anima_in_config"):
            from cli.commands.anima import cmd_create_anima
            cmd_create_anima(args)

        anima_dir = data_dir / "animas" / "worker1"
        status = json.loads(
            (anima_dir / "status.json").read_text(encoding="utf-8")
        )
        # Should use the sheet's 上司 = sakura
        assert status["supervisor"] == "sakura"


# ── Tests: Reconciliation status.json guard ──────────────────


@pytest.mark.e2e
class TestReconciliationStatusGuardE2E:
    """Test that reconciliation respects the status.json existence guard."""

    @pytest.mark.asyncio
    async def test_incomplete_anima_not_started_by_reconciliation(
        self, data_dir: Path,
    ):
        """Anima directory with identity.md but no status.json should be skipped."""
        animas_dir = data_dir / "animas"
        shared_dir = data_dir / "shared"
        run_dir = data_dir / "run"
        run_dir.mkdir(parents=True, exist_ok=True)

        # Create an incomplete anima: has identity.md but no status.json
        incomplete_dir = animas_dir / "incomplete-anima"
        incomplete_dir.mkdir(parents=True, exist_ok=True)
        (incomplete_dir / "identity.md").write_text(
            "# Incomplete\nStill being created.", encoding="utf-8"
        )

        supervisor = ProcessSupervisor(
            animas_dir=animas_dir,
            shared_dir=shared_dir,
            run_dir=run_dir,
        )

        with patch.object(supervisor, "start_anima", new_callable=AsyncMock) as mock_start:
            with patch.object(supervisor, "_reconcile_assets", new_callable=AsyncMock):
                await supervisor._reconcile()

            # start_anima should NOT have been called for incomplete-anima
            started_names = [call.args[0] for call in mock_start.call_args_list]
            assert "incomplete-anima" not in started_names, (
                "Reconciliation should skip anima without status.json"
            )

    @pytest.mark.asyncio
    async def test_complete_anima_started_by_reconciliation(
        self, data_dir: Path,
    ):
        """Anima directory with both identity.md and status.json should be started."""
        animas_dir = data_dir / "animas"
        shared_dir = data_dir / "shared"
        run_dir = data_dir / "run"
        run_dir.mkdir(parents=True, exist_ok=True)

        # Create a complete anima: has both identity.md and status.json
        complete_dir = animas_dir / "complete-anima"
        complete_dir.mkdir(parents=True, exist_ok=True)
        (complete_dir / "identity.md").write_text(
            "# Complete\nReady to run.", encoding="utf-8"
        )
        status = {
            "supervisor": "",
            "role": "worker",
            "execution_mode": "autonomous",
            "model": "claude-sonnet-4-20250514",
            "credential": "anthropic",
        }
        (complete_dir / "status.json").write_text(
            json.dumps(status, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        supervisor = ProcessSupervisor(
            animas_dir=animas_dir,
            shared_dir=shared_dir,
            run_dir=run_dir,
        )

        with patch.object(supervisor, "start_anima", new_callable=AsyncMock) as mock_start:
            with patch.object(supervisor, "_reconcile_assets", new_callable=AsyncMock):
                await supervisor._reconcile()

            # start_anima SHOULD have been called for complete-anima
            started_names = [call.args[0] for call in mock_start.call_args_list]
            assert "complete-anima" in started_names, (
                "Reconciliation should start anima with both identity.md and status.json"
            )
