# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for Mode A1 create-anima feature changes.

Covers:
1. Hiring rules in system prompt based on execution_mode (a1 vs a2)
2. Reconciliation status.json guard in _reconcile()
3. CLI --supervisor flag forwarding to create_from_md()
"""

from __future__ import annotations

import argparse
import json
import pytest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import AsyncMock, MagicMock, patch

from tests.helpers.filesystem import create_anima_dir


# ── Hiring Rules in System Prompt ─────────────────────────────────


class TestHiringRulesInSystemPrompt:
    """Test that build_system_prompt generates correct hiring rules
    depending on execution_mode and the presence of a newstaff skill.
    """

    @staticmethod
    def _setup_anima_with_newstaff(data_dir: Path, name: str = "commander") -> Path:
        """Create an anima directory with a newstaff skill file."""
        anima_dir = create_anima_dir(
            data_dir,
            name,
            identity="# Commander\nTop-level commander anima.",
            injection="## Role\nManage the team.",
            permissions="## Permissions\nAll permissions granted.",
        )
        # Create the newstaff skill file with required 概要 section
        skill_file = anima_dir / "skills" / "newstaff.md"
        skill_file.write_text(
            "# newstaff\n\n## 概要\n新しいAnimaを雇用するスキル\n\n## 手順\n...\n",
            encoding="utf-8",
        )
        return anima_dir

    @staticmethod
    def _setup_anima_without_newstaff(data_dir: Path, name: str = "worker") -> Path:
        """Create an anima directory without a newstaff skill."""
        return create_anima_dir(
            data_dir,
            name,
            identity="# Worker\nRegular worker anima.",
            injection="## Role\nDo tasks.",
            permissions="## Permissions\nBasic permissions.",
        )

    def test_no_newstaff_skill_no_hiring_rules(self, data_dir: Path) -> None:
        """Without newstaff skill, no hiring rules section in system prompt."""
        from core.memory import MemoryManager
        from core.prompt.builder import build_system_prompt

        anima_dir = self._setup_anima_without_newstaff(data_dir)
        memory = MemoryManager(anima_dir)

        prompt = build_system_prompt(memory, execution_mode="a1")

        assert "雇用ルール" not in prompt


# ── Reconciliation status.json Guard ──────────────────────────────


class TestReconciliationStatusJsonGuard:
    """Test that _reconcile() skips anima directories missing status.json."""

    @pytest.fixture
    def temp_dirs(self):
        """Create temporary directories for testing."""
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            yield {
                "animas_dir": tmp / "animas",
                "shared_dir": tmp / "shared",
                "run_dir": tmp / "run",
            }

    @pytest.fixture
    def supervisor(self, temp_dirs):
        """Create a ProcessSupervisor instance."""
        from core.supervisor.manager import (
            ProcessSupervisor,
            RestartPolicy,
            HealthConfig,
        )

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

    @pytest.mark.asyncio
    async def test_reconcile_skips_anima_without_status_json(
        self, supervisor, temp_dirs,
    ) -> None:
        """identity.md exists but status.json missing -> creation in progress, skip."""
        animas_dir = temp_dirs["animas_dir"]
        animas_dir.mkdir(parents=True)
        alice_dir = animas_dir / "alice"
        alice_dir.mkdir()
        (alice_dir / "identity.md").write_text("Alice identity", encoding="utf-8")
        # No status.json — simulates creation in progress

        supervisor.start_anima = AsyncMock()
        supervisor.stop_anima = AsyncMock()
        callback = MagicMock()
        supervisor.on_anima_added = callback

        await supervisor._reconcile()

        supervisor.start_anima.assert_not_called()
        callback.assert_not_called()

    @pytest.mark.asyncio
    async def test_reconcile_starts_anima_with_both_files(
        self, supervisor, temp_dirs,
    ) -> None:
        """identity.md AND status.json exist -> start normally."""
        animas_dir = temp_dirs["animas_dir"]
        animas_dir.mkdir(parents=True)
        alice_dir = animas_dir / "alice"
        alice_dir.mkdir()
        (alice_dir / "identity.md").write_text("Alice identity", encoding="utf-8")
        (alice_dir / "status.json").write_text(
            json.dumps({"enabled": True}), encoding="utf-8",
        )

        supervisor.start_anima = AsyncMock()
        supervisor.stop_anima = AsyncMock()
        callback = MagicMock()
        supervisor.on_anima_added = callback

        await supervisor._reconcile()

        supervisor.start_anima.assert_called_once_with("alice")
        callback.assert_called_once_with("alice")


# ── CLI --supervisor Flag ─────────────────────────────────────────


class TestCreateAnimaCLISupervisor:
    """Test that cmd_create_anima forwards --supervisor to create_from_md."""

    def test_supervisor_passed_to_create_from_md(
        self, data_dir: Path, tmp_path: Path,
    ) -> None:
        """--supervisor flag should be forwarded to create_from_md."""
        # Create a dummy character sheet MD file
        md_path = tmp_path / "test_char.md"
        md_path.write_text(
            "# キャラクターシート\n\n"
            "## 基本情報\n\n"
            "| 項目 | 値 |\n"
            "|------|----|\n"
            "| 名前 | test-subordinate |\n\n"
            "## 人格\nTest personality.\n\n"
            "## 役割・行動方針\nTest role.\n",
            encoding="utf-8",
        )

        args = argparse.Namespace(
            from_md=str(md_path),
            name=None,
            template=None,
            supervisor="boss-anima",
        )

        # Patch at the source module because cmd_create_anima uses local imports
        with patch("core.anima_factory.create_from_md") as mock_create, \
             patch("core.init.ensure_runtime_dir"), \
             patch("core.paths.get_data_dir", return_value=data_dir), \
             patch("core.paths.get_animas_dir", return_value=data_dir / "animas"), \
             patch("cli.commands.init_cmd._register_anima_in_config"):
            mock_create.return_value = data_dir / "animas" / "test-subordinate"

            from cli.commands.anima import cmd_create_anima
            cmd_create_anima(args)

            mock_create.assert_called_once()
            call_kwargs = mock_create.call_args
            # supervisor is passed as a keyword argument
            assert call_kwargs.kwargs.get("supervisor") == "boss-anima"

    def test_supervisor_none_when_not_specified(
        self, data_dir: Path, tmp_path: Path,
    ) -> None:
        """Without --supervisor flag, supervisor should be None."""
        md_path = tmp_path / "test_char2.md"
        md_path.write_text(
            "# キャラクターシート\n\n"
            "## 基本情報\n\n"
            "| 項目 | 値 |\n"
            "|------|----|\n"
            "| 名前 | test-worker |\n\n"
            "## 人格\nTest personality.\n\n"
            "## 役割・行動方針\nTest role.\n",
            encoding="utf-8",
        )

        args = argparse.Namespace(
            from_md=str(md_path),
            name=None,
            template=None,
            supervisor=None,
        )

        # Patch at the source module because cmd_create_anima uses local imports
        with patch("core.anima_factory.create_from_md") as mock_create, \
             patch("core.init.ensure_runtime_dir"), \
             patch("core.paths.get_data_dir", return_value=data_dir), \
             patch("core.paths.get_animas_dir", return_value=data_dir / "animas"), \
             patch("cli.commands.init_cmd._register_anima_in_config"):
            mock_create.return_value = data_dir / "animas" / "test-worker"

            from cli.commands.anima import cmd_create_anima
            cmd_create_anima(args)

            mock_create.assert_called_once()
            call_kwargs = mock_create.call_args
            assert call_kwargs.kwargs.get("supervisor") is None
