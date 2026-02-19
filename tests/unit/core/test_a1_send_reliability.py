"""Unit tests for A1 mode send script reliability improvements."""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

from core.execution.agent_sdk import AgentSDKExecutor
from core.anima_factory import ensure_board_scripts, ensure_send_scripts
from core.schemas import ModelConfig


# ── _build_env() ─────────────────────────────────────────


class TestBuildEnvPathAndProjectDir:
    """Verify _build_env() exposes anima_dir in PATH and sets PROJECT_DIR."""

    def _make_executor(self, anima_dir: Path) -> AgentSDKExecutor:
        mc = ModelConfig(model="claude-sonnet-4-20250514")
        return AgentSDKExecutor(model_config=mc, anima_dir=anima_dir)

    def test_anima_dir_in_path(self, tmp_path: Path) -> None:
        """PATH should start with anima_dir so `send` is discoverable."""
        anima_dir = tmp_path / "animas" / "alice"
        anima_dir.mkdir(parents=True)

        executor = self._make_executor(anima_dir)
        env = executor._build_env()

        assert "PATH" in env
        path_entries = env["PATH"].split(":")
        assert str(anima_dir) == path_entries[0], (
            "anima_dir must be the first entry in PATH"
        )

    def test_system_path_preserved(self, tmp_path: Path) -> None:
        """System PATH entries should be preserved after anima_dir."""
        anima_dir = tmp_path / "animas" / "bob"
        anima_dir.mkdir(parents=True)

        original_path = "/usr/local/bin:/usr/bin:/bin"
        with patch.dict(os.environ, {"PATH": original_path}):
            executor = self._make_executor(anima_dir)
            env = executor._build_env()

        assert env["PATH"] == f"{anima_dir}:{original_path}"

    def test_project_dir_set(self, tmp_path: Path) -> None:
        """ANIMAWORKS_PROJECT_DIR should be set to the project root."""
        from core.paths import PROJECT_DIR

        anima_dir = tmp_path / "animas" / "carol"
        anima_dir.mkdir(parents=True)

        executor = self._make_executor(anima_dir)
        env = executor._build_env()

        assert "ANIMAWORKS_PROJECT_DIR" in env
        assert env["ANIMAWORKS_PROJECT_DIR"] == str(PROJECT_DIR)

    def test_anima_dir_env_set(self, tmp_path: Path) -> None:
        """ANIMAWORKS_ANIMA_DIR should still be set."""
        anima_dir = tmp_path / "animas" / "dave"
        anima_dir.mkdir(parents=True)

        executor = self._make_executor(anima_dir)
        env = executor._build_env()

        assert env["ANIMAWORKS_ANIMA_DIR"] == str(anima_dir)

    def test_fallback_path_when_no_env(self, tmp_path: Path) -> None:
        """When PATH is not in os.environ, fall back to /usr/bin:/bin."""
        anima_dir = tmp_path / "animas" / "eve"
        anima_dir.mkdir(parents=True)

        env_without_path = {k: v for k, v in os.environ.items() if k != "PATH"}
        with patch.dict(os.environ, env_without_path, clear=True):
            executor = self._make_executor(anima_dir)
            env = executor._build_env()

        assert env["PATH"] == f"{anima_dir}:/usr/bin:/bin"


# ── ensure_send_scripts() ────────────────────────────────


class TestEnsureSendScripts:
    """Verify ensure_send_scripts() places send script for all animas."""

    def test_places_script_for_animas_without_it(self, tmp_path: Path) -> None:
        """Animas missing the send script should get it."""
        animas_dir = tmp_path / "animas"
        animas_dir.mkdir()

        # Create two anima dirs without send scripts
        for name in ("alice", "bob"):
            d = animas_dir / name
            d.mkdir()
            (d / "identity.md").write_text(f"# {name}", encoding="utf-8")

        # Create a fake blank template with a send script
        blank_dir = tmp_path / "blank"
        blank_dir.mkdir()
        (blank_dir / "send").write_text("#!/bin/bash\necho send", encoding="utf-8")

        with patch("core.anima_factory.BLANK_TEMPLATE_DIR", blank_dir):
            ensure_send_scripts(animas_dir)

        for name in ("alice", "bob"):
            send = animas_dir / name / "send"
            assert send.exists(), f"send script missing for {name}"
            assert send.stat().st_mode & 0o100, f"send script not executable for {name}"

    def test_overwrites_existing_scripts(self, tmp_path: Path) -> None:
        """Existing send scripts are overwritten to ensure template tracking."""
        animas_dir = tmp_path / "animas"
        alice_dir = animas_dir / "alice"
        alice_dir.mkdir(parents=True)
        (alice_dir / "identity.md").write_text("# alice", encoding="utf-8")

        # Pre-existing custom send script
        existing = alice_dir / "send"
        existing.write_text("#!/bin/bash\ncustom-send", encoding="utf-8")

        blank_dir = tmp_path / "blank"
        blank_dir.mkdir()
        (blank_dir / "send").write_text("#!/bin/bash\ntemplate-send", encoding="utf-8")

        with patch("core.anima_factory.BLANK_TEMPLATE_DIR", blank_dir):
            ensure_send_scripts(animas_dir)

        # Script should now match the template (overwritten)
        assert existing.read_text(encoding="utf-8") == "#!/bin/bash\ntemplate-send"

    def test_skips_dirs_without_identity(self, tmp_path: Path) -> None:
        """Directories without identity.md are not anima dirs; skip them."""
        animas_dir = tmp_path / "animas"
        animas_dir.mkdir()

        # Non-anima directory (no identity.md)
        non_anima_path = animas_dir / "shared"
        non_anima_path.mkdir()

        blank_dir = tmp_path / "blank"
        blank_dir.mkdir()
        (blank_dir / "send").write_text("#!/bin/bash\necho send", encoding="utf-8")

        with patch("core.anima_factory.BLANK_TEMPLATE_DIR", blank_dir):
            ensure_send_scripts(animas_dir)

        assert not (non_anima_path / "send").exists()

    def test_nonexistent_animas_dir(self, tmp_path: Path) -> None:
        """If animas_dir doesn't exist, ensure_send_scripts() is a no-op."""
        missing_dir = tmp_path / "nonexistent"
        # Should not raise
        ensure_send_scripts(missing_dir)


# ── ensure_board_scripts() ────────────────────────────────


class TestEnsureBoardScripts:
    """Verify ensure_board_scripts() places board script for all animas."""

    def test_places_script_for_animas_without_it(self, tmp_path: Path) -> None:
        """Animas missing the board script should get it."""
        animas_dir = tmp_path / "animas"
        animas_dir.mkdir()

        for name in ("alice", "bob"):
            d = animas_dir / name
            d.mkdir()
            (d / "identity.md").write_text(f"# {name}", encoding="utf-8")

        blank_dir = tmp_path / "blank"
        blank_dir.mkdir()
        (blank_dir / "board").write_text("#!/bin/bash\necho board", encoding="utf-8")

        with patch("core.anima_factory.BLANK_TEMPLATE_DIR", blank_dir):
            ensure_board_scripts(animas_dir)

        for name in ("alice", "bob"):
            board = animas_dir / name / "board"
            assert board.exists(), f"board script missing for {name}"
            assert board.stat().st_mode & 0o100, f"board script not executable for {name}"

    def test_overwrites_existing_scripts(self, tmp_path: Path) -> None:
        """Existing board scripts are overwritten to ensure template tracking."""
        animas_dir = tmp_path / "animas"
        alice_dir = animas_dir / "alice"
        alice_dir.mkdir(parents=True)
        (alice_dir / "identity.md").write_text("# alice", encoding="utf-8")

        # Pre-existing custom board script
        existing = alice_dir / "board"
        existing.write_text("#!/bin/bash\ncustom-board", encoding="utf-8")

        blank_dir = tmp_path / "blank"
        blank_dir.mkdir()
        (blank_dir / "board").write_text("#!/bin/bash\ntemplate-board", encoding="utf-8")

        with patch("core.anima_factory.BLANK_TEMPLATE_DIR", blank_dir):
            ensure_board_scripts(animas_dir)

        # Script should now match the template (overwritten)
        assert existing.read_text(encoding="utf-8") == "#!/bin/bash\ntemplate-board"

    def test_skips_dirs_without_identity(self, tmp_path: Path) -> None:
        """Directories without identity.md are not anima dirs; skip them."""
        animas_dir = tmp_path / "animas"
        animas_dir.mkdir()

        non_anima_path = animas_dir / "shared"
        non_anima_path.mkdir()

        blank_dir = tmp_path / "blank"
        blank_dir.mkdir()
        (blank_dir / "board").write_text("#!/bin/bash\necho board", encoding="utf-8")

        with patch("core.anima_factory.BLANK_TEMPLATE_DIR", blank_dir):
            ensure_board_scripts(animas_dir)

        assert not (non_anima_path / "board").exists()

    def test_nonexistent_animas_dir(self, tmp_path: Path) -> None:
        """If animas_dir doesn't exist, ensure_board_scripts() is a no-op."""
        missing_dir = tmp_path / "nonexistent"
        ensure_board_scripts(missing_dir)
