# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
"""E2E tests for anima lifecycle CLI commands.

Tests the full delete/disable/enable/list workflow with real filesystem
operations (temp directories) but mocked server API calls.
"""
from __future__ import annotations

import argparse
import json
import os
import zipfile
from pathlib import Path
from unittest.mock import patch

import pytest


@pytest.fixture()
def anima_env(tmp_path: Path):
    """Set up a realistic anima environment in a temp directory."""
    data_dir = tmp_path / ".animaworks"
    data_dir.mkdir()
    animas_dir = data_dir / "animas"
    animas_dir.mkdir()

    # Create config.json
    config = {
        "version": 1,
        "setup_complete": True,
        "animas": {},
        "anima_defaults": {
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 4096,
            "max_turns": 20,
            "credential": "anthropic",
            "context_threshold": 0.5,
            "max_chains": 2,
            "conversation_history_threshold": 0.3,
        },
        "credentials": {
            "anthropic": {"type": "api_key", "api_key": "test-key"},
        },
    }
    (data_dir / "config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )

    # Helper to create animas
    def create_anima(name: str, *, enabled: bool = True, role: str = "general",
                     supervisor: str | None = None):
        anima_dir = animas_dir / name
        anima_dir.mkdir()
        (anima_dir / "identity.md").write_text(f"# {name.title()}\nPersonality text.", encoding="utf-8")

        status = {"enabled": enabled, "role": role}
        if supervisor:
            status["supervisor"] = supervisor
        (anima_dir / "status.json").write_text(
            json.dumps(status, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )

        # Create some memory dirs
        for subdir in ["episodes", "knowledge", "state"]:
            (anima_dir / subdir).mkdir()
        (anima_dir / "state" / "current_task.md").write_text("status: idle\n", encoding="utf-8")

        # Register in config
        cfg = json.loads((data_dir / "config.json").read_text(encoding="utf-8"))
        cfg["animas"][name] = {}
        if supervisor:
            cfg["animas"][name]["supervisor"] = supervisor
        (data_dir / "config.json").write_text(
            json.dumps(cfg, indent=2), encoding="utf-8"
        )
        return anima_dir

    return {
        "data_dir": data_dir,
        "animas_dir": animas_dir,
        "create_anima": create_anima,
    }


class TestAnimaDeleteE2E:
    """End-to-end tests for `anima delete`."""

    def test_delete_creates_archive_and_removes_dir(self, anima_env):
        env = anima_env
        env["create_anima"]("alice")

        with (
            patch("core.paths.get_data_dir", return_value=env["data_dir"]),
            patch("core.paths.get_animas_dir", return_value=env["animas_dir"]),
        ):
            from cli.commands.anima_mgmt import cmd_anima_delete

            args = argparse.Namespace(
                anima="alice", force=True, no_archive=False, gateway_url=None
            )
            cmd_anima_delete(args)

        # Verify directory removed
        assert not (env["animas_dir"] / "alice").exists()

        # Verify archive created
        archive_dir = env["data_dir"] / "archive"
        assert archive_dir.exists()
        archives = list(archive_dir.glob("alice_*.zip"))
        assert len(archives) == 1

        # Verify ZIP contents
        with zipfile.ZipFile(archives[0]) as zf:
            names = zf.namelist()
            assert any("identity.md" in n for n in names)

        # Verify config updated
        config = json.loads((env["data_dir"] / "config.json").read_text())
        assert "alice" not in config["animas"]

    def test_delete_no_archive(self, anima_env):
        env = anima_env
        env["create_anima"]("bob")

        with (
            patch("core.paths.get_data_dir", return_value=env["data_dir"]),
            patch("core.paths.get_animas_dir", return_value=env["animas_dir"]),
        ):
            from cli.commands.anima_mgmt import cmd_anima_delete

            args = argparse.Namespace(
                anima="bob", force=True, no_archive=True, gateway_url=None
            )
            cmd_anima_delete(args)

        assert not (env["animas_dir"] / "bob").exists()
        assert not (env["data_dir"] / "archive").exists()

    def test_delete_warns_orphan_supervisors(self, anima_env, capsys):
        env = anima_env
        env["create_anima"]("sakura")
        env["create_anima"]("kotoha", supervisor="sakura")
        env["create_anima"]("rin", supervisor="sakura")

        with (
            patch("core.paths.get_data_dir", return_value=env["data_dir"]),
            patch("core.paths.get_animas_dir", return_value=env["animas_dir"]),
        ):
            from cli.commands.anima_mgmt import cmd_anima_delete

            args = argparse.Namespace(
                anima="sakura", force=True, no_archive=True, gateway_url=None
            )
            cmd_anima_delete(args)

        out = capsys.readouterr().out
        assert "Warning" in out
        assert "kotoha" in out
        assert "rin" in out

    def test_delete_abort_on_no_confirmation(self, anima_env):
        env = anima_env
        env["create_anima"]("alice")

        with (
            patch("core.paths.get_data_dir", return_value=env["data_dir"]),
            patch("core.paths.get_animas_dir", return_value=env["animas_dir"]),
            patch("builtins.input", return_value="no"),
        ):
            from cli.commands.anima_mgmt import cmd_anima_delete

            args = argparse.Namespace(
                anima="alice", force=False, no_archive=True, gateway_url=None
            )
            cmd_anima_delete(args)

        # Should NOT be deleted
        assert (env["animas_dir"] / "alice").exists()


class TestAnimaDisableEnableE2E:
    """End-to-end tests for `anima disable` and `anima enable`."""

    def test_disable_sets_enabled_false(self, anima_env):
        env = anima_env
        env["create_anima"]("alice", enabled=True, role="engineer")

        with (
            patch("core.paths.get_data_dir", return_value=env["data_dir"]),
            patch("core.paths.get_animas_dir", return_value=env["animas_dir"]),
        ):
            from cli.commands.anima_mgmt import cmd_anima_disable

            args = argparse.Namespace(anima="alice", gateway_url=None)
            cmd_anima_disable(args)

        status = json.loads((env["animas_dir"] / "alice" / "status.json").read_text())
        assert status["enabled"] is False
        # Ensure other fields preserved
        assert status["role"] == "engineer"

    def test_enable_sets_enabled_true(self, anima_env):
        env = anima_env
        env["create_anima"]("alice", enabled=False, role="writer")

        with (
            patch("core.paths.get_data_dir", return_value=env["data_dir"]),
            patch("core.paths.get_animas_dir", return_value=env["animas_dir"]),
        ):
            from cli.commands.anima_mgmt import cmd_anima_enable

            args = argparse.Namespace(anima="alice", gateway_url=None)
            cmd_anima_enable(args)

        status = json.loads((env["animas_dir"] / "alice" / "status.json").read_text())
        assert status["enabled"] is True
        assert status["role"] == "writer"

    def test_disable_enable_roundtrip(self, anima_env):
        env = anima_env
        env["create_anima"]("alice", enabled=True, role="general")

        with (
            patch("core.paths.get_data_dir", return_value=env["data_dir"]),
            patch("core.paths.get_animas_dir", return_value=env["animas_dir"]),
        ):
            from cli.commands.anima_mgmt import cmd_anima_disable, cmd_anima_enable

            # Disable
            cmd_anima_disable(argparse.Namespace(anima="alice", gateway_url=None))
            status = json.loads((env["animas_dir"] / "alice" / "status.json").read_text())
            assert status["enabled"] is False

            # Re-enable
            cmd_anima_enable(argparse.Namespace(anima="alice", gateway_url=None))
            status = json.loads((env["animas_dir"] / "alice" / "status.json").read_text())
            assert status["enabled"] is True


class TestAnimaListE2E:
    """End-to-end tests for `anima list`."""

    def test_list_local_shows_all_animas(self, anima_env, capsys):
        env = anima_env
        env["create_anima"]("alice", enabled=True, role="engineer")
        env["create_anima"]("bob", enabled=False, role="general")
        env["create_anima"]("carol", enabled=True, role="writer")

        with (
            patch("core.paths.get_data_dir", return_value=env["data_dir"]),
            patch("core.paths.get_animas_dir", return_value=env["animas_dir"]),
        ):
            from cli.commands.anima_mgmt import cmd_anima_list

            args = argparse.Namespace(local=True, gateway_url=None)
            cmd_anima_list(args)

        out = capsys.readouterr().out
        assert "alice" in out
        assert "bob" in out
        assert "carol" in out
        assert "Total: 3" in out

    def test_list_local_empty(self, anima_env, capsys):
        env = anima_env

        with (
            patch("core.paths.get_data_dir", return_value=env["data_dir"]),
            patch("core.paths.get_animas_dir", return_value=env["animas_dir"]),
        ):
            from cli.commands.anima_mgmt import cmd_anima_list

            args = argparse.Namespace(local=True, gateway_url=None)
            cmd_anima_list(args)

        out = capsys.readouterr().out
        assert "Total: 0" in out

    def test_list_skips_dirs_without_identity(self, anima_env, capsys):
        env = anima_env
        env["create_anima"]("alice")
        # Create orphan dir without identity.md
        orphan = env["animas_dir"] / "orphan"
        orphan.mkdir()

        with (
            patch("core.paths.get_data_dir", return_value=env["data_dir"]),
            patch("core.paths.get_animas_dir", return_value=env["animas_dir"]),
        ):
            from cli.commands.anima_mgmt import cmd_anima_list

            args = argparse.Namespace(local=True, gateway_url=None)
            cmd_anima_list(args)

        out = capsys.readouterr().out
        assert "alice" in out
        assert "orphan" not in out
        assert "Total: 1" in out


class TestFullLifecycleE2E:
    """Test complete lifecycle: create -> list -> disable -> enable -> delete."""

    def test_full_lifecycle(self, anima_env, capsys):
        env = anima_env

        with (
            patch("core.paths.get_data_dir", return_value=env["data_dir"]),
            patch("core.paths.get_animas_dir", return_value=env["animas_dir"]),
        ):
            from cli.commands.anima_mgmt import (
                cmd_anima_delete,
                cmd_anima_disable,
                cmd_anima_enable,
                cmd_anima_list,
            )

            # Setup: create anima manually
            env["create_anima"]("test_anima", enabled=True, role="general")

            # List — should show the anima
            capsys.readouterr()  # clear
            cmd_anima_list(argparse.Namespace(local=True, gateway_url=None))
            out = capsys.readouterr().out
            assert "test_anima" in out
            assert "Total: 1" in out

            # Disable
            cmd_anima_disable(argparse.Namespace(anima="test_anima", gateway_url=None))
            status = json.loads(
                (env["animas_dir"] / "test_anima" / "status.json").read_text()
            )
            assert status["enabled"] is False

            # Enable
            cmd_anima_enable(argparse.Namespace(anima="test_anima", gateway_url=None))
            status = json.loads(
                (env["animas_dir"] / "test_anima" / "status.json").read_text()
            )
            assert status["enabled"] is True

            # Delete
            cmd_anima_delete(argparse.Namespace(
                anima="test_anima", force=True, no_archive=False, gateway_url=None
            ))
            assert not (env["animas_dir"] / "test_anima").exists()

            # Verify archive
            archives = list((env["data_dir"] / "archive").glob("test_anima_*.zip"))
            assert len(archives) == 1

            # List — should be empty
            capsys.readouterr()  # clear
            cmd_anima_list(argparse.Namespace(local=True, gateway_url=None))
            out = capsys.readouterr().out
            assert "Total: 0" in out
