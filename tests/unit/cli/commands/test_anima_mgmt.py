"""Unit tests for cli/commands/anima_mgmt.py — Anima lifecycle CLI."""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


class TestCmdAnimaDelete:
    """Tests for cmd_anima_delete."""

    def _make_anima_dir(self, tmp_path: Path, name: str = "alice") -> Path:
        """Create a minimal anima directory structure."""
        data_dir = tmp_path / ".animaworks"
        data_dir.mkdir()
        animas_dir = data_dir / "animas"
        animas_dir.mkdir()
        anima_dir = animas_dir / name
        anima_dir.mkdir()
        (anima_dir / "identity.md").write_text("# Alice", encoding="utf-8")
        (anima_dir / "status.json").write_text(
            json.dumps({"enabled": True}), encoding="utf-8"
        )
        # Create config.json with the anima registered
        config = {
            "version": 1,
            "animas": {name: {}},
        }
        (data_dir / "config.json").write_text(
            json.dumps(config), encoding="utf-8"
        )
        return data_dir

    @patch("core.paths.get_animas_dir")
    @patch("core.paths.get_data_dir")
    def test_delete_not_found(self, mock_data_dir, mock_animas_dir, tmp_path):
        from cli.commands.anima_mgmt import cmd_anima_delete
        data_dir = tmp_path / ".animaworks"
        data_dir.mkdir()
        animas_dir = data_dir / "animas"
        animas_dir.mkdir()
        mock_data_dir.return_value = data_dir
        mock_animas_dir.return_value = animas_dir

        args = argparse.Namespace(anima="nobody", force=True, no_archive=True, gateway_url=None)
        with pytest.raises(SystemExit):
            cmd_anima_delete(args)

    @patch("core.paths.get_animas_dir")
    @patch("core.paths.get_data_dir")
    def test_delete_with_archive(self, mock_data_dir, mock_animas_dir, tmp_path, capsys):
        from cli.commands.anima_mgmt import cmd_anima_delete
        data_dir = self._make_anima_dir(tmp_path, "alice")
        mock_data_dir.return_value = data_dir
        mock_animas_dir.return_value = data_dir / "animas"

        args = argparse.Namespace(anima="alice", force=True, no_archive=False, gateway_url=None)
        cmd_anima_delete(args)

        captured = capsys.readouterr()
        assert "Archived to:" in captured.out
        assert "deleted successfully" in captured.out
        # Directory should be gone
        assert not (data_dir / "animas" / "alice").exists()
        # Archive should exist
        assert (data_dir / "archive").exists()
        archives = list((data_dir / "archive").glob("alice_*.zip"))
        assert len(archives) == 1

    @patch("core.paths.get_animas_dir")
    @patch("core.paths.get_data_dir")
    def test_delete_no_archive(self, mock_data_dir, mock_animas_dir, tmp_path, capsys):
        from cli.commands.anima_mgmt import cmd_anima_delete
        data_dir = self._make_anima_dir(tmp_path, "alice")
        mock_data_dir.return_value = data_dir
        mock_animas_dir.return_value = data_dir / "animas"

        args = argparse.Namespace(anima="alice", force=True, no_archive=True, gateway_url=None)
        cmd_anima_delete(args)

        captured = capsys.readouterr()
        assert "Archived to:" not in captured.out
        assert "deleted successfully" in captured.out
        assert not (data_dir / "animas" / "alice").exists()

    @patch("core.paths.get_animas_dir")
    @patch("core.paths.get_data_dir")
    def test_delete_aborted_by_user(self, mock_data_dir, mock_animas_dir, tmp_path, capsys):
        from cli.commands.anima_mgmt import cmd_anima_delete
        data_dir = self._make_anima_dir(tmp_path, "alice")
        mock_data_dir.return_value = data_dir
        mock_animas_dir.return_value = data_dir / "animas"

        args = argparse.Namespace(anima="alice", force=False, no_archive=True, gateway_url=None)
        with patch("builtins.input", return_value="n"):
            cmd_anima_delete(args)

        captured = capsys.readouterr()
        assert "Aborted" in captured.out
        # Directory should still exist
        assert (data_dir / "animas" / "alice").exists()

    @patch("core.paths.get_animas_dir")
    @patch("core.paths.get_data_dir")
    def test_delete_unregisters_from_config(self, mock_data_dir, mock_animas_dir, tmp_path):
        from cli.commands.anima_mgmt import cmd_anima_delete
        data_dir = self._make_anima_dir(tmp_path, "alice")
        mock_data_dir.return_value = data_dir
        mock_animas_dir.return_value = data_dir / "animas"

        args = argparse.Namespace(anima="alice", force=True, no_archive=True, gateway_url=None)
        cmd_anima_delete(args)

        config = json.loads((data_dir / "config.json").read_text(encoding="utf-8"))
        assert "alice" not in config.get("animas", {})

    @patch("core.paths.get_animas_dir")
    @patch("core.paths.get_data_dir")
    def test_delete_warns_orphan_supervisor(self, mock_data_dir, mock_animas_dir, tmp_path, capsys):
        from cli.commands.anima_mgmt import cmd_anima_delete
        data_dir = self._make_anima_dir(tmp_path, "sakura")
        animas_dir = data_dir / "animas"
        # Create a subordinate that references sakura as supervisor
        sub_dir = animas_dir / "kotoha"
        sub_dir.mkdir()
        (sub_dir / "identity.md").write_text("# Kotoha", encoding="utf-8")
        (sub_dir / "status.json").write_text(
            json.dumps({"enabled": True, "supervisor": "sakura"}), encoding="utf-8"
        )
        mock_data_dir.return_value = data_dir
        mock_animas_dir.return_value = animas_dir

        args = argparse.Namespace(anima="sakura", force=True, no_archive=True, gateway_url=None)
        cmd_anima_delete(args)

        captured = capsys.readouterr()
        assert "Warning" in captured.out
        assert "kotoha" in captured.out


class TestCmdAnimaDisable:
    """Tests for cmd_anima_disable."""

    @patch("core.paths.get_animas_dir")
    @patch("core.paths.get_data_dir")
    def test_disable_not_found(self, mock_data_dir, mock_animas_dir, tmp_path):
        from cli.commands.anima_mgmt import cmd_anima_disable
        data_dir = tmp_path / ".animaworks"
        data_dir.mkdir()
        animas_dir = data_dir / "animas"
        animas_dir.mkdir()
        mock_data_dir.return_value = data_dir
        mock_animas_dir.return_value = animas_dir

        args = argparse.Namespace(anima="nobody", gateway_url=None)
        with pytest.raises(SystemExit):
            cmd_anima_disable(args)

    @patch("core.paths.get_animas_dir")
    @patch("core.paths.get_data_dir")
    def test_disable_offline(self, mock_data_dir, mock_animas_dir, tmp_path, capsys):
        from cli.commands.anima_mgmt import cmd_anima_disable
        data_dir = tmp_path / ".animaworks"
        data_dir.mkdir()
        animas_dir = data_dir / "animas"
        animas_dir.mkdir()
        anima_dir = animas_dir / "alice"
        anima_dir.mkdir()
        (anima_dir / "identity.md").write_text("# Alice")
        (anima_dir / "status.json").write_text(json.dumps({"enabled": True, "role": "general"}))
        mock_data_dir.return_value = data_dir
        mock_animas_dir.return_value = animas_dir

        args = argparse.Namespace(anima="alice", gateway_url=None)
        cmd_anima_disable(args)

        captured = capsys.readouterr()
        assert "Disabled" in captured.out
        assert "offline" in captured.out
        status = json.loads((anima_dir / "status.json").read_text())
        assert status["enabled"] is False
        # Other fields preserved
        assert status["role"] == "general"


class TestCmdAnimaEnable:
    """Tests for cmd_anima_enable."""

    @patch("core.paths.get_animas_dir")
    @patch("core.paths.get_data_dir")
    def test_enable_not_found(self, mock_data_dir, mock_animas_dir, tmp_path):
        from cli.commands.anima_mgmt import cmd_anima_enable
        data_dir = tmp_path / ".animaworks"
        data_dir.mkdir()
        animas_dir = data_dir / "animas"
        animas_dir.mkdir()
        mock_data_dir.return_value = data_dir
        mock_animas_dir.return_value = animas_dir

        args = argparse.Namespace(anima="nobody", gateway_url=None)
        with pytest.raises(SystemExit):
            cmd_anima_enable(args)

    @patch("core.paths.get_animas_dir")
    @patch("core.paths.get_data_dir")
    def test_enable_offline(self, mock_data_dir, mock_animas_dir, tmp_path, capsys):
        from cli.commands.anima_mgmt import cmd_anima_enable
        data_dir = tmp_path / ".animaworks"
        data_dir.mkdir()
        animas_dir = data_dir / "animas"
        animas_dir.mkdir()
        anima_dir = animas_dir / "alice"
        anima_dir.mkdir()
        (anima_dir / "identity.md").write_text("# Alice")
        (anima_dir / "status.json").write_text(json.dumps({"enabled": False, "role": "general"}))
        mock_data_dir.return_value = data_dir
        mock_animas_dir.return_value = animas_dir

        args = argparse.Namespace(anima="alice", gateway_url=None)
        cmd_anima_enable(args)

        captured = capsys.readouterr()
        assert "Enabled" in captured.out
        assert "offline" in captured.out
        status = json.loads((anima_dir / "status.json").read_text())
        assert status["enabled"] is True
        assert status["role"] == "general"


class TestCmdAnimaList:
    """Tests for cmd_anima_list."""

    @patch("core.paths.get_animas_dir")
    @patch("core.paths.get_data_dir")
    def test_list_local_empty(self, mock_data_dir, mock_animas_dir, tmp_path, capsys):
        from cli.commands.anima_mgmt import cmd_anima_list
        data_dir = tmp_path / ".animaworks"
        data_dir.mkdir()
        animas_dir = data_dir / "animas"
        # No animas dir
        mock_data_dir.return_value = data_dir
        mock_animas_dir.return_value = animas_dir

        args = argparse.Namespace(local=True, gateway_url=None)
        cmd_anima_list(args)

        captured = capsys.readouterr()
        assert "No animas" in captured.out

    @patch("core.paths.get_animas_dir")
    @patch("core.paths.get_data_dir")
    def test_list_local_with_animas(self, mock_data_dir, mock_animas_dir, tmp_path, capsys):
        from cli.commands.anima_mgmt import cmd_anima_list
        data_dir = tmp_path / ".animaworks"
        data_dir.mkdir()
        animas_dir = data_dir / "animas"
        animas_dir.mkdir()

        # Create two animas
        for name, enabled, role in [("alice", True, "engineer"), ("bob", False, "general")]:
            d = animas_dir / name
            d.mkdir()
            (d / "identity.md").write_text(f"# {name}")
            (d / "status.json").write_text(json.dumps({"enabled": enabled, "role": role}))

        mock_data_dir.return_value = data_dir
        mock_animas_dir.return_value = animas_dir

        args = argparse.Namespace(local=True, gateway_url=None)
        cmd_anima_list(args)

        captured = capsys.readouterr()
        assert "alice" in captured.out
        assert "bob" in captured.out
        assert "Total: 2" in captured.out


class TestUnregisterAnimaFromConfig:
    """Tests for unregister_anima_from_config."""

    def setup_method(self):
        """Invalidate config cache before each test."""
        from core.config.models import invalidate_cache
        invalidate_cache()

    def test_unregister_existing(self, tmp_path):
        from core.config.models import unregister_anima_from_config
        config = {"version": 1, "animas": {"alice": {}, "bob": {}}}
        (tmp_path / "config.json").write_text(json.dumps(config))

        result = unregister_anima_from_config(tmp_path, "alice")
        assert result is True
        updated = json.loads((tmp_path / "config.json").read_text())
        assert "alice" not in updated["animas"]
        assert "bob" in updated["animas"]

    def test_unregister_not_present(self, tmp_path):
        from core.config.models import unregister_anima_from_config
        config = {"version": 1, "animas": {"bob": {}}}
        (tmp_path / "config.json").write_text(json.dumps(config))

        result = unregister_anima_from_config(tmp_path, "alice")
        assert result is False

    def test_unregister_no_config(self, tmp_path):
        from core.config.models import unregister_anima_from_config
        result = unregister_anima_from_config(tmp_path, "alice")
        assert result is False
