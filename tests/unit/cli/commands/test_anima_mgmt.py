"""Unit tests for cli/commands/anima_mgmt.py — Anima lifecycle CLI."""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import json
import tomllib
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


class TestCmdAnimaCodexYolo:
    """Tests for cmd_anima_codex_yolo."""

    def setup_method(self):
        from core.config.models import invalidate_cache, invalidate_models_json_cache

        invalidate_cache()
        invalidate_models_json_cache()

    def _make_runtime(
        self,
        tmp_path: Path,
        *,
        animas: dict[str, dict[str, object]],
    ) -> Path:
        data_dir = tmp_path / ".animaworks"
        animas_dir = data_dir / "animas"
        animas_dir.mkdir(parents=True)
        (data_dir / "config.json").write_text(json.dumps({"version": 1}), encoding="utf-8")
        for name, status in animas.items():
            anima_dir = animas_dir / name
            anima_dir.mkdir()
            (anima_dir / "identity.md").write_text(f"# {name}", encoding="utf-8")
            (anima_dir / "status.json").write_text(json.dumps(status), encoding="utf-8")
            (anima_dir / "permissions.json").write_text(
                json.dumps(
                    {
                        "version": 1,
                        "file_roots": [str(anima_dir)],
                        "commands": {"allow_all": True, "allow": [], "deny": []},
                        "external_tools": {"allow_all": True, "allow": [], "deny": []},
                        "tool_creation": {"personal": True, "shared": False},
                    }
                ),
                encoding="utf-8",
            )
        return data_dir

    @patch("core.paths.get_data_dir")
    def test_codex_yolo_preserves_permissions_and_updates_config(self, mock_data_dir, tmp_path, capsys):
        from cli.commands.anima_mgmt import cmd_anima_codex_yolo

        data_dir = self._make_runtime(
            tmp_path,
            animas={"sakura": {"enabled": True, "model": "codex/o4-mini"}},
        )
        mock_data_dir.return_value = data_dir

        args = argparse.Namespace(anima="sakura", all=False, restart=False, gateway_url=None)
        cmd_anima_codex_yolo(args)

        captured = capsys.readouterr()
        assert "sakura" in captured.out
        assert "Codex YOLO updated for 1 anima(s)." in captured.out

        anima_dir = data_dir / "animas" / "sakura"
        permissions = json.loads((anima_dir / "permissions.json").read_text(encoding="utf-8"))
        assert permissions["file_roots"] == [str(anima_dir)]

        config_toml = (anima_dir / ".codex_home" / "config.toml").read_text(encoding="utf-8")
        parsed = tomllib.loads(config_toml)
        assert parsed["sandbox_mode"] == "workspace-write"
        assert parsed["sandbox_workspace_write"]["network_access"] is True
        assert parsed["approval_policy"] == "never"

    @patch("core.paths.get_data_dir")
    def test_codex_yolo_all_skips_non_codex_animas(self, mock_data_dir, tmp_path, capsys):
        from cli.commands.anima_mgmt import cmd_anima_codex_yolo

        data_dir = self._make_runtime(
            tmp_path,
            animas={
                "sakura": {"enabled": True, "model": "codex/o4-mini"},
                "mei": {"enabled": True, "model": "claude-sonnet-4-6"},
            },
        )
        mock_data_dir.return_value = data_dir

        args = argparse.Namespace(anima=None, all=True, restart=False, gateway_url=None)
        cmd_anima_codex_yolo(args)

        captured = capsys.readouterr()
        assert "sakura" in captured.out
        assert "mei" not in captured.out
        sakura_permissions = json.loads(
            (data_dir / "animas" / "sakura" / "permissions.json").read_text(encoding="utf-8")
        )
        mei_permissions = json.loads(
            (data_dir / "animas" / "mei" / "permissions.json").read_text(encoding="utf-8")
        )
        assert sakura_permissions["file_roots"] == [str(data_dir / "animas" / "sakura")]
        assert mei_permissions["file_roots"] == [str(data_dir / "animas" / "mei")]
        assert (data_dir / "animas" / "sakura" / ".codex_home" / "config.toml").is_file()
        assert not (data_dir / "animas" / "mei" / ".codex_home" / "config.toml").exists()

    @patch("core.paths.get_data_dir")
    def test_codex_yolo_rejects_name_with_all(self, mock_data_dir, tmp_path, capsys):
        from cli.commands.anima_mgmt import cmd_anima_codex_yolo

        data_dir = self._make_runtime(
            tmp_path,
            animas={"sakura": {"enabled": True, "model": "codex/o4-mini"}},
        )
        mock_data_dir.return_value = data_dir

        args = argparse.Namespace(anima="sakura", all=True, restart=False, gateway_url=None)
        with pytest.raises(SystemExit):
            cmd_anima_codex_yolo(args)

        captured = capsys.readouterr()
        assert "not both" in captured.out

    @patch("core.paths.get_data_dir")
    def test_codex_yolo_single_refresh_failure_exits_nonzero(self, mock_data_dir, tmp_path, capsys):
        from cli.commands import anima_mgmt

        data_dir = self._make_runtime(
            tmp_path,
            animas={"sakura": {"enabled": True, "model": "codex/o4-mini"}},
        )
        mock_data_dir.return_value = data_dir

        args = argparse.Namespace(anima="sakura", all=False, restart=False, gateway_url=None)
        with (
            patch.object(anima_mgmt, "_refresh_codex_yolo_config", side_effect=RuntimeError("broken config")),
            pytest.raises(SystemExit),
        ):
            anima_mgmt.cmd_anima_codex_yolo(args)

        captured = capsys.readouterr()
        assert "config refresh failed" in captured.out
        assert "Codex YOLO updated for 0 anima(s)." in captured.out

    @patch("core.paths.get_data_dir")
    @patch("requests.post")
    def test_codex_yolo_restart_posts_when_server_running(
        self,
        mock_post,
        mock_data_dir,
        tmp_path,
        capsys,
    ):
        from cli.commands.anima_mgmt import cmd_anima_codex_yolo

        data_dir = self._make_runtime(
            tmp_path,
            animas={"sakura": {"enabled": True, "model": "codex/o4-mini"}},
        )
        (data_dir / "server.pid").write_text("1234", encoding="utf-8")
        mock_data_dir.return_value = data_dir
        response = MagicMock()
        response.raise_for_status.return_value = None
        mock_post.return_value = response

        args = argparse.Namespace(
            anima="sakura",
            all=False,
            restart=True,
            gateway_url="http://localhost:18500",
        )
        cmd_anima_codex_yolo(args)

        captured = capsys.readouterr()
        assert "restarted" in captured.out
        mock_post.assert_called_once_with(
            "http://localhost:18500/api/animas/sakura/restart",
            timeout=30.0,
        )


class TestCmdAnimaSetMemoryBackend:
    """Tests for cmd_anima_set_memory_backend."""

    @patch("core.paths.get_data_dir")
    def test_set_neo4j_prints_experimental_warning(self, mock_data_dir, tmp_path, capsys):
        from cli.commands.anima_mgmt import cmd_anima_set_memory_backend

        data_dir = tmp_path / ".animaworks"
        anima_dir = data_dir / "animas" / "sakura"
        anima_dir.mkdir(parents=True)
        (anima_dir / "status.json").write_text(json.dumps({"enabled": True}), encoding="utf-8")
        mock_data_dir.return_value = data_dir

        args = argparse.Namespace(anima="sakura", backend="neo4j", clear=False)
        cmd_anima_set_memory_backend(args)

        captured = capsys.readouterr()
        assert "Memory backend set to 'neo4j' for 'sakura'" in captured.out
        assert "experimental/opt-in" in captured.out
        status = json.loads((anima_dir / "status.json").read_text(encoding="utf-8"))
        assert status["memory_backend"] == "neo4j"

    @patch("core.paths.get_data_dir")
    def test_set_legacy_does_not_print_experimental_warning(self, mock_data_dir, tmp_path, capsys):
        from cli.commands.anima_mgmt import cmd_anima_set_memory_backend

        data_dir = tmp_path / ".animaworks"
        anima_dir = data_dir / "animas" / "sakura"
        anima_dir.mkdir(parents=True)
        (anima_dir / "status.json").write_text(
            json.dumps({"enabled": True, "memory_backend": "neo4j"}),
            encoding="utf-8",
        )
        mock_data_dir.return_value = data_dir

        args = argparse.Namespace(anima="sakura", backend="legacy", clear=False)
        cmd_anima_set_memory_backend(args)

        captured = capsys.readouterr()
        assert "Memory backend set to 'legacy' for 'sakura'" in captured.out
        assert "experimental/opt-in" not in captured.out
        status = json.loads((anima_dir / "status.json").read_text(encoding="utf-8"))
        assert status["memory_backend"] == "legacy"
