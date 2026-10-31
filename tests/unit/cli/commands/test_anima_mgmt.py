"""Unit tests for cli/commands/anima_mgmt.py — Anima lifecycle CLI."""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


def test_rename_dm_logs_appends_under_locked_jsonl_helper(tmp_path: Path) -> None:
    from cli.commands.anima_mgmt import _rename_dm_logs

    dm_logs = tmp_path / "shared" / "dm_logs"
    dm_logs.mkdir(parents=True)
    source = dm_logs / "alice-bob.jsonl"
    destination = dm_logs / "bob-carol.jsonl"
    source.write_text('{"from":"alice","text":"old"}\n', encoding="utf-8")
    destination.write_text('{"from":"carol","text":"existing"}\n', encoding="utf-8")

    assert _rename_dm_logs(tmp_path / "shared", "alice", "carol") == 1

    lines = destination.read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["text"] for line in lines] == ["existing", "old"]
    assert not source.exists()
    assert (dm_logs / "bob-carol.jsonl.lock").is_file()


def test_rename_rag_cleanup_deletes_old_collections_through_owner(tmp_path: Path) -> None:
    from cli.commands.anima_mgmt import _cleanup_rag_collections

    anima_dir = tmp_path / "animas" / "new_name"
    (anima_dir / "vectordb").mkdir(parents=True)
    store = MagicMock()
    store.delete_collection.return_value = True
    access_context = MagicMock()
    access_context.__enter__.return_value = SimpleNamespace(store=store)
    with patch("core.memory.rag.cli_access.open_vector_access", return_value=access_context) as open_access:
        assert _cleanup_rag_collections(anima_dir, "old_name") is False

    open_access.assert_called_once_with("new_name", anima_dir, purpose="rename")
    assert store.delete_collection.call_args_list[0].args == ("old_name_knowledge",)
    assert store.delete_collection.call_count == 6


def test_rename_rag_cleanup_queues_rebuild_when_owner_is_busy(tmp_path: Path) -> None:
    from cli.commands.anima_mgmt import _cleanup_rag_collections
    from core.memory.rag.owner_lock import VectorOwnerBusy

    anima_dir = tmp_path / "animas" / "new_name"
    (anima_dir / "vectordb").mkdir(parents=True)
    with (
        patch("core.memory.rag.cli_access.open_vector_access", side_effect=VectorOwnerBusy("busy")),
        patch("core.memory.rag.repair.state.write_repair_request_state") as write_request,
    ):
        assert _cleanup_rag_collections(anima_dir, "old_name") is True

    write_request.assert_called_once_with(
        "new_name",
        reason="anima_renamed",
        collection=None,
        source="cli",
        include_shared=True,
    )


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
        (anima_dir / "IDENTITY.md").write_text("# Alice", encoding="utf-8")
        (anima_dir / "status.json").write_text(json.dumps({"enabled": True}), encoding="utf-8")
        # Create config.json with the anima registered
        config = {
            "version": 1,
            "animas": {name: {}},
        }
        (data_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")
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
        (sub_dir / "IDENTITY.md").write_text("# Kotoha", encoding="utf-8")
        (sub_dir / "status.json").write_text(json.dumps({"enabled": True, "supervisor": "sakura"}), encoding="utf-8")
        mock_data_dir.return_value = data_dir
        mock_animas_dir.return_value = animas_dir

        args = argparse.Namespace(anima="sakura", force=True, no_archive=True, gateway_url=None)
        cmd_anima_delete(args)

        captured = capsys.readouterr()
        assert "Warning" in captured.out
        assert "kotoha" in captured.out


def test_delete_with_live_server_delegates_to_api(tmp_path, capsys):
    from cli.commands.anima_mgmt import cmd_anima_delete

    data_dir = TestCmdAnimaDelete()._make_anima_dir(tmp_path, "alice")
    animas_dir = data_dir / "animas"
    (data_dir / "server.pid").write_text("1234", encoding="utf-8")
    response = MagicMock()
    response.json.return_value = {
        "status": "deleted",
        "name": "alice",
        "archive_path": str(data_dir / "archive" / "alice.zip"),
        "supervisor_warnings": [],
    }
    args = argparse.Namespace(anima="alice", force=True, no_archive=False, gateway_url=None)

    with (
        patch("core.paths.get_data_dir", return_value=data_dir),
        patch("core.paths.get_animas_dir", return_value=animas_dir),
        patch("core.platform.process.is_process_alive", return_value=True),
        patch("cli.commands.anima_mgmt.gateway_request", return_value=response) as mock_gateway,
    ):
        cmd_anima_delete(args)

    mock_gateway.assert_called_once_with(
        args,
        "DELETE",
        "/api/animas/alice?archive=true",
        timeout=30.0,
        raw_response=True,
    )
    assert (animas_dir / "alice").is_dir()
    assert "Archived to:" in capsys.readouterr().out


def test_delete_with_stale_server_pid_uses_local_service(tmp_path):
    from cli.commands.anima_mgmt import cmd_anima_delete

    data_dir = TestCmdAnimaDelete()._make_anima_dir(tmp_path, "alice")
    animas_dir = data_dir / "animas"
    (data_dir / "server.pid").write_text("1234", encoding="utf-8")
    args = argparse.Namespace(anima="alice", force=True, no_archive=True, gateway_url=None)

    with (
        patch("core.paths.get_data_dir", return_value=data_dir),
        patch("core.paths.get_animas_dir", return_value=animas_dir),
        patch("core.platform.process.is_process_alive", return_value=False),
        patch("cli.commands.anima_mgmt.gateway_request") as mock_gateway,
    ):
        cmd_anima_delete(args)

    mock_gateway.assert_not_called()
    assert not (animas_dir / "alice").exists()
    config = json.loads((data_dir / "config.json").read_text(encoding="utf-8"))
    assert "alice" not in config["animas"]


def test_delete_refuses_local_changes_if_live_server_api_is_unreachable(tmp_path, capsys):
    from cli.commands.anima_mgmt import cmd_anima_delete

    data_dir = TestCmdAnimaDelete()._make_anima_dir(tmp_path, "alice")
    animas_dir = data_dir / "animas"
    (data_dir / "server.pid").write_text("1234", encoding="utf-8")
    args = argparse.Namespace(anima="alice", force=True, no_archive=False, gateway_url=None)

    with (
        patch("core.paths.get_data_dir", return_value=data_dir),
        patch("core.paths.get_animas_dir", return_value=animas_dir),
        patch("core.platform.process.is_process_alive", return_value=True),
        patch("cli.commands.anima_mgmt.gateway_request", side_effect=RuntimeError("gateway unavailable")),
        pytest.raises(SystemExit) as exc_info,
    ):
        cmd_anima_delete(args)

    assert exc_info.value.code == 1
    assert (animas_dir / "alice").is_dir()
    assert "gateway unavailable" in capsys.readouterr().out


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
        (anima_dir / "IDENTITY.md").write_text("# Alice")
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
        (anima_dir / "IDENTITY.md").write_text("# Alice")
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
            (d / "IDENTITY.md").write_text(f"# {name}")
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


@pytest.mark.parametrize(
    ("command_name", "endpoint", "initial_enabled", "expected_enabled", "label"),
    [
        ("cmd_anima_enable", "/api/animas/alice/enable", False, True, "Enabled"),
        ("cmd_anima_disable", "/api/animas/alice/disable", True, False, "Disabled"),
    ],
)
@pytest.mark.parametrize("server_running", [True, False])
def test_enable_disable_gateway_and_offline_paths(
    command_name, endpoint, initial_enabled, expected_enabled, label, server_running, tmp_path, capsys
):
    import cli.commands.anima_mgmt as anima_mgmt

    data_dir = tmp_path / ".animaworks"
    animas_dir = data_dir / "animas"
    anima_dir = animas_dir / "alice"
    anima_dir.mkdir(parents=True)
    (anima_dir / "IDENTITY.md").write_text("# Alice", encoding="utf-8")
    status_file = anima_dir / "status.json"
    status_file.write_text(json.dumps({"enabled": initial_enabled, "role": "general"}), encoding="utf-8")
    if server_running:
        (data_dir / "server.pid").write_text("1234", encoding="utf-8")

    response = MagicMock()
    response.json.return_value = {"ok": True}
    args = argparse.Namespace(anima="alice", gateway_url="http://custom:18501")
    with (
        patch("core.paths.get_data_dir", return_value=data_dir),
        patch("core.paths.get_animas_dir", return_value=animas_dir),
        patch("cli.commands.anima_mgmt.gateway_request", return_value=response) as mock_gateway,
    ):
        getattr(anima_mgmt, command_name)(args)

    output = capsys.readouterr().out
    status = json.loads(status_file.read_text(encoding="utf-8"))
    if server_running:
        mock_gateway.assert_called_once_with(
            args,
            "POST",
            endpoint,
            timeout=10,
            raw_response=True,
        )
        assert f"{label} anima 'alice': {{'ok': True}}" in output
        assert "offline mode" not in output
        assert status["enabled"] is initial_enabled
    else:
        mock_gateway.assert_not_called()
        assert f"{label} anima 'alice' (offline mode)" in output
        assert status["enabled"] is expected_enabled
    assert status["role"] == "general"


def test_restart_gateway_when_server_running(tmp_path, capsys):
    from cli.commands.anima_mgmt import cmd_anima_restart

    data_dir = tmp_path / ".animaworks"
    data_dir.mkdir()
    (data_dir / "server.pid").write_text("1234", encoding="utf-8")
    response = MagicMock()
    response.json.return_value = {"pid": 5678}
    args = argparse.Namespace(anima="alice", gateway_url=None)

    with (
        patch("core.paths.get_data_dir", return_value=data_dir),
        patch("cli.commands.anima_mgmt.gateway_request", return_value=response) as mock_gateway,
    ):
        cmd_anima_restart(args)

    mock_gateway.assert_called_once_with(
        args,
        "POST",
        "/api/animas/alice/restart",
        timeout=30.0,
        raw_response=True,
    )
    output = capsys.readouterr().out
    assert "Anima 'alice' restarted successfully" in output
    assert "PID: 5678" in output


def test_restart_keeps_server_stopped_message_and_exit_code(tmp_path, capsys):
    from cli.commands.anima_mgmt import cmd_anima_restart

    data_dir = tmp_path / ".animaworks"
    data_dir.mkdir()
    args = argparse.Namespace(anima="alice", gateway_url=None)

    with (
        patch("core.paths.get_data_dir", return_value=data_dir),
        patch("cli.commands.anima_mgmt.gateway_request") as mock_gateway,
        pytest.raises(SystemExit) as exc_info,
    ):
        cmd_anima_restart(args)

    assert exc_info.value.code == 1
    assert capsys.readouterr().out.strip() == "Error: Server is not running"
    mock_gateway.assert_not_called()


@pytest.mark.parametrize("server_running", [True, False])
def test_set_model_uses_root_api_when_server_is_running(server_running, tmp_path, capsys):
    from cli.commands.anima_mgmt import cmd_anima_set_model

    data_dir = tmp_path / ".animaworks"
    anima_dir = data_dir / "animas" / "alice"
    anima_dir.mkdir(parents=True)
    if server_running:
        (data_dir / "server.pid").write_text("1234", encoding="utf-8")
    args = argparse.Namespace(all=False, anima="alice", model="new-model", credential=None)

    with (
        patch("core.paths.get_data_dir", return_value=data_dir),
        patch(
            "core.config.model_config.smart_update_model",
            return_value={"family_changed": False, "execution_mode": "S"},
        ) as mock_local,
        patch("cli.commands.anima_mgmt.gateway_request") as mock_gateway,
    ):
        mock_gateway.return_value.json.return_value = {
            "family_changed": False,
            "credential": "",
            "execution_mode": "S",
        }
        cmd_anima_set_model(args)

    output = capsys.readouterr().out
    assert "Model updated to 'new-model' for 'alice'" in output
    assert ("Running anima processes were asked to reload" in output) is server_running
    assert mock_gateway.called is server_running
    assert mock_local.called is not server_running


def test_gateway_request_raw_response_preserves_caller_error_handling():
    import httpx

    from cli._gateway import gateway_request

    response = MagicMock()
    args = argparse.Namespace(gateway_url="http://gateway.test:18501")
    with patch("httpx.request", return_value=response) as mock_request:
        assert gateway_request(args, "POST", "/api/animas/alice/enable", timeout=10, raw_response=True) is response
    mock_request.assert_called_once_with(
        "POST",
        "http://gateway.test:18501/api/animas/alice/enable",
        json=None,
        timeout=10,
    )

    error = httpx.ConnectError("offline", request=httpx.Request("POST", "http://gateway.test:18501"))
    with patch("httpx.request", side_effect=error), pytest.raises(httpx.ConnectError, match="offline"):
        gateway_request(args, "POST", "/api/animas/alice/enable", timeout=10, raw_response=True)
