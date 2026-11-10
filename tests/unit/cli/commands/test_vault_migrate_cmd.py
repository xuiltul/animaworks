from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from cli.commands import vault_cmd
from core.config.vault import VaultManager


def _args(*arguments: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    vault_cmd.register_vault_command(subparsers)
    return parser.parse_args(["vault", "migrate-config", *arguments])


def _config(path: Path) -> None:
    path.write_text(
        json.dumps({"setup_complete": True, "credentials": {"svc": {"api_key": "cli-plaintext-secret"}}}),
        encoding="utf-8",
    )


def test_migrate_config_preview_lists_paths_without_values(tmp_path: Path, monkeypatch, capsys) -> None:
    config_path = tmp_path / "config.json"
    _config(config_path)
    monkeypatch.setattr(vault_cmd, "get_config_path", lambda: config_path)
    monkeypatch.delenv("ANIMAWORKS_ANIMA_DIR", raising=False)

    vault_cmd.cmd_vault(_args())

    output = capsys.readouterr().out
    payload = json.loads(output)
    assert payload == {
        "credentials": [{"path": "credentials.svc.api_key", "vault_key": "SVC_API_KEY"}],
        "committed": False,
    }
    assert "cli-plaintext-secret" not in output


def test_migrate_config_commit_references_vault_value(tmp_path: Path, monkeypatch, capsys) -> None:
    config_path = tmp_path / "config.json"
    _config(config_path)
    data_dir = tmp_path
    VaultManager(data_dir).generate_key()
    monkeypatch.setattr(vault_cmd, "get_config_path", lambda: config_path)
    monkeypatch.delenv("ANIMAWORKS_ANIMA_DIR", raising=False)

    vault_cmd.cmd_vault(_args("--commit"))

    output = capsys.readouterr().out
    assert json.loads(output) == {
        "moved": [{"path": "credentials.svc.api_key", "vault_key": "SVC_API_KEY"}],
        "committed": True,
    }
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    assert raw["credentials"]["svc"]["api_key"] == {"$vault": "SVC_API_KEY"}
    assert "cli-plaintext-secret" not in output


def test_migrate_config_refuses_anima_and_reports_running_server(tmp_path: Path, monkeypatch, capsys) -> None:
    config_path = tmp_path / "config.json"
    _config(config_path)
    monkeypatch.setattr(vault_cmd, "get_config_path", lambda: config_path)
    monkeypatch.setenv("ANIMAWORKS_ANIMA_DIR", str(tmp_path / "animas" / "worker"))
    with pytest.raises(SystemExit) as operator_exit:
        vault_cmd.cmd_vault(_args("--commit"))
    assert operator_exit.value.code == 1
    assert "オペレーター専用" in capsys.readouterr().err

    monkeypatch.delenv("ANIMAWORKS_ANIMA_DIR")
    monkeypatch.setattr(vault_cmd, "update_config", lambda *args, **kwargs: (_ for _ in ()).throw(PermissionError()))
    with pytest.raises(SystemExit) as server_exit:
        vault_cmd.cmd_vault(_args("--commit"))
    assert server_exit.value.code == 1
    assert "サーバーを停止" in capsys.readouterr().err


def test_migrate_config_refuses_even_empty_anima_environment(tmp_path: Path, monkeypatch, capsys) -> None:
    config_path = tmp_path / "config.json"
    _config(config_path)
    monkeypatch.setattr(vault_cmd, "get_config_path", lambda: config_path)
    monkeypatch.setenv("ANIMAWORKS_ANIMA_DIR", "")

    with pytest.raises(SystemExit) as result:
        vault_cmd.cmd_vault(_args())

    assert result.value.code == 1
    assert "オペレーター専用" in capsys.readouterr().err
