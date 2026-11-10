from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from cli.commands.vault_cmd import register_vault_command
from core.config.vault import VaultManager
from core.exceptions import ConfigError


def _parse(*arguments: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    register_vault_command(subparsers)
    return parser.parse_args(list(arguments))


def _make_anima_handler(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    anima_dir = tmp_path / "animas" / "sakura"
    (anima_dir / "activity_log").mkdir(parents=True)
    monkeypatch.setenv("ANIMAWORKS_ANIMA_DIR", str(anima_dir))
    memory = MagicMock()
    memory.anima_dir = anima_dir
    with (
        patch("core.tooling.handler.ExternalToolDispatcher"),
        patch("core.config.models.load_config", side_effect=ConfigError("skip")),
    ):
        from core.tooling.handler import ToolHandler

        return ToolHandler(anima_dir=anima_dir, memory=memory)


def _run(args: argparse.Namespace, vault: VaultManager, handler=None) -> None:
    with patch("core.config.vault.get_vault_manager", return_value=vault):
        if handler is None:
            args.func(args)
        else:
            with patch("cli._anima_tool.run_anima_tool", side_effect=handler.handle):
                args.func(args)


def _vault(tmp_path: Path) -> VaultManager:
    vault = VaultManager(tmp_path)
    vault.generate_key()
    return vault


def test_vault_status_summarizes_sections_without_values(tmp_path: Path, capsys) -> None:
    vault = _vault(tmp_path)
    encrypted_value = vault.encrypt("test-value")
    vault.save_vault(
        {
            "shared": {"ENCRYPTED": encrypted_value, "PLAINTEXT": "plain-value"},
            "sakura": {"EMPTY": ""},
        }
    )

    _run(_parse("vault", "status"), vault)

    output = capsys.readouterr().out
    status = json.loads(output)
    assert status == {
        "key_present": True,
        "entry_count": 3,
        "sections": {
            "shared": {"entries": 2, "encrypted": 1, "plaintext_like": 1, "invalid": 0, "onepassword_managed": 0},
            "sakura": {"entries": 1, "encrypted": 0, "plaintext_like": 1, "invalid": 0, "onepassword_managed": 0},
        },
        "onepassword": {"managed_count": 0, "last_check": None},
    }
    assert "test-value" not in output
    assert "plain-value" not in output
    assert "ENCRYPTED" not in output
    assert "PLAINTEXT" not in output


def test_vault_init_generates_key_without_anima_dir(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.delenv("ANIMAWORKS_ANIMA_DIR", raising=False)
    vault = VaultManager(tmp_path)

    _run(_parse("vault", "init"), vault)

    assert vault.has_key
    assert json.loads(capsys.readouterr().out) == {"initialized": True}


def test_vault_init_refuses_existing_key(tmp_path: Path, capsys) -> None:
    vault = _vault(tmp_path)
    original_key = vault.key_path.read_bytes()

    with pytest.raises(SystemExit) as exc_info:
        _run(_parse("vault", "init"), vault)

    assert exc_info.value.code == 1
    assert vault.key_path.read_bytes() == original_key
    assert "already exists" in capsys.readouterr().err


def test_vault_store_shared_reads_value_from_stdin(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.delenv("ANIMAWORKS_ANIMA_DIR", raising=False)
    vault = _vault(tmp_path)
    monkeypatch.setattr("sys.stdin", io.StringIO("test-input-value\n"))

    _run(_parse("vault", "store", "--shared", "API_TOKEN"), vault)

    output = capsys.readouterr().out
    assert vault.get("shared", "API_TOKEN") == "test-input-value"
    assert json.loads(output) == {"key": "API_TOKEN", "section": "shared", "stored": True}
    assert "test-input-value" not in output


def test_vault_store_shared_rejects_empty_stdin(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.delenv("ANIMAWORKS_ANIMA_DIR", raising=False)
    vault = _vault(tmp_path)
    monkeypatch.setattr("sys.stdin", io.StringIO(""))

    with pytest.raises(SystemExit) as exc_info:
        _run(_parse("vault", "store", "--shared", "API_TOKEN"), vault)

    assert exc_info.value.code == 1
    assert vault.get("shared", "API_TOKEN") is None
    assert "nothing was stored" in capsys.readouterr().err


def test_vault_get_falls_back_to_shared_section(tmp_path: Path, monkeypatch, capsys) -> None:
    handler = _make_anima_handler(tmp_path, monkeypatch)
    vault = _vault(tmp_path)
    vault.store("shared", "API_TOKEN", "shared-value")

    _run(_parse("vault", "get", "API_TOKEN"), vault, handler)

    assert capsys.readouterr().out == "shared-value\n"


def test_vault_get_prefers_anima_namespace(tmp_path: Path, monkeypatch, capsys) -> None:
    handler = _make_anima_handler(tmp_path, monkeypatch)
    vault = _vault(tmp_path)
    vault.store("shared", "API_TOKEN", "shared-value")
    vault.store("sakura", "API_TOKEN", "anima-value")

    _run(_parse("vault", "get", "API_TOKEN"), vault, handler)

    assert capsys.readouterr().out == "anima-value\n"


def test_vault_get_shared_flag_skips_anima_namespace(tmp_path: Path, monkeypatch, capsys) -> None:
    handler = _make_anima_handler(tmp_path, monkeypatch)
    vault = _vault(tmp_path)
    vault.store("sakura", "API_TOKEN", "anima-value")

    with pytest.raises(SystemExit) as exc_info:
        _run(_parse("vault", "get", "API_TOKEN", "--shared"), vault, handler)

    assert exc_info.value.code == 1
    assert json.loads(capsys.readouterr().out)["error_type"] == "NotFound"


def test_operator_can_get_shared_without_anima_dir(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.delenv("ANIMAWORKS_ANIMA_DIR", raising=False)
    vault = _vault(tmp_path)
    vault.store("shared", "API_TOKEN", "shared-value")

    _run(_parse("vault", "get", "API_TOKEN", "--shared"), vault)

    assert json.loads(capsys.readouterr().out) == {
        "key": "API_TOKEN",
        "section": "shared",
        "value": "shared-value",
    }


def test_vault_list_shows_only_anima_and_shared_keys(tmp_path: Path, monkeypatch, capsys) -> None:
    handler = _make_anima_handler(tmp_path, monkeypatch)
    vault = _vault(tmp_path)
    vault.store("shared", "API_TOKEN", "shared-value")
    vault.store("sakura", "LOCAL_KEY", "anima-value")
    vault.store("alice", "FOREIGN_KEY", "foreign-value")

    _run(_parse("vault", "list"), vault, handler)

    output = capsys.readouterr().out
    assert json.loads(output) == {
        "sections": {"sakura": ["LOCAL_KEY"], "shared": ["API_TOKEN"]},
    }
    assert "alice" not in output
    assert "FOREIGN_KEY" not in output
    assert "shared-value" not in output
    assert "anima-value" not in output


def test_vault_list_shared_flag_works_without_anima_dir(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.delenv("ANIMAWORKS_ANIMA_DIR", raising=False)
    vault = _vault(tmp_path)
    vault.store("shared", "API_TOKEN", "shared-value")

    _run(_parse("vault", "list", "--shared"), vault)

    assert json.loads(capsys.readouterr().out) == {
        "namespace": "shared",
        "keys": ["API_TOKEN"],
    }


def test_vault_store_positional_value_uses_handler_and_own_namespace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys,
) -> None:
    handler = _make_anima_handler(tmp_path, monkeypatch)
    vault = _vault(tmp_path)

    _run(_parse("vault", "store", "LOCAL_KEY", "legacy-value"), vault, handler)

    assert vault.get("sakura", "LOCAL_KEY") == "legacy-value"
    assert json.loads(capsys.readouterr().out) == {
        "status": "ok",
        "message": "Stored sakura/LOCAL_KEY",
    }


def test_anima_cannot_store_to_shared_section(tmp_path: Path, monkeypatch, capsys) -> None:
    handler = _make_anima_handler(tmp_path, monkeypatch)
    vault = _vault(tmp_path)

    with pytest.raises(SystemExit) as exc_info:
        _run(_parse("vault", "store", "API_TOKEN", "secret", "--shared"), vault, handler)

    assert exc_info.value.code == 1
    assert vault.get("shared", "API_TOKEN") is None
    assert json.loads(capsys.readouterr().out)["error_type"] == "PermissionDenied"


def test_vault_delete_removes_key_from_anima_namespace(tmp_path: Path, monkeypatch, capsys) -> None:
    anima_dir = tmp_path / "animas" / "sakura"
    anima_dir.mkdir(parents=True)
    monkeypatch.setenv("ANIMAWORKS_ANIMA_DIR", str(anima_dir))
    vault = _vault(tmp_path)
    vault.store("sakura", "LOCAL_KEY", "anima-value")

    _run(_parse("vault", "delete", "LOCAL_KEY"), vault)

    assert vault.get("sakura", "LOCAL_KEY") is None
    assert json.loads(capsys.readouterr().out) == {
        "key": "LOCAL_KEY",
        "section": "sakura",
        "deleted": True,
    }


def test_anima_cannot_delete_from_shared_section(tmp_path: Path, monkeypatch, capsys) -> None:
    anima_dir = tmp_path / "animas" / "sakura"
    anima_dir.mkdir(parents=True)
    monkeypatch.setenv("ANIMAWORKS_ANIMA_DIR", str(anima_dir))
    vault = _vault(tmp_path)
    vault.store("shared", "API_TOKEN", "shared-value")

    with pytest.raises(SystemExit) as exc_info:
        _run(_parse("vault", "delete", "API_TOKEN", "--shared"), vault)

    assert exc_info.value.code == 1
    assert vault.get("shared", "API_TOKEN") == "shared-value"
    assert "may not write vault section 'shared'" in capsys.readouterr().err


def test_operator_can_delete_from_shared_section(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.delenv("ANIMAWORKS_ANIMA_DIR", raising=False)
    vault = _vault(tmp_path)
    vault.store("shared", "API_TOKEN", "shared-value")

    _run(_parse("vault", "delete", "API_TOKEN", "--shared"), vault)

    assert vault.get("shared", "API_TOKEN") is None
    assert json.loads(capsys.readouterr().out)["section"] == "shared"


def test_vault_delete_never_cascades_into_shared(tmp_path: Path, monkeypatch, capsys) -> None:
    anima_dir = tmp_path / "animas" / "sakura"
    anima_dir.mkdir(parents=True)
    monkeypatch.setenv("ANIMAWORKS_ANIMA_DIR", str(anima_dir))
    vault = _vault(tmp_path)
    vault.store("shared", "API_TOKEN", "shared-value")

    with pytest.raises(SystemExit) as exc_info:
        _run(_parse("vault", "delete", "API_TOKEN"), vault)

    assert exc_info.value.code == 1
    assert vault.get("shared", "API_TOKEN") == "shared-value"
    assert "not found in section 'sakura'" in capsys.readouterr().err


def test_vault_store_shared_rejects_positional_value(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.delenv("ANIMAWORKS_ANIMA_DIR", raising=False)
    vault = _vault(tmp_path)

    with pytest.raises(SystemExit) as exc_info:
        _run(_parse("vault", "store", "API_TOKEN", "visible-value", "--shared"), vault)

    assert exc_info.value.code == 1
    assert vault.get("shared", "API_TOKEN") is None
    assert "must be supplied through stdin" in capsys.readouterr().err
