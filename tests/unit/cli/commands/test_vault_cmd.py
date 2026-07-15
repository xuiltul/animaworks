from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from cli.commands.vault_cmd import register_vault_command
from core.config.vault import VaultManager


def _parse(*arguments: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    register_vault_command(subparsers)
    return parser.parse_args(list(arguments))


def _run(args: argparse.Namespace, vault: VaultManager) -> None:
    with patch("core.config.vault.get_vault_manager", return_value=vault):
        args.func(args)


def test_vault_status_summarizes_sections_without_values(tmp_path: Path, capsys) -> None:
    vault = VaultManager(tmp_path)
    vault.generate_key()
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
            "shared": {"entries": 2, "encrypted": 1, "plaintext_like": 1, "invalid": 0},
            "sakura": {"entries": 1, "encrypted": 0, "plaintext_like": 1, "invalid": 0},
        },
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
    vault = VaultManager(tmp_path)
    vault.generate_key()
    original_key = vault.key_path.read_bytes()

    with pytest.raises(SystemExit) as exc_info:
        _run(_parse("vault", "init"), vault)

    assert exc_info.value.code == 1
    assert vault.key_path.read_bytes() == original_key
    assert "already exists" in capsys.readouterr().err


def test_vault_store_shared_reads_value_from_stdin(tmp_path: Path, monkeypatch, capsys) -> None:
    vault = VaultManager(tmp_path)
    vault.generate_key()
    stdin = io.StringIO("test-input-value\n")
    monkeypatch.setattr("sys.stdin", stdin)

    _run(_parse("vault", "store", "--shared", "API_TOKEN"), vault)

    output = capsys.readouterr().out
    assert vault.get("shared", "API_TOKEN") == "test-input-value"
    assert json.loads(output) == {"key": "API_TOKEN", "stored": True}
    assert "test-input-value" not in output


def test_vault_store_shared_rejects_positional_value(tmp_path: Path, capsys) -> None:
    vault = VaultManager(tmp_path)
    vault.generate_key()

    with pytest.raises(SystemExit) as exc_info:
        _run(_parse("vault", "store", "API_TOKEN", "visible-value", "--shared"), vault)

    assert exc_info.value.code == 1
    assert vault.get("shared", "API_TOKEN") is None
    assert "must be supplied through stdin" in capsys.readouterr().err


def test_vault_store_positional_value_keeps_anima_scoped_compatibility(
    tmp_path: Path,
    monkeypatch,
) -> None:
    anima_dir = tmp_path / "animas" / "sakura"
    anima_dir.mkdir(parents=True)
    monkeypatch.setenv("ANIMAWORKS_ANIMA_DIR", str(anima_dir))
    vault = VaultManager(tmp_path)
    vault.generate_key()

    _run(_parse("vault", "store", "LOCAL_KEY", "legacy-value"), vault)

    assert vault.get("sakura", "LOCAL_KEY") == "legacy-value"
