from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from cli.commands.vault_cmd import register_vault_command
from core.config.vault import VaultManager
from core.config.vault_ops import meta_path, previous_path
from core.config.vault_sources import (
    OnePasswordSettings,
    VaultSources,
    add_source,
    load_sources,
    save_sources,
)
from tests.unit.vault_fake_op import FakeOp


def _parse(*arguments: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    register_vault_command(subparsers)
    return parser.parse_args(list(arguments))


def _vault(tmp_path: Path) -> VaultManager:
    vault = VaultManager(tmp_path)
    vault.generate_key()
    return vault


def _run(args: argparse.Namespace, vault: VaultManager) -> None:
    with patch("core.config.vault.get_vault_manager", return_value=vault):
        args.func(args)


def _configure(tmp_path: Path, op: FakeOp) -> None:
    sources = load_sources(tmp_path)
    sources.onepassword = OnePasswordSettings(command=op.command, timeout_s=5.0)
    save_sources(tmp_path, sources)


@pytest.fixture
def fake_op(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> FakeOp:
    op = FakeOp(tmp_path)
    for name, value in op.env.items():
        monkeypatch.setenv(name, value)
    monkeypatch.delenv("ANIMAWORKS_ANIMA_DIR", raising=False)
    return op


class _TTY(io.StringIO):
    def isatty(self) -> bool:
        return True


def test_sources_add_list_remove(tmp_path: Path, fake_op: FakeOp, capsys) -> None:
    vault = _vault(tmp_path)
    _configure(tmp_path, fake_op)

    _run(_parse("vault", "sources", "add", "SLACK_BOT_TOKEN", "op://Vault/Item/field"), vault)
    added = json.loads(capsys.readouterr().out)
    assert added["key"] == "SLACK_BOT_TOKEN"
    assert added["ref"] == "op://Vault/Item/field"

    _run(_parse("vault", "sources", "list"), vault)
    listed = json.loads(capsys.readouterr().out)
    assert listed["entries"] == [
        {"key": "SLACK_BOT_TOKEN", "ref": "op://Vault/Item/field", "added_at": added["added_at"]}
    ]
    assert listed["onepassword"]["command"] == fake_op.command

    _run(_parse("vault", "sources", "remove", "SLACK_BOT_TOKEN"), vault)
    assert json.loads(capsys.readouterr().out) == {"key": "SLACK_BOT_TOKEN", "removed": True}
    assert load_sources(tmp_path).entries == {}


def test_sources_invalid_ref_exits_one(tmp_path: Path, fake_op: FakeOp, capsys) -> None:
    vault = _vault(tmp_path)

    with pytest.raises(SystemExit) as exc_info:
        _run(_parse("vault", "sources", "add", "KEY", "not-a-ref"), vault)

    assert exc_info.value.code == 1
    assert "op://" in capsys.readouterr().err


def test_sync_yes_applies_changes_without_values(tmp_path: Path, fake_op: FakeOp, capsys) -> None:
    vault = _vault(tmp_path)
    vault.store("shared", "CHANGED", "old-changed")
    vault.store("shared", "SAME", "same-value")
    vault.store("shared", "EMPTY", "leave-me")
    for key, ref in {
        "NEW": "op://V/New/credential",
        "CHANGED": "op://V/Changed/credential",
        "SAME": "op://V/Same/credential",
        "EMPTY": "op://V/Empty/credential",
    }.items():
        add_source(tmp_path, key, ref)
    _configure(tmp_path, fake_op)
    fake_op.set_value("op://V/New/credential", "new-secret")
    fake_op.set_value("op://V/Changed/credential", "changed-secret")
    fake_op.set_value("op://V/Same/credential", "same-value")
    fake_op.set_value("op://V/Empty/credential", "")

    _run(_parse("vault", "sync", "--yes", "--json"), vault)
    output = capsys.readouterr().out

    payload = json.loads(output)
    assert payload["applied"] == ["CHANGED", "NEW"]
    assert payload["request_count"] == 4
    assert {item["key"]: item["status"] for item in payload["items"]} == {
        "CHANGED": "changed",
        "EMPTY": "empty_remote",
        "NEW": "new",
        "SAME": "unchanged",
    }
    assert vault.get("shared", "NEW") == "new-secret"
    assert vault.get("shared", "CHANGED") == "changed-secret"
    assert vault.get("shared", "EMPTY") == "leave-me"
    for secret in ("new-secret", "changed-secret", "old-changed", "same-value", "leave-me"):
        assert secret not in output
        assert hashlib.sha256(secret.encode()).hexdigest() not in output


def test_sync_non_tty_without_yes_does_not_write(tmp_path: Path, fake_op: FakeOp, monkeypatch, capsys) -> None:
    vault = _vault(tmp_path)
    add_source(tmp_path, "NEW", "op://V/New/credential")
    _configure(tmp_path, fake_op)
    fake_op.set_value("op://V/New/credential", "new-secret")
    monkeypatch.setattr("sys.stdin", io.StringIO(""))

    with pytest.raises(SystemExit) as exc_info:
        _run(_parse("vault", "sync", "--json"), vault)

    assert exc_info.value.code == 1
    assert vault.get("shared", "NEW") is None
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["applied"] == []
    assert "new-secret" not in json.dumps(payload)
    assert "Error:" in captured.err


def test_sync_tty_prompt_applies_on_yes(tmp_path: Path, fake_op: FakeOp, monkeypatch, capsys) -> None:
    vault = _vault(tmp_path)
    add_source(tmp_path, "NEW", "op://V/New/credential")
    _configure(tmp_path, fake_op)
    fake_op.set_value("op://V/New/credential", "new-secret")
    monkeypatch.setattr("sys.stdin", _TTY("y\n"))

    _run(_parse("vault", "sync"), vault)

    assert vault.get("shared", "NEW") == "new-secret"
    out = capsys.readouterr().out
    assert "applied: NEW" in out
    assert "new-secret" not in out


def test_sync_check_exit_codes(tmp_path: Path, fake_op: FakeOp) -> None:
    vault = _vault(tmp_path)
    vault.store("shared", "KEY", "same-value")
    add_source(tmp_path, "KEY", "op://V/Key/credential")
    _configure(tmp_path, fake_op)

    fake_op.set_value("op://V/Key/credential", "same-value")
    _run(_parse("vault", "sync", "--check"), vault)

    fake_op.set_value("op://V/Key/credential", "different-value")
    with pytest.raises(SystemExit) as exc_info:
        _run(_parse("vault", "sync", "--check", "--json"), vault)
    assert exc_info.value.code == 2

    fake_op.remove_value("op://V/Key/credential")
    with pytest.raises(SystemExit) as exc_info:
        _run(_parse("vault", "sync", "--check"), vault)
    assert exc_info.value.code == 3


def test_sync_check_missing_local_exits_two(tmp_path: Path, fake_op: FakeOp) -> None:
    vault = _vault(tmp_path)
    add_source(tmp_path, "KEY", "op://V/Key/credential")
    _configure(tmp_path, fake_op)
    fake_op.set_value("op://V/Key/credential", "remote-value")

    with pytest.raises(SystemExit) as exc_info:
        _run(_parse("vault", "sync", "--check"), vault)

    assert exc_info.value.code == 2


def test_missing_op_cli_exits_three(tmp_path: Path, capsys) -> None:
    vault = _vault(tmp_path)
    save_sources(
        tmp_path,
        VaultSources(onepassword=OnePasswordSettings(command=["definitely-not-an-op-cli"])),
    )

    for arguments in (
        ("vault", "sync"),
        ("vault", "sync", "--check"),
        ("vault", "register-1p", "KEY", "--vault", "V"),
    ):
        with pytest.raises(SystemExit) as exc_info:
            _run(_parse(*arguments), vault)
        assert exc_info.value.code == 3
    assert "1Password CLI" in capsys.readouterr().err


def test_managed_key_store_and_delete_are_rejected(tmp_path: Path, fake_op: FakeOp, monkeypatch, capsys) -> None:
    vault = _vault(tmp_path)
    vault.store("shared", "MANAGED", "original")
    add_source(tmp_path, "MANAGED", "op://V/Managed/credential")
    monkeypatch.setattr("sys.stdin", io.StringIO("new-value\n"))

    with pytest.raises(SystemExit) as exc_info:
        _run(_parse("vault", "store", "MANAGED", "--shared"), vault)
    assert exc_info.value.code == 1
    assert vault.get("shared", "MANAGED") == "original"

    with pytest.raises(SystemExit) as exc_info:
        _run(_parse("vault", "delete", "MANAGED", "--shared"), vault)
    assert exc_info.value.code == 1
    assert vault.get("shared", "MANAGED") == "original"
    assert "1Password" in capsys.readouterr().err


def test_store_keeps_previous_and_delete_keeps_previous(tmp_path: Path, fake_op: FakeOp, monkeypatch, capsys) -> None:
    vault = _vault(tmp_path)

    monkeypatch.setattr("sys.stdin", io.StringIO("one\n"))
    _run(_parse("vault", "store", "API_TOKEN", "--shared"), vault)
    monkeypatch.setattr("sys.stdin", io.StringIO("two\n"))
    _run(_parse("vault", "store", "API_TOKEN", "--shared"), vault)
    previous = json.loads(previous_path(tmp_path).read_text(encoding="utf-8"))
    assert vault.decrypt(previous["shared"]["API_TOKEN"]["value"]) == "one"
    assert previous["shared"]["API_TOKEN"]["replaced_by"] == "cli"

    _run(_parse("vault", "delete", "API_TOKEN", "--shared"), vault)
    assert vault.get("shared", "API_TOKEN") is None
    previous = json.loads(previous_path(tmp_path).read_text(encoding="utf-8"))
    assert vault.decrypt(previous["shared"]["API_TOKEN"]["value"]) == "two"
    meta = json.loads(meta_path(tmp_path).read_text(encoding="utf-8"))
    assert meta["entries"]["shared/API_TOKEN"]["updated_by"] == "cli"


def test_rollback_command(tmp_path: Path, fake_op: FakeOp, capsys) -> None:
    vault = _vault(tmp_path)
    from core.config.vault_ops import set_value

    set_value(tmp_path, "shared", "KEY", "one", actor="cli")
    set_value(tmp_path, "shared", "KEY", "two", actor="cli")

    _run(_parse("vault", "rollback", "KEY"), vault)
    assert vault.get("shared", "KEY") == "one"
    assert json.loads(capsys.readouterr().out) == {"key": "KEY", "rolled_back": True}

    with pytest.raises(SystemExit) as exc_info:
        _run(_parse("vault", "rollback", "MISSING"), vault)
    assert exc_info.value.code == 1


def test_register_preview_and_commit(tmp_path: Path, fake_op: FakeOp, capsys) -> None:
    vault = _vault(tmp_path)
    vault.store("shared", "API_TOKEN", "super-secret-value")
    _configure(tmp_path, fake_op)

    _run(_parse("vault", "register-1p", "API_TOKEN", "--vault", "V"), vault)
    preview = json.loads(capsys.readouterr().out)
    assert preview["committed"] is False
    assert preview["value_length"] == len("super-secret-value")
    assert "super-secret-value" not in json.dumps(preview)
    assert fake_op.created() == []

    fake_op.set_create_id("item-42")
    fake_op.set_value("op://V/item-42/credential", "super-secret-value")
    _run(_parse("vault", "register-1p", "API_TOKEN", "--vault", "V", "--commit"), vault)
    committed = json.loads(capsys.readouterr().out)
    assert committed["committed"] is True
    assert committed["ref"] == "op://V/item-42/credential"
    assert "super-secret-value" not in json.dumps(committed)
    assert load_sources(tmp_path).entries["API_TOKEN"].ref == "op://V/item-42/credential"
    create_argv = " ".join(" ".join(argv) for argv in fake_op.argv_log() if argv[:2] == ["item", "create"])
    assert "super-secret-value" not in create_argv


def test_status_includes_onepassword_counts_and_last_check(tmp_path: Path, fake_op: FakeOp, capsys) -> None:
    vault = _vault(tmp_path)
    vault.store("shared", "MANAGED", "same-value")
    vault.store("shared", "LOCAL", "local-value")
    add_source(tmp_path, "MANAGED", "op://V/Managed/credential")
    _configure(tmp_path, fake_op)
    fake_op.set_value("op://V/Managed/credential", "same-value")
    _run(_parse("vault", "sync", "--check"), vault)
    capsys.readouterr()

    _run(_parse("vault", "status"), vault)
    output = capsys.readouterr().out
    status = json.loads(output)

    assert status["sections"]["shared"]["onepassword_managed"] == 1
    assert status["onepassword"]["managed_count"] == 1
    assert status["onepassword"]["last_check"]["counts"] == {"same": 1}
    assert "same-value" not in output
    assert "local-value" not in output


@pytest.mark.parametrize(
    "arguments",
    [
        ("vault", "sources", "list"),
        ("vault", "sync", "--check"),
        ("vault", "rollback", "KEY"),
        ("vault", "register-1p", "KEY", "--vault", "V"),
    ],
)
def test_new_commands_reject_anima_context(
    tmp_path: Path,
    fake_op: FakeOp,
    monkeypatch,
    capsys,
    arguments: tuple[str, ...],
) -> None:
    anima_dir = tmp_path / "animas" / "sakura"
    anima_dir.mkdir(parents=True)
    monkeypatch.setenv("ANIMAWORKS_ANIMA_DIR", str(anima_dir))
    vault = _vault(tmp_path)

    with pytest.raises(SystemExit) as exc_info:
        _run(_parse(*arguments), vault)

    assert exc_info.value.code == 1
    assert "ANIMAWORKS_ANIMA_DIR" in capsys.readouterr().err


def test_sync_error_item_exits_three(tmp_path: Path, fake_op: FakeOp, capsys) -> None:
    vault = _vault(tmp_path)
    add_source(tmp_path, "ERROR", "op://V/Missing/credential")
    _configure(tmp_path, fake_op)

    with pytest.raises(SystemExit) as exc_info:
        _run(_parse("vault", "sync", "--yes", "--json"), vault)

    assert exc_info.value.code == 3
    payload = json.loads(capsys.readouterr().out)
    assert payload["items"][0]["status"] == "error"
