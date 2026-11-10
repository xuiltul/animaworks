from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from core.config.onepassword import OnePasswordClient, OnePasswordError
from tests.unit.vault_fake_op import FakeOp


@pytest.fixture
def fake_op(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> FakeOp:
    op = FakeOp(tmp_path)
    for name, value in op.env.items():
        monkeypatch.setenv(name, value)
    return op


def _client(op: FakeOp, timeout_s: float = 5.0) -> OnePasswordClient:
    return OnePasswordClient(op.command, timeout_s)


def test_rejects_empty_command() -> None:
    with pytest.raises(OnePasswordError):
        OnePasswordClient([], 5.0)


def test_read_uses_no_newline_and_counts(fake_op: FakeOp) -> None:
    fake_op.set_value("op://V/Item/field", "secret-value")
    client = _client(fake_op)

    assert client.read("op://V/Item/field") == "secret-value"
    assert client.request_count == 1
    assert fake_op.argv_log()[-1][:3] == ["read", "--no-newline", "op://V/Item/field"]


def test_read_reports_only_first_stderr_line(fake_op: FakeOp, monkeypatch: pytest.MonkeyPatch) -> None:
    client = _client(fake_op)
    completed = subprocess.CompletedProcess(
        args=["op"],
        returncode=1,
        stdout="",
        stderr="permission denied\nsecret-value on another line\n",
    )
    monkeypatch.setattr(client, "_run", lambda argv, **kwargs: completed)

    with pytest.raises(OnePasswordError) as exc_info:
        client.read("op://V/Item/field")

    assert "permission denied" in str(exc_info.value)
    assert "secret-value" not in str(exc_info.value)


def test_item_exists_uses_exit_code(fake_op: FakeOp) -> None:
    client = _client(fake_op)

    assert client.item_exists("V", "Item") is False
    fake_op.set_item("V", "Item", "item-1")
    assert client.item_exists("V", "Item") is True
    assert client.request_count == 2


def test_create_api_credential_sends_value_on_stdin(fake_op: FakeOp) -> None:
    fake_op.set_create_id("item-42")
    client = _client(fake_op, timeout_s=5.0)

    item_id = client.create_api_credential("V", "Title", "credential", "super-secret-value")

    assert item_id == "item-42"
    argv = fake_op.argv_log()[-1]
    assert argv[:2] == ["item", "create"]
    assert argv[-1] == "-"
    assert "super-secret-value" not in argv
    created = fake_op.created()
    assert created[0]["template"]["title"] == "Title"
    assert created[0]["template"]["category"] == "API_CREDENTIAL"
    assert created[0]["template"]["fields"] == [
        {"id": "credential", "type": "CONCEALED", "label": "credential", "value": "super-secret-value"}
    ]
    assert client.request_count == 1


def test_create_uses_write_command(fake_op: FakeOp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    writer_dir = tmp_path / "writer"
    writer_dir.mkdir()
    writer = FakeOp(writer_dir)
    for name, value in writer.env.items():
        monkeypatch.setenv(name, value)
    client = OnePasswordClient(fake_op.command, 5.0, write_command=writer.command)
    writer.set_create_id("item-99")

    assert client.create_api_credential("V", "Title", "credential", "value") == "item-99"
    assert writer.argv_log()[-1][:2] == ["item", "create"]
    assert fake_op.argv_log() == []


def test_create_rejects_invalid_json(fake_op: FakeOp, monkeypatch: pytest.MonkeyPatch) -> None:
    client = _client(fake_op)
    completed = subprocess.CompletedProcess(args=["op"], returncode=0, stdout="not-json", stderr="")
    monkeypatch.setattr(client, "_run", lambda argv, **kwargs: completed)

    with pytest.raises(OnePasswordError):
        client.create_api_credential("V", "Title", "credential", "value")


def test_create_rejects_missing_id(fake_op: FakeOp, monkeypatch: pytest.MonkeyPatch) -> None:
    client = _client(fake_op)
    completed = subprocess.CompletedProcess(args=["op"], returncode=0, stdout=json.dumps({}), stderr="")
    monkeypatch.setattr(client, "_run", lambda argv, **kwargs: completed)

    with pytest.raises(OnePasswordError):
        client.create_api_credential("V", "Title", "credential", "value")


def test_timeout_becomes_one_password_error(fake_op: FakeOp, monkeypatch: pytest.MonkeyPatch) -> None:
    client = _client(fake_op, timeout_s=0.1)

    def _timeout(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        raise subprocess.TimeoutExpired(argv, 0.1)

    monkeypatch.setattr(subprocess, "run", _timeout)
    with pytest.raises(OnePasswordError, match="timed out"):
        client.read("op://V/Item/field")


def test_missing_executable_raises(fake_op: FakeOp) -> None:
    client = OnePasswordClient(["definitely-not-an-op-cli"], 5.0)

    assert client.available() is False
    with pytest.raises(OnePasswordError, match="Failed to run"):
        client.read("op://V/Item/field")


def test_available_uses_path(fake_op: FakeOp) -> None:
    assert _client(fake_op).available() is True
