from __future__ import annotations

import dataclasses
import hashlib
import json
import stat
from pathlib import Path

import pytest

from core.config.onepassword import OnePasswordClient, OnePasswordError
from core.config.vault import VaultManager
from core.config.vault_ops import (
    CHECK_DIFFERS,
    CHECK_ERROR,
    CHECK_MISSING_LOCAL,
    CHECK_MISSING_REMOTE,
    CHECK_SAME,
    STATUS_CHANGED,
    STATUS_EMPTY_REMOTE,
    STATUS_ERROR,
    STATUS_NEW,
    STATUS_UNCHANGED,
    VaultOpsError,
    VaultOwnershipError,
    apply_sync,
    check,
    delete_value,
    entries_overview,
    last_check_summary,
    meta_path,
    plan_sync,
    previous_path,
    register_onepassword,
    rollback,
    set_value,
)
from core.config.vault_sources import add_source, load_sources
from tests.unit.vault_fake_op import FakeOp


@pytest.fixture
def fake_op(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> FakeOp:
    op = FakeOp(tmp_path)
    for name, value in op.env.items():
        monkeypatch.setenv(name, value)
    return op


def _vault(tmp_path: Path) -> VaultManager:
    vault = VaultManager(tmp_path)
    vault.generate_key()
    return vault


def _client(op: FakeOp) -> OnePasswordClient:
    return OnePasswordClient(op.command, 5.0)


def _sources(tmp_path: Path, **refs: str) -> None:
    for key, ref in refs.items():
        add_source(tmp_path, key, ref)


def test_plan_sync_classifies_without_values(tmp_path: Path, fake_op: FakeOp) -> None:
    vault = _vault(tmp_path)
    vault.store("shared", "CHANGED", "local-changed")
    vault.store("shared", "SAME", "same-value")
    vault.store("shared", "EMPTY", "local-empty")
    _sources(
        tmp_path,
        NEW="op://V/New/credential",
        CHANGED="op://V/Changed/credential",
        SAME="op://V/Same/credential",
        EMPTY="op://V/Empty/credential",
        ERROR="op://V/Missing/credential",
    )
    fake_op.set_value("op://V/New/credential", "new-value")
    fake_op.set_value("op://V/Changed/credential", "remote-changed")
    fake_op.set_value("op://V/Same/credential", "same-value")
    fake_op.set_value("op://V/Empty/credential", "")

    plan = plan_sync(tmp_path, _client(fake_op))

    assert {item.key: item.status for item in plan} == {
        "CHANGED": STATUS_CHANGED,
        "EMPTY": STATUS_EMPTY_REMOTE,
        "ERROR": STATUS_ERROR,
        "NEW": STATUS_NEW,
        "SAME": STATUS_UNCHANGED,
    }
    for item in plan:
        exposed = json.dumps(dataclasses.asdict(item))
        assert "new-value" not in exposed
        assert "remote-changed" not in exposed
        assert "local-changed" not in exposed
        assert "same-value" not in exposed
    assert plan_sync(tmp_path, _client(fake_op), ["NEW", "SAME"])[0].status == STATUS_NEW


def test_plan_sync_unknown_key_is_error(tmp_path: Path, fake_op: FakeOp) -> None:
    plan = plan_sync(tmp_path, _client(fake_op), ["UNREGISTERED"])

    assert len(plan) == 1
    assert plan[0].status == STATUS_ERROR
    assert plan[0].error == "not registered in vault_sources.json"


def test_apply_sync_writes_only_new_and_changed(tmp_path: Path, fake_op: FakeOp) -> None:
    vault = _vault(tmp_path)
    vault.store("shared", "CHANGED", "old-changed")
    vault.store("shared", "SAME", "same-value")
    vault.store("shared", "EMPTY", "leave-me")
    _sources(
        tmp_path,
        NEW="op://V/New/credential",
        CHANGED="op://V/Changed/credential",
        SAME="op://V/Same/credential",
        EMPTY="op://V/Empty/credential",
        ERROR="op://V/Missing/credential",
    )
    fake_op.set_value("op://V/New/credential", "new-value")
    fake_op.set_value("op://V/Changed/credential", "new-changed")
    fake_op.set_value("op://V/Same/credential", "same-value")
    fake_op.set_value("op://V/Empty/credential", "")

    plan = plan_sync(tmp_path, _client(fake_op))
    applied = apply_sync(tmp_path, plan)

    assert sorted(applied) == ["CHANGED", "NEW"]
    assert vault.get("shared", "NEW") == "new-value"
    assert vault.get("shared", "CHANGED") == "new-changed"
    assert vault.get("shared", "EMPTY") == "leave-me"
    previous = json.loads(previous_path(tmp_path).read_text(encoding="utf-8"))
    assert vault.decrypt(previous["shared"]["CHANGED"]["value"]) == "old-changed"
    assert "NEW" not in previous["shared"]
    assert stat.S_IMODE(previous_path(tmp_path).stat().st_mode) == 0o600

    meta = json.loads(meta_path(tmp_path).read_text(encoding="utf-8"))
    new_meta = meta["entries"]["shared/NEW"]
    assert new_meta["updated_by"] == "sync"
    assert new_meta["last_synced_at"]
    assert meta["entries"]["shared/SAME"]["last_synced_at"]
    assert meta["entries"]["shared/EMPTY"]["last_synced_at"]
    assert "shared/ERROR" not in meta["entries"]
    assert stat.S_IMODE(meta_path(tmp_path).stat().st_mode) == 0o600


def test_ownership_blocks_human_writes_but_allows_sync(tmp_path: Path) -> None:
    vault = _vault(tmp_path)
    add_source(tmp_path, "MANAGED", "op://V/Managed/credential")

    with pytest.raises(VaultOwnershipError):
        set_value(tmp_path, "shared", "MANAGED", "value", actor="cli")
    set_value(tmp_path, "shared", "MANAGED", "value", actor="sync")
    assert vault.get("shared", "MANAGED") == "value"

    with pytest.raises(VaultOwnershipError):
        delete_value(tmp_path, "shared", "MANAGED", actor="cli")

    set_value(tmp_path, "sakura", "MANAGED", "anima-value", actor="cli")
    assert vault.get("sakura", "MANAGED") == "anima-value"


def test_set_value_rejects_empty(tmp_path: Path) -> None:
    _vault(tmp_path)
    with pytest.raises(VaultOpsError):
        set_value(tmp_path, "shared", "KEY", "", actor="cli")


def test_rollback_restores_previous_and_rotates(tmp_path: Path) -> None:
    vault = _vault(tmp_path)
    set_value(tmp_path, "shared", "KEY", "one", actor="cli")
    set_value(tmp_path, "shared", "KEY", "two", actor="cli")

    assert rollback(tmp_path, "KEY", actor="cli") is True
    assert vault.get("shared", "KEY") == "one"

    previous = json.loads(previous_path(tmp_path).read_text(encoding="utf-8"))
    assert vault.decrypt(previous["shared"]["KEY"]["value"]) == "two"
    assert previous["shared"]["KEY"]["replaced_by"] == "cli"

    assert rollback(tmp_path, "KEY", actor="cli") is True
    assert vault.get("shared", "KEY") == "two"

    assert rollback(tmp_path, "MISSING", actor="cli") is False


def test_rollback_allows_managed_key(tmp_path: Path) -> None:
    vault = _vault(tmp_path)
    set_value(tmp_path, "shared", "MANAGED", "one", actor="cli")
    add_source(tmp_path, "MANAGED", "op://V/Managed/credential")
    set_value(tmp_path, "shared", "MANAGED", "two", actor="sync")

    assert rollback(tmp_path, "MANAGED", actor="cli") is True
    assert vault.get("shared", "MANAGED") == "one"


def test_check_reports_statuses_and_records_summary(tmp_path: Path, fake_op: FakeOp) -> None:
    vault = _vault(tmp_path)
    vault.store("shared", "SAME", "same-value")
    vault.store("shared", "DIFFERS", "local-value")
    vault.store("shared", "MISSING_REMOTE", "local-value")
    _sources(
        tmp_path,
        SAME="op://V/Same/credential",
        DIFFERS="op://V/Differs/credential",
        MISSING_REMOTE="op://V/MissingRemote/credential",
        MISSING_LOCAL="op://V/MissingLocal/credential",
        ERROR="op://V/Missing/credential",
    )
    fake_op.set_value("op://V/Same/credential", "same-value")
    fake_op.set_value("op://V/Differs/credential", "remote-value")
    fake_op.set_value("op://V/MissingRemote/credential", "")
    fake_op.set_value("op://V/MissingLocal/credential", "remote-only")

    results = check(tmp_path, _client(fake_op))

    assert results == {
        "DIFFERS": CHECK_DIFFERS,
        "ERROR": CHECK_ERROR,
        "MISSING_LOCAL": CHECK_MISSING_LOCAL,
        "MISSING_REMOTE": CHECK_MISSING_REMOTE,
        "SAME": CHECK_SAME,
    }
    meta_text = meta_path(tmp_path).read_text(encoding="utf-8")
    meta = json.loads(meta_text)
    assert meta["last_check"]["results"] == results
    assert meta["last_check"]["at"]
    assert "same-value" not in meta_text
    assert "local-value" not in meta_text
    assert hashlib.sha256(b"same-value").hexdigest() not in meta_text
    summary = last_check_summary(tmp_path)
    assert summary is not None
    assert summary["counts"][CHECK_SAME] == 1


def test_register_preview_does_not_create(tmp_path: Path, fake_op: FakeOp) -> None:
    _vault(tmp_path).store("shared", "API_TOKEN", "super-secret-value")

    result = register_onepassword(tmp_path, _client(fake_op), "API_TOKEN", vault="V", commit=False)

    assert result.committed is False
    assert result.vault == "V"
    assert result.title == "API_TOKEN"
    assert result.field == "credential"
    assert result.value_length == len("super-secret-value")
    assert fake_op.created() == []
    assert load_sources(tmp_path).entries == {}
    assert all("item create" not in " ".join(argv) for argv in fake_op.argv_log())


def test_register_stops_when_item_already_exists(tmp_path: Path, fake_op: FakeOp) -> None:
    _vault(tmp_path).store("shared", "API_TOKEN", "super-secret-value")
    fake_op.set_item("V", "API_TOKEN", "existing-id")

    with pytest.raises(VaultOpsError, match="already exists"):
        register_onepassword(tmp_path, _client(fake_op), "API_TOKEN", vault="V", commit=True)

    assert fake_op.created() == []


def test_register_commit_creates_reads_back_and_adds_mapping(tmp_path: Path, fake_op: FakeOp) -> None:
    value = "super-secret-value"
    _vault(tmp_path).store("shared", "API_TOKEN", value)
    fake_op.set_create_id("item-42")
    fake_op.set_value("op://V/item-42/credential", value)

    result = register_onepassword(tmp_path, _client(fake_op), "API_TOKEN", vault="V", commit=True)

    assert result.committed is True
    assert result.item_id == "item-42"
    assert result.ref == "op://V/item-42/credential"
    assert load_sources(tmp_path).entries["API_TOKEN"].ref == "op://V/item-42/credential"
    created = fake_op.created()
    assert len(created) == 1
    assert created[0]["template"]["fields"][0]["value"] == value
    create_argv = [argv for argv in fake_op.argv_log() if argv[:2] == ["item", "create"]]
    assert len(create_argv) == 1
    assert value not in " ".join(create_argv[0])


def test_register_commit_mismatch_does_not_add_mapping(tmp_path: Path, fake_op: FakeOp) -> None:
    _vault(tmp_path).store("shared", "API_TOKEN", "super-secret-value")
    fake_op.set_create_id("item-42")
    fake_op.set_value("op://V/item-42/credential", "different-value")

    with pytest.raises(VaultOpsError, match="read-back"):
        register_onepassword(tmp_path, _client(fake_op), "API_TOKEN", vault="V", commit=True)

    assert load_sources(tmp_path).entries == {}
    assert len(fake_op.created()) == 1


def test_register_requires_local_value(tmp_path: Path, fake_op: FakeOp) -> None:
    _vault(tmp_path)
    with pytest.raises(VaultOpsError, match="non-empty"):
        register_onepassword(tmp_path, _client(fake_op), "MISSING", vault="V")


def test_one_password_client_counts_and_hides_stderr_values(tmp_path: Path, fake_op: FakeOp) -> None:
    client = _client(fake_op)
    fake_op.set_value("op://V/Item/field", "value")

    assert client.available() is True
    assert client.read("op://V/Item/field") == "value"
    assert client.request_count == 1
    assert client.item_exists("V", "Item") is False
    assert client.request_count == 2

    with pytest.raises(OnePasswordError) as exc_info:
        client.read("op://V/Missing/field")
    assert "value" not in str(exc_info.value)


def test_direct_vault_manager_writes_record_metadata(tmp_path: Path) -> None:
    vault = _vault(tmp_path)
    vault.store("shared", "DIRECT", "value")

    meta = json.loads(meta_path(tmp_path).read_text(encoding="utf-8"))
    assert meta["entries"]["shared/DIRECT"]["updated_by"] == "unknown"
    assert meta["entries"]["shared/DIRECT"]["updated_at"]

    vault.delete("shared", "DIRECT")
    meta = json.loads(meta_path(tmp_path).read_text(encoding="utf-8"))
    assert meta["entries"]["shared/DIRECT"]["updated_by"] == "unknown"


def test_entries_overview_is_value_free(tmp_path: Path, fake_op: FakeOp) -> None:
    vault = _vault(tmp_path)
    vault.store("shared", "MANAGED", "managed-value")
    vault.store("shared", "LOCAL", "local-value")
    vault.store("sakura", "OWN", "own-value")
    add_source(tmp_path, "MANAGED", "op://V/Managed/credential")
    add_source(tmp_path, "ABSENT", "op://V/Absent/credential")
    fake_op.set_value("op://V/Managed/credential", "managed-value")
    fake_op.set_value("op://V/Absent/credential", "")
    check(tmp_path, _client(fake_op))

    rows = entries_overview(tmp_path)
    by_key = {(row.section, row.key): row for row in rows}
    assert by_key[("shared", "MANAGED")].owner == "onepassword"
    assert by_key[("shared", "LOCAL")].owner == "local"
    assert by_key[("sakura", "OWN")].owner == "local"
    assert by_key[("shared", "ABSENT")].is_set is False
    assert by_key[("shared", "MANAGED")].last_check == CHECK_SAME
    exposed = json.dumps([dataclasses.asdict(row) for row in rows])
    for value in ("managed-value", "local-value", "own-value"):
        assert value not in exposed
        assert hashlib.sha256(value.encode()).hexdigest() not in exposed


def test_register_read_back_failure_keeps_item_but_not_mapping(tmp_path: Path, fake_op: FakeOp) -> None:
    _vault(tmp_path).store("shared", "API_TOKEN", "super-secret-value")
    fake_op.set_create_id("item-42")

    with pytest.raises(VaultOpsError, match="read-back failed"):
        register_onepassword(tmp_path, _client(fake_op), "API_TOKEN", vault="V", commit=True)

    assert load_sources(tmp_path).entries == {}
    assert len(fake_op.created()) == 1


def _last_check_results(tmp_path: Path) -> dict[str, str]:
    return json.loads(meta_path(tmp_path).read_text(encoding="utf-8"))["last_check"]["results"]


def test_writes_clear_stale_check_results_and_sync_marks_same(tmp_path: Path, fake_op: FakeOp) -> None:
    vault = _vault(tmp_path)
    vault.store("shared", "MANAGED", "old-value")
    vault.store("shared", "LOCAL", "local-value")
    _sources(tmp_path, MANAGED="op://V/Managed/credential")
    fake_op.set_value("op://V/Managed/credential", "new-value")

    assert check(tmp_path, _client(fake_op)) == {"MANAGED": CHECK_DIFFERS}

    apply_sync(tmp_path, plan_sync(tmp_path, _client(fake_op)))
    assert _last_check_results(tmp_path) == {"MANAGED": CHECK_SAME}

    assert rollback(tmp_path, "MANAGED", actor="cli")
    assert "MANAGED" not in _last_check_results(tmp_path)

    check(tmp_path, _client(fake_op))
    vault.store("shared", "MANAGED", "direct-write")
    assert "MANAGED" not in _last_check_results(tmp_path)
