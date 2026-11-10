from __future__ import annotations

import json
import stat
from pathlib import Path

import pytest

from core.config.vault import SHARED_VAULT_SECTION
from core.config.vault_sources import (
    VaultSourceError,
    VaultSources,
    add_source,
    is_onepassword_managed,
    load_sources,
    remove_source,
    save_sources,
    sources_path,
    validate_ref,
)


def test_missing_file_returns_defaults(tmp_path: Path) -> None:
    sources = load_sources(tmp_path)

    assert sources.version == 1
    assert sources.onepassword.command == ["op"]
    assert sources.onepassword.write_command is None
    assert sources.onepassword.timeout_s == 30.0
    assert sources.entries == {}


def test_add_source_creates_file_with_mode_and_shape(tmp_path: Path) -> None:
    entry = add_source(tmp_path, "SLACK_BOT_TOKEN", "op://VaultName/ItemName/field")

    path = sources_path(tmp_path)
    assert path.is_file()
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload == {
        "version": 1,
        "onepassword": {"command": ["op"], "write_command": None, "timeout_s": 30},
        "entries": {
            "SLACK_BOT_TOKEN": {
                "ref": "op://VaultName/ItemName/field",
                "added_at": entry.added_at,
            }
        },
    }
    loaded = load_sources(tmp_path)
    assert loaded.entries["SLACK_BOT_TOKEN"].ref == "op://VaultName/ItemName/field"
    assert loaded.entries["SLACK_BOT_TOKEN"].added_at == entry.added_at


def test_add_source_keeps_added_at_when_ref_unchanged(tmp_path: Path) -> None:
    first = add_source(tmp_path, "KEY", "op://V/Item/field")
    second = add_source(tmp_path, "KEY", "op://V/Item/field")

    assert second.added_at == first.added_at

    third = add_source(tmp_path, "KEY", "op://Other/Item/field")
    assert third.added_at != first.added_at
    assert load_sources(tmp_path).entries["KEY"].ref == "op://Other/Item/field"


def test_remove_source(tmp_path: Path) -> None:
    add_source(tmp_path, "KEY", "op://V/Item/field")

    assert remove_source(tmp_path, "KEY") is True
    assert load_sources(tmp_path).entries == {}
    assert remove_source(tmp_path, "KEY") is False


@pytest.mark.parametrize(
    "ref",
    [
        "",
        "VaultName/ItemName/field",
        "op://",
        "op://Vault/Item",
        "op://Vault//field",
    ],
)
def test_invalid_refs_are_rejected(tmp_path: Path, ref: str) -> None:
    with pytest.raises(VaultSourceError):
        add_source(tmp_path, "KEY", ref)
    assert not sources_path(tmp_path).exists()


@pytest.mark.parametrize(
    "ref",
    ["op://Vault/Item/field", "op://Vault/Item/field/sub", "op://my-vault/item_id/concealed-field"],
)
def test_valid_refs_are_accepted(tmp_path: Path, ref: str) -> None:
    assert validate_ref(ref) == ref
    add_source(tmp_path, "KEY", ref)


def test_is_onepassword_managed_only_for_shared(tmp_path: Path) -> None:
    add_source(tmp_path, "KEY", "op://Vault/Item/field")

    assert is_onepassword_managed(tmp_path, "KEY") is True
    assert is_onepassword_managed(tmp_path, "KEY", section=SHARED_VAULT_SECTION) is True
    assert is_onepassword_managed(tmp_path, "KEY", section="sakura") is False
    assert is_onepassword_managed(tmp_path, "OTHER") is False


def test_save_sources_persists_custom_settings(tmp_path: Path) -> None:
    from core.config.vault_sources import OnePasswordSettings, VaultSourceEntry

    sources = VaultSources(
        onepassword=OnePasswordSettings(
            command=["wrapper", "op"], write_command=["wrapper", "op-write"], timeout_s=5.5
        ),
        entries={"A": VaultSourceEntry(ref="op://V/A/credential", added_at="2026-01-01T00:00:00+00:00")},
    )
    save_sources(tmp_path, sources)

    loaded = load_sources(tmp_path)
    assert loaded.onepassword.command == ["wrapper", "op"]
    assert loaded.onepassword.write_command == ["wrapper", "op-write"]
    assert loaded.onepassword.effective_write_command == ["wrapper", "op-write"]
    assert loaded.onepassword.timeout_s == 5.5
    assert loaded.entries["A"].ref == "op://V/A/credential"

    default_write = VaultSources(onepassword=OnePasswordSettings(command=["op"]))
    save_sources(tmp_path, default_write)
    assert load_sources(tmp_path).onepassword.effective_write_command == ["op"]


def test_invalid_files_raise(tmp_path: Path) -> None:
    path = sources_path(tmp_path)
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(VaultSourceError):
        load_sources(tmp_path)

    path.write_text(json.dumps({"version": 2}), encoding="utf-8")
    with pytest.raises(VaultSourceError):
        load_sources(tmp_path)

    path.write_text(json.dumps({"version": 1, "onepassword": {"command": []}}), encoding="utf-8")
    with pytest.raises(VaultSourceError):
        load_sources(tmp_path)

    path.write_text(
        json.dumps({"version": 1, "entries": {"KEY": {"ref": "op://bad", "added_at": "x"}}}),
        encoding="utf-8",
    )
    with pytest.raises(VaultSourceError):
        load_sources(tmp_path)


def test_environment_data_dir_is_not_touched(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANIMAWORKS_DATA_DIR", str(tmp_path / "runtime"))
    add_source(tmp_path, "KEY", "op://Vault/Item/field")

    assert sources_path(tmp_path).is_file()
    assert not (tmp_path / "runtime").exists()
