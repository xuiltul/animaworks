"""Unit tests for scripts/migrate_chatwork_identity.py (temp dirs only)."""

from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from core.config.vault import VaultManager
from scripts.migrate_chatwork_identity import (
    _DEFAULT_GRANTS,
    _VAULT_KEY_COPIES,
    main,
    run_migration,
)


def _fp(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()[:16]


def _seed_owner_vault(data_dir: Path) -> VaultManager:
    vault = VaultManager(data_dir)
    vault.store("shared", "CHATWORK_API_TOKEN", "legacy-owner-token")
    return vault


def test_default_dry_run_owner_copy_only_no_grants(tmp_path: Path) -> None:
    """Defaults: owner copy only, empty grants, no token-file registration."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    vault = _seed_owner_vault(data_dir)
    vault.store("shared", "CHATWORK_API_TOKEN_EXTRA", "extra-token")
    config_path = data_dir / "config.json"
    config_path.write_text("{}", encoding="utf-8")
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    old_db = cache_dir / "messages.db"
    old_db.write_text("legacy-db", encoding="utf-8")
    # File present but no --register-token-file → must not be registered
    token_file = data_dir / "credentials" / "bot-token"
    token_file.parent.mkdir(parents=True)
    token_file.write_text("  bot-secret  \n", encoding="utf-8")

    before_owner = vault.get("shared", "CHATWORK_API_TOKEN")
    before_config = config_path.read_text(encoding="utf-8")
    before_db = old_db.read_text(encoding="utf-8")

    rc = run_migration(
        data_dir=data_dir,
        cache_dir=cache_dir,
        apply=False,
        finalize=False,
        vault=vault,
    )
    assert rc == 0

    assert _VAULT_KEY_COPIES == (
        ("CHATWORK_API_TOKEN", "CHATWORK_API_TOKEN__owner"),
    )
    assert _DEFAULT_GRANTS == {}
    assert vault.get("shared", "CHATWORK_API_TOKEN__owner") is None
    assert vault.get("shared", "CHATWORK_API_TOKEN__bot") is None
    assert vault.get("shared", "CHATWORK_API_TOKEN") == before_owner
    assert config_path.read_text(encoding="utf-8") == before_config
    assert old_db.read_text(encoding="utf-8") == before_db
    assert not (cache_dir / "identity_map.json").exists()


def test_apply_default_owner_copy_idempotent(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    vault = _seed_owner_vault(data_dir)
    config_path = data_dir / "config.json"
    config_path.write_text("{}", encoding="utf-8")
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    (cache_dir / "messages.db").write_text("legacy-db", encoding="utf-8")

    mock_client = MagicMock()
    mock_client.me.return_value = {"account_id": 4242}

    with patch(
        "core.tools._chatwork_client.ChatworkClient",
        return_value=mock_client,
    ):
        rc = run_migration(
            data_dir=data_dir,
            cache_dir=cache_dir,
            apply=True,
            finalize=False,
            vault=vault,
        )
    assert rc == 0

    assert vault.get("shared", "CHATWORK_API_TOKEN__owner") == "legacy-owner-token"
    # Legacy key remains until finalize
    assert vault.get("shared", "CHATWORK_API_TOKEN") == "legacy-owner-token"
    # Empty default grants → config untouched
    assert config_path.read_text(encoding="utf-8") == "{}"

    dest = cache_dir / "4242" / "messages.db"
    assert dest.is_file()
    assert dest.read_text(encoding="utf-8") == "legacy-db"
    assert not (cache_dir / "messages.db").exists()
    identity_map = json.loads(
        (cache_dir / "identity_map.json").read_text(encoding="utf-8")
    )
    assert identity_map[_fp("legacy-owner-token")] == "4242"

    vault_snapshot = {
        k: vault.get("shared", k)
        for k in ("CHATWORK_API_TOKEN", "CHATWORK_API_TOKEN__owner")
    }
    config_snapshot = config_path.read_text(encoding="utf-8")
    map_snapshot = (cache_dir / "identity_map.json").read_text(encoding="utf-8")
    db_snapshot = dest.read_text(encoding="utf-8")

    rc2 = run_migration(
        data_dir=data_dir,
        cache_dir=cache_dir,
        apply=True,
        finalize=False,
        vault=vault,
    )
    assert rc2 == 0
    for k, v in vault_snapshot.items():
        assert vault.get("shared", k) == v
    assert config_path.read_text(encoding="utf-8") == config_snapshot
    assert (cache_dir / "identity_map.json").read_text(encoding="utf-8") == map_snapshot
    assert dest.read_text(encoding="utf-8") == db_snapshot


def test_copy_key_extra_mapping(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    vault = _seed_owner_vault(data_dir)
    vault.store("shared", "CHATWORK_API_TOKEN_WRITE", "legacy-alice-token")
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    copies = (
        ("CHATWORK_API_TOKEN", "CHATWORK_API_TOKEN__owner"),
        ("CHATWORK_API_TOKEN_WRITE", "CHATWORK_API_TOKEN__alice"),
    )
    rc = run_migration(
        data_dir=data_dir,
        cache_dir=cache_dir,
        apply=True,
        finalize=False,
        vault=vault,
        copies=copies,
    )
    assert rc == 0
    assert vault.get("shared", "CHATWORK_API_TOKEN__owner") == "legacy-owner-token"
    assert vault.get("shared", "CHATWORK_API_TOKEN__alice") == "legacy-alice-token"
    # Existing dest not overwritten
    vault.store("shared", "CHATWORK_API_TOKEN__owner", "already-owner")
    # Re-seed source for second copy attempt
    vault.store("shared", "CHATWORK_API_TOKEN", "legacy-owner-token")
    rc2 = run_migration(
        data_dir=data_dir,
        cache_dir=cache_dir,
        apply=True,
        finalize=False,
        vault=vault,
        copies=copies,
    )
    assert rc2 == 0
    assert vault.get("shared", "CHATWORK_API_TOKEN__owner") == "already-owner"
    assert vault.get("shared", "CHATWORK_API_TOKEN__alice") == "legacy-alice-token"


def test_grant_writes_config(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    vault = VaultManager(data_dir)
    config_path = data_dir / "config.json"
    config_path.write_text("{}", encoding="utf-8")
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    grants = {"bob": {"owner": "read"}, "alice": {"bot": "readwrite"}}
    rc = run_migration(
        data_dir=data_dir,
        cache_dir=cache_dir,
        apply=True,
        finalize=False,
        vault=vault,
        grants=grants,
    )
    assert rc == 0
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    assert cfg["chatwork_tool"]["grants"] == grants

    # Existing non-empty grants are not overwritten
    custom = {"sakura": {"owner": "readwrite"}}
    config_path.write_text(
        json.dumps({"chatwork_tool": {"grants": custom}}, indent=2) + "\n",
        encoding="utf-8",
    )
    rc2 = run_migration(
        data_dir=data_dir,
        cache_dir=cache_dir,
        apply=True,
        finalize=False,
        vault=vault,
        grants=grants,
    )
    assert rc2 == 0
    cfg2 = json.loads(config_path.read_text(encoding="utf-8"))
    assert cfg2["chatwork_tool"]["grants"] == custom


def test_register_token_file(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    vault = VaultManager(data_dir)
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    token_file = data_dir / "credentials" / "bot-token"
    token_file.parent.mkdir(parents=True)
    token_file.write_text("  bot-secret-value  \n", encoding="utf-8")

    rc = run_migration(
        data_dir=data_dir,
        cache_dir=cache_dir,
        apply=True,
        finalize=False,
        vault=vault,
        token_files=(("bot", "credentials/bot-token"),),
    )
    assert rc == 0
    assert vault.get("shared", "CHATWORK_API_TOKEN__bot") == "bot-secret-value"

    # Destination not overwritten
    vault.store("shared", "CHATWORK_API_TOKEN__bot", "already-bot")
    rc2 = run_migration(
        data_dir=data_dir,
        cache_dir=cache_dir,
        apply=True,
        finalize=False,
        vault=vault,
        token_files=(("bot", "credentials/bot-token"),),
    )
    assert rc2 == 0
    assert vault.get("shared", "CHATWORK_API_TOKEN__bot") == "already-bot"


def test_finalize_deletes_only_copy_source_keys(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    vault = VaultManager(data_dir)
    vault.store("shared", "CHATWORK_API_TOKEN", "legacy-owner-token")
    vault.store("shared", "CHATWORK_API_TOKEN_WRITE", "legacy-alice-token")
    vault.store("shared", "CHATWORK_API_TOKEN__owner", "legacy-owner-token")
    vault.store("shared", "CHATWORK_API_TOKEN__alice", "legacy-alice-token")
    # Unrelated key must survive finalize
    vault.store("shared", "OTHER_SECRET", "keep-me")
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    copies = (
        ("CHATWORK_API_TOKEN", "CHATWORK_API_TOKEN__owner"),
        ("CHATWORK_API_TOKEN_WRITE", "CHATWORK_API_TOKEN__alice"),
    )
    rc = run_migration(
        data_dir=data_dir,
        cache_dir=cache_dir,
        apply=True,
        finalize=True,
        vault=vault,
        copies=copies,
    )
    assert rc == 0
    assert vault.get("shared", "CHATWORK_API_TOKEN") is None
    assert vault.get("shared", "CHATWORK_API_TOKEN_WRITE") is None
    assert vault.get("shared", "CHATWORK_API_TOKEN__owner") == "legacy-owner-token"
    assert vault.get("shared", "CHATWORK_API_TOKEN__alice") == "legacy-alice-token"
    assert vault.get("shared", "OTHER_SECRET") == "keep-me"


def test_finalize_default_copies_only_owner_source(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    vault = _seed_owner_vault(data_dir)
    vault.store("shared", "CHATWORK_API_TOKEN__owner", "legacy-owner-token")
    vault.store("shared", "CHATWORK_API_TOKEN_WRITE", "should-remain")
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    rc = run_migration(
        data_dir=data_dir,
        cache_dir=cache_dir,
        apply=True,
        finalize=True,
        vault=vault,
    )
    assert rc == 0
    assert vault.get("shared", "CHATWORK_API_TOKEN") is None
    assert vault.get("shared", "CHATWORK_API_TOKEN_WRITE") == "should-remain"
    assert vault.get("shared", "CHATWORK_API_TOKEN__owner") == "legacy-owner-token"


def test_cache_migrate_skips_when_api_fails(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    vault = VaultManager(data_dir)
    vault.store("shared", "CHATWORK_API_TOKEN__owner", "owner-token")
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    old_db = cache_dir / "messages.db"
    old_db.write_text("legacy-db", encoding="utf-8")

    mock_client = MagicMock()
    mock_client.me.side_effect = RuntimeError("network down")

    with patch(
        "core.tools._chatwork_client.ChatworkClient",
        return_value=mock_client,
    ):
        rc = run_migration(
            data_dir=data_dir,
            cache_dir=cache_dir,
            apply=True,
            finalize=False,
            vault=vault,
        )
    assert rc == 0
    assert old_db.is_file()
    assert not (cache_dir / "identity_map.json").exists()


def test_cli_dry_run_default_exits_zero(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    rc = main(
        [
            "--data-dir",
            str(data_dir),
            "--cache-dir",
            str(cache_dir),
        ]
    )
    assert rc == 0


def test_cli_copy_key_grant_register_token_file(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    vault = VaultManager(data_dir)
    vault.store("shared", "CHATWORK_API_TOKEN", "legacy-owner-token")
    vault.store("shared", "CHATWORK_API_TOKEN_WRITE", "legacy-alice-token")
    token_file = data_dir / "credentials" / "bot-token"
    token_file.parent.mkdir(parents=True)
    token_file.write_text("bot-from-file\n", encoding="utf-8")
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    # main() builds its own VaultManager when --data-dir is set; seed via path
    # so the CLI path exercises argparse parsing end-to-end.
    rc = main(
        [
            "--data-dir",
            str(data_dir),
            "--cache-dir",
            str(cache_dir),
            "--apply",
            "--copy-key",
            "CHATWORK_API_TOKEN_WRITE=CHATWORK_API_TOKEN__alice",
            "--grant",
            "bob=owner:read",
            "--grant",
            "alice=bot:readwrite",
            "--register-token-file",
            "bot=credentials/bot-token",
        ]
    )
    assert rc == 0

    vault2 = VaultManager(data_dir)
    assert vault2.get("shared", "CHATWORK_API_TOKEN__owner") == "legacy-owner-token"
    assert vault2.get("shared", "CHATWORK_API_TOKEN__alice") == "legacy-alice-token"
    assert vault2.get("shared", "CHATWORK_API_TOKEN__bot") == "bot-from-file"
    cfg = json.loads((data_dir / "config.json").read_text(encoding="utf-8"))
    assert cfg["chatwork_tool"]["grants"] == {
        "bob": {"owner": "read"},
        "alice": {"bot": "readwrite"},
    }
