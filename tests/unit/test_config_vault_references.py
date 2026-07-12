"""Tests for recursive config vault references and credential migration."""

from __future__ import annotations

import json
import stat
from pathlib import Path

import pytest

from core.config.io import invalidate_cache, load_config, save_config
from core.config.vault import VaultError, VaultManager
from scripts.migrate_credentials_to_vault import main


def _write_config(data_dir: Path, credentials: dict) -> Path:
    path = data_dir / "config.json"
    path.write_text(json.dumps({"credentials": credentials}), encoding="utf-8")
    return path


def test_load_config_resolves_nested_vault_reference(tmp_path: Path) -> None:
    vault = VaultManager(tmp_path)
    vault.save_vault({"shared": {"AZURE_API_KEY": "dummy-azure-value"}})
    config_path = _write_config(
        tmp_path,
        {
            "azure": {
                "api_key": {"$vault": "AZURE_API_KEY"},
                "keys": {"nested": {"$vault": "AZURE_API_KEY"}},
                "base_url": "https://example.invalid",
            },
            "plaintext": {"api_key": "existing-plaintext"},
        },
    )

    invalidate_cache()
    config = load_config(config_path)
    assert config.credentials["azure"].api_key == "dummy-azure-value"
    assert config.credentials["azure"].keys["nested"] == "dummy-azure-value"
    assert config.credentials["plaintext"].api_key == "existing-plaintext"


def test_missing_vault_key_is_config_error(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path, {"azure": {"api_key": {"$vault": "MISSING"}}})
    invalidate_cache()
    with pytest.raises(VaultError, match="MISSING"):
        load_config(config_path)


def test_empty_vault_value_is_distinct_from_missing_key(tmp_path: Path) -> None:
    VaultManager(tmp_path).save_vault({"shared": {"EMPTY_KEY": ""}})
    config_path = _write_config(tmp_path, {"azure": {"api_key": {"$vault": "EMPTY_KEY"}}})
    invalidate_cache()
    assert load_config(config_path).credentials["azure"].api_key == ""


@pytest.mark.parametrize("key", ["", 123, None])
def test_invalid_vault_reference_is_config_error(tmp_path: Path, key: object) -> None:
    config_path = _write_config(tmp_path, {"azure": {"api_key": {"$vault": key}}})
    invalidate_cache()
    with pytest.raises(VaultError, match="non-empty string"):
        load_config(config_path)


def test_save_config_preserves_reference_on_disk(tmp_path: Path) -> None:
    VaultManager(tmp_path).save_vault({"shared": {"AZURE_API_KEY": "dummy-azure-value"}})
    config_path = _write_config(tmp_path, {"azure": {"api_key": {"$vault": "AZURE_API_KEY"}}})
    invalidate_cache()
    config = load_config(config_path)
    config.server.session_ttl_days = 91
    save_config(config, config_path)
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    assert raw["credentials"]["azure"]["api_key"] == {"$vault": "AZURE_API_KEY"}


def test_migration_dry_run_never_changes_files_or_prints_values(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config_path = _write_config(
        tmp_path,
        {
            "azure": {"api_key": "dummy-azure-value"},
            "bedrock": {
                "keys": {
                    "aws_access_key_id": "dummy-aws-id",
                    "aws_secret_access_key": "dummy-aws-secret",
                    "aws_region_name": "dummy-region",
                }
            },
        },
    )
    before = config_path.read_bytes()
    assert main(["--data-dir", str(tmp_path)]) == 0
    output = capsys.readouterr().out
    assert "DRY-RUN: 3 credential value(s)" in output
    assert "AZURE_API_KEY" in output
    assert "AWS_ACCESS_KEY_ID" in output
    assert "dummy-" not in output
    assert config_path.read_bytes() == before
    assert not (tmp_path / "vault.json").exists()


def test_migration_apply_creates_backups_references_and_secure_files(tmp_path: Path) -> None:
    config_path = _write_config(
        tmp_path,
        {
            "azure": {"api_key": "dummy-azure-value"},
            "bedrock": {"keys": {"aws_secret_access_key": "dummy-aws-secret"}},
        },
    )
    original_config = config_path.read_text(encoding="utf-8")
    vault = VaultManager(tmp_path)
    vault.generate_key()
    vault.save_vault({"shared": {"EXISTING_KEY": "existing-value"}})
    original_vault = vault.vault_path.read_text(encoding="utf-8")

    assert main(["--data-dir", str(tmp_path), "--apply"]) == 0
    config = json.loads(config_path.read_text(encoding="utf-8"))
    assert config["credentials"]["azure"]["api_key"] == {"$vault": "AZURE_API_KEY"}
    assert config["credentials"]["bedrock"]["keys"]["aws_secret_access_key"] == {"$vault": "AWS_SECRET_ACCESS_KEY"}
    assert (tmp_path / "config.json.bak").is_file()
    assert (tmp_path / "vault.json.bak").is_file()
    assert (tmp_path / "config.json.bak").read_text(encoding="utf-8") == original_config
    assert (tmp_path / "vault.json.bak").read_text(encoding="utf-8") == original_vault
    assert VaultManager(tmp_path).get("shared", "AZURE_API_KEY") == "dummy-azure-value"
    raw_vault = json.loads((tmp_path / "vault.json").read_text(encoding="utf-8"))
    assert raw_vault["shared"]["AZURE_API_KEY"] != "dummy-azure-value"
    for name in ("config.json", "vault.json", "config.json.bak", "vault.json.bak"):
        assert stat.S_IMODE((tmp_path / name).stat().st_mode) == 0o600


def test_migration_qualifies_conflicting_aws_profiles(tmp_path: Path) -> None:
    _write_config(
        tmp_path,
        {
            "bedrock": {"keys": {"aws_access_key_id": "dummy-one"}},
            "bedrock-use1": {"keys": {"aws_access_key_id": "dummy-two"}},
        },
    )
    assert main(["--data-dir", str(tmp_path), "--apply"]) == 0
    config = json.loads((tmp_path / "config.json").read_text(encoding="utf-8"))
    assert config["credentials"]["bedrock"]["keys"]["aws_access_key_id"] == {"$vault": "BEDROCK_AWS_ACCESS_KEY_ID"}
    assert config["credentials"]["bedrock-use1"]["keys"]["aws_access_key_id"] == {
        "$vault": "BEDROCK_USE1_AWS_ACCESS_KEY_ID"
    }
