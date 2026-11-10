from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.config import invalidate_cache
from core.config.io import load_config, save_config, update_config
from core.config.vault import VaultManager
from core.config.vault_sources import add_source
from core.exceptions import ConfigError


def _config_path(data_dir: Path, credentials: dict) -> Path:
    path = data_dir / "config.json"
    path.write_text(json.dumps({"setup_complete": True, "credentials": credentials}), encoding="utf-8")
    invalidate_cache()
    return path


def test_update_config_rejects_changing_onepassword_managed_reference(data_dir: Path) -> None:
    vault = VaultManager(data_dir)
    vault.generate_key()
    vault.store("shared", "MANAGED", "original-secret")
    add_source(data_dir, "MANAGED", "op://V/Managed/credential")
    path = _config_path(data_dir, {"svc": {"api_key": {"$vault": "MANAGED"}}})

    def change_api_key(config) -> None:
        config.credentials["svc"].api_key = "replacement-secret"

    with pytest.raises(ConfigError, match="1Password"):
        update_config(change_api_key, path=path)

    raw = json.loads(path.read_text(encoding="utf-8"))
    assert raw["credentials"]["svc"]["api_key"] == {"$vault": "MANAGED"}
    assert vault.get("shared", "MANAGED") == "original-secret"


def test_save_config_rejects_plaintext_move_to_onepassword_managed_key(data_dir: Path) -> None:
    vault = VaultManager(data_dir)
    vault.generate_key()
    add_source(data_dir, "SVC_API_KEY", "op://V/Managed/credential")
    path = _config_path(data_dir, {"svc": {"api_key": "plaintext-secret"}})
    config = load_config(path)

    with pytest.raises(ConfigError, match="1Password"):
        save_config(config, path)

    raw = json.loads(path.read_text(encoding="utf-8"))
    assert raw["credentials"]["svc"]["api_key"] == "plaintext-secret"
    assert vault.get("shared", "SVC_API_KEY") is None
