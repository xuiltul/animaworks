# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for VaultManager credential encryption.

Covers:
  - Key generation and persistence
  - Encrypt / decrypt round-trip (single values)
  - Config credentials dict encrypt / decrypt
  - vault.json CRUD (store / get / delete / load / save)
  - shared/credentials.json migration
  - PyNaCl not-installed fallback (plaintext passthrough)
"""

from __future__ import annotations

import base64
import json
import os
import stat
from pathlib import Path
from unittest.mock import patch

import pytest


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture()
def data_dir(tmp_path: Path) -> Path:
    """Create a temporary data directory for vault tests."""
    d = tmp_path / ".animaworks"
    d.mkdir()
    return d


@pytest.fixture()
def vault(data_dir: Path):
    """Create a VaultManager with a freshly generated key."""
    from core.config.vault import VaultManager

    vm = VaultManager(data_dir)
    vm.generate_key()
    return vm


@pytest.fixture()
def vault_no_key(data_dir: Path):
    """Create a VaultManager without generating a key."""
    from core.config.vault import VaultManager

    return VaultManager(data_dir)


# ── Key generation ───────────────────────────────────────────────────────────


class TestKeyGeneration:
    def test_generate_creates_key_file(self, data_dir: Path):
        from core.config.vault import VaultManager

        vm = VaultManager(data_dir)
        assert not vm.has_key
        result = vm.generate_key()
        assert result is True
        assert vm.has_key
        assert vm.key_path.is_file()

    def test_key_file_permissions(self, vault):
        mode = stat.S_IMODE(vault.key_path.stat().st_mode)
        assert mode == 0o600

    def test_key_file_is_valid_base64(self, vault):
        raw = vault.key_path.read_text(encoding="utf-8").strip()
        decoded = base64.b64decode(raw)
        assert len(decoded) == 32  # Curve25519 private key = 32 bytes

    def test_generate_key_idempotent(self, vault, data_dir: Path):
        key1 = vault.key_path.read_text(encoding="utf-8").strip()
        from core.config.vault import VaultManager

        vm2 = VaultManager(data_dir)
        vm2.generate_key()
        key2 = vm2.key_path.read_text(encoding="utf-8").strip()
        # Second generate overwrites — keys differ
        assert key1 != key2


# ── Encrypt / Decrypt round-trip ─────────────────────────────────────────────


class TestEncryptDecrypt:
    def test_round_trip_basic(self, vault):
        plaintext = "sk-ant-api03-secret-key-12345"
        encrypted = vault.encrypt(plaintext)
        assert encrypted != plaintext
        decrypted = vault.decrypt(encrypted)
        assert decrypted == plaintext

    def test_round_trip_unicode(self, vault):
        plaintext = "日本語のAPIキー_テスト_🔑"
        encrypted = vault.encrypt(plaintext)
        decrypted = vault.decrypt(encrypted)
        assert decrypted == plaintext

    def test_round_trip_empty_string(self, vault):
        assert vault.encrypt("") == ""
        assert vault.decrypt("") == ""

    def test_encrypted_is_base64(self, vault):
        encrypted = vault.encrypt("test-value")
        decoded = base64.b64decode(encrypted)
        assert len(decoded) > 0

    def test_different_encryptions_differ(self, vault):
        """SealedBox uses ephemeral keys, so encrypting the same value twice
        produces different ciphertexts."""
        enc1 = vault.encrypt("same-value")
        enc2 = vault.encrypt("same-value")
        assert enc1 != enc2
        # Both decrypt to the same value
        assert vault.decrypt(enc1) == "same-value"
        assert vault.decrypt(enc2) == "same-value"

    def test_decrypt_plaintext_fallback(self, vault):
        """If decryption fails (e.g. value was never encrypted), return as-is."""
        result = vault.decrypt("not-actually-encrypted")
        assert result == "not-actually-encrypted"


# ── Config credentials encrypt / decrypt ─────────────────────────────────────


class TestConfigCredentials:
    def test_encrypt_decrypt_round_trip(self, vault):
        credentials = {
            "anthropic": {
                "type": "api_key",
                "api_key": "sk-ant-xxx",
                "keys": {},
                "base_url": None,
            },
            "chatwork": {
                "type": "api_token",
                "api_key": "cwt-yyy",
                "keys": {"room_id": "12345"},
                "base_url": "https://api.chatwork.com",
            },
        }
        encrypted = vault.encrypt_config_credentials(credentials)

        # api_key should be encrypted (different from original)
        assert encrypted["anthropic"]["api_key"] != "sk-ant-xxx"
        assert encrypted["chatwork"]["api_key"] != "cwt-yyy"
        assert encrypted["chatwork"]["keys"]["room_id"] != "12345"

        # Non-sensitive fields should be preserved
        assert encrypted["anthropic"]["type"] == "api_key"
        assert encrypted["chatwork"]["base_url"] == "https://api.chatwork.com"

        # Decrypt should restore original
        decrypted = vault.decrypt_config_credentials(encrypted)
        assert decrypted["anthropic"]["api_key"] == "sk-ant-xxx"
        assert decrypted["chatwork"]["api_key"] == "cwt-yyy"
        assert decrypted["chatwork"]["keys"]["room_id"] == "12345"

    def test_encrypt_with_pydantic_model(self, vault):
        from core.config.models import CredentialConfig

        credentials = {
            "anthropic": CredentialConfig(
                api_key="sk-ant-xxx",
                keys={"extra": "val"},
            ),
        }
        encrypted = vault.encrypt_config_credentials(credentials)
        assert encrypted["anthropic"]["api_key"] != "sk-ant-xxx"
        assert encrypted["anthropic"]["keys"]["extra"] != "val"

        decrypted = vault.decrypt_config_credentials(encrypted)
        assert decrypted["anthropic"]["api_key"] == "sk-ant-xxx"
        assert decrypted["anthropic"]["keys"]["extra"] == "val"

    def test_empty_credentials(self, vault):
        encrypted = vault.encrypt_config_credentials({})
        assert encrypted == {}
        decrypted = vault.decrypt_config_credentials({})
        assert decrypted == {}

    def test_empty_api_key_preserved(self, vault):
        credentials = {
            "empty": {
                "type": "api_key",
                "api_key": "",
                "keys": {},
                "base_url": None,
            },
        }
        encrypted = vault.encrypt_config_credentials(credentials)
        assert encrypted["empty"]["api_key"] == ""
        decrypted = vault.decrypt_config_credentials(encrypted)
        assert decrypted["empty"]["api_key"] == ""


# ── vault.json CRUD ──────────────────────────────────────────────────────────


class TestVaultCRUD:
    def test_store_and_get(self, vault):
        vault.store("shared", "MY_TOKEN", "secret-123")
        result = vault.get("shared", "MY_TOKEN")
        assert result == "secret-123"

    def test_get_nonexistent_section(self, vault):
        assert vault.get("nonexistent", "key") is None

    def test_get_nonexistent_key(self, vault):
        vault.store("section", "key1", "val1")
        assert vault.get("section", "key2") is None

    def test_delete_existing(self, vault):
        vault.store("section", "key", "value")
        assert vault.delete("section", "key") is True
        assert vault.get("section", "key") is None

    def test_delete_nonexistent(self, vault):
        assert vault.delete("section", "key") is False

    def test_delete_removes_empty_section(self, vault):
        vault.store("section", "key", "value")
        vault.delete("section", "key")
        data = vault.load_vault()
        assert "section" not in data

    def test_save_and_load_vault(self, vault):
        test_data = {"credentials": {"test": "encrypted_value"}, "version": 1}
        vault.save_vault(test_data)
        loaded = vault.load_vault()
        assert loaded == test_data

    def test_vault_file_permissions(self, vault):
        vault.save_vault({"test": "data"})
        mode = stat.S_IMODE(vault.vault_path.stat().st_mode)
        assert mode == 0o600

    def test_load_empty_vault(self, vault_no_key):
        assert vault_no_key.load_vault() == {}

    def test_multiple_sections(self, vault):
        vault.store("shared", "KEY_A", "val-a")
        vault.store("credentials", "KEY_B", "val-b")
        assert vault.get("shared", "KEY_A") == "val-a"
        assert vault.get("credentials", "KEY_B") == "val-b"


# ── shared/credentials.json migration ───────────────────────────────────────


class TestMigration:
    def test_migrate_shared_credentials(self, vault, data_dir: Path):
        shared_dir = data_dir / "shared"
        shared_dir.mkdir(parents=True, exist_ok=True)
        creds = {
            "CHATWORK_API_TOKEN": "cwt-secret",
            "SLACK_BOT_TOKEN": "xoxb-secret",
        }
        (shared_dir / "credentials.json").write_text(
            json.dumps(creds), encoding="utf-8"
        )

        count = vault.migrate_shared_credentials()

        assert count == 2
        assert vault.get("shared", "CHATWORK_API_TOKEN") == "cwt-secret"
        assert vault.get("shared", "SLACK_BOT_TOKEN") == "xoxb-secret"

    def test_migrate_creates_bak(self, vault, data_dir: Path):
        shared_dir = data_dir / "shared"
        shared_dir.mkdir(parents=True, exist_ok=True)
        cred_path = shared_dir / "credentials.json"
        cred_path.write_text(json.dumps({"KEY": "val"}), encoding="utf-8")

        vault.migrate_shared_credentials()

        assert not cred_path.is_file()
        bak_path = shared_dir / "credentials.json.bak"
        assert bak_path.is_file()

    def test_migrate_no_file(self, vault):
        count = vault.migrate_shared_credentials()
        assert count == 0

    def test_migrate_empty_dict(self, vault, data_dir: Path):
        shared_dir = data_dir / "shared"
        shared_dir.mkdir(parents=True, exist_ok=True)
        (shared_dir / "credentials.json").write_text("{}", encoding="utf-8")

        count = vault.migrate_shared_credentials()
        assert count == 0

    def test_migrate_skips_non_string_values(self, vault, data_dir: Path):
        shared_dir = data_dir / "shared"
        shared_dir.mkdir(parents=True, exist_ok=True)
        creds = {"VALID_KEY": "valid-value", "INVALID": 12345, "EMPTY": ""}
        (shared_dir / "credentials.json").write_text(
            json.dumps(creds), encoding="utf-8"
        )

        count = vault.migrate_shared_credentials()
        assert count == 1
        assert vault.get("shared", "VALID_KEY") == "valid-value"

    def test_migrate_invalid_json(self, vault, data_dir: Path):
        shared_dir = data_dir / "shared"
        shared_dir.mkdir(parents=True, exist_ok=True)
        (shared_dir / "credentials.json").write_text(
            "not valid json", encoding="utf-8"
        )

        count = vault.migrate_shared_credentials()
        assert count == 0


# ── No-key fallback ──────────────────────────────────────────────────────────


class TestNoKeyFallback:
    def test_encrypt_without_key_returns_plaintext(self, vault_no_key):
        result = vault_no_key.encrypt("my-secret")
        assert result == "my-secret"

    def test_decrypt_without_key_returns_as_is(self, vault_no_key):
        result = vault_no_key.decrypt("some-value")
        assert result == "some-value"


# ── PyNaCl not installed fallback ────────────────────────────────────────────


class TestNaClNotInstalled:
    """Test behaviour when PyNaCl is not available."""

    def test_encrypt_plaintext_passthrough(self, data_dir: Path):
        with patch("core.config.vault._HAS_NACL", False):
            from core.config.vault import VaultManager

            vm = VaultManager(data_dir)
            result = vm.encrypt("my-secret")
            assert result == "my-secret"

    def test_decrypt_plaintext_passthrough(self, data_dir: Path):
        with patch("core.config.vault._HAS_NACL", False):
            from core.config.vault import VaultManager

            vm = VaultManager(data_dir)
            result = vm.decrypt("my-secret")
            assert result == "my-secret"

    def test_generate_key_returns_false(self, data_dir: Path):
        with patch("core.config.vault._HAS_NACL", False):
            from core.config.vault import VaultManager

            vm = VaultManager(data_dir)
            result = vm.generate_key()
            assert result is False
            assert not vm.has_key

    def test_config_credentials_passthrough(self, data_dir: Path):
        with patch("core.config.vault._HAS_NACL", False):
            from core.config.vault import VaultManager

            vm = VaultManager(data_dir)
            credentials = {
                "anthropic": {
                    "type": "api_key",
                    "api_key": "sk-ant-xxx",
                    "keys": {"extra": "val"},
                    "base_url": None,
                },
            }
            encrypted = vm.encrypt_config_credentials(credentials)
            # Without PyNaCl, values pass through as plaintext
            assert encrypted["anthropic"]["api_key"] == "sk-ant-xxx"
            assert encrypted["anthropic"]["keys"]["extra"] == "val"

            decrypted = vm.decrypt_config_credentials(encrypted)
            assert decrypted["anthropic"]["api_key"] == "sk-ant-xxx"

    def test_store_and_get_plaintext(self, data_dir: Path):
        with patch("core.config.vault._HAS_NACL", False):
            from core.config.vault import VaultManager

            vm = VaultManager(data_dir)
            vm.store("shared", "KEY", "value")
            assert vm.get("shared", "KEY") == "value"


# ── Singleton ────────────────────────────────────────────────────────────────


class TestSingleton:
    def test_get_vault_manager_returns_same_instance(self, data_dir: Path):
        from core.config.vault import (
            get_vault_manager,
            invalidate_vault_cache,
        )

        invalidate_vault_cache()
        vm1 = get_vault_manager(data_dir)
        vm2 = get_vault_manager(data_dir)
        assert vm1 is vm2
        invalidate_vault_cache()

    def test_invalidate_resets_singleton(self, data_dir: Path):
        from core.config.vault import (
            get_vault_manager,
            invalidate_vault_cache,
        )

        invalidate_vault_cache()
        vm1 = get_vault_manager(data_dir)
        invalidate_vault_cache()
        vm2 = get_vault_manager(data_dir)
        assert vm1 is not vm2
        invalidate_vault_cache()
