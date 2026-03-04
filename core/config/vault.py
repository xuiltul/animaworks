# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.

"""Credential vault with PyNaCl SealedBox encryption.

Provides transparent encryption/decryption of API keys and other secrets
stored in ``vault.json``.  When PyNaCl is not installed the vault falls
back to plaintext passthrough with a warning log, so existing tests and
environments without the dependency continue to work.

Key file: ``{data_dir}/vault.key`` (Curve25519 private key, base64, mode 0o600)
Vault file: ``{data_dir}/vault.json`` (encrypted credential store, mode 0o600)
"""

from __future__ import annotations

import base64
import json
import logging
import os
import threading
from pathlib import Path
from typing import Any

logger = logging.getLogger("animaworks.config.vault")

# ── PyNaCl import guard ──────────────────────────────────────────────────────

try:
    from nacl.public import PrivateKey, SealedBox

    _HAS_NACL = True
except ImportError:  # pragma: no cover – optional dependency
    _HAS_NACL = False
    PrivateKey = None  # type: ignore[assignment,misc]
    SealedBox = None  # type: ignore[assignment,misc]

# ── Exceptions ───────────────────────────────────────────────────────────────

from core.exceptions import ConfigError


class VaultError(ConfigError):
    """Vault-specific errors (missing key, decryption failure)."""


# ── VaultManager ─────────────────────────────────────────────────────────────


class VaultManager:
    """Manages encrypted credential storage using PyNaCl SealedBox.

    The vault stores sensitive credential values (API keys, tokens) in
    ``vault.json`` encrypted with a Curve25519 keypair kept in ``vault.key``.

    When PyNaCl is unavailable, all encrypt/decrypt methods become identity
    functions (plaintext passthrough) and emit a warning.  This ensures zero
    breakage in environments where the optional dependency is not installed.

    Thread-safety: vault.json writes are serialised via an internal
    :class:`threading.Lock`.
    """

    def __init__(self, data_dir: Path) -> None:
        self._data_dir = data_dir
        self._key_path = data_dir / "vault.key"
        self._vault_path = data_dir / "vault.json"
        self._lock = threading.Lock()
        self._private_key: Any = None  # nacl.public.PrivateKey | None

    # ── Properties ───────────────────────────────────────────────

    @property
    def is_encryption_available(self) -> bool:
        """Return True if PyNaCl is installed and usable."""
        return _HAS_NACL

    @property
    def has_key(self) -> bool:
        """Return True if a vault key file exists on disk."""
        return self._key_path.is_file()

    @property
    def key_path(self) -> Path:
        return self._key_path

    @property
    def vault_path(self) -> Path:
        return self._vault_path

    # ── Key management ───────────────────────────────────────────

    def generate_key(self) -> bool:
        """Generate a new Curve25519 private key and persist it.

        Returns:
            True if the key was generated, False if PyNaCl is missing.
        """
        if not _HAS_NACL:
            logger.warning(
                "PyNaCl is not installed; vault key generation skipped. "
                "Install with: pip install PyNaCl>=1.5.0"
            )
            return False

        sk = PrivateKey.generate()
        self._key_path.parent.mkdir(parents=True, exist_ok=True)
        self._key_path.write_text(
            base64.b64encode(bytes(sk)).decode("ascii") + "\n",
            encoding="utf-8",
        )
        os.chmod(self._key_path, 0o600)
        self._private_key = sk
        logger.info("Vault key generated at %s", self._key_path)
        return True

    def _load_key(self) -> Any:
        """Load the private key from disk (cached after first load).

        Returns:
            The PrivateKey instance, or None if not available.
        """
        if self._private_key is not None:
            return self._private_key
        if not _HAS_NACL or not self._key_path.is_file():
            return None
        try:
            raw = base64.b64decode(
                self._key_path.read_text(encoding="utf-8").strip()
            )
            self._private_key = PrivateKey(raw)
            return self._private_key
        except Exception as exc:
            logger.error("Failed to load vault key from %s: %s", self._key_path, exc)
            return None

    # ── Encrypt / Decrypt primitives ─────────────────────────────

    def encrypt(self, plaintext: str) -> str:
        """Encrypt a plaintext string with SealedBox.

        Returns:
            Base64-encoded ciphertext.  If PyNaCl is missing or no key
            exists, returns the plaintext unchanged (with a warning).
        """
        if not plaintext:
            return plaintext

        if not _HAS_NACL:
            logger.warning("PyNaCl not installed; credential stored in plaintext")
            return plaintext

        sk = self._load_key()
        if sk is None:
            logger.warning("Vault key not found; credential stored in plaintext")
            return plaintext

        box = SealedBox(sk.public_key)
        encrypted = box.encrypt(plaintext.encode("utf-8"))
        return base64.b64encode(encrypted).decode("ascii")

    def decrypt(self, ciphertext: str) -> str:
        """Decrypt a base64-encoded ciphertext.

        Returns:
            The original plaintext.  If PyNaCl is missing or no key exists,
            returns the input unchanged (plaintext passthrough).
        """
        if not ciphertext:
            return ciphertext

        if not _HAS_NACL:
            logger.warning("PyNaCl not installed; returning value as-is")
            return ciphertext

        sk = self._load_key()
        if sk is None:
            logger.warning("Vault key not found; returning value as-is")
            return ciphertext

        try:
            raw = base64.b64decode(ciphertext)
            return box_decrypt(sk, raw)
        except Exception:
            # Might be a plaintext value from before encryption was enabled
            logger.debug("Decryption failed; returning value as-is (likely plaintext)")
            return ciphertext

    # ── Config credentials encrypt / decrypt ─────────────────────

    def encrypt_config_credentials(
        self,
        credentials: dict[str, Any],
    ) -> dict[str, Any]:
        """Encrypt all sensitive fields in a config credentials dict.

        Accepts either raw dicts or CredentialConfig instances.  Non-sensitive
        fields (``type``, ``base_url``) are preserved in cleartext.

        Returns:
            A new dict with encrypted ``api_key`` and ``keys`` values.
        """
        result: dict[str, Any] = {}
        for name, cred in credentials.items():
            if hasattr(cred, "model_dump"):
                cred = cred.model_dump(mode="json")

            entry: dict[str, Any] = {
                "type": cred.get("type", "api_key"),
                "base_url": cred.get("base_url"),
            }
            api_key = cred.get("api_key", "")
            entry["api_key"] = self.encrypt(api_key) if api_key else ""

            encrypted_keys: dict[str, str] = {}
            for k, v in cred.get("keys", {}).items():
                encrypted_keys[k] = self.encrypt(v) if v else ""
            entry["keys"] = encrypted_keys

            result[name] = entry
        return result

    def decrypt_config_credentials(
        self,
        encrypted: dict[str, Any],
    ) -> dict[str, Any]:
        """Decrypt all sensitive fields from an encrypted credentials dict.

        Returns:
            A new dict with plaintext ``api_key`` and ``keys`` values,
            suitable for constructing ``CredentialConfig`` instances.
        """
        result: dict[str, Any] = {}
        for name, entry in encrypted.items():
            api_key = entry.get("api_key", "")
            decrypted_key = self.decrypt(api_key) if api_key else ""

            decrypted_keys: dict[str, str] = {}
            for k, v in entry.get("keys", {}).items():
                decrypted_keys[k] = self.decrypt(v) if v else ""

            result[name] = {
                "type": entry.get("type", "api_key"),
                "api_key": decrypted_key,
                "keys": decrypted_keys,
                "base_url": entry.get("base_url"),
            }
        return result

    # ── vault.json CRUD ──────────────────────────────────────────

    def load_vault(self) -> dict[str, Any]:
        """Load vault.json from disk.

        Returns:
            Parsed JSON dict, or empty dict if the file does not exist.
        """
        if not self._vault_path.is_file():
            return {}
        try:
            return json.loads(self._vault_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            logger.error("Failed to load vault from %s: %s", self._vault_path, exc)
            return {}

    def save_vault(self, data: dict[str, Any]) -> None:
        """Persist vault data to disk with atomic write (mode 0o600)."""
        with self._lock:
            self._vault_path.parent.mkdir(parents=True, exist_ok=True)
            text = json.dumps(data, indent=2, ensure_ascii=False) + "\n"
            tmp = self._vault_path.with_suffix(".tmp")
            tmp.write_text(text, encoding="utf-8")
            os.chmod(tmp, 0o600)
            tmp.rename(self._vault_path)

    def store(self, section: str, key: str, value: str) -> None:
        """Store a single encrypted value in vault.json."""
        with self._lock:
            data = self.load_vault()
            if section not in data:
                data[section] = {}
            data[section][key] = self.encrypt(value)
            self._save_vault_unlocked(data)

    def get(self, section: str, key: str) -> str | None:
        """Retrieve and decrypt a single value from vault.json.

        Returns:
            Decrypted plaintext, or None if not found.
        """
        data = self.load_vault()
        section_data = data.get(section, {})
        encrypted = section_data.get(key)
        if encrypted is None:
            return None
        return self.decrypt(encrypted)

    def delete(self, section: str, key: str) -> bool:
        """Remove a key from vault.json.

        Returns:
            True if the key was found and removed, False otherwise.
        """
        with self._lock:
            data = self.load_vault()
            section_data = data.get(section, {})
            if key not in section_data:
                return False
            del section_data[key]
            if not section_data:
                del data[section]
            self._save_vault_unlocked(data)
            return True

    def _save_vault_unlocked(self, data: dict[str, Any]) -> None:
        """Internal save without acquiring lock (caller must hold it)."""
        self._vault_path.parent.mkdir(parents=True, exist_ok=True)
        text = json.dumps(data, indent=2, ensure_ascii=False) + "\n"
        tmp = self._vault_path.with_suffix(".tmp")
        tmp.write_text(text, encoding="utf-8")
        os.chmod(tmp, 0o600)
        tmp.rename(self._vault_path)

    # ── Migration ────────────────────────────────────────────────

    def migrate_shared_credentials(self) -> int:
        """Migrate ``shared/credentials.json`` into vault.json's ``shared`` section.

        After successful migration the original file is renamed to
        ``credentials.json.bak``.

        Returns:
            Number of entries migrated (0 if nothing to migrate).
        """
        shared_path = self._data_dir / "shared" / "credentials.json"
        if not shared_path.is_file():
            logger.debug("No shared/credentials.json found; nothing to migrate")
            return 0

        try:
            creds = json.loads(shared_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning(
                "Failed to read shared/credentials.json for migration: %s", exc
            )
            return 0

        if not isinstance(creds, dict) or not creds:
            return 0

        with self._lock:
            data = self.load_vault()
            if "shared" not in data:
                data["shared"] = {}

            count = 0
            for key, value in creds.items():
                if isinstance(value, str) and value:
                    data["shared"][key] = self.encrypt(value)
                    count += 1

            if count > 0:
                self._save_vault_unlocked(data)

        # Rename original to .bak
        bak_path = shared_path.with_suffix(".json.bak")
        try:
            shared_path.rename(bak_path)
            logger.info(
                "Migrated %d entries from shared/credentials.json → vault.json "
                "(backup: %s)",
                count,
                bak_path,
            )
        except OSError as exc:
            logger.warning("Failed to rename credentials.json to .bak: %s", exc)

        return count


# ── Module-level helper ──────────────────────────────────────────────────────


def box_decrypt(sk: Any, raw: bytes) -> str:
    """Decrypt raw bytes using a SealedBox with private key *sk*."""
    box = SealedBox(sk)
    return box.decrypt(raw).decode("utf-8")


# ── Singleton ────────────────────────────────────────────────────────────────

_vault_manager: VaultManager | None = None


def get_vault_manager(data_dir: Path | None = None) -> VaultManager:
    """Return the module-level VaultManager singleton.

    Lazily creates the instance on first call.  If *data_dir* is not
    provided, it is resolved via ``core.paths.get_data_dir()``.
    """
    global _vault_manager
    if _vault_manager is not None:
        return _vault_manager
    if data_dir is None:
        from core.paths import get_data_dir

        data_dir = get_data_dir()
    _vault_manager = VaultManager(data_dir)
    return _vault_manager


def invalidate_vault_cache() -> None:
    """Reset the module-level VaultManager singleton."""
    global _vault_manager
    _vault_manager = None
