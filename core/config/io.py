# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.

"""Configuration I/O: singleton cache, load, and save."""

from __future__ import annotations

import json
import logging
import re
import threading
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.config.schemas import AnimaWorksConfig
from core.config.vault import VaultManager, resolve_vault_references
from core.exceptions import ConfigError
from core.platform.atomic_io import update_json
from core.platform.process_role import assert_settings_write_allowed

logger = logging.getLogger("animaworks.config")

_SENSITIVE_KEY_PARTS = ("key", "secret", "token", "password")
_MISSING = object()
_vault_unavailable_warned = False
_vault_unavailable_warn_lock = threading.Lock()


@dataclass(frozen=True)
class PlaintextCredential:
    """A credential leaf that will be moved to the vault on the next save."""

    path: str
    vault_key: str


# ---------------------------------------------------------------------------
# Singleton cache
# ---------------------------------------------------------------------------

_config: AnimaWorksConfig | None = None
_config_path: Path | None = None
_config_mtime: float = 0.0
_config_vault_values: dict[tuple[str | int, ...], Any] = {}


def invalidate_cache() -> None:
    """Reset the module-level singleton cache."""
    global _config, _config_path, _config_mtime, _config_vault_values
    _config = None
    _config_path = None
    _config_mtime = 0.0
    _config_vault_values = {}


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


def get_config_path(data_dir: Path | None = None) -> Path:
    """Return the path to config.json inside *data_dir*.

    If *data_dir* is not given, it is resolved via ``core.paths.get_data_dir``
    (imported lazily to avoid circular imports).
    """
    if data_dir is None:
        from core.paths import get_data_dir

        data_dir = get_data_dir()
    return data_dir / "config.json"


# ---------------------------------------------------------------------------
# Load / Save
# ---------------------------------------------------------------------------


def load_config(path: Path | None = None) -> AnimaWorksConfig:
    """Load configuration from disk, returning cached instance when possible.

    If *path* is ``None``, :func:`get_config_path` determines the location.
    When the file does not exist the default configuration is returned.

    The cache is automatically invalidated when the file's mtime changes,
    so external edits (org_sync, manual changes) are picked up without
    requiring a server restart.
    """
    global _config, _config_path, _config_mtime, _config_vault_values

    if path is None:
        path = get_config_path()

    # Check whether the on-disk file has been modified since last load.
    if _config is not None and _config_path == path:
        try:
            disk_mtime = path.stat().st_mtime
        except OSError:
            disk_mtime = 0.0
        if disk_mtime == _config_mtime:
            return _config
        logger.debug("Config file changed on disk (mtime %.3f → %.3f); reloading", _config_mtime, disk_mtime)

    if path.is_file():
        logger.debug("Loading config from %s", path)
        try:
            raw_text = path.read_text(encoding="utf-8")
            raw_data: dict[str, Any] = json.loads(raw_text)
            data = resolve_vault_references(raw_data, path.parent)
            config = AnimaWorksConfig.model_validate(data)
            _config_vault_values = _collect_vault_reference_values(raw_data, data)
        except json.JSONDecodeError as exc:
            logger.error("Failed to parse %s: %s", path, exc)
            raise ConfigError(f"Invalid JSON in {path}: {exc}") from exc
        except ConfigError:
            raise
        except Exception as exc:
            logger.error("Failed to load config from %s: %s", path, exc)
            raise ConfigError(f"Failed to load config from {path}: {exc}") from exc
    else:
        logger.info("Config file not found at %s; using defaults", path)
        config = AnimaWorksConfig()
        _config_vault_values = {}

    _config = config
    _config_path = path
    try:
        _config_mtime = path.stat().st_mtime
    except OSError:
        _config_mtime = 0.0
    return config


def save_config(config: AnimaWorksConfig, path: Path | None = None) -> None:
    """Persist *config* under an inter-process lock (mode 0o600).

    Callers that derive a change from the current config should prefer
    :func:`update_config` so their read-modify-write cycle also happens under
    the lock.
    """
    if path is None:
        path = get_config_path()
    assert_settings_write_allowed(path)

    payload = config.model_dump(mode="json")

    def replace_config(existing: dict[str, Any]) -> dict[str, Any]:
        disk_payload, vault_updates = _prepare_disk_payload(
            payload,
            existing,
            loaded_values=_config_vault_values if _config_path == path else {},
            data_dir=path.parent,
        )
        if vault_updates:
            _apply_vault_updates(path.parent, vault_updates)
        return disk_payload

    update_json(
        path,
        replace_config,
        mode=0o600,
        always_write=True,
        after_update=lambda saved: _refresh_config_cache(config, path, saved),
    )
    logger.debug("Config saved to %s", path)


def update_config(
    fn: Callable[[AnimaWorksConfig], AnimaWorksConfig | dict[str, Any] | None],
    path: Path | None = None,
) -> AnimaWorksConfig:
    """Lock, load, modify, and persist config.json as one transaction.

    The callback receives the latest validated config while the sibling lock
    is held. It may mutate that instance and return ``None``, return a
    replacement model, or return a raw JSON object for lossless migrations
    that must preserve sparse/unknown legacy fields. Vault references and the
    in-process mtime cache are kept in sync with :func:`save_config`.
    """
    if path is None:
        path = get_config_path()
    assert_settings_write_allowed(path)

    updated_config: AnimaWorksConfig | None = None

    def apply_update(existing: dict[str, Any]) -> dict[str, Any]:
        nonlocal updated_config
        raw_data: dict[str, Any] = existing or AnimaWorksConfig().model_dump(mode="json")
        resolved = resolve_vault_references(raw_data, path.parent)
        current_config = AnimaWorksConfig.model_validate(resolved)
        loaded_values = _collect_vault_reference_values(raw_data, resolved)
        result = fn(current_config)
        if result is None:
            updated_config = current_config
            payload = updated_config.model_dump(mode="json")
        elif isinstance(result, AnimaWorksConfig):
            updated_config = result
            payload = updated_config.model_dump(mode="json")
        elif isinstance(result, dict):
            payload = result
            resolved_payload = resolve_vault_references(payload, path.parent)
            updated_config = AnimaWorksConfig.model_validate(resolved_payload)
        else:
            raise TypeError("update_config callback must return None, AnimaWorksConfig, or a JSON object")
        disk_payload, vault_updates = _prepare_disk_payload(
            payload,
            raw_data,
            loaded_values=loaded_values,
            data_dir=path.parent,
        )
        if vault_updates:
            _apply_vault_updates(path.parent, vault_updates)
        if not existing and not vault_updates and payload == AnimaWorksConfig().model_dump(mode="json"):
            return existing
        return disk_payload

    def refresh_cache(saved: dict[str, Any]) -> None:
        assert updated_config is not None
        _refresh_config_cache(updated_config, path, saved)

    update_json(path, apply_update, mode=0o600, write_if_missing=False, after_update=refresh_cache)
    assert updated_config is not None
    logger.debug("Config updated transactionally at %s", path)
    return updated_config


def _refresh_config_cache(config: AnimaWorksConfig, path: Path, payload: dict[str, Any]) -> None:
    """Refresh the singleton cache after a successful disk write."""
    global _config, _config_path, _config_mtime, _config_vault_values
    _config = config
    _config_path = path
    try:
        _config_mtime = path.stat().st_mtime
    except OSError:
        _config_mtime = 0.0
    _config_vault_values = _collect_vault_reference_values(payload, config.model_dump(mode="json"))


def _collect_vault_reference_values(
    references: Any,
    resolved: Any,
    path: tuple[str | int, ...] = (),
) -> dict[tuple[str | int, ...], Any]:
    """Collect resolved values corresponding to vault reference leaves."""
    if isinstance(references, dict) and set(references) == {"$vault"}:
        return {path: resolved}
    found: dict[tuple[str | int, ...], Any] = {}
    if isinstance(references, dict) and isinstance(resolved, dict):
        for key, item in references.items():
            if key in resolved:
                found.update(_collect_vault_reference_values(item, resolved[key], (*path, key)))
    elif isinstance(references, list) and isinstance(resolved, list):
        for index, item in enumerate(references):
            if index < len(resolved):
                found.update(_collect_vault_reference_values(item, resolved[index], (*path, index)))
    return found


def _preserve_vault_references(
    value: Any,
    existing: Any,
    *,
    loaded_values: dict[tuple[str | int, ...], Any],
    path: tuple[str | int, ...] = (),
) -> tuple[Any, dict[str, str]]:
    """Preserve reference leaves and identify intentional value changes."""
    if isinstance(existing, dict) and set(existing) == {"$vault"}:
        key = existing["$vault"]
        if isinstance(key, str) and key:
            updates: dict[str, str] = {}
            if path in loaded_values and value != loaded_values[path]:
                if not isinstance(value, str):
                    raise ConfigError(f"Value for vault key {key} must be a string")
                updates[key] = value
            return existing, updates
    if isinstance(value, dict) and isinstance(existing, dict):
        result: dict[str, Any] = {}
        updates: dict[str, str] = {}
        for key, item in value.items():
            result[key], child_updates = _preserve_vault_references(
                item,
                existing.get(key),
                loaded_values=loaded_values,
                path=(*path, key),
            )
            for vault_key, vault_value in child_updates.items():
                if vault_key in updates and updates[vault_key] != vault_value:
                    raise ConfigError(f"Conflicting updates for vault key: {vault_key}")
                updates[vault_key] = vault_value
        return result, updates
    if isinstance(value, list) and isinstance(existing, list):
        result_list: list[Any] = []
        updates = {}
        for index, item in enumerate(value):
            child, child_updates = _preserve_vault_references(
                item,
                existing[index] if index < len(existing) else None,
                loaded_values=loaded_values,
                path=(*path, index),
            )
            result_list.append(child)
            for vault_key, vault_value in child_updates.items():
                if vault_key in updates and updates[vault_key] != vault_value:
                    raise ConfigError(f"Conflicting updates for vault key: {vault_key}")
                updates[vault_key] = vault_value
        return result_list, updates
    return value, {}


def _prepare_disk_payload(
    payload: dict[str, Any],
    existing: dict[str, Any],
    *,
    loaded_values: dict[tuple[str | int, ...], Any],
    data_dir: Path,
) -> tuple[dict[str, Any], dict[str, str]]:
    """Preserve existing references and move plaintext credentials into the vault."""
    disk_payload, vault_updates = _preserve_vault_references(
        payload,
        existing,
        loaded_values=loaded_values,
    )
    disk_payload, vault_updates = _reference_plaintext_credentials(disk_payload, data_dir, vault_updates)
    _guard_onepassword_config_keys(data_dir, vault_updates)
    return disk_payload, vault_updates


def _reference_plaintext_credentials(
    payload: dict[str, Any],
    data_dir: Path,
    vault_updates: dict[str, str],
) -> tuple[dict[str, Any], dict[str, str]]:
    leaves = _plaintext_credential_leaves(payload)
    if not leaves:
        return payload, vault_updates

    vault = VaultManager(data_dir)
    if not _vault_store_available(vault):
        _warn_vault_unavailable_once()
        return payload, vault_updates

    shared = _load_shared_section(vault)
    updates = dict(vault_updates)
    moves = _plan_credential_moves(leaves, shared, updates, vault)
    _guard_onepassword_config_keys(data_dir, (vault_key for _, vault_key, _ in moves))
    for path, vault_key, _value in moves:
        _set_path(payload, path, {"$vault": vault_key})
    return payload, updates


def _guard_onepassword_config_keys(data_dir: Path, keys: Iterable[str]) -> None:
    """Reject config writes that would change a key managed by 1Password."""
    from core.config.vault_sources import is_onepassword_managed
    from core.i18n import t

    for key in keys:
        if is_onepassword_managed(data_dir, key, section="shared"):
            raise ConfigError(t("vault.managed_config", name=key))


def _plan_credential_moves(
    leaves: list[tuple[tuple[str, ...], str, str, str]],
    shared: dict[str, Any],
    pending: dict[str, str],
    vault: VaultManager,
) -> list[tuple[tuple[str, ...], str, str]]:
    moves: list[tuple[tuple[str, ...], str, str]] = []
    for path, name, leaf, value in leaves:
        vault_key = _choose_vault_key(name, leaf, value, shared, pending, vault)
        current = _existing_vault_value(vault_key, shared, pending, vault)
        if current is _MISSING or current != value:
            if vault_key in pending and pending[vault_key] != value:
                raise ConfigError(f"Conflicting updates for vault key: {vault_key}")
            pending[vault_key] = value
        moves.append((path, vault_key, value))
    return moves


def _plaintext_credential_leaves(raw: dict[str, Any]) -> list[tuple[tuple[str, ...], str, str, str]]:
    """Return ``(path, credential name, leaf name, value)`` for plaintext secrets."""
    credentials = raw.get("credentials")
    if not isinstance(credentials, dict):
        return []

    leaves: list[tuple[tuple[str, ...], str, str, str]] = []
    for name, entry in credentials.items():
        if not isinstance(name, str) or not isinstance(entry, dict):
            continue
        api_key = entry.get("api_key")
        if isinstance(api_key, str) and api_key:
            leaves.append((("credentials", name, "api_key"), name, "api_key", api_key))
        keys = entry.get("keys")
        if isinstance(keys, dict):
            for key_name, value in keys.items():
                if not isinstance(key_name, str) or not _is_sensitive_key_name(key_name):
                    continue
                if isinstance(value, str) and value:
                    leaves.append((("credentials", name, "keys", key_name), name, key_name, value))
    return leaves


def _is_sensitive_key_name(name: str) -> bool:
    lowered = name.lower()
    return any(part in lowered for part in _SENSITIVE_KEY_PARTS)


def _env_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", value).upper().strip("_")


def _base_vault_key(name: str, leaf: str) -> str:
    return f"{_env_name(name)}_{_env_name(leaf)}".strip("_")


def _vault_store_available(vault: VaultManager) -> bool:
    from core.config.vault import _HAS_NACL

    return bool(_HAS_NACL) and vault.has_key


def _load_shared_section(vault: VaultManager) -> dict[str, Any]:
    shared = vault.load_vault().get("shared", {})
    if not isinstance(shared, dict):
        raise ConfigError("vault.json shared section must be an object")
    return shared


def _existing_vault_value(
    key: str,
    shared: dict[str, Any],
    pending: dict[str, str],
    vault: VaultManager,
) -> Any:
    if key in pending:
        return pending[key]
    stored = shared.get(key, _MISSING)
    if stored is _MISSING or not isinstance(stored, str):
        return _MISSING
    return vault.decrypt(stored)


def _choose_vault_key(
    name: str,
    leaf: str,
    value: str,
    shared: dict[str, Any],
    pending: dict[str, str],
    vault: VaultManager,
) -> str:
    base = _base_vault_key(name, leaf)
    existing = _existing_vault_value(base, shared, pending, vault)
    if existing is not _MISSING and existing != value:
        return f"CONFIG_{base}"
    return base


def _set_path(data: dict[str, Any], path: tuple[str, ...], value: Any) -> None:
    target: Any = data
    for part in path[:-1]:
        target = target[part]
    target[path[-1]] = value


def find_plaintext_credentials(
    raw: dict[str, Any],
    data_dir: Path | None = None,
) -> list[PlaintextCredential]:
    """List credential leaves that a save would move into the vault.

    Only paths and planned vault key names are returned, never the values.
    """
    leaves = _plaintext_credential_leaves(raw)
    if not leaves:
        return []

    if data_dir is None:
        from core.paths import get_data_dir

        data_dir = get_data_dir()

    vault = VaultManager(data_dir)
    if not _vault_store_available(vault):
        return [
            PlaintextCredential(path=".".join(path), vault_key=_base_vault_key(name, leaf))
            for path, name, leaf, _value in leaves
        ]

    shared = _load_shared_section(vault)
    pending: dict[str, str] = {}
    moves = _plan_credential_moves(leaves, shared, pending, vault)
    return [PlaintextCredential(path=".".join(path), vault_key=vault_key) for path, vault_key, _value in moves]


def _warn_vault_unavailable_once() -> None:
    global _vault_unavailable_warned
    with _vault_unavailable_warn_lock:
        if _vault_unavailable_warned:
            return
        _vault_unavailable_warned = True
    logger.warning(
        "Vault is unavailable (PyNaCl missing or vault.key not found); "
        "credential values in config.json will be stored in plaintext"
    )


def _apply_vault_updates(data_dir: Path, updates: dict[str, str]) -> None:
    """Atomically update changed values in the vault's shared section."""
    vault = VaultManager(data_dir)
    shared = vault.load_vault().get("shared", {})
    if not isinstance(shared, dict):
        raise ConfigError("vault.json shared section must be an object")
    for key, value in updates.items():
        vault.store("shared", key, value)


__all__ = [
    "PlaintextCredential",
    "find_plaintext_credentials",
    "get_config_path",
    "invalidate_cache",
    "load_config",
    "save_config",
    "update_config",
]
