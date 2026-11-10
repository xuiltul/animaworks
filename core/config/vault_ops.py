# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.

"""Core operator operations for the vault and its 1Password mapping.

The CLI and the future GUI call these functions.  They never return or log
secret values except through the vault itself; metadata files contain only
timestamps, actors and state names.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from core.config.onepassword import OnePasswordClient, OnePasswordError
from core.config.vault import SHARED_VAULT_SECTION, VaultManager
from core.config.vault_sources import (
    VaultSourceError,
    add_source,
    is_onepassword_managed,
    load_sources,
    validate_key,
    validate_ref,
)
from core.exceptions import ConfigError

logger = logging.getLogger(__name__)

_META_FILENAME = "vault_meta.json"
_PREVIOUS_FILENAME = "vault.previous.json"
_VAULT_LOCK = threading.RLock()

STATUS_UNCHANGED = "unchanged"
STATUS_NEW = "new"
STATUS_CHANGED = "changed"
STATUS_EMPTY_REMOTE = "empty_remote"
STATUS_ERROR = "error"

CHECK_SAME = "same"
CHECK_DIFFERS = "differs"
CHECK_MISSING_REMOTE = "missing_remote"
CHECK_MISSING_LOCAL = "missing_local"
CHECK_ERROR = "error"


class VaultOpsError(ConfigError):
    """Raised when an operator vault operation cannot be completed."""


class VaultOwnershipError(PermissionError):
    """Raised when a locally requested write targets a 1Password-managed key."""


@dataclass
class SyncItem:
    """One key's sync state.  The value itself is kept off this dataclass."""

    key: str
    status: str
    error: str | None = None


@dataclass
class RegisterResult:
    """Preview or result of registering a shared key with 1Password."""

    key: str
    vault: str
    title: str
    field: str
    value_length: int
    committed: bool = False
    ref: str | None = None
    item_id: str | None = None


@dataclass
class EntryInfo:
    """Value-free overview row for a vault entry."""

    section: str
    key: str
    owner: str
    is_set: bool
    updated_at: str | None = None
    updated_by: str | None = None
    last_synced_at: str | None = None
    last_check: str | None = None


# ── Paths and file helpers ────────────────────────────────────────────────


def meta_path(data_dir: Path) -> Path:
    return Path(data_dir) / _META_FILENAME


def previous_path(data_dir: Path) -> Path:
    return Path(data_dir) / _PREVIOUS_FILENAME


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _atomic_write_json(path: Path, payload: dict[str, Any], *, mode: int = 0o600) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    tmp = path.with_suffix(".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.chmod(tmp, mode)
    tmp.rename(path)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Failed to read %s: %s", path, exc)
        return {}
    return raw if isinstance(raw, dict) else {}


def _read_meta(data_dir: Path) -> dict[str, Any]:
    meta = _read_json(meta_path(data_dir))
    entries = meta.get("entries")
    if not isinstance(entries, dict):
        entries = {}
    meta["entries"] = entries
    return meta


def _write_meta(data_dir: Path, meta: dict[str, Any]) -> None:
    try:
        _atomic_write_json(meta_path(data_dir), meta, mode=0o600)
    except OSError as exc:
        logger.warning("Failed to update %s: %s", _META_FILENAME, exc)


def _read_previous(data_dir: Path) -> dict[str, Any]:
    previous = _read_json(previous_path(data_dir))
    shared = previous.get("shared")
    if not isinstance(shared, dict):
        shared = {}
    previous["shared"] = shared
    return previous


def _meta_key(section: str, key: str) -> str:
    return f"{section}/{key}"


def _update_meta_entry(
    data_dir: Path,
    section: str,
    key: str,
    *,
    updated_at: str | None = None,
    updated_by: str | None = None,
    last_synced_at: str | None = None,
) -> None:
    with _VAULT_LOCK:
        meta = _read_meta(data_dir)
        entry = meta["entries"].get(_meta_key(section, key))
        if not isinstance(entry, dict):
            entry = {}
        if updated_at is not None:
            entry["updated_at"] = updated_at
        if updated_by is not None:
            entry["updated_by"] = updated_by
        if last_synced_at is not None:
            entry["last_synced_at"] = last_synced_at
        meta["entries"][_meta_key(section, key)] = entry
        if updated_at is not None and section == SHARED_VAULT_SECTION:
            # The value changed, so the last 1Password comparison no longer applies.
            _set_last_check_result(meta, key, None)
        _write_meta(data_dir, meta)


def _set_last_check_result(meta: dict[str, Any], key: str, result: str | None) -> None:
    last_check = meta.get("last_check")
    if not isinstance(last_check, dict):
        return
    results = last_check.get("results")
    if not isinstance(results, dict):
        return
    if result is None:
        results.pop(key, None)
    else:
        results[key] = result


def record_direct_write(data_dir: Path, section: str, key: str) -> None:
    """Record a write made through the legacy ``VaultManager`` API.

    The caller's identity is unknown on that path, so ``updated_by`` is
    ``"unknown"``.  A metadata failure never fails the vault write.
    """
    try:
        _update_meta_entry(data_dir, section, key, updated_at=_now(), updated_by="unknown")
    except Exception as exc:  # pragma: no cover - defensive boundary
        logger.warning("Failed to record vault metadata for %s/%s: %s", section, key, exc)


def _push_previous(data_dir: Path, section: str, key: str, value: str, actor: str) -> None:
    if section != SHARED_VAULT_SECTION:
        return
    vm = VaultManager(data_dir)
    with _VAULT_LOCK:
        previous = _read_previous(data_dir)
        previous["shared"][key] = {
            "value": vm.encrypt(value),
            "replaced_at": _now(),
            "replaced_by": actor,
        }
        try:
            _atomic_write_json(previous_path(data_dir), previous, mode=0o600)
        except OSError as exc:
            logger.warning("Failed to update %s: %s", _PREVIOUS_FILENAME, exc)


# ── Value operations ──────────────────────────────────────────────────────


def set_value(data_dir: Path, section: str, key: str, value: str, *, actor: str) -> None:
    """Store a value, keeping one previous generation for ``shared``.

    A 1Password-managed shared key cannot be overwritten by a human actor;
    only the ``sync`` actor may write it.
    """
    validate_key(key)
    if not isinstance(value, str) or value == "":
        raise VaultOpsError("Value must be a non-empty string")
    if section == SHARED_VAULT_SECTION and actor != "sync" and is_onepassword_managed(data_dir, key, section):
        raise VaultOwnershipError(
            f"{key} is managed by 1Password; change it in 1Password and run `animaworks vault sync`"
        )

    vm = VaultManager(data_dir)
    with _VAULT_LOCK:
        existing = vm.get(section, key)
        if existing is not None:
            _push_previous(data_dir, section, key, existing, actor)
        vm.store(section, key, value)
        _update_meta_entry(data_dir, section, key, updated_at=_now(), updated_by=actor)


def delete_value(data_dir: Path, section: str, key: str, *, actor: str) -> bool:
    """Delete a value, keeping the removed value as the previous generation."""
    validate_key(key)
    if section == SHARED_VAULT_SECTION and is_onepassword_managed(data_dir, key, section):
        raise VaultOwnershipError(
            f"{key} is managed by 1Password; change it in 1Password and run `animaworks vault sync`"
        )

    vm = VaultManager(data_dir)
    with _VAULT_LOCK:
        existing = vm.get(section, key)
        if existing is None:
            return False
        _push_previous(data_dir, section, key, existing, actor)
        deleted = vm.delete(section, key)
        if deleted:
            _update_meta_entry(data_dir, section, key, updated_at=_now(), updated_by=actor)
        return deleted


def rollback(data_dir: Path, key: str, *, actor: str) -> bool:
    """Restore the previous shared value for *key*.

    Managed keys may be rolled back because this is an explicit human action.
    The value being replaced becomes the new previous generation.
    """
    validate_key(key)
    vm = VaultManager(data_dir)
    with _VAULT_LOCK:
        previous = _read_previous(data_dir)
        entry = previous["shared"].get(key)
        if not isinstance(entry, dict):
            return False
        encrypted_previous = entry.get("value")
        if not isinstance(encrypted_previous, str) or not encrypted_previous:
            return False

        current = vm.get(SHARED_VAULT_SECTION, key)
        restored = vm.decrypt(encrypted_previous)
        vm.store(SHARED_VAULT_SECTION, key, restored)

        if current is None:
            previous["shared"].pop(key, None)
        else:
            previous["shared"][key] = {
                "value": vm.encrypt(current),
                "replaced_at": _now(),
                "replaced_by": actor,
            }
        try:
            _atomic_write_json(previous_path(data_dir), previous, mode=0o600)
        except OSError as exc:
            logger.warning("Failed to update %s: %s", _PREVIOUS_FILENAME, exc)
        _update_meta_entry(data_dir, SHARED_VAULT_SECTION, key, updated_at=_now(), updated_by=actor)
        return True


# ── Sync planning and application ─────────────────────────────────────────


def _selected_keys(keys: list[str] | None, sources_entries: dict[str, Any]) -> list[str]:
    if keys is None:
        return sorted(sources_entries)
    ordered: list[str] = []
    seen: set[str] = set()
    for key in keys:
        if key not in seen:
            seen.add(key)
            ordered.append(key)
    return ordered


def plan_sync(data_dir: Path, client: OnePasswordClient, keys: list[str] | None = None) -> list[SyncItem]:
    """Compare 1Password values with local shared values without exposing them."""
    sources = load_sources(data_dir)
    vm = VaultManager(data_dir)
    plan: list[SyncItem] = []
    for key in _selected_keys(keys, sources.entries):
        entry = sources.entries.get(key)
        if entry is None:
            plan.append(SyncItem(key=key, status=STATUS_ERROR, error="not registered in vault_sources.json"))
            continue
        try:
            remote = client.read(entry.ref)
        except OnePasswordError as exc:
            plan.append(SyncItem(key=key, status=STATUS_ERROR, error=str(exc)))
            continue

        item = SyncItem(key=key, status=STATUS_UNCHANGED)
        # Not a dataclass field: repr/asdict/JSON never include the value.
        item._remote_value = remote  # type: ignore[attr-defined]
        if remote == "":
            item.status = STATUS_EMPTY_REMOTE
            plan.append(item)
            continue
        local = vm.get(SHARED_VAULT_SECTION, key)
        if local is None or local == "":
            item.status = STATUS_NEW
        elif local == remote:
            item.status = STATUS_UNCHANGED
        else:
            item.status = STATUS_CHANGED
        plan.append(item)
    return plan


def apply_sync(data_dir: Path, plan: list[SyncItem], *, actor: str = "sync") -> list[str]:
    """Write ``new`` and ``changed`` items through ``set_value``.

    ``empty_remote`` and ``error`` items are left untouched.  Readable keys
    get their ``last_synced_at`` timestamp updated.
    """
    applied: list[str] = []
    readable = {STATUS_UNCHANGED, STATUS_NEW, STATUS_CHANGED, STATUS_EMPTY_REMOTE}
    synced_at = _now()
    for item in plan:
        if item.status in (STATUS_NEW, STATUS_CHANGED):
            value = getattr(item, "_remote_value", None)
            if value is None:
                continue
            set_value(data_dir, SHARED_VAULT_SECTION, item.key, value, actor=actor)
            applied.append(item.key)
        if item.status in readable:
            _update_meta_entry(
                data_dir,
                SHARED_VAULT_SECTION,
                item.key,
                last_synced_at=synced_at,
            )
    _record_sync_results(data_dir, plan, set(applied))
    return applied


def _record_sync_results(data_dir: Path, plan: list[SyncItem], applied: set[str]) -> None:
    """After a sync the compared keys are known to match (or be empty remotely)."""
    with _VAULT_LOCK:
        meta = _read_meta(data_dir)
        for item in plan:
            if item.status == STATUS_UNCHANGED or item.key in applied:
                _set_last_check_result(meta, item.key, CHECK_SAME)
            elif item.status == STATUS_EMPTY_REMOTE:
                _set_last_check_result(meta, item.key, CHECK_MISSING_REMOTE)
        _write_meta(data_dir, meta)


def _sha256(value: str) -> bytes:
    return hashlib.sha256(value.encode("utf-8")).digest()


def check(data_dir: Path, client: OnePasswordClient, keys: list[str] | None = None) -> dict[str, str]:
    """Compare local and remote values and record the result in metadata."""
    sources = load_sources(data_dir)
    vm = VaultManager(data_dir)
    results: dict[str, str] = {}
    for key in _selected_keys(keys, sources.entries):
        entry = sources.entries.get(key)
        if entry is None:
            results[key] = CHECK_ERROR
            continue
        try:
            remote = client.read(entry.ref)
        except OnePasswordError:
            results[key] = CHECK_ERROR
            continue
        local = vm.get(SHARED_VAULT_SECTION, key)
        if local is None or local == "":
            results[key] = CHECK_MISSING_LOCAL
        elif remote == "":
            results[key] = CHECK_MISSING_REMOTE
        elif _sha256(local) == _sha256(remote):
            results[key] = CHECK_SAME
        else:
            results[key] = CHECK_DIFFERS

    with _VAULT_LOCK:
        meta = _read_meta(data_dir)
        meta["last_check"] = {"at": _now(), "results": dict(sorted(results.items()))}
        _write_meta(data_dir, meta)
    return results


# ── Registration ──────────────────────────────────────────────────────────


def register_onepassword(
    data_dir: Path,
    client: OnePasswordClient,
    key: str,
    *,
    vault: str,
    title: str | None = None,
    field: str = "credential",
    commit: bool = False,
) -> RegisterResult:
    """Plan or perform one-way registration of a shared key with 1Password."""
    validate_key(key)
    if not vault:
        raise VaultOpsError("Vault name is required")
    if not field:
        raise VaultOpsError("Field name is required")

    vm = VaultManager(data_dir)
    value = vm.get(SHARED_VAULT_SECTION, key)
    if value is None or value == "":
        raise VaultOpsError(f"No non-empty local value for shared/{key}")

    sources = load_sources(data_dir)
    if key in sources.entries:
        raise VaultOpsError(f"{key} is already registered with 1Password")

    item_title = title or key
    if client.item_exists(vault, item_title):
        raise VaultOpsError(f"1Password item '{item_title}' already exists in vault '{vault}'; it was left unchanged")

    result = RegisterResult(
        key=key,
        vault=vault,
        title=item_title,
        field=field,
        value_length=len(value),
    )
    if not commit:
        return result

    item_id = client.create_api_credential(vault, item_title, field, value)
    ref = validate_ref(f"op://{vault}/{item_id}/{field}")
    try:
        read_back = client.read(ref)
    except OnePasswordError as exc:
        raise VaultOpsError(
            f"1Password item '{item_title}' was created (id {item_id}) but the read-back failed; "
            f"the mapping was not added: {exc}"
        ) from exc
    if read_back != value:
        raise VaultOpsError(
            f"1Password item '{item_title}' was created (id {item_id}) but the read-back value did not match; "
            "the mapping was not added"
        )
    add_source(data_dir, key, ref)
    result.committed = True
    result.ref = ref
    result.item_id = item_id
    return result


# ── Overview ──────────────────────────────────────────────────────────────


def entries_overview(data_dir: Path) -> list[EntryInfo]:
    """Return value-free information about every vault entry."""
    vm = VaultManager(data_dir)
    data = vm.load_vault()
    meta = _read_meta(data_dir)
    last_check_meta = meta.get("last_check")
    last_check_results = last_check_meta.get("results") if isinstance(last_check_meta, dict) else {}
    if not isinstance(last_check_results, dict):
        last_check_results = {}

    try:
        source_entries = load_sources(data_dir).entries
    except VaultSourceError:
        source_entries = {}

    rows: list[EntryInfo] = []
    seen: set[str] = set()
    for section, entries in sorted(data.items()):
        if not isinstance(entries, dict):
            continue
        for key, value in sorted(entries.items()):
            meta_entry = meta["entries"].get(_meta_key(section, key))
            if not isinstance(meta_entry, dict):
                meta_entry = {}
            owner = "onepassword" if section == SHARED_VAULT_SECTION and key in source_entries else "local"
            rows.append(
                EntryInfo(
                    section=section,
                    key=key,
                    owner=owner,
                    is_set=isinstance(value, str) and value != "",
                    updated_at=meta_entry.get("updated_at"),
                    updated_by=meta_entry.get("updated_by"),
                    last_synced_at=meta_entry.get("last_synced_at"),
                    last_check=last_check_results.get(key) if section == SHARED_VAULT_SECTION else None,
                )
            )
            if section == SHARED_VAULT_SECTION:
                seen.add(key)

    for key in sorted(source_entries):
        if key in seen:
            continue
        meta_entry = meta["entries"].get(_meta_key(SHARED_VAULT_SECTION, key))
        if not isinstance(meta_entry, dict):
            meta_entry = {}
        rows.append(
            EntryInfo(
                section=SHARED_VAULT_SECTION,
                key=key,
                owner="onepassword",
                is_set=False,
                updated_at=meta_entry.get("updated_at"),
                updated_by=meta_entry.get("updated_by"),
                last_synced_at=meta_entry.get("last_synced_at"),
                last_check=last_check_results.get(key),
            )
        )
    return rows


def last_check_summary(data_dir: Path) -> dict[str, Any] | None:
    """Return a value-free summary of the most recent check, if any."""
    meta = _read_meta(data_dir)
    last_check = meta.get("last_check")
    if not isinstance(last_check, dict):
        return None
    results = last_check.get("results")
    if not isinstance(results, dict):
        results = {}
    counts: dict[str, int] = {}
    for status in results.values():
        if isinstance(status, str):
            counts[status] = counts.get(status, 0) + 1
    return {"at": last_check.get("at"), "counts": dict(sorted(counts.items()))}


__all__ = [
    "CHECK_DIFFERS",
    "CHECK_ERROR",
    "CHECK_MISSING_LOCAL",
    "CHECK_MISSING_REMOTE",
    "CHECK_SAME",
    "EntryInfo",
    "RegisterResult",
    "STATUS_CHANGED",
    "STATUS_EMPTY_REMOTE",
    "STATUS_ERROR",
    "STATUS_NEW",
    "STATUS_UNCHANGED",
    "SyncItem",
    "VaultOpsError",
    "VaultOwnershipError",
    "apply_sync",
    "check",
    "delete_value",
    "entries_overview",
    "last_check_summary",
    "meta_path",
    "plan_sync",
    "previous_path",
    "record_direct_write",
    "register_onepassword",
    "rollback",
    "set_value",
]
