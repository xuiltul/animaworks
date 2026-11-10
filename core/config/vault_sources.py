# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.

"""Mapping between vault keys and 1Password references.

The mapping is deliberately kept outside ``config.json`` because the running
server owns that file.  ``vault_sources.json`` contains no secret values; it
only records which shared vault key corresponds to which ``op://`` reference.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from core.config.vault import SHARED_VAULT_SECTION
from core.exceptions import ConfigError

logger = logging.getLogger(__name__)

_SOURCES_FILENAME = "vault_sources.json"
_SOURCES_LOCK = threading.RLock()


class VaultSourceError(ConfigError):
    """Raised when the vault source mapping is invalid."""


@dataclass
class OnePasswordSettings:
    """How to invoke the 1Password CLI."""

    command: list[str] = field(default_factory=lambda: ["op"])
    write_command: list[str] | None = None
    timeout_s: float = 30.0

    @property
    def effective_write_command(self) -> list[str]:
        return list(self.write_command) if self.write_command else list(self.command)


@dataclass
class VaultSourceEntry:
    """A single ``shared`` key mapped to a 1Password reference."""

    ref: str
    added_at: str


@dataclass
class VaultSources:
    """Contents of ``vault_sources.json``."""

    version: int = 1
    onepassword: OnePasswordSettings = field(default_factory=OnePasswordSettings)
    entries: dict[str, VaultSourceEntry] = field(default_factory=dict)


def sources_path(data_dir: Path) -> Path:
    return Path(data_dir) / _SOURCES_FILENAME


def validate_ref(ref: str) -> str:
    """Validate and return an ``op://`` reference.

    A valid reference has at least three non-empty path components, e.g.
    ``op://VaultName/ItemName/field``.
    """
    if not isinstance(ref, str) or not ref.startswith("op://"):
        raise VaultSourceError("1Password references must start with 'op://'")
    parts = ref[len("op://") :].split("/")
    if len(parts) < 3 or any(not part for part in parts):
        raise VaultSourceError("1Password references must have at least three '/'-separated parts")
    return ref


def validate_key(key: str) -> str:
    if not isinstance(key, str) or not key or key != key.strip() or "/" in key:
        raise VaultSourceError("Vault key must be a non-empty string without surrounding whitespace or '/'")
    return key


def _validate_command(value: Any, *, field_name: str) -> list[str]:
    if not isinstance(value, list) or not value or not all(isinstance(part, str) and part for part in value):
        raise VaultSourceError(f"{field_name} must be a non-empty array of strings")
    return list(value)


def _parse_entry(key: str, raw: Any) -> VaultSourceEntry:
    validate_key(key)
    if not isinstance(raw, dict):
        raise VaultSourceError(f"Entry '{key}' must be an object")
    ref = validate_ref(raw.get("ref"))
    added_at = raw.get("added_at")
    if not isinstance(added_at, str) or not added_at:
        raise VaultSourceError(f"Entry '{key}' is missing added_at")
    return VaultSourceEntry(ref=ref, added_at=added_at)


def _parse_sources(raw: Any) -> VaultSources:
    if not isinstance(raw, dict):
        raise VaultSourceError("vault_sources.json must contain a JSON object")
    version = raw.get("version", 1)
    if version != 1:
        raise VaultSourceError(f"Unsupported vault_sources version: {version!r}")

    op_raw = raw.get("onepassword", {})
    if not isinstance(op_raw, dict):
        raise VaultSourceError("onepassword must be an object")
    command = _validate_command(op_raw.get("command", ["op"]), field_name="onepassword.command")
    write_command_raw = op_raw.get("write_command")
    write_command = (
        None
        if write_command_raw is None
        else _validate_command(write_command_raw, field_name="onepassword.write_command")
    )
    timeout_s = op_raw.get("timeout_s", 30.0)
    if isinstance(timeout_s, bool) or not isinstance(timeout_s, (int, float)) or timeout_s <= 0:
        raise VaultSourceError("onepassword.timeout_s must be a positive number")

    entries_raw = raw.get("entries", {})
    if not isinstance(entries_raw, dict):
        raise VaultSourceError("entries must be an object")
    entries = {key: _parse_entry(key, value) for key, value in entries_raw.items()}
    return VaultSources(
        version=1,
        onepassword=OnePasswordSettings(command=command, write_command=write_command, timeout_s=float(timeout_s)),
        entries=entries,
    )


def load_sources(data_dir: Path) -> VaultSources:
    """Load the mapping, returning defaults when the file does not exist."""
    path = sources_path(data_dir)
    if not path.is_file():
        return VaultSources()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise VaultSourceError(f"Failed to load {path.name}: {exc}") from exc
    return _parse_sources(raw)


def _serialise(sources: VaultSources) -> dict[str, Any]:
    return {
        "version": 1,
        "onepassword": {
            "command": list(sources.onepassword.command),
            "write_command": (
                list(sources.onepassword.write_command) if sources.onepassword.write_command is not None else None
            ),
            "timeout_s": sources.onepassword.timeout_s,
        },
        "entries": {
            key: {"ref": entry.ref, "added_at": entry.added_at} for key, entry in sorted(sources.entries.items())
        },
    }


def save_sources(data_dir: Path, sources: VaultSources) -> None:
    """Persist the mapping atomically with mode 0o600."""
    path = sources_path(data_dir)
    text = json.dumps(_serialise(sources), indent=2, ensure_ascii=False) + "\n"
    with _SOURCES_LOCK:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(text, encoding="utf-8")
        os.chmod(tmp, 0o600)
        tmp.rename(path)


def add_source(data_dir: Path, key: str, ref: str) -> VaultSourceEntry:
    """Add or update a mapping entry."""
    validate_key(key)
    validate_ref(ref)
    with _SOURCES_LOCK:
        sources = load_sources(data_dir)
        existing = sources.entries.get(key)
        now = datetime.now(UTC).isoformat()
        added_at = existing.added_at if existing is not None and existing.ref == ref else now
        entry = VaultSourceEntry(ref=ref, added_at=added_at)
        sources.entries[key] = entry
        save_sources(data_dir, sources)
        return entry


def remove_source(data_dir: Path, key: str) -> bool:
    """Remove a mapping entry.  Returns True when it existed."""
    with _SOURCES_LOCK:
        sources = load_sources(data_dir)
        if key not in sources.entries:
            return False
        del sources.entries[key]
        save_sources(data_dir, sources)
        return True


def is_onepassword_managed(data_dir: Path, key: str, section: str = SHARED_VAULT_SECTION) -> bool:
    """Return whether *key* in *section* is managed by 1Password.

    Only the ``shared`` section can be managed; all other sections are local.
    """
    if section != SHARED_VAULT_SECTION:
        return False
    try:
        return key in load_sources(data_dir).entries
    except VaultSourceError as exc:
        logger.warning("Could not read vault sources while checking %s: %s", key, exc)
        return False
