#!/usr/bin/env python3
"""Move plaintext config credentials to vault references.

The command is a dry-run unless ``--apply`` is explicitly supplied.  Output
contains config paths and vault key names only, never credential values.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.config.vault import VaultManager
from core.paths import get_data_dir

_SENSITIVE_KEY_PARTS = ("api_key", "access_key", "secret", "token", "password")


class MigrationError(RuntimeError):
    """Raised when migration cannot proceed without risking data loss."""


@dataclass(frozen=True)
class CredentialMove:
    path: tuple[str, ...]
    vault_key: str
    value: str


def _env_name(value: str) -> str:
    return re.sub(r"[^A-Z0-9]+", "_", value.upper()).strip("_")


def _is_sensitive(field: str) -> bool:
    lowered = field.lower()
    return any(part in lowered for part in _SENSITIVE_KEY_PARTS)


def discover_credentials(config: dict[str, Any]) -> list[CredentialMove]:
    """Find non-empty plaintext credential leaves under ``credentials``."""
    credentials = config.get("credentials", {})
    if not isinstance(credentials, dict):
        return []

    candidates: list[tuple[tuple[str, ...], str, str]] = []
    for credential_name, entry in credentials.items():
        if not isinstance(entry, dict):
            continue
        for field, value in entry.items():
            if field == "api_key" and isinstance(value, str) and value:
                candidates.append(
                    (("credentials", credential_name, field), f"{_env_name(credential_name)}_API_KEY", value)
                )
            elif field == "keys" and isinstance(value, dict):
                for key_name, key_value in value.items():
                    if _is_sensitive(key_name) and isinstance(key_value, str) and key_value:
                        candidates.append(
                            (("credentials", credential_name, field, key_name), _env_name(key_name), key_value)
                        )

    # A conventional key may be shared when values are identical.  If two
    # profiles use different values, qualify the key with the profile name.
    values_by_key: dict[str, set[str]] = {}
    for _, vault_key, value in candidates:
        values_by_key.setdefault(vault_key, set()).add(value)

    moves: list[CredentialMove] = []
    for path, vault_key, value in candidates:
        if len(values_by_key[vault_key]) > 1:
            vault_key = f"{_env_name(path[1])}_{vault_key}"
        moves.append(CredentialMove(path, vault_key, value))
    return moves


def _set_path(data: dict[str, Any], path: tuple[str, ...], value: Any) -> None:
    target: dict[str, Any] = data
    for part in path[:-1]:
        target = target[part]
    target[path[-1]] = value


def build_migration(
    config: dict[str, Any],
    vault: VaultManager,
) -> tuple[dict[str, Any], dict[str, Any], list[CredentialMove]]:
    """Return updated documents without writing either file."""
    moves = discover_credentials(config)
    new_config = copy.deepcopy(config)
    new_vault = copy.deepcopy(vault.load_vault())
    shared = new_vault.setdefault("shared", {})
    if not isinstance(shared, dict):
        raise MigrationError("vault.json shared section must be an object")

    for move in moves:
        if move.vault_key in shared:
            existing = vault.decrypt(shared[move.vault_key])
            if existing != move.value:
                raise MigrationError(f"Vault key already contains a different value: {move.vault_key}")
        else:
            shared[move.vault_key] = vault.encrypt(move.value)
        _set_path(new_config, move.path, {"$vault": move.vault_key})
    return new_config, new_vault, moves


def _backup(path: Path) -> Path:
    backup = path.with_name(f"{path.name}.bak")
    shutil.copy2(path, backup)
    os.chmod(backup, 0o600)
    return backup


def apply_migration(data_dir: Path) -> list[CredentialMove]:
    config_path = data_dir / "config.json"
    if not config_path.is_file():
        raise MigrationError(f"Config file not found: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    vault = VaultManager(data_dir)
    new_config, new_vault, moves = build_migration(config, vault)
    if not moves:
        return []

    _backup(config_path)
    if vault.vault_path.is_file():
        _backup(vault.vault_path)

    # Write the vault first so config references never point at a missing key.
    vault.save_vault(new_vault)
    config_tmp = config_path.with_name(f".{config_path.name}.{os.getpid()}.tmp")
    config_tmp.write_text(json.dumps(new_config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.chmod(config_tmp, 0o600)
    os.replace(config_tmp, config_path)
    os.chmod(vault.vault_path, 0o600)
    os.chmod(config_path, 0o600)
    return moves


def dry_run(data_dir: Path) -> list[CredentialMove]:
    config_path = data_dir / "config.json"
    if not config_path.is_file():
        raise MigrationError(f"Config file not found: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    _, _, moves = build_migration(config, VaultManager(data_dir))
    return moves


def _print_plan(moves: list[CredentialMove], *, applied: bool) -> None:
    mode = "APPLIED" if applied else "DRY-RUN"
    print(f"{mode}: {len(moves)} credential value(s)")
    for move in moves:
        print(f"  {'.'.join(move.path)} -> vault.shared.{move.vault_key}")
    if not applied and moves:
        print("No files changed. Re-run with --apply to migrate.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--apply", action="store_true", help="write changes (default: dry-run)")
    args = parser.parse_args(argv)
    data_dir = args.data_dir.expanduser().resolve() if args.data_dir else get_data_dir()
    try:
        moves = apply_migration(data_dir) if args.apply else dry_run(data_dir)
    except (MigrationError, json.JSONDecodeError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    _print_plan(moves, applied=args.apply)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
