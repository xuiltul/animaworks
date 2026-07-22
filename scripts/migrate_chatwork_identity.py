#!/usr/bin/env python3
from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Migrate legacy Chatwork credentials/cache/config to the identity model.

Idempotent deploy-time helper. Never prints token values.

With no extra options the script is dry-run and only applies the generic
default vault key copy (``CHATWORK_API_TOKEN`` → ``CHATWORK_API_TOKEN__owner``).
Site-specific copies, grants, and token-file registration are opt-in via CLI.

Usage::

    python scripts/migrate_chatwork_identity.py            # dry-run (default)
    python scripts/migrate_chatwork_identity.py --apply
    python scripts/migrate_chatwork_identity.py --apply --finalize
    python scripts/migrate_chatwork_identity.py --apply \\
        --copy-key CHATWORK_API_TOKEN_WRITE=CHATWORK_API_TOKEN__alice \\
        --grant bob=owner:read \\
        --register-token-file bot=credentials/bot-token
"""

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

# Vault key renames (source → destination). Old keys are kept unless --finalize.
# Generic default only; add more with --copy-key SRC=DST.
_VAULT_KEY_COPIES: tuple[tuple[str, str], ...] = (("CHATWORK_API_TOKEN", "CHATWORK_API_TOKEN__owner"),)

# Empty by default; pass --grant ANIMA=TARGET:LEVEL to configure.
_DEFAULT_GRANTS: dict[str, dict[str, str]] = {}


def _print(msg: str) -> None:
    print(msg)


def _vault_has(vault: Any, key: str) -> bool:
    return vault.get("shared", key) is not None


def _legacy_keys_from_copies(copies: tuple[tuple[str, str], ...]) -> tuple[str, ...]:
    """Derive finalize-delete keys from copy sources (order-preserving unique)."""
    seen: set[str] = set()
    result: list[str] = []
    for src, _dst in copies:
        if src not in seen:
            seen.add(src)
            result.append(src)
    return tuple(result)


def step_vault_key_copies(
    *,
    vault: Any,
    data_dir: Path,
    apply: bool,
    copies: tuple[tuple[str, str], ...],
    token_files: tuple[tuple[str, str], ...],
) -> None:
    """(a) Copy legacy vault keys and optionally register tokens from files."""
    for src, dst in copies:
        if _vault_has(vault, dst):
            _print(f"SKIP vault copy {src} → {dst}: destination already exists")
            continue
        value = vault.get("shared", src)
        if value is None:
            _print(f"SKIP vault copy {src} → {dst}: source key not found")
            continue
        if apply:
            vault.store("shared", dst, value)
            _print(f"APPLY vault copy {src} → {dst}")
        else:
            _print(f"DRY-RUN vault copy {src} → {dst}")

    for name, rel in token_files:
        dst = f"CHATWORK_API_TOKEN__{name}"
        rel_path = Path(rel)
        if _vault_has(vault, dst):
            _print(f"SKIP vault register {dst}: destination already exists")
            continue

        token_path = data_dir / rel_path
        if not token_path.is_file():
            _print(f"SKIP vault register {dst}: file not found ({rel_path.as_posix()})")
            continue

        try:
            raw = token_path.read_text(encoding="utf-8").strip()
        except OSError as exc:
            _print(f"SKIP vault register {dst}: cannot read file ({exc})")
            continue

        if not raw:
            _print(f"SKIP vault register {dst}: file is empty")
            continue

        if apply:
            vault.store("shared", dst, raw)
            _print(f"APPLY vault register {dst} from {rel_path.as_posix()}")
        else:
            _print(f"DRY-RUN vault register {dst} from {rel_path.as_posix()}")


def step_config_grants(
    *,
    config_path: Path,
    apply: bool,
    grants: dict[str, dict[str, str]],
) -> None:
    """(b) Write chatwork_tool.grants when unset and grants are provided."""
    if not grants:
        _print("SKIP config.json grants: no grants configured (pass --grant)")
        return

    if not config_path.is_file():
        if apply:
            payload = {"version": 1, "chatwork_tool": {"grants": grants}}
            config_path.parent.mkdir(parents=True, exist_ok=True)
            config_path.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            _print(f"APPLY config.json: created with chatwork_tool.grants at {config_path}")
        else:
            _print(f"DRY-RUN config.json: would create with chatwork_tool.grants at {config_path}")
        return

    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        _print(f"SKIP config.json grants: cannot read/parse ({exc})")
        return

    if not isinstance(raw, dict):
        _print("SKIP config.json grants: root is not an object")
        return

    tool_cfg = raw.get("chatwork_tool")
    if isinstance(tool_cfg, dict):
        existing = tool_cfg.get("grants")
        if isinstance(existing, dict) and existing:
            _print("SKIP config.json grants: already configured")
            return

    if not apply:
        _print("DRY-RUN config.json: write chatwork_tool.grants")
        return

    if not isinstance(tool_cfg, dict):
        raw["chatwork_tool"] = {"grants": dict(grants)}
    else:
        tool_cfg["grants"] = dict(grants)
        raw["chatwork_tool"] = tool_cfg

    config_path.write_text(
        json.dumps(raw, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    _print("APPLY config.json: wrote chatwork_tool.grants")


def _token_fingerprint(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()[:16]


def _fetch_account_id(token: str) -> str | None:
    """Call Chatwork /me; return account_id string or None on failure."""
    try:
        from core.tools._chatwork_client import ChatworkClient

        me = ChatworkClient(api_token=token).me()
    except Exception as exc:
        _print(f"SKIP cache migrate: Chatwork /me failed ({exc})")
        return None

    if not isinstance(me, dict):
        _print("SKIP cache migrate: /me response is not an object")
        return None
    account_id = me.get("account_id")
    if account_id in (None, ""):
        _print("SKIP cache migrate: /me returned no account_id")
        return None
    return str(account_id)


def step_cache_migrate(*, vault: Any, cache_dir: Path, apply: bool) -> None:
    """(c) Move legacy messages.db under account_id/ and update identity_map."""
    old_db = cache_dir / "messages.db"
    if not old_db.is_file():
        _print(f"SKIP cache migrate: no legacy DB at {old_db}")
        return

    owner_token = vault.get("shared", "CHATWORK_API_TOKEN__owner")
    if not owner_token:
        # After vault copies, dry-run may not have written yet — also check legacy
        owner_token = vault.get("shared", "CHATWORK_API_TOKEN")
    if not owner_token:
        _print("SKIP cache migrate: CHATWORK_API_TOKEN__owner (and legacy CHATWORK_API_TOKEN) not in vault")
        return

    if not apply:
        _print(
            f"DRY-RUN cache migrate: would call /me and move {old_db} "
            f"to {cache_dir}/<account_id>/messages.db and update identity_map.json"
        )
        return

    account_id = _fetch_account_id(owner_token)
    if account_id is None:
        return

    dest_dir = cache_dir / account_id
    dest_db = dest_dir / "messages.db"
    if dest_db.exists():
        _print(f"SKIP cache migrate: destination already exists at {dest_db} (leaving legacy {old_db} in place)")
        # Still ensure identity_map entry if missing.
    else:
        dest_dir.mkdir(parents=True, exist_ok=True)
        shutil.move(str(old_db), str(dest_db))
        _print(f"APPLY cache migrate: moved {old_db} → {dest_db}")

    map_path = cache_dir / "identity_map.json"
    identity_map: dict[str, str] = {}
    if map_path.is_file():
        try:
            raw_map = json.loads(map_path.read_text(encoding="utf-8"))
            if isinstance(raw_map, dict):
                identity_map = {str(k): str(v) for k, v in raw_map.items()}
        except (OSError, json.JSONDecodeError):
            identity_map = {}

    fp = _token_fingerprint(owner_token)
    if identity_map.get(fp) == account_id:
        _print(f"SKIP identity_map: {fp} already maps to {account_id}")
    else:
        identity_map[fp] = account_id
        tmp = map_path.with_suffix(".json.tmp")
        tmp.write_text(
            json.dumps(identity_map, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        tmp.replace(map_path)
        _print(f"APPLY identity_map: registered token fingerprint → {account_id}")


def step_finalize(
    *,
    vault: Any,
    apply: bool,
    finalize: bool,
    legacy_keys: tuple[str, ...],
) -> None:
    """(d) Optionally delete legacy vault keys after verification."""
    if not finalize:
        _print("SKIP finalize: pass --finalize to delete legacy vault keys")
        return

    for key in legacy_keys:
        if not _vault_has(vault, key):
            _print(f"SKIP finalize delete {key}: not present")
            continue
        if apply:
            vault.delete("shared", key)
            _print(f"APPLY finalize delete {key}")
        else:
            _print(f"DRY-RUN finalize delete {key}")


def run_migration(
    *,
    data_dir: Path,
    cache_dir: Path | None = None,
    apply: bool = False,
    finalize: bool = False,
    vault: Any | None = None,
    copies: tuple[tuple[str, str], ...] | None = None,
    grants: dict[str, dict[str, str]] | None = None,
    token_files: tuple[tuple[str, str], ...] | None = None,
) -> int:
    """Execute migration steps. Returns process exit code (always 0 on soft skips)."""
    if vault is None:
        from core.config.vault import get_vault_manager

        vault = get_vault_manager(data_dir)

    if cache_dir is None:
        from core.tools._chatwork_cache import DEFAULT_CACHE_DIR

        cache_dir = DEFAULT_CACHE_DIR

    if copies is None:
        copies = _VAULT_KEY_COPIES
    if grants is None:
        grants = dict(_DEFAULT_GRANTS)
    if token_files is None:
        token_files = ()

    mode = "APPLY" if apply else "DRY-RUN"
    _print(f"=== migrate_chatwork_identity ({mode}) data_dir={data_dir} ===")

    step_vault_key_copies(
        vault=vault,
        data_dir=data_dir,
        apply=apply,
        copies=copies,
        token_files=token_files,
    )
    step_config_grants(
        config_path=data_dir / "config.json",
        apply=apply,
        grants=grants,
    )
    step_cache_migrate(vault=vault, cache_dir=cache_dir, apply=apply)
    step_finalize(
        vault=vault,
        apply=apply,
        finalize=finalize,
        legacy_keys=_legacy_keys_from_copies(copies),
    )

    _print("=== done ===")
    return 0


def _parse_copy_key(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError(f"--copy-key expects SRC=DST, got {value!r}")
    src, dst = value.split("=", 1)
    if not src or not dst:
        raise argparse.ArgumentTypeError(f"--copy-key expects non-empty SRC=DST, got {value!r}")
    return src, dst


def _parse_grant(value: str) -> tuple[str, str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError(f"--grant expects ANIMA=TARGET:LEVEL, got {value!r}")
    anima, rest = value.split("=", 1)
    if ":" not in rest:
        raise argparse.ArgumentTypeError(f"--grant expects ANIMA=TARGET:LEVEL, got {value!r}")
    target, level = rest.split(":", 1)
    if not anima or not target:
        raise argparse.ArgumentTypeError(f"--grant expects non-empty ANIMA and TARGET, got {value!r}")
    if level not in ("read", "readwrite"):
        raise argparse.ArgumentTypeError(f"--grant LEVEL must be 'read' or 'readwrite', got {level!r}")
    return anima, target, level


def _parse_register_token_file(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError(f"--register-token-file expects NAME=RELPATH, got {value!r}")
    name, rel = value.split("=", 1)
    if not name or not rel:
        raise argparse.ArgumentTypeError(f"--register-token-file expects non-empty NAME=RELPATH, got {value!r}")
    return name, rel


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Migrate Chatwork vault keys, config grants, and message cache to the identity model (idempotent)."
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply changes (default is dry-run)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show planned changes without writing (default)",
    )
    parser.add_argument(
        "--finalize",
        action="store_true",
        help="Also delete legacy vault keys after copies (use after verification)",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Override data directory (default: core.paths.get_data_dir())",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Override Chatwork cache directory (default: core.tools._chatwork_cache.DEFAULT_CACHE_DIR)",
    )
    parser.add_argument(
        "--copy-key",
        action="append",
        type=_parse_copy_key,
        default=None,
        metavar="SRC=DST",
        help="Additional vault key copy (repeatable). "
        "Default already includes CHATWORK_API_TOKEN→CHATWORK_API_TOKEN__owner",
    )
    parser.add_argument(
        "--grant",
        action="append",
        type=_parse_grant,
        default=None,
        metavar="ANIMA=TARGET:LEVEL",
        help="Grant mapping for chatwork_tool.grants (repeatable). "
        "LEVEL is read or readwrite. Default is empty (no grants written)",
    )
    parser.add_argument(
        "--register-token-file",
        action="append",
        type=_parse_register_token_file,
        default=None,
        metavar="NAME=RELPATH",
        help="Register file contents at data_dir/RELPATH as CHATWORK_API_TOKEN__NAME (repeatable)",
    )
    return parser.parse_args(argv)


def _build_grants(
    grant_args: list[tuple[str, str, str]] | None,
) -> dict[str, dict[str, str]]:
    grants: dict[str, dict[str, str]] = dict(_DEFAULT_GRANTS)
    if not grant_args:
        return grants
    for anima, target, level in grant_args:
        grants.setdefault(anima, {})[target] = level
    return grants


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    # --dry-run is the default; --apply wins when both are passed.
    apply = bool(args.apply)

    if args.data_dir is not None:
        data_dir = args.data_dir
    else:
        from core.paths import get_data_dir

        data_dir = get_data_dir()

    # When --data-dir is given, construct a dedicated VaultManager so we do not
    # pollute the process-wide singleton (important for tests and one-shot runs).
    vault = None
    if args.data_dir is not None:
        from core.config.vault import VaultManager

        vault = VaultManager(data_dir)

    copies: list[tuple[str, str]] = list(_VAULT_KEY_COPIES)
    if args.copy_key:
        copies.extend(args.copy_key)

    grants = _build_grants(args.grant)
    token_files: tuple[tuple[str, str], ...] = tuple(args.register_token_file or ())

    return run_migration(
        data_dir=data_dir,
        cache_dir=args.cache_dir,
        apply=apply,
        finalize=bool(args.finalize),
        vault=vault,
        copies=tuple(copies),
        grants=grants,
        token_files=token_files,
    )


if __name__ == "__main__":
    raise SystemExit(main())
