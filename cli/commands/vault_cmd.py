from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""CLI subcommand for vault key-value storage.

Values live in two kinds of sections of ``vault.json``: the Anima-scoped
section (named after ``ANIMAWORKS_ANIMA_DIR``) and the ``shared`` section
that credential resolution reads from.  ``get`` and ``list`` cover both so
that a value written with ``store --shared`` stays visible afterwards.

Usage via animaworks-tool:
    animaworks-tool vault get KEY [--shared]
    animaworks-tool vault store KEY VALUE
    animaworks-tool vault list [--shared]
    animaworks-tool vault delete KEY [--shared]
    animaworks vault status
    animaworks vault init
    printf 'VALUE' | animaworks vault store --shared KEY
"""

import argparse
import getpass
import json
import os
import sys
from pathlib import Path

SHARED_SECTION = "shared"


def cmd_vault(args: argparse.Namespace) -> None:
    """Dispatch vault subcommand."""
    from core.config.vault import get_vault_manager

    vm = get_vault_manager()
    sub = getattr(args, "vault_command", None)
    shared = bool(getattr(args, "shared", False))

    if sub == "status":
        _cmd_status(vm)
    elif sub == "init":
        _cmd_init(vm)
    elif sub == "store":
        namespace = SHARED_SECTION if shared else _get_anima_namespace()
        _cmd_store(args, vm, namespace)
    elif sub == "get":
        _cmd_get(args, vm, shared)
    elif sub == "list":
        _cmd_list(vm, shared)
    elif sub == "delete":
        namespace = SHARED_SECTION if shared else _get_anima_namespace()
        _cmd_delete(args, vm, namespace)
    else:
        print("Usage: animaworks vault {status|init|get|store|list|delete}", file=sys.stderr)
        sys.exit(1)


def _get_anima_namespace() -> str:
    anima_dir_str = os.environ.get("ANIMAWORKS_ANIMA_DIR", "")
    if not anima_dir_str:
        print("Error: ANIMAWORKS_ANIMA_DIR not set", file=sys.stderr)
        sys.exit(1)

    anima_dir = Path(anima_dir_str)
    if not anima_dir.is_dir():
        print(f"Error: anima_dir not found: {anima_dir}", file=sys.stderr)
        sys.exit(1)
    return anima_dir.name


def _search_sections(shared_only: bool) -> list[str]:
    """Return the sections to search, Anima-scoped first then ``shared``."""
    if shared_only:
        return [SHARED_SECTION]
    return [_get_anima_namespace(), SHARED_SECTION]


def _cmd_get(args: argparse.Namespace, vm, shared_only: bool) -> None:
    key = getattr(args, "key", "")
    if not key:
        print("Error: key is required", file=sys.stderr)
        sys.exit(1)

    sections = _search_sections(shared_only)
    for section in sections:
        value = vm.get(section, key)
        if value is None:
            continue
        result = {"key": key, "section": section, "value": value}
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    print(
        f"Error: key not found: {key} (searched sections: {', '.join(sections)})",
        file=sys.stderr,
    )
    sys.exit(1)


def _cmd_store(args: argparse.Namespace, vm, namespace: str) -> None:
    key = getattr(args, "key", "")
    if not key:
        print("Error: key is required", file=sys.stderr)
        sys.exit(1)

    value = getattr(args, "value", None)
    if getattr(args, "shared", False):
        if value is not None:
            print("Error: --shared values must be supplied through stdin or interactive input", file=sys.stderr)
            sys.exit(1)
        value = getpass.getpass("Value: ") if sys.stdin.isatty() else sys.stdin.read().rstrip("\r\n")
        if not value:
            print("Error: empty value received; nothing was stored", file=sys.stderr)
            sys.exit(1)
    elif value is None:
        print("Error: value is required", file=sys.stderr)
        sys.exit(1)

    vm.store(namespace, key, value)
    result = {"key": key, "section": namespace, "stored": True}
    print(json.dumps(result, ensure_ascii=False, indent=2))


def _cmd_delete(args: argparse.Namespace, vm, namespace: str) -> None:
    """Remove a key from exactly one section.

    Unlike ``get``, this never cascades into ``shared``: an Anima deleting its
    own key must not take out a credential the rest of the fleet depends on.
    """
    key = getattr(args, "key", "")
    if not key:
        print("Error: key is required", file=sys.stderr)
        sys.exit(1)

    if not vm.delete(namespace, key):
        print(f"Error: key not found in section '{namespace}': {key}", file=sys.stderr)
        sys.exit(1)

    result = {"key": key, "section": namespace, "deleted": True}
    print(json.dumps(result, ensure_ascii=False, indent=2))


def _section_keys(data: dict, section: str) -> list[str]:
    entries = data.get(section)
    if not isinstance(entries, dict):
        return []
    return sorted(entries)


def _cmd_list(vm, shared_only: bool) -> None:
    data = vm.load_vault()
    if shared_only:
        result: dict = {"namespace": SHARED_SECTION, "keys": _section_keys(data, SHARED_SECTION)}
    else:
        namespace = _get_anima_namespace()
        result = {
            "namespace": namespace,
            "keys": _section_keys(data, namespace),
            "shared": _section_keys(data, SHARED_SECTION),
        }
    print(json.dumps(result, ensure_ascii=False, indent=2))


def _cmd_status(vm) -> None:
    data = vm.load_vault()
    sections: dict[str, dict[str, int]] = {}
    entry_count = 0
    for section, entries in data.items():
        if not isinstance(entries, dict):
            sections[section] = {"entries": 0, "encrypted": 0, "plaintext_like": 0, "invalid": 1}
            continue
        encrypted = sum(1 for value in entries.values() if isinstance(value, str) and vm.is_encrypted_value(value))
        count = len(entries)
        entry_count += count
        sections[section] = {
            "entries": count,
            "encrypted": encrypted,
            "plaintext_like": count - encrypted,
            "invalid": 0,
        }

    result = {
        "key_present": vm.has_key,
        "entry_count": entry_count,
        "sections": sections,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


def _cmd_init(vm) -> None:
    from core.config.vault import VaultError

    try:
        generated = vm.generate_key()
    except VaultError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
    if not generated:
        print("Error: vault key generation is unavailable", file=sys.stderr)
        sys.exit(1)
    print(json.dumps({"initialized": True}, ensure_ascii=False, indent=2))


def register_vault_command(subparsers) -> None:
    """Register the vault subcommand under animaworks-tool."""
    p_vault = subparsers.add_parser("vault", help="Manage encrypted vault values")
    vault_sub = p_vault.add_subparsers(dest="vault_command")

    vault_sub.add_parser("status", help="Show key and encryption status without values")
    vault_sub.add_parser("init", help="Generate a vault key if one does not exist")

    p_get = vault_sub.add_parser("get", help="Get a value by key")
    p_get.add_argument("key", help="Key to retrieve")
    p_get.add_argument(
        "--shared",
        action="store_true",
        help="Look only in the shared section (default: Anima namespace, then shared)",
    )

    p_store = vault_sub.add_parser("store", help="Store a key-value pair")
    p_store.add_argument("key", help="Key to store")
    p_store.add_argument("value", nargs="?", help="Value to store (Anima-scoped compatibility mode only)")
    p_store.add_argument(
        "--shared",
        action="store_true",
        help="Store in the shared section, reading the value from stdin or a hidden prompt",
    )

    p_list = vault_sub.add_parser("list", help="List keys in the anima namespace and the shared section")
    p_list.add_argument(
        "--shared",
        action="store_true",
        help="List only the shared section (does not require ANIMAWORKS_ANIMA_DIR)",
    )

    p_delete = vault_sub.add_parser("delete", help="Remove a key from one section")
    p_delete.add_argument("key", help="Key to remove")
    p_delete.add_argument(
        "--shared",
        action="store_true",
        help="Delete from the shared section instead of the Anima namespace (never cascades)",
    )

    p_vault.set_defaults(func=cmd_vault)
