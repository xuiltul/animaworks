from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""CLI subcommand for vault key-value storage.

Anima ``get/store/list`` commands delegate to ToolHandler. Animas may read
only their own namespace and ``shared``; writes stay in their own namespace.
Operator-only shared writes/deletes remain available through the explicit
``--shared`` CLI path when ``ANIMAWORKS_ANIMA_DIR`` is not set.

Usage via animaworks-tool:
    animaworks-tool vault get KEY [--shared]
    animaworks-tool vault store KEY VALUE
    animaworks-tool vault list [--shared]
    animaworks-tool vault delete KEY [--shared]
    animaworks vault status
    animaworks vault init
    animaworks vault migrate-config [--commit]
    printf 'VALUE' | animaworks vault store --shared KEY
"""

import argparse
import dataclasses
import getpass
import json
import sys
from pathlib import Path

from cli._anima_tool import run_and_print
from core.config.io import find_plaintext_credentials, get_config_path, update_config
from core.config.onepassword import OnePasswordClient, OnePasswordError
from core.config.vault import SHARED_VAULT_SECTION as SHARED_SECTION
from core.config.vault import VaultAccessError, resolve_vault_sections
from core.config.vault_ops import (
    CHECK_ERROR,
    CHECK_SAME,
    STATUS_CHANGED,
    STATUS_ERROR,
    STATUS_NEW,
    VaultOpsError,
    VaultOwnershipError,
    apply_sync,
    check,
    delete_value,
    entries_overview,
    last_check_summary,
    plan_sync,
    register_onepassword,
    rollback,
    set_value,
)
from core.config.vault_sources import VaultSourceError, add_source, is_onepassword_managed, load_sources, remove_source
from core.i18n import t
from core.platform.env import anima_dir_env


def cmd_vault(args: argparse.Namespace) -> None:
    """Dispatch vault subcommand."""
    sub = getattr(args, "vault_command", None)
    shared = bool(getattr(args, "shared", False))

    if sub in {"get", "store", "list"} and anima_dir_env():
        if sub == "get":
            _cmd_tool_get(args, shared)
        elif sub == "store":
            _cmd_tool_store(args, shared)
        else:
            _cmd_tool_list(shared)
        return

    if sub == "migrate-config":
        if anima_dir_env() is not None:
            print(
                f"Error: {t('vault.operator_only', command='animaworks vault migrate-config')}",
                file=sys.stderr,
            )
            sys.exit(1)
        _cmd_migrate_config(args)
        return

    from core.config.vault import get_vault_manager

    vm = get_vault_manager()
    operator_commands = {"sources", "sync", "rollback", "register-1p"}
    if sub in operator_commands and anima_dir_env():
        print(
            f"Error: {t('vault.operator_only', command=f'animaworks vault {sub}')}",
            file=sys.stderr,
        )
        sys.exit(1)

    if sub == "status":
        _cmd_status(vm)
    elif sub == "init":
        _cmd_init(vm)
    elif sub == "store":
        anima_name = _get_anima_namespace(required=False)
        namespace = SHARED_SECTION if shared else (anima_name or _get_anima_namespace())
        _cmd_store(args, vm, namespace, anima_name)
    elif sub == "get":
        _cmd_get(args, vm, shared)
    elif sub == "list":
        _cmd_list(vm, shared)
    elif sub == "delete":
        anima_name = _get_anima_namespace(required=False)
        namespace = SHARED_SECTION if shared else (anima_name or _get_anima_namespace())
        _cmd_delete(args, vm, namespace, anima_name)
    elif sub == "sources":
        _cmd_sources(args, vm)
    elif sub == "sync":
        _cmd_sync(args, vm)
    elif sub == "rollback":
        _cmd_rollback(args, vm)
    elif sub == "register-1p":
        _cmd_register_1p(args, vm)
    else:
        print(
            "Usage: animaworks vault {status|init|get|store|list|delete|sources|sync|rollback|register-1p}",
            file=sys.stderr,
        )
        sys.exit(1)


def _cmd_tool_get(args: argparse.Namespace, shared_only: bool) -> None:
    tool_args = {"key": getattr(args, "key", "")}
    if shared_only:
        tool_args["section"] = SHARED_SECTION
    run_and_print("vault_get", tool_args)


def _cmd_tool_store(args: argparse.Namespace, shared: bool) -> None:
    tool_args = {"key": getattr(args, "key", "")}
    value = getattr(args, "value", None)
    if value is not None:
        tool_args["value"] = value
    if shared:
        # The handler rejects anima writes to shared before looking at a value.
        tool_args["section"] = SHARED_SECTION
    run_and_print("vault_store", tool_args)


def _cmd_tool_list(shared_only: bool) -> None:
    tool_args = {"section": SHARED_SECTION} if shared_only else {}
    run_and_print("vault_list", tool_args)


def _get_anima_namespace(*, required: bool = True) -> str | None:
    anima_dir_str = anima_dir_env() or ""
    if not anima_dir_str:
        if required:
            print(
                "Error: ANIMAWORKS_ANIMA_DIR not set (set automatically inside an anima's tool context)",
                file=sys.stderr,
            )
            sys.exit(1)
        return None

    anima_dir = Path(anima_dir_str)
    if not anima_dir.is_dir():
        print(f"Error: anima_dir not found: {anima_dir}", file=sys.stderr)
        sys.exit(1)
    return anima_dir.name


def _search_sections(shared_only: bool) -> list[str]:
    """Return sections for the minimal operator read path."""
    if shared_only:
        return resolve_vault_sections(None, section=SHARED_SECTION, operator=True)
    anima_name = _get_anima_namespace()
    return resolve_vault_sections(anima_name)


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


def _cmd_store(args: argparse.Namespace, vm, namespace: str, anima_name: str | None) -> None:
    key = getattr(args, "key", "")
    if not key:
        print("Error: key is required", file=sys.stderr)
        sys.exit(1)

    try:
        namespace = resolve_vault_sections(
            anima_name,
            section=namespace,
            write=True,
            operator=anima_name is None,
        )[0]
    except VaultAccessError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    if namespace == SHARED_SECTION and is_onepassword_managed(vm.data_dir, key):
        print(f"Error: {t('vault.managed_store', name=key)}", file=sys.stderr)
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

    try:
        set_value(vm.data_dir, namespace, key, value, actor="cli")
    except (VaultOwnershipError, VaultOpsError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
    result = {"key": key, "section": namespace, "stored": True}
    print(json.dumps(result, ensure_ascii=False, indent=2))


def _cmd_delete(args: argparse.Namespace, vm, namespace: str, anima_name: str | None) -> None:
    """Remove a key from exactly one section, respecting anima write scope."""
    key = getattr(args, "key", "")
    if not key:
        print("Error: key is required", file=sys.stderr)
        sys.exit(1)

    try:
        namespace = resolve_vault_sections(
            anima_name,
            section=namespace,
            write=True,
            operator=anima_name is None,
        )[0]
    except VaultAccessError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    if namespace == SHARED_SECTION and is_onepassword_managed(vm.data_dir, key):
        print(f"Error: {t('vault.managed_delete', name=key)}", file=sys.stderr)
        sys.exit(1)

    try:
        deleted = delete_value(vm.data_dir, namespace, key, actor="cli")
    except (VaultOwnershipError, VaultOpsError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
    if not deleted:
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
        section = resolve_vault_sections(None, section=SHARED_SECTION, operator=True)[0]
        result: dict = {"namespace": section, "keys": _section_keys(data, section)}
    else:
        sections = _search_sections(shared_only=False)
        namespace = sections[0]
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

    overview = entries_overview(vm.data_dir)
    managed_by_section: dict[str, int] = {}
    for entry in overview:
        if entry.owner == "onepassword":
            managed_by_section[entry.section] = managed_by_section.get(entry.section, 0) + 1
    for section, row in sections.items():
        row["onepassword_managed"] = managed_by_section.get(section, 0)

    result = {
        "key_present": vm.has_key,
        "entry_count": entry_count,
        "sections": sections,
        "onepassword": {
            "managed_count": sum(managed_by_section.values()),
            "last_check": last_check_summary(vm.data_dir),
        },
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


def _make_client(vm) -> OnePasswordClient:
    try:
        settings = load_sources(vm.data_dir).onepassword
    except VaultSourceError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
    client = OnePasswordClient(settings.command, settings.timeout_s, write_command=settings.write_command)
    if not client.available():
        print(
            f"Error: {t('vault.op_not_found', command=' '.join(settings.command))}",
            file=sys.stderr,
        )
        sys.exit(3)
    return client


def _plan_payload(plan) -> list[dict[str, object]]:
    return [{"key": item.key, "status": item.status, "error": item.error} for item in plan]


def _cmd_sources(args: argparse.Namespace, vm) -> None:
    sub = getattr(args, "sources_command", None)
    try:
        if sub == "list":
            sources = load_sources(vm.data_dir)
            payload = {
                "version": sources.version,
                "onepassword": {
                    "command": list(sources.onepassword.command),
                    "write_command": (
                        list(sources.onepassword.write_command)
                        if sources.onepassword.write_command is not None
                        else None
                    ),
                    "timeout_s": sources.onepassword.timeout_s,
                },
                "entries": [
                    {"key": key, "ref": entry.ref, "added_at": entry.added_at}
                    for key, entry in sorted(sources.entries.items())
                ],
            }
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        elif sub == "add":
            entry = add_source(vm.data_dir, args.key, args.ref)
            print(json.dumps({"key": args.key, "ref": entry.ref, "added_at": entry.added_at}, indent=2))
        elif sub == "remove":
            if not remove_source(vm.data_dir, args.key):
                print(f"Error: source not found: {args.key}", file=sys.stderr)
                sys.exit(1)
            print(json.dumps({"key": args.key, "removed": True}, indent=2))
        else:
            print("Usage: animaworks vault sources {list|add|remove}", file=sys.stderr)
            sys.exit(1)
    except VaultSourceError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


def _cmd_sync(args: argparse.Namespace, vm) -> None:
    client = _make_client(vm)
    keys = list(getattr(args, "keys", []) or [])

    if getattr(args, "check", False):
        try:
            results = check(vm.data_dir, client, keys or None)
        except VaultSourceError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            sys.exit(1)
        if getattr(args, "json", False):
            print(
                json.dumps(
                    {"results": results, "request_count": client.request_count},
                    ensure_ascii=False,
                    indent=2,
                )
            )
        else:
            for key, status in results.items():
                print(f"{key}: {status}")
            print(f"1Password requests: {client.request_count}")
        if CHECK_ERROR in results.values():
            sys.exit(3)
        if any(status != CHECK_SAME for status in results.values()):
            sys.exit(2)
        return

    try:
        plan = plan_sync(vm.data_dir, client, keys or None)
    except VaultSourceError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    pending = [item for item in plan if item.status in (STATUS_NEW, STATUS_CHANGED)]
    applied: list[str] = []
    cancelled = False
    exit_code = 0
    if pending:
        if getattr(args, "yes", False):
            applied = apply_sync(vm.data_dir, plan)
        elif sys.stdin.isatty():
            try:
                answer = input(t("vault.sync_confirm", count=len(pending)))
            except EOFError:
                answer = ""
            if answer.strip().lower() in {"y", "yes"}:
                applied = apply_sync(vm.data_dir, plan)
            else:
                cancelled = True
        else:
            exit_code = 1

    if getattr(args, "json", False):
        print(
            json.dumps(
                {
                    "items": _plan_payload(plan),
                    "applied": applied,
                    "cancelled": cancelled,
                    "request_count": client.request_count,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    else:
        for item in plan:
            print(f"{item.key}: {item.status}")
            if item.error:
                print(f"  error: {item.error}")
        if applied:
            print(f"applied: {', '.join(applied)}")
        if cancelled:
            print(t("vault.sync_cancelled"))
        print(f"1Password requests: {client.request_count}")

    if exit_code == 1:
        print(f"Error: {t('vault.non_tty_pending')}", file=sys.stderr)
        sys.exit(1)
    if any(item.status == STATUS_ERROR for item in plan):
        sys.exit(3)


def _cmd_rollback(args: argparse.Namespace, vm) -> None:
    key = getattr(args, "key", "")
    if not rollback(vm.data_dir, key, actor="cli"):
        print(f"Error: {t('vault.rollback_missing', name=key)}", file=sys.stderr)
        sys.exit(1)
    print(json.dumps({"key": key, "rolled_back": True}, ensure_ascii=False, indent=2))


def _cmd_register_1p(args: argparse.Namespace, vm) -> None:
    client = _make_client(vm)
    try:
        result = register_onepassword(
            vm.data_dir,
            client,
            args.key,
            vault=args.vault,
            title=args.title,
            field=args.field,
            commit=bool(args.commit),
        )
    except (VaultSourceError, VaultOpsError, VaultOwnershipError, OnePasswordError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
    print(json.dumps(dataclasses.asdict(result), ensure_ascii=False, indent=2))


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


def _read_plaintext_credentials(config_path: Path, data_dir: Path) -> list[dict[str, str]]:
    if not config_path.is_file():
        raw = {}
    else:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        return []
    return [dataclasses.asdict(item) for item in find_plaintext_credentials(raw, data_dir)]


def _cmd_migrate_config(args: argparse.Namespace) -> None:
    config_path = get_config_path()
    data_dir = config_path.parent
    try:
        before = _read_plaintext_credentials(config_path, data_dir)
    except Exception:
        print(f"Error: {t('vault.migrate_failed')}", file=sys.stderr)
        sys.exit(1)

    if not getattr(args, "commit", False):
        print(json.dumps({"credentials": before, "committed": False}, ensure_ascii=False, indent=2))
        return

    try:
        update_config(lambda config: None, path=config_path)
        after = _read_plaintext_credentials(config_path, data_dir)
    except PermissionError:
        print(f"Error: {t('vault.migrate_server_running')}", file=sys.stderr)
        sys.exit(1)
    except Exception:
        print(f"Error: {t('vault.migrate_failed')}", file=sys.stderr)
        sys.exit(1)

    remaining = {(item["path"], item["vault_key"]) for item in after}
    moved = [item for item in before if (item["path"], item["vault_key"]) not in remaining]
    print(json.dumps({"moved": moved, "committed": True}, ensure_ascii=False, indent=2))


def register_vault_command(subparsers) -> None:
    """Register the vault subcommand under animaworks-tool."""
    p_vault = subparsers.add_parser("vault", help="Manage encrypted vault values")
    vault_sub = p_vault.add_subparsers(dest="vault_command")

    vault_sub.add_parser("status", help="Show key and encryption status without values")
    vault_sub.add_parser("init", help="Generate a vault key if one does not exist")

    p_migrate = vault_sub.add_parser("migrate-config", help="Move plaintext config credentials into the vault")
    p_migrate.add_argument("--commit", action="store_true", help="Move the listed credentials now")

    p_get = vault_sub.add_parser("get", help="Get a value by key")
    p_get.add_argument("key", help="Key to retrieve")
    p_get.add_argument(
        "--shared",
        action="store_true",
        help="Look only in the shared section (default: own namespace, then shared)",
    )

    p_store = vault_sub.add_parser("store", help="Store a key-value pair")
    p_store.add_argument("key", help="Key to store")
    p_store.add_argument("value", nargs="?", help="Value to store in the anima namespace")
    p_store.add_argument(
        "--shared",
        action="store_true",
        help="Operator-only: store in shared via stdin or a hidden prompt",
    )

    p_list = vault_sub.add_parser("list", help="List keys in the own and shared namespaces")
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
        help="Operator-only: delete from shared (never cascades)",
    )

    p_sources = vault_sub.add_parser("sources", help="Manage the shared-key to 1Password reference mapping")
    sources_sub = p_sources.add_subparsers(dest="sources_command")
    p_sources_list = sources_sub.add_parser("list", help="List registered 1Password references")
    p_sources_list.add_argument("--json", action="store_true", help="Emit JSON (the default)")
    p_sources_add = sources_sub.add_parser("add", help="Register a 1Password reference for a shared key")
    p_sources_add.add_argument("key", help="Shared vault key")
    p_sources_add.add_argument("ref", help="1Password reference (op://Vault/Item/field)")
    p_sources_remove = sources_sub.add_parser("remove", help="Remove a 1Password reference")
    p_sources_remove.add_argument("key", help="Shared vault key")

    p_sync = vault_sub.add_parser("sync", help="Sync shared keys from 1Password")
    p_sync.add_argument("keys", nargs="*", help="Only sync these keys (default: all registered keys)")
    p_sync.add_argument("--check", action="store_true", help="Only compare values; do not write")
    p_sync.add_argument("--yes", action="store_true", help="Apply changes without prompting")
    p_sync.add_argument("--json", action="store_true", help="Emit JSON without secret values")

    p_rollback = vault_sub.add_parser("rollback", help="Restore the previous shared value for a key")
    p_rollback.add_argument("key", help="Shared vault key")

    p_register = vault_sub.add_parser("register-1p", help="Register a shared key with 1Password")
    p_register.add_argument("key", help="Shared vault key")
    p_register.add_argument("--vault", required=True, help="1Password vault name")
    p_register.add_argument("--title", default=None, help="1Password item title (default: the key)")
    p_register.add_argument("--field", default="credential", help="1Password field name (default: credential)")
    p_register.add_argument("--commit", action="store_true", help="Create the item (default: preview only)")

    p_vault.set_defaults(func=cmd_vault)
