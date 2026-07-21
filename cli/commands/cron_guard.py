# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
"""CLI commands for inspecting and re-enabling cron guard tasks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def register_cron_guard_command(subparsers: argparse._SubParsersAction) -> None:
    """Register the ``cron-guard`` command group."""
    parser = subparsers.add_parser("cron-guard", help="Inspect and re-enable auto-disabled cron tasks")
    commands = parser.add_subparsers(dest="cron_guard_command", required=True)

    list_parser = commands.add_parser("list", help="List auto-disabled cron tasks")
    list_parser.add_argument("anima", help="Anima name")
    list_parser.set_defaults(func=cmd_cron_guard_list)

    enable_parser = commands.add_parser("enable", help="Re-enable an auto-disabled cron task")
    enable_parser.add_argument("anima", help="Anima name")
    enable_parser.add_argument("task", help="Cron task name")
    enable_parser.set_defaults(func=cmd_cron_guard_enable)


def _read_object(path: Path) -> dict[str, Any]:
    try:
        if not path.is_file():
            return {}
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return {}


def _write_object(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(f"{path.suffix}.tmp")
    tmp_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    tmp_path.replace(path)


def _resolve_anima_dir(name: str) -> Path | None:
    from core.paths import get_animas_dir

    animas_dir = get_animas_dir().resolve()
    anima_dir = (animas_dir / name).resolve()
    if not anima_dir.is_relative_to(animas_dir) or not anima_dir.is_dir():
        print(f"Error: Anima '{name}' not found at {anima_dir}")
        return None
    return anima_dir


def cmd_cron_guard_list(args: argparse.Namespace) -> None:
    """Print disabled task details and their latest persisted statistics."""
    anima_dir = _resolve_anima_dir(args.anima)
    if anima_dir is None:
        return

    state_dir = anima_dir / "state"
    disabled = _read_object(state_dir / "cron_disabled.json")
    stats = _read_object(state_dir / "cron_stats.json")
    if not disabled:
        print(f"No cron tasks are disabled for '{args.anima}'.")
        return

    print(f"Disabled cron tasks for '{args.anima}':")
    for task_name in sorted(disabled):
        details = disabled[task_name] if isinstance(disabled[task_name], dict) else {}
        current_stats = stats.get(task_name, {})
        print(f"- {task_name}")
        print(f"  disabled_at: {details.get('disabled_at', '')}")
        print(f"  reason: {details.get('reason', '')}")
        print(f"  stats: {json.dumps(current_stats, ensure_ascii=False, sort_keys=True)}")


def cmd_cron_guard_enable(args: argparse.Namespace) -> None:
    """Remove one task from the disabled sidecar; reload restores scheduling."""
    anima_dir = _resolve_anima_dir(args.anima)
    if anima_dir is None:
        return

    disabled_path = anima_dir / "state" / "cron_disabled.json"
    disabled = _read_object(disabled_path)
    if args.task not in disabled:
        print(f"Cron task '{args.task}' is not disabled for '{args.anima}'.")
        return

    del disabled[args.task]
    _write_object(disabled_path, disabled)
    print(f"Enabled cron task '{args.task}' for '{args.anima}'. Reload the scheduler to register it.")
