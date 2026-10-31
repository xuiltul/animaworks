"""Operator-only, cohort-scoped task migration and current-state export."""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from core.anima.workspace_files import IDENTITY_FILE
from core.i18n import t
from core.schemas import TaskEntry
from core.time_utils import now_iso


@contextmanager
def _offline_anima(runtime: Path, anima: str) -> Iterator[None]:
    """Reject a live server and hold the worker's own exclusion lock."""
    from cli.commands.server import _find_server_pid_by_process
    from core.platform.locks import acquire_file_lock, release_file_lock
    from core.platform.process import is_process_alive

    pid_file = runtime / "server.pid"
    if pid_file.exists():
        try:
            if is_process_alive(int(pid_file.read_text().strip())):
                raise RuntimeError(t("task_store.offline_required"))
        except ValueError as exc:
            raise RuntimeError(t("task_store.invalid_pid")) from exc
    if _find_server_pid_by_process() is not None:
        raise RuntimeError(t("task_store.offline_required"))
    lock_path = runtime / "run" / "animas" / f"{anima}.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+") as lock:
        try:
            acquire_file_lock(lock, exclusive=True, blocking=False)
        except OSError as exc:
            raise RuntimeError(t("task_store.offline_required")) from exc
        try:
            yield
        finally:
            release_file_lock(lock)


def _migrate_pending_command_tasks(store: Any, anima_dir: Path) -> dict[str, Any]:
    """Import only untouched legacy command descriptors; preserve every source file."""
    pending_dir = anima_dir / "state" / "background_tasks" / "pending"
    report: dict[str, Any] = {
        "pending_files": 0,
        "imported": 0,
        "already_imported": 0,
        "unimportable": 0,
        "processing_files_preserved": 0,
        "failed_files_preserved": 0,
        "warnings": [],
    }
    warnings: list[str] = report["warnings"]

    for path in sorted(pending_dir.glob("*.json")) if pending_dir.is_dir() else []:
        report["pending_files"] += 1
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            warnings.append(f"Could not read legacy command descriptor {path.name}: {exc}")
            report["unimportable"] += 1
            continue
        if not isinstance(raw, dict):
            warnings.append(f"Legacy command descriptor {path.name} is not a JSON object")
            report["unimportable"] += 1
            continue

        task_id = raw.get("task_id")
        tool_name = raw.get("tool_name")
        raw_args = raw.get("raw_args", [])
        subcommand = raw.get("subcommand", "")
        if (
            not isinstance(task_id, str)
            or not task_id
            or task_id in {".", ".."}
            or "/" in task_id
            or "\\" in task_id
            or path.stem != task_id
            or not isinstance(tool_name, str)
            or not tool_name.strip()
            or not isinstance(subcommand, str)
            or not isinstance(raw_args, list)
            or any(not isinstance(arg, str) for arg in raw_args)
            or raw.get("task_type") not in (None, "command")
            or raw.get("status") not in (None, "pending", "submitted")
            or raw.get("anima_name") not in (None, anima_dir.name)
        ):
            warnings.append(f"Legacy command descriptor {path.name} has invalid or conflicting fields")
            report["unimportable"] += 1
            continue

        description = raw.get("description")
        if not isinstance(description, str) or not description:
            description = shlex.join(["animaworks-tool", tool_name, *raw_args])
        title = raw.get("title")
        if not isinstance(title, str) or not title:
            title = f"{tool_name}:{subcommand}" if subcommand else tool_name
        payload = {
            "task_type": "command",
            "task_id": task_id,
            "tool_name": tool_name,
            "subcommand": subcommand,
            "raw_args": raw_args,
            "anima_name": anima_dir.name,
            "anima_dir": str(anima_dir),
            "submitted_at": raw.get("submitted_at", 0.0),
            "submitted_by": raw.get("submitted_by", anima_dir.name),
            "status": "pending",
            "title": title,
            "description": description,
        }

        existing = store.get_input(anima_dir.name, task_id)
        if existing is not None:
            same_command = all(
                existing.get(key) == payload[key]
                for key in ("task_type", "tool_name", "subcommand", "raw_args", "anima_dir")
            )
            if same_command:
                report["already_imported"] += 1
            else:
                warnings.append(f"Task ID collision for legacy command descriptor {path.name}; file retained")
                report["unimportable"] += 1
            continue
        if store.get(anima_dir.name, task_id) is not None:
            warnings.append(f"Task ID {task_id} already belongs to another task; {path.name} retained")
            report["unimportable"] += 1
            continue
        result_file = anima_dir / "state" / "background_tasks" / f"{task_id}.json"
        if result_file.exists():
            warnings.append(f"A background result already exists for {task_id}; {path.name} retained for review")
            report["unimportable"] += 1
            continue

        now = now_iso()
        entry = TaskEntry(
            task_id=task_id,
            ts=now,
            updated_at=now,
            source="anima",
            original_instruction=description,
            assignee=anima_dir.name,
            status="pending",
            summary=title,
            meta={"executor": "command", "migration_source": "legacy_background_pending"},
        )
        if store.submit(anima_dir.name, entry, payload):
            report["imported"] += 1
        else:
            warnings.append(f"Could not import legacy command descriptor {path.name}; file retained")
            report["unimportable"] += 1

    for directory, key in (("processing", "processing_files_preserved"), ("failed", "failed_files_preserved")):
        subdir = pending_dir / directory
        if subdir.is_dir():
            count = sum(1 for _ in subdir.glob("*.json"))
            report[key] = count
            if count:
                warnings.append(
                    f"Left {count} legacy command descriptor(s) in {directory}/ untouched for manual review"
                )
    return report


def run_maintenance(args: argparse.Namespace) -> dict[str, Any]:
    from core.paths import get_data_dir
    from core.tasks.board.tasks import TaskStore, task_database_path

    runtime = get_data_dir().resolve()
    anima_dir = (runtime / "animas" / args.anima).resolve()
    if anima_dir.parent != runtime / "animas" or not (anima_dir / IDENTITY_FILE).is_file():
        raise ValueError(t("task_store.invalid_anima", name=args.anima))
    store = TaskStore(task_database_path(anima_dir))
    action = args.task_store_action
    if action == "status":
        return store.maintenance_status(args.anima)
    if action in {"quiesce", "resume"}:
        store.pause_claims(args.anima, paused=action == "quiesce")
        return store.maintenance_status(args.anima)
    if action == "backup":
        store.backup(args.destination)
        return {"database_backup": str(args.destination.resolve()), "scope": "whole taskboard database"}

    with _offline_anima(runtime, args.anima):
        store.pause_claims(args.anima)
        status = store.maintenance_status(args.anima)
        if status["active_attempts"]:
            raise RuntimeError(t("task_store.active_attempts", count=status["active_attempts"]))
        if action == "migrate":
            from core.tasks.board.readiness import require_task_store_ready

            store.backup(args.backup)
            with store.transaction():
                report = store.import_legacy(anima_dir)
                if report["invalid_rows"]:
                    raise ValueError(t("task_store.invalid_rows", count=report["invalid_rows"]))
                command_report = _migrate_pending_command_tasks(store, anima_dir)
            require_task_store_ready(anima_dir)
            return {
                "migration": report,
                "command_migration": command_report,
                "backup": str(args.backup.resolve()),
                **store.maintenance_status(args.anima),
            }
        if action == "export":
            destination = args.destination.resolve()
            # A fresh destination avoids overlaying stale runnable descriptors.
            destination.mkdir(parents=True, exist_ok=False)
            snapshot_anima = destination / "animas" / args.anima
            count = store.export(anima_dir, destination=snapshot_anima, descriptors=True)
            store.backup(destination / "taskboard.sqlite3")
            manifest = {
                "source_database": str(store.db_path),
                "anima": args.anima,
                "tasks": count,
                "ready_descriptors": len(list((snapshot_anima / "state/pending").glob("*.json"))),
                "claims_remain_paused": True,
                "complete": True,
                "rollback_policy": "Use this current export, never an older database backup, for legacy rollback.",
                "scope": "task state only; identity, configuration and business artifacts remain in the source runtime",
            }
            from core.platform.atomic_io import atomic_write_text

            atomic_write_text(destination / "manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
            return manifest
    raise ValueError(action)


def cmd_task_store(args: argparse.Namespace) -> None:
    try:
        result = run_maintenance(args)
        command_migration = result.get("command_migration", {})
        for warning in command_migration.get("warnings", []):
            print(f"Warning: {warning}", file=sys.stderr)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except (OSError, ValueError, RuntimeError) as exc:
        print(t("task_store.error", error=str(exc)), file=sys.stderr)
        raise SystemExit(1) from exc


def register_task_store_command(sub: argparse._SubParsersAction) -> None:
    parser = sub.add_parser("task-store", help=t("task_store.help"))
    actions = parser.add_subparsers(dest="task_store_action", required=True)
    for action in ("status", "quiesce", "resume", "migrate", "backup", "export"):
        child = actions.add_parser(action, help=t(f"task_store.{action}_help"))
        child.add_argument("--anima", required=True, help=t("task_store.anima_help"))
        if action in {"backup", "export"}:
            child.add_argument("--destination", type=Path, required=True, help=t("task_store.destination_help"))
        if action == "migrate":
            child.add_argument("--backup", type=Path, required=True, help=t("task_store.backup_help"))
        child.set_defaults(func=cmd_task_store)
