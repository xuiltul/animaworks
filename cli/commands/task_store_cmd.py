"""Operator-only, cohort-scoped task migration and current-state export."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from core.i18n import t


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


def run_maintenance(args: argparse.Namespace) -> dict[str, Any]:
    from core.paths import get_data_dir
    from core.taskboard.tasks import TaskStore, task_database_path

    runtime = get_data_dir().resolve()
    anima_dir = (runtime / "animas" / args.anima).resolve()
    if anima_dir.parent != runtime / "animas" or not (anima_dir / "identity.md").is_file():
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
            store.backup(args.backup)
            with store.transaction():
                report = store.import_legacy(anima_dir)
                if report["invalid_rows"]:
                    raise ValueError(t("task_store.invalid_rows", count=report["invalid_rows"]))
            return {"migration": report, "backup": str(args.backup.resolve()), **store.maintenance_status(args.anima)}
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
            from core.memory._io import atomic_write_text

            atomic_write_text(destination / "manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
            return manifest
    raise ValueError(action)


def cmd_task_store(args: argparse.Namespace) -> None:
    try:
        print(json.dumps(run_maintenance(args), ensure_ascii=False, indent=2))
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
