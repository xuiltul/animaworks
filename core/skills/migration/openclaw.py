from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""OpenClaw dry-run-first migration importer."""

from dataclasses import dataclass
from pathlib import Path

from core.time_utils import now_iso

from ._common import (
    append_import_lock,
    detect_redacted_credentials,
    load_import_lock,
    provenance_header,
    redact_credentials,
    rel_to,
    safe_anima_name,
    source_fingerprint,
    write_backup_manifest,
    write_text_once,
)
from .report import MigrationItem, MigrationReport


@dataclass(slots=True)
class OpenClawImportOptions:
    source_path: Path
    data_dir: Path
    target_anima: str
    apply: bool = False
    replace: bool = False


def import_openclaw(options: OpenClawImportOptions) -> MigrationReport:
    source = options.source_path.expanduser().resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"OpenClaw source path not found: {source}")
    safe_anima_name(options.target_anima)

    batch_id = f"openclaw_{now_iso().replace(':', '').replace('-', '').replace('+', '_')}"
    report = MigrationReport(
        source_system="openclaw",
        source_path=str(source),
        mode="apply" if options.apply else "dry_run",
        batch_id=batch_id,
        target_anima=options.target_anima,
    )
    migration_dir = _migration_dir(options)
    import_lock_path = migration_dir / "import_lock.jsonl"
    seen = load_import_lock(import_lock_path)

    for warning in detect_redacted_credentials(source):
        report.add_warning(warning)
    if report.warnings:
        report.add_warning("credentials are never imported automatically")

    if options.apply:
        backup_path = migration_dir / f"{batch_id}_backup_manifest.json"
        write_backup_manifest(
            backup_path,
            data_dir=options.data_dir,
            targets=[
                _anima_dir(options) / "identity.md",
                _anima_dir(options) / "injection.md",
                _anima_dir(options) / "permissions.md",
                _anima_dir(options) / "permissions.json",
                migration_dir / "drafts",
                migration_dir / "proposals",
                migration_dir / "import_lock.jsonl",
            ],
            batch_id=batch_id,
        )
        report.backup_manifest_path = rel_to(backup_path, options.data_dir)

    _draft_soul(source, options, report, import_lock_path, seen)
    _draft_permissions(source, options, report, import_lock_path, seen)
    _draft_memory(source, options, report, import_lock_path, seen)
    _propose_tasks(source, options, report, import_lock_path, seen)

    if options.apply:
        report_path = migration_dir / f"{batch_id}_import.md"
        report.write_markdown(report_path)
        report.report_path = rel_to(report_path, options.data_dir)
    return report


def _draft_soul(
    source: Path,
    options: OpenClawImportOptions,
    report: MigrationReport,
    import_lock_path: Path,
    seen: set[str],
) -> None:
    soul = source / "SOUL.md"
    if not soul.is_file():
        return
    body = (
        provenance_header("openclaw", soul, report.batch_id)
        + "# OpenClaw Identity / Injection Draft\n\n"
        + "This draft is not applied to identity.md or injection.md automatically.\n\n"
        + redact_credentials(soul.read_text(encoding="utf-8"))
    )
    target = _migration_dir(options) / "drafts" / "openclaw_SOUL_identity_injection_draft.md"
    _write_item(soul, target, body, "identity_injection_draft", options, report, import_lock_path, seen)


def _draft_permissions(
    source: Path,
    options: OpenClawImportOptions,
    report: MigrationReport,
    import_lock_path: Path,
    seen: set[str],
) -> None:
    for name in ("permissions.md", "permissions.json", "allowlist.json", "settings.json"):
        path = source / name
        if not path.is_file():
            continue
        body = (
            provenance_header("openclaw", path, report.batch_id)
            + "# OpenClaw Permissions Draft\n\n"
            + "This draft is not applied to permissions.md or permissions.json automatically.\n\n"
            + redact_credentials(path.read_text(encoding="utf-8"))
        )
        target = _migration_dir(options) / "drafts" / f"openclaw_{path.stem}_permissions_draft.md"
        _write_item(path, target, body, "permissions_draft", options, report, import_lock_path, seen)


def _draft_memory(
    source: Path,
    options: OpenClawImportOptions,
    report: MigrationReport,
    import_lock_path: Path,
    seen: set[str],
) -> None:
    for dirname in ("memory", "memories", "knowledge"):
        root = source / dirname
        if not root.is_dir():
            continue
        for path in sorted(p for p in root.rglob("*") if p.is_file()):
            body = provenance_header("openclaw", path, report.batch_id) + redact_credentials(
                path.read_text(encoding="utf-8")
            )
            target = _migration_dir(options) / "drafts" / f"openclaw_{dirname}_{path.stem}.md"
            _write_item(path, target, body, "memory_draft", options, report, import_lock_path, seen)


def _propose_tasks(
    source: Path,
    options: OpenClawImportOptions,
    report: MigrationReport,
    import_lock_path: Path,
    seen: set[str],
) -> None:
    task_files = [source / "tasks.md", source / "tasks.json"]
    existing = [path for path in task_files if path.is_file()]
    if not existing:
        return
    target = _migration_dir(options) / "proposals" / "openclaw_taskboard_import_proposal.md"
    body = "# OpenClaw TaskBoard Import Proposal\n\n"
    body += "Tasks are proposed for TaskBoard import only; no legacy task_queue JSONL extension is used.\n\n"
    for path in existing:
        body += provenance_header("openclaw", path, report.batch_id)
        body += redact_credentials(path.read_text(encoding="utf-8")) + "\n\n"
    source_for_fingerprint = existing[0]
    _write_item(
        source_for_fingerprint, target, body, "taskboard_import_proposal", options, report, import_lock_path, seen
    )


def _write_item(
    source_path: Path,
    target: Path,
    body: str,
    action: str,
    options: OpenClawImportOptions,
    report: MigrationReport,
    import_lock_path: Path,
    seen: set[str],
) -> None:
    target_path = rel_to(target, options.data_dir)
    fp = source_fingerprint(f"openclaw_{action}", source_path, target_path)
    if fp in seen and not options.replace:
        report.add_item(MigrationItem(f"openclaw_{action}", str(source_path), target_path, action, "skipped", fp))
        return
    status = "planned"
    if options.apply:
        status = "applied" if write_text_once(target, body, replace=options.replace) else "skipped"
        if status == "applied":
            append_import_lock(
                import_lock_path,
                fingerprint=fp,
                source_system="openclaw",
                action=action,
                target_path=target_path,
                batch_id=report.batch_id,
            )
            seen.add(fp)
    report.add_item(
        MigrationItem(
            f"openclaw_{action}",
            str(source_path),
            target_path,
            action,
            status,
            fp,
            manual_action="review draft and apply manually if desired",
        )
    )


def _migration_dir(options: OpenClawImportOptions) -> Path:
    return _anima_dir(options) / "state" / "migrations"


def _anima_dir(options: OpenClawImportOptions) -> Path:
    return options.data_dir / "animas" / safe_anima_name(options.target_anima)
