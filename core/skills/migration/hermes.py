from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Hermes Agent dry-run-first migration importer."""

import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.memory.task_queue import TaskQueueManager
from core.skills.hub import SkillHub
from core.taskboard.store import TaskBoardStore
from core.time_utils import now_iso

from ._common import (
    append_import_lock,
    load_import_lock,
    provenance_header,
    read_json,
    redact_credentials,
    rel_to,
    safe_anima_name,
    source_fingerprint,
    write_backup_manifest,
    write_text_once,
)
from .hermes_format import (
    coerce_lock_entries,
    coerce_task_entries,
    convert_cron_skills_field,
    parse_hermes_usage,
    task_status,
)
from .report import MigrationItem, MigrationReport


@dataclass(slots=True)
class HermesImportOptions:
    source_path: Path
    data_dir: Path
    target_anima: str | None = None
    common_skills: bool = False
    apply: bool = False
    replace: bool = False


def import_hermes(options: HermesImportOptions) -> MigrationReport:
    source = options.source_path.expanduser().resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"Hermes source path not found: {source}")
    if not options.common_skills:
        safe_anima_name(options.target_anima)

    batch_id = f"hermes_{now_iso().replace(':', '').replace('-', '').replace('+', '_')}"
    report = MigrationReport(
        source_system="hermes",
        source_path=str(source),
        mode="apply" if options.apply else "dry_run",
        batch_id=batch_id,
        target_anima=options.target_anima,
        common_skills=options.common_skills,
    )
    migration_dir = _migration_dir(options)
    import_lock_path = migration_dir / "import_lock.jsonl"
    seen = load_import_lock(import_lock_path)

    if options.apply:
        backup_targets = _planned_backup_targets(source, options, migration_dir)
        backup_path = migration_dir / f"{batch_id}_backup_manifest.json"
        write_backup_manifest(backup_path, data_dir=options.data_dir, targets=backup_targets, batch_id=batch_id)
        report.backup_manifest_path = rel_to(backup_path, options.data_dir)

    _migrate_skills(source, options, report, import_lock_path, seen)
    _migrate_usage(source, options, report, import_lock_path, seen)
    _migrate_hub_lock(source, options, report, import_lock_path, seen)
    _migrate_memory_and_profile(source, options, report, import_lock_path, seen)
    _migrate_cron(source, options, report, import_lock_path, seen)
    _migrate_tasks(source, options, report, import_lock_path, seen)

    report.add_warning("Hermes .usage.json is sparse; missing usage is not treated as immediate non-use.")
    if options.apply:
        report_path = migration_dir / f"{batch_id}_import.md"
        report.write_markdown(report_path)
        report.report_path = rel_to(report_path, options.data_dir)
    return report


def _migrate_skills(
    source: Path,
    options: HermesImportOptions,
    report: MigrationReport,
    import_lock_path: Path,
    seen: set[str],
) -> None:
    skills_dir = source / "skills"
    if not skills_dir.is_dir():
        return
    for skill_dir in sorted(p for p in skills_dir.iterdir() if p.is_dir() and not p.name.startswith(".")):
        if not (skill_dir / "SKILL.md").is_file():
            continue
        target = "common" if options.common_skills else "personal"
        target_path = _skill_target_path(skill_dir.name, options)
        fp = source_fingerprint("hermes_skill", skill_dir, target_path)
        if fp in seen and not options.replace:
            report.add_item(
                MigrationItem(
                    "hermes_skill", str(skill_dir), target_path, "skill_import", "skipped", fp, "already imported"
                )
            )
            continue

        try:
            hub_result = _run_hub_install(skill_dir, options, target=target)
        except Exception as exc:
            report.add_item(
                MigrationItem(
                    "hermes_skill",
                    str(skill_dir),
                    target_path,
                    "skill_import",
                    "error",
                    fp,
                    f"{type(exc).__name__}: {exc}",
                    manual_action="review failed skill import",
                )
            )
            continue
        item_status = _skill_status(hub_result.status, options.apply)
        item = MigrationItem(
            "hermes_skill",
            str(skill_dir),
            hub_result.installed_path or target_path,
            "skill_import",
            item_status,
            fp,
            hub_result.message,
            scan_verdict=hub_result.scan_verdict,
            manual_action="review skill approval requirement" if hub_result.status == "approval_required" else "",
            metadata=hub_result.to_dict(),
        )
        report.add_item(item)
        if options.apply and hub_result.status in {"installed", "quarantine"}:
            append_import_lock(
                import_lock_path,
                fingerprint=fp,
                source_system="hermes",
                action="skill_import",
                target_path=item.target_path,
                batch_id=report.batch_id,
            )
            seen.add(fp)


def _migrate_usage(
    source: Path,
    options: HermesImportOptions,
    report: MigrationReport,
    import_lock_path: Path,
    seen: set[str],
) -> None:
    usage_path = source / "skills" / ".usage.json"
    if not usage_path.is_file():
        return
    if not options.target_anima:
        report.add_item(
            MigrationItem(
                "hermes_usage",
                str(usage_path),
                "(no target anima)",
                "usage_import",
                "skipped",
                message="--target-anima is required to import per-anima usage",
            )
        )
        return
    anima_dir = _anima_dir(options)
    target_path = rel_to(anima_dir / "state" / "skill_usage.jsonl", options.data_dir)
    events, source_skill_names = parse_hermes_usage(usage_path, import_time=report.generated_at)
    source_skill_dirs = _hermes_skill_names(source)
    missing = sorted(source_skill_dirs - source_skill_names)
    if missing:
        report.add_warning(f"usage_missing_from_source:{len(missing)} skills")

    for index, event in enumerate(events):
        fp = source_fingerprint(
            "hermes_usage", usage_path, target_path, extra=f"{index}:{event['skill_name']}:{event['event_type']}"
        )
        if fp in seen and not options.replace:
            report.add_item(MigrationItem("hermes_usage", str(usage_path), target_path, "usage_import", "skipped", fp))
            continue
        payload = {
            **event,
            "source_system": "hermes",
            "source_usage_completeness": "sparse",
            "import_batch_id": report.batch_id,
            "source_fingerprint": fp,
        }
        if options.apply:
            _append_jsonl(anima_dir / "state" / "skill_usage.jsonl", payload)
            append_import_lock(
                import_lock_path,
                fingerprint=fp,
                source_system="hermes",
                action="usage_import",
                target_path=target_path,
                batch_id=report.batch_id,
            )
            seen.add(fp)
        report.add_item(
            MigrationItem(
                "hermes_usage",
                str(usage_path),
                target_path,
                "usage_import",
                "applied" if options.apply else "planned",
                fp,
                metadata={"skill_name": event["skill_name"], "event_type": event["event_type"]},
            )
        )


def _migrate_hub_lock(
    source: Path,
    options: HermesImportOptions,
    report: MigrationReport,
    import_lock_path: Path,
    seen: set[str],
) -> None:
    lock_path = source / "skills" / ".hub" / "lock.json"
    if not lock_path.is_file():
        return
    target_path = rel_to(_skill_hub_lock_path(options), options.data_dir)
    for index, raw in enumerate(coerce_lock_entries(read_json(lock_path))):
        skill_name = str(raw.get("skill_name") or raw.get("name") or raw.get("id") or "unknown")
        fp = source_fingerprint("hermes_hub_lock", lock_path, target_path, extra=f"{index}:{skill_name}")
        if fp in seen and not options.replace:
            report.add_item(
                MigrationItem("hermes_hub_lock", str(lock_path), target_path, "hub_lock_import", "skipped", fp)
            )
            continue
        entry = {
            "ts": raw.get("ts") or raw.get("created_at") or now_iso(),
            "action": "hermes_lock_import",
            "skill_name": skill_name,
            "target": "common" if options.common_skills else "personal",
            "source_type": raw.get("source_type") or raw.get("type") or "hermes",
            "source_identifier": raw.get("source_identifier") or raw.get("source") or str(lock_path),
            "resolved_commit": raw.get("resolved_commit") or raw.get("commit"),
            "scan_verdict": raw.get("scan_verdict") or raw.get("verdict"),
            "installed_path": raw.get("installed_path") or raw.get("path"),
            "actor": "migration",
            "reason": f"imported_from_hermes batch={report.batch_id}",
            "source_fingerprint": fp,
        }
        if options.apply:
            _append_jsonl(_skill_hub_lock_path(options), entry)
            append_import_lock(
                import_lock_path,
                fingerprint=fp,
                source_system="hermes",
                action="hub_lock_import",
                target_path=target_path,
                batch_id=report.batch_id,
            )
            seen.add(fp)
        report.add_item(
            MigrationItem(
                "hermes_hub_lock",
                str(lock_path),
                target_path,
                "hub_lock_import",
                "applied" if options.apply else "planned",
                fp,
                metadata={"skill_name": skill_name},
            )
        )


def _migrate_memory_and_profile(
    source: Path,
    options: HermesImportOptions,
    report: MigrationReport,
    import_lock_path: Path,
    seen: set[str],
) -> None:
    draft_dir = _migration_dir(options) / "drafts"
    for memory_root_name in ("memory", "memories", "knowledge"):
        memory_root = source / memory_root_name
        if not memory_root.is_dir():
            continue
        for path in sorted(p for p in memory_root.rglob("*") if p.is_file()):
            target = draft_dir / f"hermes_{memory_root_name}_{path.stem}.md"
            _write_draft_item(path, target, "memory_draft", options, report, import_lock_path, seen)

    for profile_name in ("profile.md", "user.md", "user_profile.md"):
        profile = source / profile_name
        if profile.is_file():
            target = options.data_dir / "shared" / "users" / "drafts" / f"{options.target_anima or 'hermes'}_profile.md"
            _write_draft_item(profile, target, "user_profile_draft", options, report, import_lock_path, seen)


def _migrate_cron(
    source: Path,
    options: HermesImportOptions,
    report: MigrationReport,
    import_lock_path: Path,
    seen: set[str],
) -> None:
    cron = source / "cron.md"
    if not cron.is_file():
        return
    target = _migration_dir(options) / "proposals" / "hermes_cron_patch.md"
    body = provenance_header("hermes", cron, report.batch_id) + "# Hermes Cron Patch Proposal\n\n"
    body += convert_cron_skills_field(redact_credentials(cron.read_text(encoding="utf-8")))
    _write_text_item(cron, target, body, "cron_patch_proposal", options, report, import_lock_path, seen)


def _migrate_tasks(
    source: Path,
    options: HermesImportOptions,
    report: MigrationReport,
    import_lock_path: Path,
    seen: set[str],
) -> None:
    if not options.target_anima:
        return
    for name in ("kanban.json", "tasks.json"):
        path = source / name
        if not path.is_file():
            continue
        tasks = coerce_task_entries(read_json(path))
        for index, task in enumerate(tasks):
            summary = str(task.get("summary") or task.get("title") or task.get("name") or f"Hermes task {index + 1}")
            task_id = f"hermes_{source_fingerprint('hermes_task_id', path, summary, extra=str(index))[:12]}"
            target_path = rel_to(_anima_dir(options) / "state" / "task_queue.jsonl", options.data_dir)
            fp = source_fingerprint("hermes_task", path, target_path, extra=f"{index}:{summary}")
            if fp in seen and not options.replace:
                report.add_item(MigrationItem("hermes_task", str(path), target_path, "taskboard_import", "skipped", fp))
                continue
            if options.apply:
                entry = TaskQueueManager(_anima_dir(options)).add_task(
                    source="anima",
                    original_instruction=str(task.get("description") or summary),
                    assignee=options.target_anima,
                    summary=summary,
                    task_id=task_id,
                    status=task_status(str(task.get("status") or "pending")),
                    meta={"source_system": "hermes", "import_batch_id": report.batch_id, "source_fingerprint": fp},
                )
                TaskBoardStore(options.data_dir / "shared" / "taskboard.sqlite3").upsert_metadata(
                    anima_name=options.target_anima,
                    task_id=entry.task_id,
                    actor="migration",
                    source_ref=f"hermes://{path.name}#{index}",
                )
                append_import_lock(
                    import_lock_path,
                    fingerprint=fp,
                    source_system="hermes",
                    action="taskboard_import",
                    target_path=target_path,
                    batch_id=report.batch_id,
                )
                seen.add(fp)
            report.add_item(
                MigrationItem(
                    "hermes_task",
                    str(path),
                    target_path,
                    "taskboard_import",
                    "applied" if options.apply else "planned",
                    fp,
                    metadata={"task_id": task_id, "summary": summary},
                )
            )


def _run_hub_install(skill_dir: Path, options: HermesImportOptions, *, target: str):
    if options.apply:
        hub = SkillHub(data_dir=options.data_dir, actor="migration")
        return hub.install(
            str(skill_dir),
            target=target,
            anima=options.target_anima,
            dry_run=False,
            replace=options.replace,
        )
    with tempfile.TemporaryDirectory(prefix="animaworks-hermes-dry-run-") as tmp:
        hub = SkillHub(data_dir=Path(tmp), actor="migration")
        return hub.install(
            str(skill_dir),
            target=target,
            anima=options.target_anima,
            dry_run=True,
            replace=options.replace,
        )


def _write_draft_item(
    source_path: Path,
    target: Path,
    action: str,
    options: HermesImportOptions,
    report: MigrationReport,
    import_lock_path: Path,
    seen: set[str],
) -> None:
    body = provenance_header("hermes", source_path, report.batch_id) + redact_credentials(
        source_path.read_text(encoding="utf-8")
    )
    _write_text_item(source_path, target, body, action, options, report, import_lock_path, seen)


def _write_text_item(
    source_path: Path,
    target: Path,
    body: str,
    action: str,
    options: HermesImportOptions,
    report: MigrationReport,
    import_lock_path: Path,
    seen: set[str],
) -> None:
    target_path = rel_to(target, options.data_dir)
    fp = source_fingerprint(f"hermes_{action}", source_path, target_path)
    if fp in seen and not options.replace:
        report.add_item(MigrationItem(f"hermes_{action}", str(source_path), target_path, action, "skipped", fp))
        return
    status = "planned"
    if options.apply:
        status = "applied" if write_text_once(target, body, replace=options.replace) else "skipped"
        if status == "applied":
            append_import_lock(
                import_lock_path,
                fingerprint=fp,
                source_system="hermes",
                action=action,
                target_path=target_path,
                batch_id=report.batch_id,
            )
            seen.add(fp)
    report.add_item(MigrationItem(f"hermes_{action}", str(source_path), target_path, action, status, fp))


def _planned_backup_targets(source: Path, options: HermesImportOptions, migration_dir: Path) -> list[Path]:
    targets = [migration_dir / "import_lock.jsonl", migration_dir / "drafts", migration_dir / "proposals"]
    if options.target_anima:
        targets.extend(
            [
                _anima_dir(options) / "state" / "skill_usage.jsonl",
                _anima_dir(options) / "state" / "task_queue.jsonl",
                _anima_dir(options) / "state" / "skill_hub_lock.jsonl",
            ]
        )
    if options.common_skills:
        targets.append(options.data_dir / "shared" / "skill_hub_lock.jsonl")
    if (source / "profile.md").exists():
        targets.append(options.data_dir / "shared" / "users" / "drafts")
    return targets


def _migration_dir(options: HermesImportOptions) -> Path:
    if options.target_anima:
        return _anima_dir(options) / "state" / "migrations"
    return options.data_dir / "shared" / "migrations"


def _anima_dir(options: HermesImportOptions) -> Path:
    return options.data_dir / "animas" / safe_anima_name(options.target_anima)


def _skill_hub_lock_path(options: HermesImportOptions) -> Path:
    if options.common_skills:
        return options.data_dir / "shared" / "skill_hub_lock.jsonl"
    return _anima_dir(options) / "state" / "skill_hub_lock.jsonl"


def _skill_target_path(source_name: str, options: HermesImportOptions) -> str:
    if options.common_skills:
        return f"common_skills/community/{source_name}/SKILL.md"
    return f"animas/{safe_anima_name(options.target_anima)}/skills/{source_name}/SKILL.md"


def _skill_status(hub_status: str, apply_mode: bool) -> str:
    if hub_status in {"blocked", "approval_required"}:
        return hub_status
    return "applied" if apply_mode and hub_status in {"installed", "quarantine"} else "planned"


def _hermes_skill_names(source: Path) -> set[str]:
    skills_dir = source / "skills"
    if not skills_dir.is_dir():
        return set()
    return {
        p.name for p in skills_dir.iterdir() if p.is_dir() and not p.name.startswith(".") and (p / "SKILL.md").is_file()
    }


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
