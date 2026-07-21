from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Unified housekeeping engine for periodic disk cleanup.

Runs as a single daily job from ProcessSupervisor's scheduler, cleaning up
all data types that lack their own rotation mechanisms.
"""

import asyncio
import errno
import logging
import os
import re
import shutil
import subprocess
from collections.abc import Iterable
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

from core.i18n import t
from core.time_utils import now_local, today_local

logger = logging.getLogger("animaworks.housekeeping")
_PRESERVED_TMP_SUBDIRS = frozenset({"attachments", "skill_hub"})
_ANIMA_LOG_DATE_RE = re.compile(r"(20\d{6})")
_CODEX_LOG_DB_NAME = "logs_2.sqlite"
_PROTECTED_PREFIXES = ("current_session_", "streaming_journal")


# ── Public API ──────────────────────────────────────────────────


async def run_housekeeping(
    data_dir: Path,
    *,
    prompt_log_retention_days: int = 3,
    daemon_log_max_size_mb: int = 50,
    daemon_log_keep_generations: int = 5,
    anima_log_retention_days: int = 30,
    anima_log_total_max_size_mb: int = 200,
    frontend_log_backup_count: int = 7,
    dm_log_archive_retention_days: int = 30,
    cron_log_retention_days: int = 30,
    shortterm_retention_days: int = 7,
    shortterm_archive_retention_days: int = 30,
    shortterm_thread_gc_days: int = 30,
    facts_lock_stale_hours: int = 24,
    curator_report_retention_days: int = 30,
    task_results_retention_days: int = 7,
    pending_failed_retention_days: int = 14,
    corrupt_vectordb_keep_generations: int = 2,
    tmp_retention_days: int = 14,
    backup_retention_days: int = 90,
    codex_log_max_size_mb: int = 200,
    codex_tmp_retention_hours: int = 12,
    anima_tmp_gitdirs_retention_days: int = 14,
    anima_local_log_retention_days: int = 30,
    pending_processing_stale_hours: int = 24,
    background_running_stale_hours: int = 48,
    current_state_stale_hours: int = 24,
    taskboard_suppressed_retention_days: int = 30,
    taskboard_orphan_metadata_stale_hours: int = 24,
    suppressed_messages_max_size_mb: int = 10,
    suppressed_messages_keep_generations: int = 5,
    archive_superseded_retention_days: int = 7,
    hygiene_grace_days: int = 21,
    inbox_ttl_hours: float = 24.0,
    inbox_expired_retention_days: int = 7,
    inbox_processed_retention_days: int = 30,
    inbox_quarantine_retention_days: int = 30,
) -> dict[str, Any]:
    """Run all housekeeping tasks. Returns summary of actions taken."""
    loop = asyncio.get_running_loop()
    results: dict[str, Any] = {}

    animas_dir = data_dir / "animas"

    # Memory hygiene scan and stale semantic-cleanup fallback
    try:
        r = await loop.run_in_executor(
            None,
            _archive_stale_merge_leftovers,
            animas_dir,
            hygiene_grace_days,
        )
        results["memory_hygiene"] = r
    except Exception:
        logger.exception("Housekeeping: memory hygiene fallback failed")
        results["memory_hygiene"] = {"error": True}

    # 1. Prompt logs
    try:
        r = await loop.run_in_executor(
            None,
            _rotate_prompt_logs_all,
            animas_dir,
            prompt_log_retention_days,
        )
        results["prompt_logs"] = r
    except Exception:
        logger.exception("Housekeeping: prompt_logs rotation failed")
        results["prompt_logs"] = {"error": True}

    # 2. Daemon log
    try:
        r = await loop.run_in_executor(
            None,
            _rotate_daemon_log,
            data_dir / "logs" / "server-daemon.log",
            daemon_log_max_size_mb,
            daemon_log_keep_generations,
        )
        results["daemon_log"] = r
    except Exception:
        logger.exception("Housekeeping: daemon_log rotation failed")
        results["daemon_log"] = {"error": True}

    try:
        r = await loop.run_in_executor(
            None,
            _rotate_daemon_log,
            data_dir / "logs" / "vector-worker.log",
            daemon_log_max_size_mb,
            daemon_log_keep_generations,
        )
        results["vector_worker_log"] = r
    except Exception:
        logger.exception("Housekeeping: vector worker log rotation failed")
        results["vector_worker_log"] = {"error": True}

    # 2b. Per-Anima runtime logs
    try:
        r = await loop.run_in_executor(
            None,
            _cleanup_anima_runtime_logs,
            data_dir / "logs" / "animas",
            anima_log_retention_days,
            anima_log_total_max_size_mb,
        )
        results["anima_logs"] = r
    except Exception:
        logger.exception("Housekeeping: anima runtime log cleanup failed")
        results["anima_logs"] = {"error": True}

    try:
        r = await loop.run_in_executor(
            None,
            _cleanup_frontend_logs,
            data_dir / "logs" / "frontend",
            frontend_log_backup_count,
        )
        results["frontend_logs"] = r
    except Exception:
        logger.exception("Housekeeping: frontend log cleanup failed")
        results["frontend_logs"] = {"error": True}

    # 3. DM archives
    try:
        r = await loop.run_in_executor(
            None,
            _cleanup_dm_archives,
            data_dir / "shared" / "dm_logs",
            dm_log_archive_retention_days,
        )
        results["dm_archives"] = r
    except Exception:
        logger.exception("Housekeeping: dm_archives cleanup failed")
        results["dm_archives"] = {"error": True}

    # 4. Cron logs
    try:
        r = await loop.run_in_executor(
            None,
            _cleanup_cron_logs,
            animas_dir,
            cron_log_retention_days,
        )
        results["cron_logs"] = r
    except Exception:
        logger.exception("Housekeeping: cron_logs cleanup failed")
        results["cron_logs"] = {"error": True}

    # 5. Shortterm
    try:
        r = await loop.run_in_executor(
            None,
            _cleanup_shortterm,
            animas_dir,
            shortterm_retention_days,
            shortterm_archive_retention_days,
            shortterm_thread_gc_days,
        )
        results["shortterm"] = r
    except Exception:
        logger.exception("Housekeeping: shortterm cleanup failed")
        results["shortterm"] = {"error": True}

    # 5b. Stale facts lock files
    try:
        r = await loop.run_in_executor(
            None,
            _cleanup_facts_locks,
            animas_dir,
            facts_lock_stale_hours,
        )
        results["facts_locks"] = r
    except Exception:
        logger.exception("Housekeeping: facts lock cleanup failed")
        results["facts_locks"] = {"error": True}

    # 5c. Skill Curator reports
    try:
        r = await loop.run_in_executor(
            None,
            _cleanup_curator_reports,
            animas_dir,
            curator_report_retention_days,
        )
        results["curator_reports"] = r
    except Exception:
        logger.exception("Housekeeping: curator report cleanup failed")
        results["curator_reports"] = {"error": True}

    # 6. Task results
    try:
        r = await loop.run_in_executor(
            None,
            _cleanup_task_results,
            animas_dir,
            task_results_retention_days,
        )
        results["task_results"] = r
    except Exception:
        logger.exception("Housekeeping: task_results cleanup failed")
        results["task_results"] = {"error": True}

    # 7. Pending failed tasks
    try:
        r = await loop.run_in_executor(
            None,
            _cleanup_pending_failed,
            animas_dir,
            pending_failed_retention_days,
        )
        results["pending_failed"] = r
    except Exception:
        logger.exception("Housekeeping: pending_failed cleanup failed")
        results["pending_failed"] = {"error": True}

    # 8. Runtime bloat retention
    try:
        r = await loop.run_in_executor(
            None,
            _cleanup_corrupt_vectordb_archives,
            animas_dir,
            corrupt_vectordb_keep_generations,
        )
        results["corrupt_vectordb_archives"] = r
    except Exception:
        logger.exception("Housekeeping: corrupt vectordb archive cleanup failed")
        results["corrupt_vectordb_archives"] = {"error": True}

    try:
        r = await loop.run_in_executor(None, _cleanup_runtime_tmp, data_dir / "tmp", tmp_retention_days)
        results["runtime_tmp"] = r
    except Exception:
        logger.exception("Housekeeping: runtime tmp cleanup failed")
        results["runtime_tmp"] = {"error": True}

    try:
        r = await loop.run_in_executor(None, _cleanup_backup_dirs, animas_dir, backup_retention_days)
        results["backup_dirs"] = r
    except Exception:
        logger.exception("Housekeeping: backup dir cleanup failed")
        results["backup_dirs"] = {"error": True}

    try:
        r = await loop.run_in_executor(None, _cleanup_codex_execution_logs, animas_dir, codex_log_max_size_mb)
        results["codex_execution_logs"] = r
    except Exception:
        logger.exception("Housekeeping: Codex execution log cleanup failed")
        results["codex_execution_logs"] = {"error": True}

    try:
        r = await loop.run_in_executor(None, _cleanup_codex_tmp_dirs, animas_dir, codex_tmp_retention_hours)
        results["codex_tmp"] = r
    except Exception:
        logger.exception("Housekeeping: Codex tmp cleanup failed")
        results["codex_tmp"] = {"error": True}

    try:
        r = await loop.run_in_executor(
            None,
            _cleanup_anima_runtime_artifacts,
            animas_dir,
            anima_tmp_gitdirs_retention_days,
            anima_local_log_retention_days,
        )
        results["anima_runtime_artifacts"] = r
    except Exception:
        logger.exception("Housekeeping: Anima runtime artifact cleanup failed")
        results["anima_runtime_artifacts"] = {"error": True}

    try:
        from core.memory.taskboard_housekeeping import cleanup_taskboard_stale_artifacts

        r = await loop.run_in_executor(
            None,
            cleanup_taskboard_stale_artifacts,
            data_dir,
            pending_processing_stale_hours,
            background_running_stale_hours,
            current_state_stale_hours,
            taskboard_suppressed_retention_days,
            taskboard_orphan_metadata_stale_hours,
        )
        results["taskboard_stale"] = r
    except Exception:
        logger.exception("Housekeeping: taskboard stale cleanup failed")
        results["taskboard_stale"] = {"error": True}

    try:
        r = await loop.run_in_executor(
            None,
            _rotate_suppressed_message_logs,
            data_dir,
            suppressed_messages_max_size_mb,
            suppressed_messages_keep_generations,
        )
        results["suppressed_messages"] = r
    except Exception:
        logger.exception("Housekeeping: suppressed message log rotation failed")
        results["suppressed_messages"] = {"error": True}

    # 9. Archive/superseded rotation
    try:
        r = await loop.run_in_executor(
            None,
            _rotate_archive_superseded,
            animas_dir,
            archive_superseded_retention_days,
        )
        results["archive_superseded"] = r
    except Exception:
        logger.exception("Housekeeping: archive_superseded rotation failed")
        results["archive_superseded"] = {"error": True}

    # 10. Shared inbox stale-file cleanup
    try:
        r = await loop.run_in_executor(
            None,
            _cleanup_shared_inbox,
            data_dir / "shared" / "inbox",
            inbox_ttl_hours,
            inbox_expired_retention_days,
            inbox_processed_retention_days,
            inbox_quarantine_retention_days,
        )
        results["shared_inbox"] = r
    except Exception:
        logger.exception("Housekeeping: shared inbox cleanup failed")
        results["shared_inbox"] = {"error": True}

    # 11. Skill Curator report
    try:
        r = await loop.run_in_executor(None, _run_skill_curator_reports, animas_dir)
        results["skill_curator"] = r
    except Exception:
        logger.exception("Housekeeping: skill curator report failed")
        results["skill_curator"] = {"error": True}

    return results


# ── Sub-functions ───────────────────────────────────────────────


def _archive_stale_merge_leftovers(animas_dir: Path, hygiene_grace_days: int) -> dict[str, Any]:
    """Move stale merge leftovers to the canonical archive without deleting them."""
    if not animas_dir.is_dir():
        return {"skipped": True, "scanned_animas": 0, "moved_items": 0}

    from core.memory.hygiene import scan_memory_hygiene

    scanned_animas = 0
    moved_items = 0
    for anima_dir in sorted(path for path in animas_dir.iterdir() if path.is_dir()):
        try:
            report = scan_memory_hygiene(anima_dir)
            scanned_animas += 1
            moved_for_anima = _archive_stale_hygiene_entries(
                anima_dir,
                report,
                hygiene_grace_days,
            )
            moved_items += moved_for_anima
            if moved_for_anima:
                # Refresh all categories so entries nested below a moved inherited
                # directory also disappear while unaffected first_seen dates remain.
                scan_memory_hygiene(anima_dir)
        except Exception:
            logger.exception("Memory hygiene fallback failed for %s", anima_dir)

    return {"scanned_animas": scanned_animas, "moved_items": moved_items}


def _archive_stale_hygiene_entries(
    anima_dir: Path,
    report: dict[str, list[dict[str, Any]]],
    hygiene_grace_days: int,
) -> int:
    knowledge_dir = anima_dir / "knowledge"
    archive_dir = knowledge_dir / "archive" / "unmerged"
    moved = 0

    # Moving an inherited directory first prevents separately moving leftovers
    # contained inside that same directory.
    for category in ("inherited_dirs", "merged_leftovers"):
        for entry in report.get(category, []):
            if not _hygiene_entry_is_stale(entry, hygiene_grace_days):
                continue
            source = _validated_hygiene_source(anima_dir, knowledge_dir, entry, category)
            if source is None or not source.exists():
                continue
            try:
                relative = source.relative_to(knowledge_dir)
                destination_base = archive_dir / relative
                if not _prepare_hygiene_destination_parent(
                    knowledge_dir,
                    archive_dir,
                    destination_base.parent,
                ):
                    continue
                destination = _unique_hygiene_destination(destination_base)
                shutil.move(str(source), str(destination))
                moved += 1
                logger.info("Archived stale memory hygiene item: %s -> %s", source, destination)
            except (OSError, RuntimeError):
                logger.exception("Failed to archive stale memory hygiene item: %s", source)

    moved += _archive_stale_noncanonical_episodes(anima_dir, report, hygiene_grace_days)
    return moved


def _archive_stale_noncanonical_episodes(
    anima_dir: Path,
    report: dict[str, list[dict[str, Any]]],
    hygiene_grace_days: int,
) -> int:
    """Move stale non-canonical episode files into ``episodes/archive/``."""
    episodes_dir = anima_dir / "episodes"
    archive_dir = episodes_dir / "archive"
    moved = 0

    for entry in report.get("noncanonical_episodes", []):
        if not _hygiene_entry_is_stale(entry, hygiene_grace_days):
            continue
        source = _validated_noncanonical_episode_source(anima_dir, episodes_dir, entry)
        if source is None or not source.exists():
            continue
        try:
            archive_dir.mkdir(parents=True, exist_ok=True)
            if not _prepare_episode_archive_destination(episodes_dir, archive_dir):
                continue
            destination = _unique_hygiene_destination(archive_dir / source.name)
            relative_source = source.relative_to(anima_dir).as_posix()
            shutil.move(str(source), str(destination))
            moved += 1
            logger.info("Archived stale noncanonical episode: %s -> %s", source, destination)
            # Index deletion is best-effort; file move already succeeded.
            _delete_episode_index_entry(anima_dir, relative_source)
        except (OSError, RuntimeError):
            logger.exception("Failed to archive stale noncanonical episode: %s", source)
    return moved


def _validated_noncanonical_episode_source(
    anima_dir: Path,
    episodes_dir: Path,
    entry: dict[str, Any],
) -> Path | None:
    """Validate a hygiene entry path is a file directly under episodes/."""
    raw_path = entry.get("path")
    if not isinstance(raw_path, str):
        return None
    relative = Path(raw_path)
    if relative.is_absolute() or ".." in relative.parts:
        return None
    source = anima_dir / relative
    try:
        resolved = source.resolve()
        resolved.relative_to(episodes_dir.resolve())
    except (OSError, ValueError):
        return None
    # Only direct children of episodes/ (not archive/ or nested dirs).
    if source.parent != episodes_dir or not source.is_file():
        return None
    if source.name == "archive":
        return None
    return source


def _prepare_episode_archive_destination(episodes_dir: Path, archive_dir: Path) -> bool:
    """Validate ``episodes/archive`` is a real directory (no symlink escape)."""
    try:
        if archive_dir.is_symlink():
            logger.warning("Skipping episode archive through symlink: %s", archive_dir)
            return False
        archive_dir.mkdir(parents=True, exist_ok=True)
        if archive_dir.is_symlink():
            logger.warning("Skipping episode archive through symlink: %s", archive_dir)
            return False
        resolved_episodes = episodes_dir.resolve(strict=True)
        resolved_archive = archive_dir.resolve(strict=True)
        resolved_archive.relative_to(resolved_episodes)
    except (OSError, ValueError):
        logger.warning("Skipping unsafe episode archive destination: %s", archive_dir)
        return False
    return True


def _delete_episode_index_entry(anima_dir: Path, source_file: str) -> None:
    """Best-effort removal of vector-index chunks for an archived episode file.

    Failures are logged as warnings and never raise — the file has already
    been moved successfully.
    """
    anima_name = anima_dir.name
    collection_name = f"{anima_name}_episodes"
    try:
        from core.memory.rag.singleton import get_vector_store

        store = get_vector_store(anima_name)
        if store is None:
            return
        results = store.get_by_metadata(collection_name, {"source_file": source_file}, limit=10_000)
        ids = [result.document.id for result in results]
        if not ids:
            return
        if not store.delete_documents(collection_name, ids):
            logger.warning(
                "Failed to delete indexed episode chunks for %s/%s",
                collection_name,
                source_file,
            )
    except Exception:
        logger.warning(
            "Failed to delete indexed episode chunks for %s/%s",
            collection_name,
            source_file,
            exc_info=True,
        )


def _hygiene_entry_is_stale(entry: dict[str, Any], hygiene_grace_days: int) -> bool:
    first_seen = entry.get("first_seen")
    if not isinstance(first_seen, str):
        return False
    try:
        first_seen_date = date.fromisoformat(first_seen[:10])
    except ValueError:
        return False
    return (today_local() - first_seen_date).days > hygiene_grace_days


def _validated_hygiene_source(
    anima_dir: Path,
    knowledge_dir: Path,
    entry: dict[str, Any],
    category: str,
) -> Path | None:
    raw_path = entry.get("path")
    if not isinstance(raw_path, str):
        return None
    relative = Path(raw_path)
    if relative.is_absolute() or ".." in relative.parts:
        return None
    source = anima_dir / relative
    try:
        source.resolve().relative_to(knowledge_dir.resolve())
    except (OSError, ValueError):
        return None
    knowledge_relative = source.relative_to(knowledge_dir)
    if "archive" in knowledge_relative.parts:
        return None
    if category == "inherited_dirs":
        if source.parent != knowledge_dir or not source.name.startswith("inherited-") or not source.is_dir():
            return None
    elif not source.name.startswith("_merged_") or not source.is_file():
        return None
    return source


def _unique_hygiene_destination(destination: Path) -> Path:
    if not destination.exists():
        return destination
    stem = destination.stem
    suffix = destination.suffix
    for index in range(1, 10_000):
        candidate = destination.with_name(f"{stem}-{index}{suffix}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Could not allocate archive destination for {destination}")


def _prepare_hygiene_destination_parent(
    knowledge_dir: Path,
    archive_dir: Path,
    destination_parent: Path,
) -> bool:
    """Create and validate an archive parent without traversing symlinks."""
    try:
        relative_parent = destination_parent.relative_to(knowledge_dir)
        destination_parent.relative_to(archive_dir)
    except ValueError:
        logger.warning("Skipping memory hygiene archive outside unmerged directory: %s", destination_parent)
        return False

    current = knowledge_dir
    for part in relative_parent.parts:
        current /= part
        if current.is_symlink():
            logger.warning("Skipping memory hygiene archive through symlink: %s", current)
            return False

    destination_parent.mkdir(parents=True, exist_ok=True)

    # Recheck after creation and confirm the physical path remains within both
    # the canonical unmerged directory and the resolved knowledge directory.
    current = knowledge_dir
    for part in relative_parent.parts:
        current /= part
        if current.is_symlink():
            logger.warning("Skipping memory hygiene archive through symlink: %s", current)
            return False
    try:
        resolved_knowledge = knowledge_dir.resolve(strict=True)
        resolved_archive = archive_dir.resolve(strict=True)
        resolved_parent = destination_parent.resolve(strict=True)
        resolved_archive.relative_to(resolved_knowledge)
        resolved_parent.relative_to(resolved_archive)
    except (OSError, ValueError):
        logger.warning("Skipping unsafe memory hygiene archive destination: %s", destination_parent)
        return False
    return True


def _run_skill_curator_reports(animas_dir: Path) -> dict[str, Any]:
    """Generate daily deterministic Skill Curator reports for all Animas."""
    if not animas_dir.is_dir():
        return {"skipped": True}
    generated = 0
    for anima_dir in sorted(p for p in animas_dir.iterdir() if p.is_dir()):
        try:
            from core.paths import get_common_skills_dir
            from core.skills.curator import SkillCurator
            from core.skills.index import SkillIndex

            index = SkillIndex(
                anima_dir / "skills",
                get_common_skills_dir(),
                anima_dir / "procedures",
                anima_dir=anima_dir,
            )
            index.build_index()
            report = SkillCurator(anima_dir, common_skills_dir=get_common_skills_dir()).generate_report(
                index.search("", include_blocked=True)
            )
            report_dir = anima_dir / "state" / "skill_curator"
            report_dir.mkdir(parents=True, exist_ok=True)
            report_path = report_dir / f"report-{today_local().isoformat()}.json"
            import json

            report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
            generated += 1
        except Exception:
            logger.debug("Skill Curator report failed for %s", anima_dir, exc_info=True)
    return {"generated_reports": generated}


def _rotate_prompt_logs_all(
    animas_dir: Path,
    retention_days: int,
) -> dict[str, Any]:
    """Rotate prompt logs for all Animas."""
    from core._agent_prompt_log import rotate_all_prompt_logs

    if not animas_dir.exists():
        return {"skipped": True}
    result = rotate_all_prompt_logs(animas_dir, retention_days=retention_days)
    total = sum(result.values())
    if total:
        logger.info("Prompt log rotation: deleted %d files across %d animas", total, len(result))
    return {"deleted_files": total, "per_anima": result}


def _rotate_daemon_log(
    log_path: Path,
    max_size_mb: int,
    keep_generations: int,
) -> dict[str, Any]:
    """Size-based rotation for daemon logs using copytruncate strategy.

    Copies current log to .1, shifts .1 → .2, etc., then truncates the
    original path so live processes keep writing through their open fd.
    """
    if not log_path.exists():
        return {"skipped": True, "reason": "file_not_found"}

    parent = log_path.parent
    stem = log_path.name
    deleted = _delete_daemon_log_generations(parent, stem, keep_generations)

    size_mb = log_path.stat().st_size / (1024 * 1024)
    if size_mb < max_size_mb:
        return {
            "skipped": True,
            "current_size_mb": round(size_mb, 1),
            "deleted_generations": deleted,
        }

    # Shift existing generations: .N → .N+1 (highest first)
    for gen in range(keep_generations, 0, -1):
        src = parent / f"{stem}.{gen}"
        dst = parent / f"{stem}.{gen + 1}"
        if src.exists():
            if gen >= keep_generations:
                src.unlink()
            else:
                src.rename(dst)

    # Current → .1, then truncate in place for live redirected stdout/stderr fds.
    gen1 = parent / f"{stem}.1"
    shutil.copyfile(log_path, gen1)
    os.truncate(log_path, 0)

    # Delete over-limit generations
    deleted += _delete_daemon_log_generations(parent, stem, keep_generations)

    logger.info(
        "Daemon log rotated: %.1f MB → %s (deleted %d old generations)",
        size_mb,
        gen1.name,
        deleted,
    )
    return {"rotated": True, "size_mb": round(size_mb, 1), "deleted_generations": deleted}


def _delete_daemon_log_generations(parent: Path, stem: str, keep_generations: int) -> int:
    deleted = 0
    for gen in range(keep_generations + 1, keep_generations + 20):
        old = parent / f"{stem}.{gen}"
        if old.exists():
            old.unlink()
            deleted += 1
    return deleted


def _rotate_suppressed_message_logs(
    data_dir: Path,
    max_size_mb: int,
    keep_generations: int,
) -> dict[str, Any]:
    """Rotate legacy ``suppressed_messages.jsonl`` files by size."""
    if not data_dir.exists():
        return {"skipped": True}

    files = sorted(path for path in data_dir.rglob("suppressed_messages.jsonl") if path.is_file())
    rotated = 0
    skipped = 0
    deleted_generations = 0
    for path in files:
        result = _rotate_daemon_log(path, max_size_mb=max_size_mb, keep_generations=keep_generations)
        if result.get("rotated"):
            rotated += 1
        if result.get("skipped"):
            skipped += 1
        deleted_generations += int(result.get("deleted_generations", 0) or 0)

    return {
        "files": len(files),
        "rotated": rotated,
        "skipped": skipped,
        "deleted_generations": deleted_generations,
    }


def _cleanup_anima_runtime_logs(
    animas_log_dir: Path,
    retention_days: int,
    max_total_size_mb: int,
) -> dict[str, Any]:
    """Delete old per-Anima runtime logs and cap each Anima log directory.

    The active ``current.log`` target and ``stderr.log`` are preserved. This
    complements ``TimedRotatingFileHandler`` because Anima workers use a
    date-stamped base filename on restart, which can leave old files outside
    the handler's backup cleanup.
    """
    if not animas_log_dir.exists():
        return {"skipped": True}

    cutoff_date = today_local() - timedelta(days=retention_days)
    cutoff_ts = (now_local() - timedelta(days=retention_days)).timestamp()
    max_total_bytes = max_total_size_mb * 1024 * 1024
    open_paths = _collect_open_paths(animas_log_dir)

    deleted_files = 0
    capped_files = 0
    freed_bytes = 0
    skipped_open = 0
    per_anima: dict[str, dict[str, int]] = {}

    for anima_log_dir in sorted(p for p in animas_log_dir.iterdir() if p.is_dir()):
        protected = _protected_anima_log_paths(anima_log_dir)
        anima_deleted = 0
        anima_capped = 0
        anima_freed = 0

        for log_file in _iter_anima_log_files(anima_log_dir):
            if _is_protected_path(log_file, protected):
                continue
            try:
                if not _is_old_anima_log(log_file, cutoff_date, cutoff_ts):
                    continue
                if _path_has_open_files(log_file, open_paths):
                    skipped_open += 1
                    continue
                size = log_file.stat().st_size
                log_file.unlink()
                deleted_files += 1
                anima_deleted += 1
                freed_bytes += size
                anima_freed += size
            except OSError:
                logger.warning("Failed to delete old Anima runtime log: %s", log_file, exc_info=True)

        if max_total_bytes > 0:
            total_size = 0
            candidates: list[Path] = []
            for log_file in _iter_anima_log_files(anima_log_dir):
                try:
                    size = log_file.stat().st_size
                except OSError:
                    continue
                total_size += size
                if not _is_protected_path(log_file, protected):
                    candidates.append(log_file)

            for log_file in sorted(candidates, key=_anima_log_delete_sort_key):
                if total_size <= max_total_bytes:
                    break
                try:
                    if _path_has_open_files(log_file, open_paths):
                        skipped_open += 1
                        continue
                    size = log_file.stat().st_size
                    log_file.unlink()
                    total_size -= size
                    deleted_files += 1
                    capped_files += 1
                    anima_deleted += 1
                    anima_capped += 1
                    freed_bytes += size
                    anima_freed += size
                except OSError:
                    logger.warning("Failed to delete capped Anima runtime log: %s", log_file, exc_info=True)

        if anima_deleted:
            per_anima[anima_log_dir.name] = {
                "deleted_files": anima_deleted,
                "capped_files": anima_capped,
                "freed_bytes": anima_freed,
            }

    if deleted_files:
        logger.info(
            "Anima runtime log cleanup: deleted=%d capped=%d freed=%d bytes",
            deleted_files,
            capped_files,
            freed_bytes,
        )
    return {
        "deleted_files": deleted_files,
        "capped_files": capped_files,
        "freed_bytes": freed_bytes,
        "skipped_open": skipped_open,
        "per_anima": per_anima,
    }


def _iter_anima_log_files(anima_log_dir: Path) -> list[Path]:
    return [
        p
        for p in anima_log_dir.iterdir()
        if p.is_file() and not p.is_symlink() and p.name not in {"current.log", "stderr.log"}
    ]


def _protected_anima_log_paths(anima_log_dir: Path) -> set[Path]:
    protected: set[Path] = set()
    current_link = anima_log_dir / "current.log"
    stderr_log = anima_log_dir / "stderr.log"
    if stderr_log.exists():
        protected.add(stderr_log)
    try:
        if current_link.is_symlink():
            protected.add((anima_log_dir / current_link.readlink()).resolve(strict=False))
        elif current_link.is_file():
            target_name = current_link.read_text(encoding="utf-8").strip()
            if target_name:
                protected.add((anima_log_dir / target_name).resolve(strict=False))
    except OSError:
        logger.debug("Failed to resolve current Anima log link: %s", current_link, exc_info=True)
    return {p.resolve(strict=False) for p in protected}


def _is_protected_path(path: Path, protected: set[Path]) -> bool:
    return path.resolve(strict=False) in protected


def _is_old_anima_log(log_file: Path, cutoff_date: date, cutoff_ts: float) -> bool:
    match = _ANIMA_LOG_DATE_RE.search(log_file.name)
    if match:
        try:
            return datetime.strptime(match.group(1), "%Y%m%d").date() < cutoff_date
        except ValueError:
            pass
    return log_file.stat().st_mtime < cutoff_ts


def _anima_log_delete_sort_key(path: Path) -> tuple[float, str]:
    try:
        return (path.stat().st_mtime, path.name)
    except OSError:
        return (0.0, path.name)


def _cleanup_frontend_logs(frontend_log_dir: Path, backup_count: int) -> dict[str, Any]:
    """Keep only the newest rotated frontend log files."""
    if not frontend_log_dir.exists():
        return {"skipped": True}

    backups = [
        p
        for p in frontend_log_dir.iterdir()
        if p.is_file() and not p.is_symlink() and p.name != "frontend.jsonl" and _frontend_log_sort_key(p)[0]
    ]
    backups.sort(key=_frontend_log_sort_key, reverse=True)

    deleted_files = 0
    freed_bytes = 0
    open_paths = _collect_open_file_paths(backups[max(backup_count, 0) :])
    skipped_open = 0

    for log_file in backups[max(backup_count, 0) :]:
        try:
            if _path_has_open_files(log_file, open_paths):
                skipped_open += 1
                continue
            size = log_file.stat().st_size
            log_file.unlink()
            deleted_files += 1
            freed_bytes += size
        except OSError:
            logger.warning("Failed to delete frontend log backup: %s", log_file, exc_info=True)

    if deleted_files:
        logger.info("Frontend log cleanup: deleted=%d freed=%d bytes", deleted_files, freed_bytes)
    return {"deleted_files": deleted_files, "freed_bytes": freed_bytes, "skipped_open": skipped_open}


def _frontend_log_sort_key(path: Path) -> tuple[str, str]:
    if path.name.startswith("frontend.jsonl."):
        stamp = path.name.rsplit(".", 1)[-1]
        if re.fullmatch(r"\d{8}", stamp):
            return (stamp, path.name)
    if path.name.endswith(".jsonl") and re.fullmatch(r"\d{8}", path.stem):
        return (path.stem, path.name)
    return ("", path.name)


def _cleanup_shared_inbox(
    inbox_root: Path,
    ttl_hours: float,
    expired_retention_days: int,
    processed_retention_days: int,
    quarantine_retention_days: int,
) -> dict[str, Any]:
    """Sweep stale files and rotate archives under shared/inbox/<anima>/."""
    if not inbox_root.exists():
        return {"skipped": True}

    from core.messenger import Messenger

    shared_dir = inbox_root.parent
    totals: dict[str, Any] = {
        "animas": 0,
        "expired": 0,
        "protected": 0,
        "quarantined": 0,
        "deleted_expired": 0,
        "deleted_processed": 0,
        "deleted_quarantine": 0,
        "errors": 0,
    }
    per_anima: dict[str, dict[str, int]] = {}

    for inbox_dir in sorted(inbox_root.iterdir()):
        if not inbox_dir.is_dir() or inbox_dir.name.startswith("."):
            continue
        messenger = Messenger(shared_dir, inbox_dir.name)
        result = messenger.sweep_expired(
            ttl_hours=ttl_hours,
            expired_retention_days=expired_retention_days,
            processed_retention_days=processed_retention_days,
            quarantine_retention_days=quarantine_retention_days,
        )
        totals["animas"] += 1
        for key, value in result.items():
            totals[key] = totals.get(key, 0) + value
        if any(result.values()):
            per_anima[inbox_dir.name] = result

    if per_anima:
        logger.info("Shared inbox cleanup: %s", per_anima)
    totals["per_anima"] = per_anima
    return totals


def _cleanup_dm_archives(
    dm_logs_dir: Path,
    retention_days: int,
) -> dict[str, Any]:
    """Delete DM log archive files older than *retention_days*."""
    if not dm_logs_dir.exists():
        return {"skipped": True}

    cutoff = now_local() - timedelta(days=retention_days)
    cutoff_ts = cutoff.timestamp()
    deleted = 0

    for f in dm_logs_dir.glob("*.archive.jsonl"):
        try:
            if f.stat().st_mtime < cutoff_ts:
                f.unlink()
                deleted += 1
        except OSError:
            logger.warning("Failed to delete dm archive: %s", f)

    if deleted:
        logger.info("DM archive cleanup: deleted %d files", deleted)
    return {"deleted_files": deleted}


def _cleanup_cron_logs(
    animas_dir: Path,
    retention_days: int,
) -> dict[str, Any]:
    """Delete cron log date files older than *retention_days*."""
    if not animas_dir.exists():
        return {"skipped": True}

    cutoff = (today_local() - timedelta(days=retention_days)).isoformat()
    total_deleted = 0

    for anima_dir in sorted(animas_dir.iterdir()):
        if not anima_dir.is_dir():
            continue
        cron_log_dir = anima_dir / "state" / "cron_logs"
        if not cron_log_dir.is_dir():
            continue
        for f in cron_log_dir.glob("*.jsonl"):
            # Filename format: YYYY-MM-DD.jsonl
            if f.stem < cutoff:
                try:
                    f.unlink()
                    total_deleted += 1
                except OSError:
                    logger.warning("Failed to delete cron log: %s", f)

    if total_deleted:
        logger.info("Cron log cleanup: deleted %d files", total_deleted)
    return {"deleted_files": total_deleted}


def _episodeify_abandoned_session(anima_dir: Path, json_path: Path) -> bool:
    """Persist an abandoned short-term session as an episode before deletion.

    An expired ``session_state.json`` represents a chat session that crossed
    the context-window threshold, externalized its state, but was never
    finalized into long-term memory. Rather than losing it at retention expiry,
    fold its accumulated content into ``episodes/`` so it stays searchable.

    Returns True when the state file may be deleted (episode saved, or the file
    holds no recoverable content), and False when the episode write failed and
    deletion should be retried on the next housekeeping run.
    """
    import json

    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        # Truly corrupt state carries no recoverable memory.
        return True
    except OSError:
        # Transient I/O failure (e.g. FD exhaustion): the content may still be
        # recoverable, so keep the file and retry on the next run.
        logger.warning(
            "Failed to read abandoned shortterm session %s; deferring deletion",
            json_path,
            exc_info=True,
        )
        return False
    if not isinstance(data, dict):
        return True

    prompt = (data.get("original_prompt") or "").strip()
    response = (data.get("accumulated_response") or "").strip()
    notes = (data.get("notes") or "").strip()
    if not prompt and not response and not notes:
        return True

    body_parts: list[str] = []
    if prompt:
        body_parts.append(f"{t('shortterm.original_request')}\n{prompt}")
    if response:
        body_parts.append(f"{t('shortterm.work_so_far')}\n{response}")
    if notes:
        body_parts.append(f"{t('shortterm.notes_header')}\n{notes}")
    entry = f"## {t('shortterm.title')}\n\n" + "\n\n".join(body_parts) + "\n"

    try:
        from core.memory.manager import MemoryManager

        MemoryManager(anima_dir).append_episode(entry, origin="shortterm_recovery")
        return True
    except Exception:
        logger.warning(
            "Failed to episode-ify abandoned shortterm session %s; deferring deletion",
            json_path,
            exc_info=True,
        )
        return False


def _cleanup_shortterm(
    animas_dir: Path,
    retention_days: int,
    archive_retention_days: int = 30,
    thread_gc_days: int = 30,
) -> dict[str, Any]:
    """Delete old session files from shortterm/ directories.

    Recurses into per-thread subdirectories (``shortterm/chat/{thread_id}/``)
    as well as the supported lifecycle folders. Archive files use their own
    retention window. Before removing an expired ``session_state.json`` from a
    chat tree, its unfinalized content is saved as an episode; if that save
    fails, the file and its thread directory are kept for retry.

    Skips ``current_session_*.json`` and ``streaming_journal*.jsonl``
    during individual file cleanup. Stale chat thread directories are removed
    as a unit once every file in them is older than the thread GC window. The
    directory is rescanned immediately before deletion to reduce TOCTOU risk;
    a write after that final scan remains an accepted residual race.
    """
    if not animas_dir.exists():
        return {"skipped": True}

    retention_days = max(retention_days, 0)
    archive_cleanup_enabled = archive_retention_days > 0
    thread_gc_enabled = thread_gc_days > 0
    skipped_substeps: dict[str, str] = {}
    if not archive_cleanup_enabled:
        skipped_substeps["archive_cleanup"] = "archive_retention_days_must_be_positive"
    if not thread_gc_enabled:
        skipped_substeps["thread_gc"] = "thread_gc_days_must_be_positive"
    now = now_local()
    cutoff_ts = (now - timedelta(days=retention_days)).timestamp()
    archive_cutoff_ts = (now - timedelta(days=archive_retention_days)).timestamp() if archive_cleanup_enabled else None
    thread_gc_cutoff_ts = (now - timedelta(days=thread_gc_days)).timestamp() if thread_gc_enabled else None
    total_deleted = 0
    total_episodified = 0
    archive_deleted = 0
    thread_dirs_deleted = 0
    deleted_by_subdir = {sub: 0 for sub in ("chat", "heartbeat", "cron", "inbox", "task")}

    for anima_dir in sorted(animas_dir.iterdir()):
        if not anima_dir.is_dir():
            continue
        shortterm_dir = anima_dir / "shortterm"
        if not shortterm_dir.is_dir():
            continue
        gc_protected_thread_dirs: set[Path] = set()
        for sub in deleted_by_subdir:
            sub_dir = shortterm_dir / sub
            if not sub_dir.is_dir():
                continue
            for f in list(sub_dir.rglob("*")):
                if not f.is_file():
                    continue
                relative_parts = f.relative_to(sub_dir).parts
                is_archive = "archive" in relative_parts
                if is_archive and not archive_cleanup_enabled:
                    continue
                if any(f.name.startswith(p) for p in _PROTECTED_PREFIXES):
                    continue
                try:
                    file_cutoff_ts = archive_cutoff_ts if is_archive else cutoff_ts
                    if file_cutoff_ts is None:
                        continue
                    if f.stat().st_mtime >= file_cutoff_ts:
                        continue
                except OSError:
                    logger.warning("Failed to stat shortterm file: %s", f)
                    continue
                # Preserve abandoned chat sessions as episodes before deletion.
                episodified = False
                if sub == "chat" and f.name == "session_state.json":
                    if not _episodeify_abandoned_session(anima_dir, f):
                        if len(relative_parts) > 1 and relative_parts[0] != "archive":
                            gc_protected_thread_dirs.add(sub_dir / relative_parts[0])
                        continue  # keep the file; retry next run
                    episodified = True
                try:
                    f.unlink()
                    total_deleted += 1
                    deleted_by_subdir[sub] += 1
                    if is_archive:
                        archive_deleted += 1
                    if episodified:
                        total_episodified += 1
                except OSError:
                    if episodified:
                        # Already saved as an episode: rename so the next run
                        # does not episodify it again (duplicate episode). The
                        # renamed file keeps its expired mtime and is swept by
                        # the generic deletion above on a later run.
                        try:
                            f.rename(f.with_name(f.name + ".episodified.bak"))
                            total_episodified += 1
                            logger.warning(
                                "Failed to delete episodified shortterm state; renamed for later sweep: %s",
                                f,
                            )
                        except OSError:
                            logger.error(
                                "Failed to delete or rename episodified shortterm state %s; "
                                "a duplicate episode may result on the next run",
                                f,
                            )
                    else:
                        logger.warning("Failed to delete shortterm file: %s", f)

        chat_dir = shortterm_dir / "chat"
        if not thread_gc_enabled or thread_gc_cutoff_ts is None or not chat_dir.is_dir():
            continue
        for thread_dir in sorted(path for path in chat_dir.iterdir() if path.is_dir()):
            if thread_dir.name == "archive" or thread_dir in gc_protected_thread_dirs:
                continue
            snapshot = _scan_shortterm_thread_for_gc(thread_dir, thread_gc_cutoff_ts)
            if snapshot is None:
                continue
            files, max_mtime, recent_protected = snapshot
            if recent_protected:
                continue
            if max_mtime is not None and max_mtime >= thread_gc_cutoff_ts:
                continue

            # Recheck immediately before rmtree. A write after this scan is the
            # documented residual race.
            rechecked = _scan_shortterm_thread_for_gc(thread_dir, thread_gc_cutoff_ts)
            if rechecked is None:
                continue
            files, max_mtime, recent_protected = rechecked
            if recent_protected or (max_mtime is not None and max_mtime >= thread_gc_cutoff_ts):
                continue
            deleted_file_count = len(files)
            deleted_archive_count = sum("archive" in f.relative_to(thread_dir).parts for f in files)
            try:
                shutil.rmtree(thread_dir)
            except OSError:
                logger.warning("Failed to delete stale shortterm thread directory: %s", thread_dir)
                continue
            thread_dirs_deleted += 1
            total_deleted += deleted_file_count
            archive_deleted += deleted_archive_count
            deleted_by_subdir["chat"] += deleted_file_count

    if total_deleted or total_episodified or thread_dirs_deleted:
        logger.info(
            "Shortterm cleanup: deleted=%d archive_deleted=%d thread_dirs_deleted=%d episodified=%d by_subdir=%s",
            total_deleted,
            archive_deleted,
            thread_dirs_deleted,
            total_episodified,
            deleted_by_subdir,
        )
    return {
        "deleted_files": total_deleted,
        "archive_deleted": archive_deleted,
        "thread_dirs_deleted": thread_dirs_deleted,
        "deleted_by_subdir": deleted_by_subdir,
        "episodified_sessions": total_episodified,
        "skipped_substeps": skipped_substeps,
    }


def _scan_shortterm_thread_for_gc(
    thread_dir: Path,
    cutoff_ts: float,
) -> tuple[list[Path], float | None, bool] | None:
    """Return a conservative thread snapshot, or ``None`` on any stat failure."""
    try:
        files = [path for path in thread_dir.rglob("*") if path.is_file()]
    except OSError:
        logger.warning("Failed to scan shortterm thread directory: %s", thread_dir)
        return None
    max_mtime: float | None = None
    recent_protected = False
    for path in files:
        try:
            mtime = path.stat().st_mtime
        except OSError:
            logger.warning("Failed to stat shortterm thread file: %s", path)
            return None
        max_mtime = mtime if max_mtime is None else max(max_mtime, mtime)
        if mtime >= cutoff_ts and any(path.name.startswith(prefix) for prefix in _PROTECTED_PREFIXES):
            recent_protected = True
    return files, max_mtime, recent_protected


def _cleanup_facts_locks(animas_dir: Path, stale_hours: int) -> dict[str, Any]:
    """Delete stale empty ``facts/*.lock`` files for every Anima."""
    if not animas_dir.exists():
        return {"skipped": True}
    if stale_hours <= 0:
        return {
            "skipped": True,
            "reason": "stale_hours_must_be_positive",
            "scanned_files": 0,
            "deleted_files": 0,
        }

    try:
        import fcntl
    except ImportError:
        logger.info("Facts lock cleanup skipped: fcntl unavailable")
        return {
            "skipped": True,
            "reason": "fcntl_unavailable",
            "scanned_files": 0,
            "deleted_files": 0,
        }

    cutoff_ts = (now_local() - timedelta(hours=stale_hours)).timestamp()
    deleted_files = 0
    scanned_files = 0
    locked_files = 0
    lock_failures = 0

    for anima_dir in sorted(animas_dir.iterdir()):
        if not anima_dir.is_dir():
            continue
        facts_dir = anima_dir / "facts"
        if not facts_dir.is_dir():
            continue
        for lock_file in sorted(facts_dir.glob("*.lock")):
            if not lock_file.is_file():
                continue
            scanned_files += 1
            try:
                with lock_file.open("r+b") as lock_handle:
                    try:
                        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except OSError as exc:
                        if exc.errno in (errno.EACCES, errno.EAGAIN):
                            locked_files += 1
                        else:
                            lock_failures += 1
                            logger.warning("Failed to acquire facts lock file: %s", lock_file)
                        continue

                    stat = os.fstat(lock_handle.fileno())
                    path_stat = lock_file.stat()
                    if (stat.st_dev, stat.st_ino) != (path_stat.st_dev, path_stat.st_ino):
                        lock_failures += 1
                        continue
                    if stat.st_size != 0 or stat.st_mtime >= cutoff_ts:
                        continue
                    # Keep LOCK_EX held until after unlink; closing the handle
                    # at the end of this block releases the lock safely.
                    lock_file.unlink()
                    deleted_files += 1
            except OSError:
                logger.warning("Failed to inspect or delete facts lock file: %s", lock_file)
                lock_failures += 1

    if deleted_files:
        logger.info(
            "Facts lock cleanup: scanned=%d deleted=%d stale_hours=%d",
            scanned_files,
            deleted_files,
            stale_hours,
        )
    return {
        "scanned_files": scanned_files,
        "deleted_files": deleted_files,
        "locked_files": locked_files,
        "lock_failures": lock_failures,
    }


def _cleanup_curator_reports(
    animas_dir: Path,
    retention_days: int,
) -> dict[str, Any]:
    """Delete Skill Curator report files older than *retention_days*.

    Reports are generated daily into ``state/skill_curator/report-*.json`` but
    only the newest is ever consumed, so older ones accumulate unbounded. The
    filename carries the ISO date (``report-YYYY-MM-DD.json``); anything past
    the cutoff is removed.
    """
    if not animas_dir.exists():
        return {"skipped": True}

    cutoff = (today_local() - timedelta(days=retention_days)).isoformat()
    total_deleted = 0

    for anima_dir in sorted(animas_dir.iterdir()):
        if not anima_dir.is_dir():
            continue
        report_dir = anima_dir / "state" / "skill_curator"
        if not report_dir.is_dir():
            continue
        for f in report_dir.glob("report-*.json"):
            # Filename format: report-YYYY-MM-DD.json
            date_part = f.stem[len("report-") :]
            if date_part < cutoff:
                try:
                    f.unlink()
                    total_deleted += 1
                except OSError:
                    logger.warning("Failed to delete curator report: %s", f)

    if total_deleted:
        logger.info("Curator report cleanup: deleted %d files", total_deleted)
    return {"deleted_files": total_deleted}


def _cleanup_task_results(
    animas_dir: Path,
    retention_days: int,
) -> dict[str, Any]:
    """Delete old task result files from state/task_results/."""
    if not animas_dir.exists():
        return {"skipped": True}

    cutoff_ts = (now_local() - timedelta(days=retention_days)).timestamp()
    total_deleted = 0

    for anima_dir in sorted(animas_dir.iterdir()):
        if not anima_dir.is_dir():
            continue
        results_dir = anima_dir / "state" / "task_results"
        if not results_dir.is_dir():
            continue
        for f in results_dir.glob("*.md"):
            try:
                if f.stat().st_mtime < cutoff_ts:
                    f.unlink()
                    total_deleted += 1
            except OSError:
                logger.warning("Failed to delete task result: %s", f)

    if total_deleted:
        logger.info("Task results cleanup: deleted %d files", total_deleted)
    return {"deleted_files": total_deleted}


def _rotate_archive_superseded(
    animas_dir: Path,
    retention_days: int,
) -> dict[str, Any]:
    """Delete archived files in archive/superseded/ older than *retention_days*."""
    if not animas_dir.exists():
        return {"skipped": True}

    cutoff_ts = (now_local() - timedelta(days=retention_days)).timestamp()
    total_deleted = 0

    for anima_dir in sorted(animas_dir.iterdir()):
        if not anima_dir.is_dir():
            continue
        archive_dir = anima_dir / "archive" / "superseded"
        if not archive_dir.is_dir():
            continue
        for f in archive_dir.iterdir():
            if not f.is_file():
                continue
            try:
                # mtime is preserved by shutil.move, so a file archived today can
                # carry an old mtime and be deleted immediately. ctime is updated
                # by the move itself, so max(mtime, ctime) approximates the time
                # the file entered archive/superseded.
                st = f.stat()
                if max(st.st_mtime, st.st_ctime) < cutoff_ts:
                    f.unlink()
                    total_deleted += 1
            except OSError:
                logger.warning("Failed to delete archived file: %s", f)

    if total_deleted:
        logger.info("Archive/superseded cleanup: deleted %d files", total_deleted)
    return {"deleted_files": total_deleted}


def _cleanup_pending_failed(
    animas_dir: Path,
    retention_days: int,
) -> dict[str, Any]:
    from core.memory.pending_housekeeping import cleanup_pending_failed

    return cleanup_pending_failed(animas_dir, retention_days)


_CORRUPT_VECTORDB_RE = re.compile(r"^(?:vectordb-corrupt|corrupt-vectordb)[-_](?P<stamp>\d{8}[-_]?\d{6}|\d{14})")


def _corrupt_archive_sort_key(path: Path) -> tuple[str, str]:
    match = _CORRUPT_VECTORDB_RE.match(path.name)
    if match:
        return (match.group("stamp").replace("_", "").replace("-", ""), path.name)
    return ("", path.name)


def _path_size(path: Path) -> int:
    if path.is_symlink() or not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    return sum(child.stat().st_size for child in path.rglob("*") if child.is_file() and not child.is_symlink())


def _remove_path(path: Path) -> None:
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink(missing_ok=True)


def _cleanup_corrupt_vectordb_archives(
    animas_dir: Path,
    keep_generations: int,
) -> dict[str, Any]:
    """Keep only the newest corrupt vectordb archives per Anima."""
    if not animas_dir.exists():
        return {"skipped": True}

    deleted_dirs = 0
    freed_bytes = 0
    per_anima: dict[str, int] = {}

    for anima_dir in sorted(p for p in animas_dir.iterdir() if p.is_dir()):
        archive_dir = anima_dir / "archive"
        if not archive_dir.is_dir():
            continue
        archives = [
            p
            for p in archive_dir.iterdir()
            if p.is_dir() and (p.name.startswith("vectordb-corrupt-") or p.name.startswith("corrupt-vectordb-"))
        ]
        archives.sort(key=_corrupt_archive_sort_key, reverse=True)
        for archive in archives[max(keep_generations, 0) :]:
            try:
                size = _path_size(archive)
                _remove_path(archive)
                deleted_dirs += 1
                freed_bytes += size
                per_anima[anima_dir.name] = per_anima.get(anima_dir.name, 0) + 1
            except OSError:
                logger.warning("Failed to delete corrupt vectordb archive: %s", archive, exc_info=True)

    if deleted_dirs:
        logger.info("Corrupt vectordb archive cleanup: deleted=%d freed=%d bytes", deleted_dirs, freed_bytes)
    return {"deleted_dirs": deleted_dirs, "freed_bytes": freed_bytes, "per_anima": per_anima}


def _cleanup_runtime_tmp(tmp_dir: Path, retention_days: int) -> dict[str, Any]:
    """Delete old top-level entries from runtime tmp."""
    if not tmp_dir.exists():
        return {"skipped": True}

    cutoff_ts = (now_local() - timedelta(days=retention_days)).timestamp()
    open_paths = _collect_open_tmp_paths(tmp_dir)
    deleted_entries = 0
    freed_bytes = 0
    skipped_count = 0
    errors: list[str] = []

    for entry in sorted(tmp_dir.iterdir()):
        if entry.name.startswith("."):
            continue
        if entry.is_symlink():
            skipped_count += 1
            continue
        try:
            if entry.name in _PRESERVED_TMP_SUBDIRS and entry.is_dir():
                nested = _cleanup_runtime_tmp_contents(entry, cutoff_ts, open_paths)
                deleted_entries += nested["deleted_entries"]
                freed_bytes += nested["freed_bytes"]
                skipped_count += nested["skipped_count"]
                errors.extend(nested["errors"])
                continue
            if not _path_tree_older_than(entry, cutoff_ts):
                continue
            if _path_has_open_files(entry, open_paths):
                skipped_count += 1
                logger.warning("Runtime tmp entry is open; skipping: %s", entry)
                continue
            size = _path_size(entry)
            _remove_path(entry)
            deleted_entries += 1
            freed_bytes += size
        except OSError as exc:
            logger.warning("Failed to delete runtime tmp entry: %s", entry, exc_info=True)
            skipped_count += 1
            errors.append(f"{entry}: {exc}")

    if deleted_entries:
        logger.info("Runtime tmp cleanup: deleted=%d freed=%d bytes", deleted_entries, freed_bytes)
    return {
        "deleted_entries": deleted_entries,
        "freed_bytes": freed_bytes,
        "skipped_count": skipped_count,
        "errors": errors,
    }


def _cleanup_runtime_tmp_contents(root: Path, cutoff_ts: float, open_paths: set[Path]) -> dict[str, Any]:
    """Clean old children inside a preserved tmp subdir without deleting it."""
    deleted_entries = 0
    freed_bytes = 0
    skipped_count = 0
    errors: list[str] = []

    for entry in sorted(root.iterdir()):
        if entry.name.startswith(".") or entry.is_symlink():
            skipped_count += 1
            continue
        try:
            if not _path_tree_older_than(entry, cutoff_ts):
                continue
            if _path_has_open_files(entry, open_paths):
                skipped_count += 1
                logger.warning("Runtime tmp entry is open; skipping: %s", entry)
                continue
            size = _path_size(entry)
            _remove_path(entry)
            deleted_entries += 1
            freed_bytes += size
        except OSError as exc:
            logger.warning("Failed to delete runtime tmp entry: %s", entry, exc_info=True)
            skipped_count += 1
            errors.append(f"{entry}: {exc}")

    return {
        "deleted_entries": deleted_entries,
        "freed_bytes": freed_bytes,
        "skipped_count": skipped_count,
        "errors": errors,
    }


def _path_tree_older_than(path: Path, cutoff_ts: float) -> bool:
    try:
        if path.stat().st_mtime >= cutoff_ts:
            return False
    except FileNotFoundError:
        return False
    except OSError:
        return False
    if path.is_dir() and not path.is_symlink():
        for child in path.rglob("*"):
            try:
                if child.stat().st_mtime >= cutoff_ts:
                    return False
            except FileNotFoundError:
                continue
            except OSError:
                return False
    return True


def _collect_open_tmp_paths(root: Path) -> set[Path]:
    return _collect_open_paths(root)


def _collect_open_paths(root: Path) -> set[Path]:
    lsof = shutil.which("lsof")
    if not lsof:
        return set()
    try:
        result = subprocess.run(
            [lsof, "-F", "n", "+D", str(root)],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            check=False,
            timeout=10,
        )
    except subprocess.TimeoutExpired:
        logger.warning("Runtime tmp lsof scan timed out; proceeding without open-file skips for %s", root)
        return set()
    if result.returncode not in (0, 1):
        logger.debug("Runtime tmp lsof scan exited with status %s for %s", result.returncode, root)
    paths: set[Path] = set()
    for line in result.stdout.splitlines():
        if not line.startswith("n"):
            continue
        raw_path = line[1:].split(" (", 1)[0]
        if raw_path:
            paths.add(Path(raw_path).resolve(strict=False))
    return paths


def _collect_open_file_paths(paths_to_check: Iterable[Path]) -> set[Path]:
    paths_list = [p for p in paths_to_check if p.exists()]
    if not paths_list:
        return set()
    lsof = shutil.which("lsof")
    if not lsof:
        return set()
    try:
        result = subprocess.run(
            [lsof, "-F", "n", "--", *(str(p) for p in paths_list)],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            check=False,
            timeout=5,
        )
    except subprocess.TimeoutExpired:
        logger.warning("Open-file scan timed out; proceeding without open-file skips for %s", paths_list)
        return set()
    paths: set[Path] = set()
    for line in result.stdout.splitlines():
        if not line.startswith("n"):
            continue
        raw_path = line[1:].split(" (", 1)[0]
        if raw_path:
            paths.add(Path(raw_path).resolve(strict=False))
    return paths


def _path_has_open_files(path: Path, open_paths: set[Path]) -> bool:
    if not open_paths:
        return False
    resolved = path.resolve(strict=False)
    if resolved in open_paths:
        return True
    if path.is_dir():
        return any(open_path.is_relative_to(resolved) for open_path in open_paths)
    return False


def _cleanup_backup_dirs(animas_dir: Path, retention_days: int) -> dict[str, Any]:
    """Delete old explicit backup directories such as assets_backup_*."""
    if not animas_dir.exists():
        return {"skipped": True}

    cutoff_ts = (now_local() - timedelta(days=retention_days)).timestamp()
    deleted_dirs = 0
    freed_bytes = 0

    for anima_dir in sorted(p for p in animas_dir.iterdir() if p.is_dir()):
        for backup_dir in sorted(p for p in anima_dir.iterdir() if p.is_dir() and not p.is_symlink()):
            if "_backup_" not in backup_dir.name:
                continue
            try:
                if backup_dir.stat().st_mtime >= cutoff_ts:
                    continue
                size = _path_size(backup_dir)
                _remove_path(backup_dir)
                deleted_dirs += 1
                freed_bytes += size
            except OSError:
                logger.warning("Failed to delete backup dir: %s", backup_dir, exc_info=True)

    if deleted_dirs:
        logger.info("Backup dir cleanup: deleted=%d freed=%d bytes", deleted_dirs, freed_bytes)
    return {"deleted_dirs": deleted_dirs, "freed_bytes": freed_bytes}


def _cleanup_codex_execution_logs(animas_dir: Path, max_size_mb: int) -> dict[str, Any]:
    """Delete oversized Codex execution log databases, preserving state/session DBs."""
    if not animas_dir.exists():
        return {"skipped": True}

    max_bytes = max_size_mb * 1024 * 1024
    deleted_databases = 0
    deleted_files = 0
    freed_bytes = 0
    skipped_open = 0

    for db in sorted(animas_dir.glob(f"*/.codex_home/{_CODEX_LOG_DB_NAME}")):
        try:
            if db.stat().st_size <= max_bytes:
                continue
            bundle = [
                p
                for p in (db, db.with_name(f"{_CODEX_LOG_DB_NAME}-wal"), db.with_name(f"{_CODEX_LOG_DB_NAME}-shm"))
                if p.exists()
            ]
            open_paths = _collect_open_file_paths(bundle)
            if any(_path_has_open_files(p, open_paths) for p in bundle):
                skipped_open += 1
                logger.warning("Codex execution log DB is open; skipping: %s", db)
                continue
            bundle_size = sum(_path_size(p) for p in bundle)
            for path in bundle:
                path.unlink(missing_ok=True)
                deleted_files += 1
            deleted_databases += 1
            freed_bytes += bundle_size
        except OSError:
            logger.warning("Failed to delete Codex execution log DB: %s", db, exc_info=True)

    if deleted_databases:
        logger.info(
            "Codex execution log cleanup: databases=%d files=%d freed=%d bytes",
            deleted_databases,
            deleted_files,
            freed_bytes,
        )
    return {
        "deleted_databases": deleted_databases,
        "deleted_files": deleted_files,
        "freed_bytes": freed_bytes,
        "skipped_open": skipped_open,
    }


def _cleanup_codex_tmp_dirs(animas_dir: Path, retention_hours: int) -> dict[str, Any]:
    """Delete stale temporary entries under per-Anima CODEX_HOME directories."""
    if not animas_dir.exists():
        return {"skipped": True}

    cutoff_ts = (now_local() - timedelta(hours=retention_hours)).timestamp()
    deleted_entries = 0
    freed_bytes = 0
    skipped_open = 0

    for codex_home in sorted(animas_dir.glob("*/.codex_home")):
        for tmp_root_name in (".tmp", "tmp"):
            tmp_root = codex_home / tmp_root_name
            if not tmp_root.is_dir():
                continue
            open_paths = _collect_open_paths(tmp_root)
            for entry in sorted(tmp_root.iterdir()):
                if entry.is_symlink():
                    continue
                try:
                    if not _path_tree_older_than(entry, cutoff_ts):
                        continue
                    if _path_has_open_files(entry, open_paths):
                        skipped_open += 1
                        logger.warning("Codex tmp entry is open; skipping: %s", entry)
                        continue
                    size = _path_size(entry)
                    _remove_path(entry)
                    deleted_entries += 1
                    freed_bytes += size
                except OSError:
                    logger.warning("Failed to delete Codex tmp entry: %s", entry, exc_info=True)

    if deleted_entries:
        logger.info("Codex tmp cleanup: deleted=%d freed=%d bytes", deleted_entries, freed_bytes)
    return {"deleted_entries": deleted_entries, "freed_bytes": freed_bytes, "skipped_open": skipped_open}


def _cleanup_anima_runtime_artifacts(
    animas_dir: Path,
    tmp_gitdirs_retention_days: int,
    local_log_retention_days: int,
) -> dict[str, Any]:
    """Delete old per-Anima temporary git dirs and local runtime log files."""
    if not animas_dir.exists():
        return {"skipped": True}

    tmp_cutoff_ts = (now_local() - timedelta(days=tmp_gitdirs_retention_days)).timestamp()
    log_cutoff_ts = (now_local() - timedelta(days=local_log_retention_days)).timestamp()
    tmp_gitdirs_deleted = 0
    local_logs_deleted = 0
    freed_bytes = 0
    skipped_open = 0

    for anima_dir in sorted(p for p in animas_dir.iterdir() if p.is_dir()):
        tmp_gitdirs = anima_dir / "tmp_gitdirs"
        if tmp_gitdirs.is_dir():
            open_paths = _collect_open_paths(tmp_gitdirs)
            for entry in sorted(tmp_gitdirs.iterdir()):
                if entry.is_symlink():
                    continue
                try:
                    if not _path_tree_older_than(entry, tmp_cutoff_ts):
                        continue
                    if _path_has_open_files(entry, open_paths):
                        skipped_open += 1
                        continue
                    size = _path_size(entry)
                    _remove_path(entry)
                    tmp_gitdirs_deleted += 1
                    freed_bytes += size
                except OSError:
                    logger.warning("Failed to delete tmp gitdir artifact: %s", entry, exc_info=True)

        local_logs = anima_dir / "logs"
        if local_logs.is_dir():
            open_paths = _collect_open_paths(local_logs)
            for log_file in sorted(p for p in local_logs.rglob("*") if p.is_file() and not p.is_symlink()):
                try:
                    if log_file.stat().st_mtime >= log_cutoff_ts:
                        continue
                    if _path_has_open_files(log_file, open_paths):
                        skipped_open += 1
                        continue
                    size = log_file.stat().st_size
                    log_file.unlink()
                    local_logs_deleted += 1
                    freed_bytes += size
                except OSError:
                    logger.warning("Failed to delete local Anima runtime log: %s", log_file, exc_info=True)

    if tmp_gitdirs_deleted or local_logs_deleted:
        logger.info(
            "Anima runtime artifact cleanup: tmp_gitdirs=%d local_logs=%d freed=%d bytes",
            tmp_gitdirs_deleted,
            local_logs_deleted,
            freed_bytes,
        )
    return {
        "tmp_gitdirs_deleted": tmp_gitdirs_deleted,
        "local_logs_deleted": local_logs_deleted,
        "freed_bytes": freed_bytes,
        "skipped_open": skipped_open,
    }
