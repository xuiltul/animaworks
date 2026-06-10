from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Unified housekeeping engine for periodic disk cleanup.

Runs as a single daily job from ProcessSupervisor's scheduler, cleaning up
all data types that lack their own rotation mechanisms.
"""

import asyncio
import logging
import re
import shutil
import subprocess
from datetime import timedelta
from pathlib import Path
from typing import Any

from core.time_utils import now_local, today_local

logger = logging.getLogger("animaworks.housekeeping")
_PRESERVED_TMP_SUBDIRS = frozenset({"attachments", "skill_hub"})


# ── Public API ──────────────────────────────────────────────────


async def run_housekeeping(
    data_dir: Path,
    *,
    prompt_log_retention_days: int = 3,
    daemon_log_max_size_mb: int = 200,
    daemon_log_keep_generations: int = 2,
    dm_log_archive_retention_days: int = 30,
    cron_log_retention_days: int = 30,
    shortterm_retention_days: int = 7,
    task_results_retention_days: int = 7,
    pending_failed_retention_days: int = 14,
    corrupt_vectordb_keep_generations: int = 3,
    tmp_retention_days: int = 14,
    backup_retention_days: int = 90,
    pending_processing_stale_hours: int = 24,
    background_running_stale_hours: int = 48,
    current_state_stale_hours: int = 24,
    taskboard_suppressed_retention_days: int = 30,
    archive_superseded_retention_days: int = 7,
    inbox_ttl_hours: float = 24.0,
    inbox_expired_retention_days: int = 7,
    inbox_processed_retention_days: int = 30,
    inbox_quarantine_retention_days: int = 30,
) -> dict[str, Any]:
    """Run all housekeeping tasks. Returns summary of actions taken."""
    loop = asyncio.get_running_loop()
    results: dict[str, Any] = {}

    animas_dir = data_dir / "animas"

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
        )
        results["shortterm"] = r
    except Exception:
        logger.exception("Housekeeping: shortterm cleanup failed")
        results["shortterm"] = {"error": True}

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
        from core.memory.taskboard_housekeeping import cleanup_taskboard_stale_artifacts

        r = await loop.run_in_executor(
            None,
            cleanup_taskboard_stale_artifacts,
            data_dir,
            pending_processing_stale_hours,
            background_running_stale_hours,
            current_state_stale_hours,
            taskboard_suppressed_retention_days,
        )
        results["taskboard_stale"] = r
    except Exception:
        logger.exception("Housekeeping: taskboard stale cleanup failed")
        results["taskboard_stale"] = {"error": True}

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
    """Size-based rotation for server-daemon.log using rename strategy.

    Renames current log to .1, shifts .1 → .2, etc., and deletes
    generations beyond *keep_generations*.
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

    # Current → .1
    gen1 = parent / f"{stem}.1"
    log_path.rename(gen1)

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


def _cleanup_shortterm(
    animas_dir: Path,
    retention_days: int,
) -> dict[str, Any]:
    """Delete old session files from shortterm/ directories.

    Skips ``current_session_*.json`` and ``streaming_journal_*.jsonl``
    which are managed by the session lifecycle.
    """
    if not animas_dir.exists():
        return {"skipped": True}

    cutoff = now_local() - timedelta(days=retention_days)
    cutoff_ts = cutoff.timestamp()
    total_deleted = 0

    _PROTECTED_PREFIXES = ("current_session_", "streaming_journal_")

    for anima_dir in sorted(animas_dir.iterdir()):
        if not anima_dir.is_dir():
            continue
        shortterm_dir = anima_dir / "shortterm"
        if not shortterm_dir.is_dir():
            continue
        # Walk chat/ and heartbeat/ subdirs
        for sub in ("chat", "heartbeat"):
            sub_dir = shortterm_dir / sub
            if not sub_dir.is_dir():
                continue
            for f in sub_dir.iterdir():
                if not f.is_file():
                    continue
                if any(f.name.startswith(p) for p in _PROTECTED_PREFIXES):
                    continue
                try:
                    if f.stat().st_mtime < cutoff_ts:
                        f.unlink()
                        total_deleted += 1
                except OSError:
                    logger.warning("Failed to delete shortterm file: %s", f)

    if total_deleted:
        logger.info("Shortterm cleanup: deleted %d files", total_deleted)
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
                if f.stat().st_mtime < cutoff_ts:
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
                nested = _cleanup_runtime_tmp_contents(entry, cutoff_ts)
                deleted_entries += nested["deleted_entries"]
                freed_bytes += nested["freed_bytes"]
                skipped_count += nested["skipped_count"]
                errors.extend(nested["errors"])
                continue
            if not _path_tree_older_than(entry, cutoff_ts):
                continue
            if _path_has_open_files(entry):
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


def _cleanup_runtime_tmp_contents(root: Path, cutoff_ts: float) -> dict[str, Any]:
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
            if _path_has_open_files(entry):
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
    if path.stat().st_mtime >= cutoff_ts:
        return False
    if path.is_dir() and not path.is_symlink():
        for child in path.rglob("*"):
            if child.stat().st_mtime >= cutoff_ts:
                return False
    return True


def _path_has_open_files(path: Path) -> bool:
    lsof = shutil.which("lsof")
    if not lsof:
        return False
    cmd = [lsof, "+D", str(path)] if path.is_dir() else [lsof, "--", str(path)]
    result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
    return result.returncode == 0


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
