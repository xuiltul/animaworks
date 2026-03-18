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
from datetime import timedelta
from pathlib import Path
from typing import Any

from core.time_utils import now_local, today_local

logger = logging.getLogger("animaworks.housekeeping")


# ── Public API ──────────────────────────────────────────────────


async def run_housekeeping(
    data_dir: Path,
    *,
    prompt_log_retention_days: int = 3,
    daemon_log_max_size_mb: int = 100,
    daemon_log_keep_generations: int = 5,
    dm_log_archive_retention_days: int = 30,
    cron_log_retention_days: int = 30,
    shortterm_retention_days: int = 7,
    task_results_retention_days: int = 7,
    pending_failed_retention_days: int = 14,
    archive_superseded_retention_days: int = 7,
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

    # 8. Archive/superseded rotation
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

    return results


# ── Sub-functions ───────────────────────────────────────────────


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

    size_mb = log_path.stat().st_size / (1024 * 1024)
    if size_mb < max_size_mb:
        return {"skipped": True, "current_size_mb": round(size_mb, 1)}

    # Shift existing generations: .N → .N+1 (highest first)
    parent = log_path.parent
    stem = log_path.name
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
    deleted = 0
    for gen in range(keep_generations + 1, keep_generations + 20):
        old = parent / f"{stem}.{gen}"
        if old.exists():
            old.unlink()
            deleted += 1
        else:
            break

    logger.info(
        "Daemon log rotated: %.1f MB → %s (deleted %d old generations)",
        size_mb,
        gen1.name,
        deleted,
    )
    return {"rotated": True, "size_mb": round(size_mb, 1), "deleted_generations": deleted}


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
    """Delete old failed task files from state/pending/failed/ and
    state/background_tasks/pending/failed/."""
    if not animas_dir.exists():
        return {"skipped": True}

    cutoff_ts = (now_local() - timedelta(days=retention_days)).timestamp()
    total_deleted = 0

    _FAILED_SUBDIRS = (
        Path("state") / "pending" / "failed",
        Path("state") / "background_tasks" / "pending" / "failed",
    )

    for anima_dir in sorted(animas_dir.iterdir()):
        if not anima_dir.is_dir():
            continue
        for rel in _FAILED_SUBDIRS:
            failed_dir = anima_dir / rel
            if not failed_dir.is_dir():
                continue
            for f in failed_dir.glob("*.json"):
                try:
                    if f.stat().st_mtime < cutoff_ts:
                        f.unlink()
                        total_deleted += 1
                except OSError:
                    logger.warning("Failed to delete failed task: %s", f)

    if total_deleted:
        logger.info("Pending failed cleanup: deleted %d files", total_deleted)
    return {"deleted_files": total_deleted}
