from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""TaskBoard-aware stale runtime artifact cleanup."""

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from core.time_utils import ensure_aware, now_local, today_local

logger = logging.getLogger("animaworks.housekeeping.taskboard")


def cleanup_taskboard_stale_artifacts(
    data_dir: Path,
    pending_processing_stale_hours: int,
    background_running_stale_hours: int,
    current_state_stale_hours: int,
    taskboard_suppressed_retention_days: int,
    taskboard_orphan_metadata_stale_hours: int = 24,
) -> dict[str, Any]:
    """Clean runtime artifacts that can resurface stale TaskBoard work."""
    animas_dir = data_dir / "animas"
    if not animas_dir.exists():
        return {"skipped": True}

    store = _taskboard_store_for_housekeeping(data_dir)
    results: dict[str, Any] = {}

    # LLM task state and attempts live in the canonical store. Legacy pending
    # files remain migration evidence, never a recovery or execution signal.

    background = _cleanup_background_running(animas_dir, background_running_stale_hours)
    results.update({f"background_{key}": value for key, value in background.items()})

    current_state = _cleanup_current_state(animas_dir, current_state_stale_hours, store)
    results.update({f"current_state_{key}": value for key, value in current_state.items()})

    orphan = _cleanup_orphan_metadata(animas_dir, store, taskboard_orphan_metadata_stale_hours)
    results.update({f"orphan_{key}": value for key, value in orphan.items()})

    purged = _purge_stale_metadata(store, taskboard_suppressed_retention_days)
    results.update({f"purged_{key}": value for key, value in purged.items()})

    return results


def _cleanup_background_running(animas_dir: Path, stale_hours: int) -> dict[str, int]:
    cutoff_ts = time.time() - (stale_hours * 3600)
    deleted = 0
    errors = 0

    for anima_dir in _iter_anima_dirs(animas_dir):
        background_dir = anima_dir / "state" / "background_tasks"
        if not background_dir.is_dir():
            continue
        for path in sorted(background_dir.glob("*.json")):
            try:
                payload, valid_json = _read_json_object(path)
                if not valid_json or payload.get("status") != "running":
                    continue
                created_at = _float_or_none(payload.get("created_at"))
                if created_at is not None and created_at < cutoff_ts:
                    path.unlink()
                    deleted += 1
            except OSError:
                errors += 1
                logger.warning("Failed to delete stale background task: %s", path, exc_info=True)

    return {"running_deleted": deleted, "errors": errors}


def _cleanup_current_state(
    animas_dir: Path,
    stale_hours: int,
    store: Any | None,
) -> dict[str, int]:
    cutoff_ts = time.time() - (stale_hours * 3600)
    now = now_local()
    archived = 0
    active_visible = 0
    skipped_idle = 0
    changed = 0
    errors = 0

    for anima_dir in _iter_anima_dirs(animas_dir):
        state_path = anima_dir / "state" / "current_state.md"
        try:
            if not state_path.is_file():
                continue
            observed_mtime = state_path.stat().st_mtime
            if observed_mtime >= cutoff_ts:
                continue
            content = state_path.read_text(encoding="utf-8")
            if not content.strip() or content.strip() == "status: idle":
                skipped_idle += 1
                continue
            if _has_active_visible_task(anima_dir, store, now):
                active_visible += 1
                continue
            outcome = _archive_current_state_for_housekeeping(
                anima_dir,
                state_path,
                content,
                expected_mtime=observed_mtime,
            )
            if outcome == "archived":
                archived += 1
            elif outcome == "changed":
                changed += 1
            else:
                errors += 1
        except OSError:
            errors += 1
            logger.warning("Failed to cleanup current_state: %s", state_path, exc_info=True)

    return {
        "archived": archived,
        "active_visible": active_visible,
        "skipped_idle": skipped_idle,
        "changed": changed,
        "errors": errors,
    }


_ORPHAN_VISIBILITIES = frozenset({"active", "snoozed"})
_PURGE_VISIBILITIES = frozenset({"archived", "expired", "tombstoned"})


def _cleanup_orphan_metadata(
    animas_dir: Path,
    store: Any | None,
    stale_hours: int,
) -> dict[str, int]:
    """Archive active/snoozed metadata rows whose live queue entry is gone."""
    archived = 0
    errors = 0
    if store is None:
        return {"archived": archived, "errors": errors}

    now = now_local()
    cutoff = now.timestamp() - (stale_hours * 3600)
    live_task_ids_cache: dict[str, set[str] | None] = {}

    try:
        rows = store.list_metadata()
    except Exception:
        logger.warning("Failed to list TaskBoard metadata for orphan cleanup", exc_info=True)
        return {"archived": 0, "errors": 1}

    for meta in rows:
        try:
            visibility = getattr(meta.visibility, "value", meta.visibility)
            if visibility not in _ORPHAN_VISIBILITIES:
                continue

            updated_at = _parse_datetime_value(getattr(meta, "updated_at", None))
            if updated_at is None or updated_at.timestamp() >= cutoff:
                continue

            anima_name = meta.anima_name
            task_id = meta.task_id
            anima_dir = animas_dir / anima_name

            if not anima_dir.is_dir():
                missing_from_queue = True
            else:
                if anima_name not in live_task_ids_cache:
                    live_task_ids_cache[anima_name] = _live_queue_task_ids(anima_dir)
                live_ids = live_task_ids_cache[anima_name]
                if live_ids is None:
                    # Queue unreadable — skip this anima rather than mass-archive.
                    continue
                missing_from_queue = task_id not in live_ids

            if not missing_from_queue:
                continue

            store.upsert_metadata(
                anima_name=anima_name,
                task_id=task_id,
                actor="housekeeping",
                event_type="archived",
                visibility="archived",
                column="done",
                tombstone_reason="queue_missing_reconciled",
            )
            archived += 1
        except Exception:
            errors += 1
            logger.warning(
                "Failed to archive orphan TaskBoard metadata: %s/%s",
                getattr(meta, "anima_name", "?"),
                getattr(meta, "task_id", "?"),
                exc_info=True,
            )

    if archived:
        logger.info("TaskBoard orphan metadata cleanup: archived %d rows", archived)
    return {"archived": archived, "errors": errors}


def _purge_stale_metadata(store: Any | None, retention_days: int) -> dict[str, int]:
    """Physically delete archived/expired/tombstoned metadata past retention."""
    deleted = 0
    errors = 0
    if store is None:
        return {"deleted": deleted, "errors": errors}

    now = now_local()
    cutoff = now.timestamp() - (retention_days * 86400)

    try:
        rows = store.list_metadata()
    except Exception:
        logger.warning("Failed to list TaskBoard metadata for purge", exc_info=True)
        return {"deleted": 0, "errors": 1}

    for meta in rows:
        try:
            visibility = getattr(meta.visibility, "value", meta.visibility)
            if visibility not in _PURGE_VISIBILITIES:
                continue

            updated_at = _parse_datetime_value(getattr(meta, "updated_at", None))
            if updated_at is None or updated_at.timestamp() >= cutoff:
                continue

            if store.delete_metadata(meta.anima_name, meta.task_id):
                deleted += 1
        except Exception:
            errors += 1
            logger.warning(
                "Failed to purge stale TaskBoard metadata: %s/%s",
                getattr(meta, "anima_name", "?"),
                getattr(meta, "task_id", "?"),
                exc_info=True,
            )

    if deleted:
        logger.info("TaskBoard metadata purge: deleted %d rows", deleted)
    return {"deleted": deleted, "errors": errors}


def _live_queue_task_ids(anima_dir: Path) -> set[str] | None:
    """Return task_ids present in the live queue, or None if unreadable."""
    try:
        from core.memory.task_queue import TaskQueueManager

        return set(TaskQueueManager(anima_dir)._load_all().keys())
    except Exception:
        logger.debug("Failed to load live queue for %s", anima_dir.name, exc_info=True)
        return None


def _iter_anima_dirs(animas_dir: Path) -> list[Path]:
    if not animas_dir.exists():
        return []
    return [path for path in sorted(animas_dir.iterdir()) if path.is_dir()]


def _taskboard_store_for_housekeeping(data_dir: Path) -> Any | None:
    try:
        from core.taskboard.store import TaskBoardStore

        return TaskBoardStore(data_dir / "shared" / "taskboard.sqlite3")
    except Exception:
        logger.debug("TaskBoard store unavailable for housekeeping", exc_info=True)
        return None


def _read_json_object(path: Path) -> tuple[dict[str, Any], bool]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}, False
    if not isinstance(payload, dict):
        return {}, False
    return payload, True


def _parse_datetime_value(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        return ensure_aware(datetime.fromisoformat(value))
    except (TypeError, ValueError):
        return None


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _has_active_visible_task(anima_dir: Path, store: Any | None, now: datetime) -> bool:
    """Retain working notes while any canonical task is unfinished.

    Board visibility is presentation metadata, never evidence that work ended.
    """
    try:
        from core.memory.task_queue import TaskQueueManager

        return bool(TaskQueueManager(anima_dir).load_active_tasks())
    except Exception:
        logger.debug("Failed to read canonical tasks for %s", anima_dir.name, exc_info=True)
        return True


def _archive_current_state_for_housekeeping(
    anima_dir: Path,
    state_path: Path,
    content: str,
    *,
    expected_mtime: float | None = None,
) -> str:
    try:
        from core.memory._io import atomic_write_text

        if expected_mtime is not None and state_path.stat().st_mtime != expected_mtime:
            return "changed"
        episodes_dir = anima_dir / "episodes"
        episodes_dir.mkdir(parents=True, exist_ok=True)
        episode_path = episodes_dir / f"{today_local().isoformat()}.md"
        existing = (
            episode_path.read_text(encoding="utf-8") if episode_path.exists() else f"# {today_local().isoformat()}\n"
        )
        entry = f"\n## Working notes archived by TaskBoard housekeeping\n\n{content.rstrip()}\n"
        atomic_write_text(episode_path, existing.rstrip() + "\n\n" + entry.lstrip())
        atomic_write_text(state_path, "status: idle\n")
        return "archived"
    except Exception:
        logger.warning("Failed to archive current_state for housekeeping: %s", state_path, exc_info=True)
        return "error"
