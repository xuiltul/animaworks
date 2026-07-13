from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.

"""Channel E: Pending task queue summary + active parallel tasks."""

import asyncio
import logging
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path

from core.file_access_policy import find_denied_root, load_denied_roots
from core.i18n import t
from core.memory.priming.constants import _BUDGET_PENDING_TASKS
from core.paths import get_animas_dir
from core.time_utils import now_local

logger = logging.getLogger("animaworks.priming")


def _resolved_readable_path(path: Path, denied_roots: tuple[Path, ...]) -> Path | None:
    """Resolve a prompt source and reject explicit deny roots before use."""
    if not denied_roots:
        return path
    try:
        resolved = path.resolve()
    except (OSError, RuntimeError):
        return None
    return resolved if find_denied_root(resolved, denied_roots) is None else None


def format_elapsed(started_at: str) -> str:
    """Format elapsed time from an ISO timestamp."""
    if not started_at:
        return ""
    try:
        start = datetime.fromisoformat(started_at)
        if start.tzinfo is None:
            start = start.replace(tzinfo=UTC)
        elapsed_s = (datetime.now(UTC) - start).total_seconds()
        if elapsed_s < 60:
            return f"{int(elapsed_s)}s"
        if elapsed_s < 3600:
            return f"{int(elapsed_s / 60)}m"
        return f"{elapsed_s / 3600:.1f}h"
    except (ValueError, TypeError):
        return ""


async def channel_e_pending_tasks(
    anima_dir: Path,
    get_active_parallel_tasks: Callable[[], dict[str, dict]] | None,
) -> str:
    """Channel E: Pending task queue summary + active parallel tasks.

    Retrieves pending tasks from the persistent task queue.
    Human-origin tasks are marked with 🔴 HIGH priority.
    Also includes currently running parallel tasks (Level 2 format:
    title + description summary + status + elapsed time).
    Budget: 300 tokens.

    Uses asyncio.to_thread to avoid blocking the event loop
    since TaskQueueManager performs synchronous file I/O.
    """
    parts: list[str] = []
    resolver = None
    denied_roots = load_denied_roots(anima_dir)
    unresolved_queue_path = anima_dir / "state" / "task_queue.jsonl"
    queue_path = _resolved_readable_path(unresolved_queue_path, denied_roots)
    if denied_roots and unresolved_queue_path.is_symlink():
        queue_path = None

    from core.taskboard.attention_resolver import taskboard_db_path_for_anima

    taskboard_path = _resolved_readable_path(taskboard_db_path_for_anima(anima_dir), denied_roots)

    try:
        if queue_path is None or taskboard_path is None:
            raise PermissionError("pending task source is explicitly denied")
        from core.taskboard.attention_resolver import AttentionResolver
        from core.taskboard.formatting import format_tasks_for_priming
        from core.taskboard.projector import project_anima
        from core.taskboard.store import TaskBoardStore

        resolver = AttentionResolver(TaskBoardStore(taskboard_path))
        board_tasks = await asyncio.to_thread(
            project_anima,
            anima_dir,
            resolver.store,
            anima_name=anima_dir.name,
            include_missing=True,
            include_archived=True,
        )
        visible_tasks = resolver.filter_for_priming(anima_dir.name, board_tasks, now_local())
        animas_dir = anima_dir.parent if anima_dir.parent.name == "animas" else get_animas_dir()
        queue_summary = format_tasks_for_priming(visible_tasks, _BUDGET_PENDING_TASKS, animas_dir=animas_dir)
        if queue_summary:
            parts.append(queue_summary)
    except Exception:
        logger.debug("Channel E TaskBoard projection failed; falling back to task_queue formatter", exc_info=True)
        from core.memory.task_queue import TaskQueueManager

        try:
            if queue_path is None:
                raise PermissionError("task queue is explicitly denied")
            manager = TaskQueueManager(anima_dir)
            queue_summary = await asyncio.to_thread(
                manager.format_for_priming,
                _BUDGET_PENDING_TASKS,
            )
            if queue_summary:
                parts.append(queue_summary)
        except Exception:
            logger.debug("Channel E (pending_tasks) failed", exc_info=True)

    active = get_active_parallel_tasks() if get_active_parallel_tasks else {}
    if active:
        lines = [t("priming.active_parallel_tasks_header")]
        for tid, info in active.items():
            elapsed = format_elapsed(info.get("started_at", ""))
            status = info.get("status", "running")
            deps = info.get("depends_on", [])
            dep_str = f", depends_on: {','.join(deps)}" if deps else ""
            lines.append(f"- [{tid}] {info.get('title', '?')} ({status} {elapsed}{dep_str})")
            desc = info.get("description", "")
            if desc:
                lines.append(f"  {desc[:100]}")
        parts.append("\n".join(lines))

    # ── Overflow inbox summary ──
    overflow_dir = _resolved_readable_path(anima_dir / "state" / "overflow_inbox", denied_roots)
    if overflow_dir is not None and overflow_dir.is_dir():
        try:
            files = sorted(
                [
                    resolved
                    for f in overflow_dir.iterdir()
                    if f.suffix == ".md" and (resolved := _resolved_readable_path(f, denied_roots)) is not None
                ],
                key=lambda f: f.name,
                reverse=True,
            )
            if files:
                names = [f.name for f in files[:5]]
                listing = ", ".join(names)
                remaining = f" (+{len(files) - 5})" if len(files) > 5 else ""
                parts.append(
                    t(
                        "dedup.overflow_inbox_summary",
                        count=len(files),
                        listing=listing,
                        remaining=remaining,
                    )
                )
        except Exception:
            logger.debug("Channel E: overflow_inbox read failed", exc_info=True)

    results_dir = _resolved_readable_path(anima_dir / "state" / "task_results", denied_roots)
    if results_dir is not None and results_dir.is_dir():
        try:
            now = now_local()
            result_files = []
            readable_result_files = [
                resolved
                for candidate in results_dir.glob("*.md")
                if (resolved := _resolved_readable_path(candidate, denied_roots)) is not None
            ]
            for rf in sorted(readable_result_files, key=lambda p: p.stat().st_mtime, reverse=True):
                if _should_show_task_result(anima_dir, rf, resolver, now):
                    result_files.append(rf)
                if len(result_files) >= 5:
                    break
            if result_files:
                lines = [t("priming.completed_bg_tasks_header")]
                for rf in result_files:
                    try:
                        content = rf.read_text(encoding="utf-8").strip()
                        task_id = rf.stem
                        preview = content[:150].replace("\n", " ")
                        lines.append(f"- [{task_id}] {preview}")
                    except Exception:
                        logger.debug("Channel E: failed to read %s", rf.name, exc_info=True)
                if len(lines) > 1:
                    parts.append("\n".join(lines))
        except Exception:
            logger.debug("Channel E: task_results read failed", exc_info=True)

    return "\n\n".join(parts)


def _should_show_task_result(anima_dir: Path, result_file: Path, resolver: object | None, now: datetime) -> bool:
    try:
        result_mtime = result_file.stat().st_mtime
    except OSError:
        return False

    if resolver is None:
        modified_at = datetime.fromtimestamp(result_mtime, tz=now.tzinfo)
        return now - modified_at <= timedelta(hours=24)

    try:
        return bool(
            resolver.should_show_task_result(
                anima_dir.name,
                result_file.stem,
                result_mtime,
                now,
            )
        )
    except Exception:
        logger.debug("Channel E: TaskBoard task_result gate failed", exc_info=True)
        modified_at = datetime.fromtimestamp(result_mtime, tz=now.tzinfo)
        return now - modified_at <= timedelta(hours=24)
