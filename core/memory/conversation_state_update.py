# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.

"""State update functions for conversation memory finalization."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from core.i18n import t
from core.memory.conversation_models import (
    _ERROR_PATTERN,
    _RESOLVED_PATTERN,
    ConversationTurn,
    ParsedSessionSummary,
)
from core.time_utils import now_iso

logger = logging.getLogger("animaworks.conversation_memory")


def _update_state_from_summary(
    anima_dir: Path,
    memory_mgr: Any,
    parsed: ParsedSessionSummary,
) -> None:
    """Route session summary outcomes to task_queue.jsonl.

    New tasks are registered in the persistent task queue instead of
    appending free-form markers to current_state.md.  Resolved items
    update matching task_queue entries to 'done'.
    """
    from core.memory.task_queue import TaskQueueManager

    anima_name = memory_mgr.anima_dir.name
    try:
        tqm = TaskQueueManager(memory_mgr.anima_dir)
    except Exception:
        logger.warning(
            "Failed to initialise TaskQueueManager; skipping state update",
            exc_info=True,
        )
        return

    active_tasks = tqm.load_active_tasks()

    for item in parsed.resolved_items:
        for task in active_tasks.values():
            if item in task.summary or task.summary in item:
                tqm.update_status(task.task_id, "done", summary=item)
                logger.info("Task marked done from session summary: %s", task.task_id)
                break

    existing_summaries = {t.summary for t in active_tasks.values()}
    for task_text in parsed.new_tasks:
        if not any(task_text in s or s in task_text for s in existing_summaries):
            tqm.add_task(
                source="anima",
                original_instruction=task_text,
                assignee=anima_name,
                summary=task_text,
                meta={"origin": "session_summary_auto_detected"},
            )
            logger.info("New task registered from session summary: %s", task_text[:60])


def _auto_track_procedure_outcomes(
    anima_dir: Path,
    memory_mgr: Any,
    new_turns: list[ConversationTurn],
    injected_procedures: list[Path] | None = None,
    session_id: str = "",
) -> None:
    """Auto-track outcomes for procedures that were injected during this session."""
    try:
        if not injected_procedures:
            return

        # Only check the LAST assistant turn, not all turns
        assistant_turns = [t for t in new_turns if t.role == "assistant"]
        if assistant_turns:
            last_turn = assistant_turns[-1]
            has_error = bool(_ERROR_PATTERN.search(last_turn.content))
            if has_error and _RESOLVED_PATTERN.search(last_turn.content):
                has_error = False  # Resolution context overrides error detection
        else:
            has_error = False

        for proc_path in injected_procedures:
            if not proc_path.exists():
                continue

            meta = memory_mgr.read_procedure_metadata(proc_path)
            if not meta:
                continue

            # Skip if already reported via explicit tool in this session
            if session_id and meta.get("_reported_session_id") == session_id:
                logger.debug(
                    "Skipping auto-track for %s: already reported in session %s",
                    proc_path.name,
                    session_id,
                )
                continue

            if has_error:
                meta["failure_count"] = meta.get("failure_count", 0) + 1
            else:
                meta["success_count"] = meta.get("success_count", 0) + 1

            meta["last_used"] = now_iso()

            s = meta.get("success_count", 0)
            f = meta.get("failure_count", 0)
            meta["confidence"] = s / max(1, s + f)

            body = memory_mgr.read_procedure_content(proc_path)
            memory_mgr.write_procedure_with_meta(proc_path, body, meta)

            logger.debug(
                "Auto-tracked procedure outcome: %s success=%s confidence=%.2f",
                proc_path.name,
                not has_error,
                meta["confidence"],
            )

    except Exception:
        logger.debug("Failed to auto-track procedure outcomes", exc_info=True)


def _record_resolutions(
    anima_dir: Path,
    memory_mgr: Any,
    resolved_items: list[str],
) -> None:
    """Record resolution events to ActivityLogger and shared registry."""
    from core.memory.activity import ActivityLogger

    activity = ActivityLogger(anima_dir)

    for item in resolved_items:
        # Layer 1: ActivityLogger issue_resolved event
        try:
            activity.log(
                "issue_resolved",
                content=item,
                summary=t("conversation.resolution_summary", item=item[:100]),
            )
        except Exception:
            logger.debug("Failed to log issue_resolved event", exc_info=True)

        # Layer 3: shared/resolutions.jsonl cross-org record
        try:
            memory_mgr.append_resolution(
                issue=item,
                resolver=anima_dir.name,
            )
        except Exception:
            logger.debug("Failed to write resolution registry", exc_info=True)
