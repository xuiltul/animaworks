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
from core.memory.conversation_models import ParsedSessionSummary

logger = logging.getLogger("animaworks.conversation_memory")


def _update_state_from_summary(
    anima_dir: Path,
    memory_mgr: Any,
    parsed: ParsedSessionSummary,
) -> None:
    """Route session summary outcomes through the canonical task API.

    New tasks are registered in the persistent task queue instead of
    appending free-form markers to current_state.md.  Resolved items
    update matching task_queue entries to 'done'.
    """
    from core.memory.task_queue import TaskQueueManager

    try:
        tqm = TaskQueueManager(memory_mgr.anima_dir)
    except Exception:
        logger.warning(
            "Failed to initialise TaskQueueManager; skipping state update",
            exc_info=True,
        )
        return

    active_tasks = tqm.load_active_tasks()

    _MIN_FUZZY_MATCH_LEN = 8
    for item in parsed.resolved_items:
        for task in active_tasks.values():
            shorter = min(len(item), len(task.summary))
            if shorter >= _MIN_FUZZY_MATCH_LEN and (item in task.summary or task.summary in item):
                tqm.update_status(task.task_id, "done", summary=item)
                logger.info("Task marked done from session summary: %s", task.task_id)
                break

    # NOTE: Auto-detection of new tasks from session summaries is disabled.
    # Investigation showed zero cases where auto-detected tasks drove useful
    # work; they produced noise (wrong assignees, stale "待ち" items, literal
    # "なし" entries) and duplicated what heartbeat Observe→Plan already covers.
    if parsed.new_tasks:
        logger.debug(
            "Session summary contained %d new_tasks (auto-registration disabled)",
            len(parsed.new_tasks),
        )


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
