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

logger = logging.getLogger("animaworks.conversation_memory")


def _record_resolutions(
    anima_dir: Path,
    memory_mgr: Any,
    resolved_items: list[str],
) -> None:
    """Record resolution events to ActivityLogger and shared registry."""
    from core.activity.logger import ActivityLogger

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

        # Company timeline: this is the only writer of ``issue_resolved``
        # (verified with grep), so the resolution is recorded exactly once.
        from core.hub.timeline_activity import record_activity

        record_activity(anima_dir.name, via="issue", text=item)

        # Layer 3: shared/resolutions.jsonl cross-org record
        try:
            memory_mgr.append_resolution(
                issue=item,
                resolver=anima_dir.name,
            )
        except Exception:
            logger.debug("Failed to write resolution registry", exc_info=True)
