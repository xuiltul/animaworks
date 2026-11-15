from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.

"""Record Anima activity (heartbeat / cron / task) on the company timeline.

Only the runs that actually did something are recorded.  The company and
enabled/disabled decision is the same one used by the notification timeline
writer: ``hub.timeline_enabled`` must be true and the Anima's company must
match ``hub.company``.  Recording failures never propagate to the caller;
heartbeat and cron execution must keep working even when the timeline is
unavailable.
"""

import logging
from datetime import datetime, timedelta
from typing import Any

from core.i18n import t
from core.time_utils import ensure_aware

logger = logging.getLogger(__name__)


def record_activity(
    anima_name: str,
    *,
    via: str,
    text: str,
    label: str | None = None,
    failed: bool = False,
) -> str | None:
    """Append one ``kind="activity"`` company timeline item.

    Returns the created item id, or ``None`` when the hub is disabled, the
    Anima belongs to another company, the text is empty, or writing failed.
    """
    try:
        name = str(anima_name or "").strip()
        if not name:
            return None

        body = str(text or "").strip()
        if not body:
            return None

        from core.config.models import load_config, read_anima_company_checked
        from core.paths import get_animas_dir, get_shared_dir

        hub = load_config().hub
        if not hub.timeline_enabled:
            return None

        readable, anima_company = read_anima_company_checked(get_animas_dir() / name)
        if not readable or not anima_company or anima_company != hub.company:
            return None

        if failed:
            body = f"{t('hub.activity_failed_prefix')} {body}"

        from core.hub.timeline import append_item

        item = append_item(
            get_shared_dir(),
            from_name=name,
            company=anima_company,
            kind="activity",
            text=body,
            source="anima",
            via=via,
            label=label,
        )
        return str(item["id"])
    except Exception:
        logger.warning("Failed to record timeline activity for %s", anima_name, exc_info=True)
        return None


def _window_entries(activity: Any, cron_entry: Any) -> list[Any] | None:
    """Load activity entries inside the cron execution window.

    Returns ``None`` when the window cannot be determined (missing or invalid
    timestamp / duration).  ``ActivityLogger._load_entries`` accepts
    ``since``/``until`` and already scans across date boundaries.
    """
    end_ts = getattr(cron_entry, "ts", None)
    meta = getattr(cron_entry, "meta", None)
    # Only a real dict can be inspected; this also keeps mocked entries from
    # leaving un-awaited AsyncMock coroutines behind in tests.
    duration_ms = meta.get("duration_ms") if isinstance(meta, dict) else None
    if not isinstance(end_ts, str) or not isinstance(duration_ms, (int, float)) or duration_ms <= 0:
        return None

    try:
        end = ensure_aware(datetime.fromisoformat(end_ts.replace("Z", "+00:00")))
    except (TypeError, ValueError):
        return None
    start = end - timedelta(milliseconds=float(duration_ms))

    loader = getattr(activity, "_load_entries", None)
    if not callable(loader):
        return None
    return list(loader(since=start, until=end))


def record_cron_activity(
    anima_name: str,
    *,
    activity: Any,
    cron_entry: Any,
    task_name: str,
    text: str,
    failed: bool = False,
    stop_kind: str | None = None,
) -> str | None:
    """Record one LLM cron run unless it did nothing.

    ``cron_entry`` is the ``cron_executed`` activity entry for this run;
    its ``ts`` and ``duration_ms`` define the window used for the noop
    decision.  Failed runs are always recorded.  ``budget_skipped`` runs
    (the LLM was not started) are never recorded.
    """
    if stop_kind == "budget_skipped":
        return None

    if failed:
        return record_activity(
            anima_name,
            via="cron",
            text=text,
            label=task_name,
            failed=True,
        )

    try:
        from core.memory.maintenance.cron_noop import is_noop_cron_execution

        window = _window_entries(activity, cron_entry)
        if window is not None and is_noop_cron_execution(
            cron_entry,
            window,
            ignore_memory_writes=True,
        ):
            return None
    except Exception:
        logger.warning(
            "Failed to judge cron activity for %s (%s)",
            anima_name,
            task_name,
            exc_info=True,
        )
        return None

    return record_activity(anima_name, via="cron", text=text, label=task_name)
