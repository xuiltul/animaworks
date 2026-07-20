# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.

"""Multi-source external tasks collector with per-source fault isolation.

Each source runs independently: a failure (exception or missing credentials)
keeps the previous snapshot's tasks for that source and marks health as
``unavailable``. Other sources continue unaffected.

Task id convention (source implementations MUST follow):
    ``"{source_type}-{stable_external_id}"``
e.g. ``github-pr-12345``, ``slack-C01-1234567890.123456``.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from datetime import datetime, timedelta
from urllib.parse import urlparse

from core.config.schemas import ExternalTasksConfig
from core.external_tasks.models import ExternalTask, Snapshot, SourceHealth
from core.external_tasks.sources.chatwork import collect_chatwork
from core.external_tasks.sources.github import collect_github
from core.external_tasks.sources.gmail import collect_gmail
from core.external_tasks.sources.slack import collect_slack
from core.time_utils import ensure_aware

logger = logging.getLogger("animaworks.external_tasks.collector")

# Control characters: C0 (\x00-\x1f) and DEL (\x7f)
_CONTROL_CHARS_RE = re.compile(r"[\x00-\x1f\x7f]")
_TITLE_MAX_LEN = 200
_URL_MAX_LEN = 2048
_PRIORITY_AGE_DAYS = 3
_PRIORITY_DECAY = 20
_PRIORITY_FLOOR = 10
_ALLOWED_URL_SCHEMES = frozenset({"http", "https"})

CollectFn = Callable[[], list[ExternalTask]]

SOURCE_REGISTRY: dict[str, CollectFn] = {
    "github": collect_github,
    "slack": collect_slack,
    "chatwork": collect_chatwork,
    "gmail": collect_gmail,
}

_SOURCE_ATTRS = ("github", "slack", "chatwork", "gmail")


class CredentialNotFoundError(Exception):
    """Raised by a source collector when required credentials are missing."""


def _error_code(exc: BaseException) -> str:
    """Map exception to a short public error code (no raw exception text)."""
    if isinstance(exc, CredentialNotFoundError):
        return "credential_missing"
    return "collection_failed"


def collect_all(
    config: ExternalTasksConfig,
    previous: Snapshot,
    now: datetime,
) -> Snapshot:
    """Collect tasks from all enabled sources with fault isolation.

    - Sources disabled in *config* are skipped (no health entry).
    - On success, that source's tasks are replaced and health is ``ok``.
    - On failure (any exception, including :class:`CredentialNotFoundError`),
      tasks for that source are carried over from *previous* and health is
      ``unavailable`` with a short error code (details stay in server logs).
    - New tasks get title/URL sanitize + priority decay; carry-over tasks get
      title/URL sanitize only (no re-decay).
    """
    resolved_now = ensure_aware(now)
    collected_at = resolved_now.isoformat()

    previous_by_source: dict[str, list[ExternalTask]] = {}
    for task in previous.tasks:
        previous_by_source.setdefault(task.source_type, []).append(task)

    new_sources: dict[str, SourceHealth] = {}
    new_tasks: list[ExternalTask] = []

    for source_name in _SOURCE_ATTRS:
        if not getattr(config.sources, source_name, False):
            continue
        collect_fn = SOURCE_REGISTRY.get(source_name)
        if collect_fn is None:
            continue

        try:
            source_tasks = collect_fn()
            for task in source_tasks:
                new_tasks.append(
                    _normalize_task(task, resolved_now, apply_priority_decay=True)
                )
            new_sources[source_name] = SourceHealth(
                status="ok",
                collected_at=collected_at,
                error=None,
            )
        except Exception as exc:
            err_code = _error_code(exc)
            prev_health = previous.sources.get(source_name)
            # Suppress repeated identical credential/source errors to avoid WARN noise
            # every collection interval (e.g. every 5 minutes).
            same_error = (
                prev_health is not None
                and prev_health.error is not None
                and prev_health.error == err_code
            )
            log_fn = logger.debug if same_error else logger.warning
            log_fn(
                "External tasks source %s failed: %s",
                source_name,
                exc,
                exc_info=True,
            )
            for task in previous_by_source.get(source_name, []):
                # Carry-over: sanitize only; do not re-apply priority decay.
                new_tasks.append(
                    _normalize_task(task, resolved_now, apply_priority_decay=False)
                )
            # Keep previous collected_at on failure (None if never succeeded).
            prev_collected_at = (
                prev_health.collected_at if prev_health is not None else None
            )
            new_sources[source_name] = SourceHealth(
                status="unavailable",
                collected_at=prev_collected_at,
                error=err_code,
            )

    return Snapshot(
        version=1,
        last_collected_at=collected_at,
        sources=new_sources,
        tasks=new_tasks,
    )


def _normalize_task(
    task: ExternalTask,
    now: datetime,
    *,
    apply_priority_decay: bool = True,
) -> ExternalTask:
    """Apply title/URL sanitize; optionally age-based priority decay."""
    title = _normalize_title(task.title)
    source_url = _sanitize_url(task.source_url)
    if apply_priority_decay:
        priority = _adjust_priority(task.priority, task.last_updated_at, now)
    else:
        priority = task.priority
    if (
        title == task.title
        and source_url == task.source_url
        and priority == task.priority
    ):
        return task
    return task.model_copy(
        update={
            "title": title,
            "source_url": source_url,
            "priority": priority,
        }
    )


def _normalize_title(title: str) -> str:
    cleaned = _CONTROL_CHARS_RE.sub("", title)
    if len(cleaned) > _TITLE_MAX_LEN:
        return cleaned[:_TITLE_MAX_LEN]
    return cleaned


def _sanitize_url(url: str | None) -> str | None:
    if url is None:
        return None
    stripped = url.strip()
    if not stripped or len(stripped) > _URL_MAX_LEN:
        return None
    if _CONTROL_CHARS_RE.search(stripped):
        return None
    try:
        parsed = urlparse(stripped)
    except ValueError:
        return None
    if parsed.scheme.lower() not in _ALLOWED_URL_SCHEMES:
        return None
    if not parsed.netloc:
        return None
    return stripped


def _adjust_priority(priority: int, last_updated_at: str, now: datetime) -> int:
    updated = _parse_datetime(last_updated_at)
    if updated is None:
        return priority
    age = now - ensure_aware(updated)
    if age > timedelta(days=_PRIORITY_AGE_DAYS):
        return max(_PRIORITY_FLOOR, priority - _PRIORITY_DECAY)
    return priority


def _parse_datetime(value: str) -> datetime | None:
    try:
        # Support trailing Z
        if value.endswith("Z"):
            value = value[:-1] + "+00:00"
        return datetime.fromisoformat(value)
    except (TypeError, ValueError):
        return None
