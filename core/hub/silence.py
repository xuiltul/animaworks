from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Human alerts for hub members that have stopped visiting."""

import asyncio
import logging
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from core.hub.members import (
    expect_every_seconds,
    last_seen,
    load_members,
    member_path,
    member_state,
    parse_iso,
    update_member_state,
)
from core.hub.wake import notify_human
from core.i18n import t

logger = logging.getLogger("animaworks.hub.silence")

DEFAULT_INTERVAL_S = 15 * 60


def _member_created_at(shared_dir: Path, member_id: str) -> str | None:
    try:
        stamp = member_path(shared_dir, member_id).stat().st_mtime
    except OSError:
        return None
    return datetime.fromtimestamp(stamp, UTC).isoformat()


def _effective_last_seen(shared_dir: Path, member_id: str) -> str | None:
    recorded = last_seen(shared_dir, member_id)
    if recorded is not None and parse_iso(recorded) is not None:
        return recorded
    return _member_created_at(shared_dir, member_id)


def check_silent_members(
    shared_dir: Path,
    *,
    now: datetime | None = None,
    notifier: Any | None = None,
) -> list[str]:
    """Alert once per silence window for enabled members with ``expect_every``.

    Returns the member ids that were alerted.  The alert marker is stored in
    ``shared/hub/state/<id>.json`` as ``silence_alerted_for`` so a later
    ``last_seen`` update can trigger a fresh alert.
    """
    current = now or datetime.now(UTC)
    if current.tzinfo is None:
        current = current.replace(tzinfo=UTC)
    alerted: list[str] = []
    for member_id, member in sorted(load_members(shared_dir).items()):
        if not member.enabled or not member.expect_every:
            continue
        expected = expect_every_seconds(member.expect_every)
        if expected is None:
            continue
        effective = _effective_last_seen(shared_dir, member_id)
        if effective is None:
            continue
        seen_at = parse_iso(effective)
        if seen_at is None:
            continue
        if (current - seen_at).total_seconds() <= expected:
            continue

        if member_state(shared_dir, member_id).get("silence_alerted_for") == effective:
            continue

        notify_human(
            t("hub.silence_subject", name=member.display_name),
            t("hub.silence_body", name=member.display_name, last_seen=effective),
            "normal",
            notifier=notifier,
        )
        update_member_state(shared_dir, member_id, silence_alerted_for=effective)
        alerted.append(member_id)
    return alerted


async def silence_alert_loop(
    shared_dir: Path,
    stop_event: threading.Event,
    *,
    interval_s: float = DEFAULT_INTERVAL_S,
) -> None:
    """Run :func:`check_silent_members` every *interval_s* until stopped."""
    while not stop_event.is_set():
        await asyncio.to_thread(stop_event.wait, max(0.0, interval_s))
        if stop_event.is_set():
            break
        try:
            await asyncio.to_thread(check_silent_members, shared_dir)
        except Exception:
            logger.warning("Hub silence check failed", exc_info=True)
