from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.

"""Timezone-aware datetime helpers.

Provides ``now_local()`` and ``now_iso()`` as drop-in replacements for
``datetime.now()`` throughout the codebase, ensuring all timestamps are
timezone-aware in the configured application timezone.

The application timezone is resolved as:
1. Explicit IANA name via ``configure_timezone("America/New_York")``
2. System timezone auto-detected by *tzlocal* when called with ``""``
3. Fallback to ``Asia/Tokyo`` if detection fails or is never configured
"""

import logging
from datetime import date, datetime
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)

_FALLBACK_TZ = ZoneInfo("Asia/Tokyo")
_app_tz: ZoneInfo | None = None


# ── Configuration ──────────────────────────────────────────


def configure_timezone(tz_name: str = "") -> None:
    """Set application timezone.

    Must be called once at startup before any concurrent access.

    Args:
        tz_name: IANA timezone name (e.g. ``"America/New_York"``).
                 Empty string triggers auto-detection from the system.
    """
    global _app_tz  # noqa: PLW0603
    if tz_name:
        try:
            _app_tz = ZoneInfo(tz_name)
            logger.info("Application timezone set to %s", tz_name)
        except (KeyError, Exception):
            logger.warning("Invalid timezone '%s', falling back to %s", tz_name, _FALLBACK_TZ.key)
            _app_tz = _FALLBACK_TZ
    else:
        try:
            from tzlocal import get_localzone

            detected = get_localzone()
            _app_tz = ZoneInfo(detected.key)
            logger.info("Auto-detected system timezone: %s", _app_tz.key)
        except Exception:
            logger.warning(
                "Failed to detect system timezone, falling back to %s",
                _FALLBACK_TZ.key,
            )
            _app_tz = _FALLBACK_TZ


def get_app_timezone() -> ZoneInfo:
    """Return the configured application timezone.

    Falls back to ``Asia/Tokyo`` if ``configure_timezone`` has not been called.
    """
    if _app_tz is None:
        logger.debug("Timezone not configured; using fallback %s", _FALLBACK_TZ.key)
        return _FALLBACK_TZ
    return _app_tz


# ── Datetime helpers ───────────────────────────────────────


def now_local() -> datetime:
    """Return current time as timezone-aware datetime in the app timezone."""
    return datetime.now(tz=get_app_timezone())


now_jst = now_local  # backward-compatibility alias


def today_local() -> date:
    """Return today's date in the app timezone."""
    return now_local().date()


def now_iso() -> str:
    """Return current time as ISO 8601 string with app timezone offset."""
    return now_local().isoformat()


def ensure_aware(dt: datetime) -> datetime:
    """Ensure *dt* is timezone-aware.  Naive datetimes assume app timezone."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=get_app_timezone())
    return dt
