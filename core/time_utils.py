from __future__ import annotations
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.

"""Timezone-aware datetime helpers.

Provides ``now_jst()`` and ``now_iso()`` as drop-in replacements for
``datetime.now()`` throughout the codebase, ensuring all timestamps are
timezone-aware (JST / Asia/Tokyo).
"""

from datetime import datetime
from zoneinfo import ZoneInfo

_DEFAULT_TZ = ZoneInfo("Asia/Tokyo")


def now_jst() -> datetime:
    """Return current time as timezone-aware JST datetime."""
    return datetime.now(tz=_DEFAULT_TZ)


def now_iso() -> str:
    """Return current time as ISO8601 string with JST offset."""
    return now_jst().isoformat()


def ensure_aware(dt: datetime) -> datetime:
    """Ensure *dt* is timezone-aware.  Naive datetimes are assumed JST."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=_DEFAULT_TZ)
    return dt
