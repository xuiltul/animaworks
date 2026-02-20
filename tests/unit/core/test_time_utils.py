"""Unit tests for core/time_utils.py — timezone-aware helpers."""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import re
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from core.time_utils import _DEFAULT_TZ, ensure_aware, now_iso, now_jst


class TestNowJst:
    def test_returns_datetime(self) -> None:
        assert isinstance(now_jst(), datetime)

    def test_timezone_aware(self) -> None:
        dt = now_jst()
        assert dt.tzinfo is not None

    def test_jst_offset(self) -> None:
        dt = now_jst()
        assert dt.utcoffset().total_seconds() == 9 * 3600  # +09:00

    def test_tzname(self) -> None:
        dt = now_jst()
        assert dt.tzinfo == ZoneInfo("Asia/Tokyo")


class TestNowIso:
    def test_returns_string(self) -> None:
        assert isinstance(now_iso(), str)

    def test_contains_jst_offset(self) -> None:
        iso = now_iso()
        assert "+09:00" in iso

    def test_iso_pattern(self) -> None:
        iso = now_iso()
        assert re.match(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}", iso)

    def test_roundtrip(self) -> None:
        iso = now_iso()
        dt = datetime.fromisoformat(iso)
        assert dt.tzinfo is not None


class TestEnsureAware:
    def test_naive_becomes_jst(self) -> None:
        naive = datetime(2026, 1, 1, 12, 0, 0)
        aware = ensure_aware(naive)
        assert aware.tzinfo == _DEFAULT_TZ
        assert aware.hour == 12  # value unchanged

    def test_aware_unchanged(self) -> None:
        utc_dt = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        result = ensure_aware(utc_dt)
        assert result.tzinfo == timezone.utc  # not replaced
        assert result is utc_dt  # same object

    def test_jst_aware_unchanged(self) -> None:
        jst_dt = datetime(2026, 1, 1, 12, 0, 0, tzinfo=ZoneInfo("Asia/Tokyo"))
        result = ensure_aware(jst_dt)
        assert result is jst_dt

    def test_arithmetic_naive_vs_aware(self) -> None:
        """ensure_aware allows safe subtraction of mixed datetimes."""
        aware = now_jst()
        naive = datetime(2026, 1, 1, 0, 0, 0)
        # This would raise TypeError without ensure_aware
        diff = aware - ensure_aware(naive)
        assert diff.total_seconds() > 0
