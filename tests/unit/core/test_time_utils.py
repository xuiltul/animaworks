"""Unit tests for core/time_utils.py — timezone-aware helpers."""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import re
from datetime import date, datetime, timezone
from unittest.mock import patch
from zoneinfo import ZoneInfo

import pytest

import core.time_utils as tu
from core.time_utils import (
    configure_timezone,
    ensure_aware,
    get_app_timezone,
    now_iso,
    now_jst,
    now_local,
    today_local,
)


@pytest.fixture(autouse=True)
def _reset_app_tz():
    """Reset _app_tz before and after each test."""
    original = tu._app_tz
    tu._app_tz = None
    yield
    tu._app_tz = original


# ── configure_timezone ─────────────────────────────────────


class TestConfigureTimezone:
    def test_explicit_iana_name(self) -> None:
        configure_timezone("America/New_York")
        tz = get_app_timezone()
        assert tz.key == "America/New_York"

    def test_explicit_asia_tokyo(self) -> None:
        configure_timezone("Asia/Tokyo")
        tz = get_app_timezone()
        assert tz.key == "Asia/Tokyo"

    def test_explicit_utc(self) -> None:
        configure_timezone("UTC")
        tz = get_app_timezone()
        assert tz.key == "UTC"

    def test_explicit_europe(self) -> None:
        configure_timezone("Europe/London")
        tz = get_app_timezone()
        assert tz.key == "Europe/London"

    def test_invalid_tz_falls_back(self) -> None:
        configure_timezone("Invalid/NotATimezone")
        tz = get_app_timezone()
        assert tz.key == "Asia/Tokyo"

    def test_empty_string_auto_detects(self) -> None:
        configure_timezone("")
        tz = get_app_timezone()
        assert isinstance(tz, ZoneInfo)
        assert tz.key  # IANA name should be non-empty

    def test_auto_detect_failure_falls_back(self) -> None:
        with patch("tzlocal.get_localzone", side_effect=RuntimeError("no tz")):
            configure_timezone("")
        tz = get_app_timezone()
        assert tz.key == "Asia/Tokyo"

    def test_unconfigured_returns_fallback(self) -> None:
        tz = get_app_timezone()
        assert tz.key == "Asia/Tokyo"


# ── now_local / now_jst ────────────────────────────────────


class TestNowLocal:
    def test_returns_datetime(self) -> None:
        assert isinstance(now_local(), datetime)

    def test_timezone_aware(self) -> None:
        dt = now_local()
        assert dt.tzinfo is not None

    def test_uses_configured_tz(self) -> None:
        configure_timezone("America/New_York")
        dt = now_local()
        assert dt.tzinfo == ZoneInfo("America/New_York")

    def test_default_uses_fallback(self) -> None:
        dt = now_local()
        assert dt.tzinfo == ZoneInfo("Asia/Tokyo")

    def test_jst_alias_works(self) -> None:
        assert now_jst is now_local

    def test_jst_alias_returns_same_tz(self) -> None:
        configure_timezone("Europe/Berlin")
        dt = now_jst()
        assert dt.tzinfo == ZoneInfo("Europe/Berlin")


# ── today_local ────────────────────────────────────────────


class TestTodayLocal:
    def test_returns_date(self) -> None:
        assert isinstance(today_local(), date)

    def test_matches_now_local_date(self) -> None:
        assert today_local() == now_local().date()

    def test_uses_configured_tz(self) -> None:
        configure_timezone("Pacific/Auckland")
        d = today_local()
        expected = datetime.now(tz=ZoneInfo("Pacific/Auckland")).date()
        assert d == expected


# ── now_iso ────────────────────────────────────────────────


class TestNowIso:
    def test_returns_string(self) -> None:
        assert isinstance(now_iso(), str)

    def test_iso_pattern(self) -> None:
        iso = now_iso()
        assert re.match(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}", iso)

    def test_roundtrip(self) -> None:
        iso = now_iso()
        dt = datetime.fromisoformat(iso)
        assert dt.tzinfo is not None

    def test_contains_configured_offset(self) -> None:
        configure_timezone("Asia/Tokyo")
        iso = now_iso()
        assert "+09:00" in iso

    def test_utc_offset(self) -> None:
        configure_timezone("UTC")
        iso = now_iso()
        assert "+00:00" in iso


# ── ensure_aware ───────────────────────────────────────────


class TestEnsureAware:
    def test_naive_becomes_app_tz(self) -> None:
        configure_timezone("America/New_York")
        naive = datetime(2026, 1, 1, 12, 0, 0)
        aware = ensure_aware(naive)
        assert aware.tzinfo == ZoneInfo("America/New_York")
        assert aware.hour == 12

    def test_naive_default_fallback(self) -> None:
        naive = datetime(2026, 1, 1, 12, 0, 0)
        aware = ensure_aware(naive)
        assert aware.tzinfo == ZoneInfo("Asia/Tokyo")

    def test_aware_unchanged(self) -> None:
        utc_dt = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        result = ensure_aware(utc_dt)
        assert result.tzinfo == timezone.utc
        assert result is utc_dt

    def test_jst_aware_unchanged(self) -> None:
        jst_dt = datetime(2026, 1, 1, 12, 0, 0, tzinfo=ZoneInfo("Asia/Tokyo"))
        result = ensure_aware(jst_dt)
        assert result is jst_dt

    def test_arithmetic_naive_vs_aware(self) -> None:
        aware = now_local()
        naive = datetime(2026, 1, 1, 0, 0, 0)
        diff = aware - ensure_aware(naive)
        assert diff.total_seconds() > 0
