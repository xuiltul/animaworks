"""Unit tests for activity timeline fix — JS pattern validation and timestamp format."""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

import pytest

# ── Paths ──────────────────────────────────────────────

REPO_ROOT = Path(__file__).resolve().parents[2]
TIMELINE_JS = REPO_ROOT / "server" / "static" / "workspace" / "modules" / "timeline.js"
APP_JS = REPO_ROOT / "server" / "static" / "workspace" / "modules" / "app.js"


# ── Test 1: timeline.js exports localISOString ─────────


class TestTimelineLocalISOString:
    """Verify timeline.js contains and exports the localISOString helper."""

    def test_localISOString_exported(self) -> None:
        src = TIMELINE_JS.read_text(encoding="utf-8")
        assert "export function localISOString()" in src

    def test_localISOString_no_Z_suffix(self) -> None:
        """The helper must NOT append a 'Z' suffix (naive local time)."""
        src = TIMELINE_JS.read_text(encoding="utf-8")
        # Extract the localISOString function body
        match = re.search(
            r"export function localISOString\(\)\s*\{(.*?)\n\}",
            src,
            re.DOTALL,
        )
        assert match, "localISOString function not found"
        body = match.group(1)
        # Must not contain a Z suffix in the template literal
        assert '"Z"' not in body and "'Z'" not in body and "`Z`" not in body


# ── Test 2: timeline.js sort uses Date comparison ──────


class TestTimelineSortFix:
    """Verify sort functions use Date comparison instead of localeCompare."""

    def test_no_localeCompare_in_sort(self) -> None:
        """localeCompare must not appear in any sort callback."""
        src = TIMELINE_JS.read_text(encoding="utf-8")
        # After the fix, localeCompare should not be used for event sorting
        assert "localeCompare" not in src

    def test_tsDescending_comparator_exists(self) -> None:
        src = TIMELINE_JS.read_text(encoding="utf-8")
        assert "_tsDescending" in src

    def test_sort_calls_use_tsDescending(self) -> None:
        src = TIMELINE_JS.read_text(encoding="utf-8")
        assert src.count("_events.sort(_tsDescending)") >= 2


# ── Test 3: app.js imports localISOString ──────────────


class TestAppJsImport:
    """Verify app.js imports localISOString from timeline.js."""

    def test_import_localISOString(self) -> None:
        src = APP_JS.read_text(encoding="utf-8")
        assert "localISOString" in src
        assert 'from "./timeline.js"' in src


# ── Test 4: app.js uses localISOString for all addTimelineEvent calls ──


class TestAppJsTimestampUsage:
    """All addTimelineEvent calls must use localISOString() instead of new Date().toISOString()."""

    def test_no_raw_toISOString_in_addTimelineEvent(self) -> None:
        """new Date().toISOString() must not appear near addTimelineEvent calls."""
        src = APP_JS.read_text(encoding="utf-8")
        # Find all addTimelineEvent blocks
        blocks = re.findall(
            r"addTimelineEvent\(\{[^}]+\}\)",
            src,
            re.DOTALL,
        )
        assert len(blocks) > 0, "No addTimelineEvent calls found"
        for block in blocks:
            assert "new Date().toISOString()" not in block, (
                f"Found raw toISOString in addTimelineEvent block: {block[:100]}..."
            )

    def test_localISOString_usage_count(self) -> None:
        """localISOString() should appear at least 5 times (4 existing + 1 active handler).

        Note: chat.response handler was removed (dead code — no server emit).
        anima.notification handler is a no-op (Arch-1 fix for double event).
        """
        src = APP_JS.read_text(encoding="utf-8")
        count = src.count("localISOString()")
        assert count >= 5, f"Expected ≥5 localISOString() usages, got {count}"


# ── Test 5: New event handlers exist in app.js ─────────


class TestNewEventHandlers:
    """Verify the three new WebSocket event handlers exist."""

    @pytest.mark.parametrize("event_name", [
        "anima.proactive_message",
        "anima.notification",
    ])
    def test_handler_registered(self, event_name: str) -> None:
        src = APP_JS.read_text(encoding="utf-8")
        assert f'onEvent("{event_name}"' in src

    def test_chat_response_handler_removed(self) -> None:
        """chat.response handler was dead code (no server emit) and has been removed."""
        src = APP_JS.read_text(encoding="utf-8")
        assert 'onEvent("chat.response"' not in src

    def test_proactive_message_uses_dm_sent_type(self) -> None:
        src = APP_JS.read_text(encoding="utf-8")
        idx = src.index('onEvent("anima.proactive_message"')
        block = src[idx:idx + 400]
        assert 'type: "dm_sent"' in block

    def test_notification_handler_is_noop(self) -> None:
        """anima.notification handler is a no-op (timeline handled by proactive_message)."""
        src = APP_JS.read_text(encoding="utf-8")
        idx = src.index('onEvent("anima.notification"')
        block = src[idx:idx + 400]
        # After Arch-1 fix, the handler no longer adds timeline events
        assert 'type: "human_notify"' not in block
        assert "addTimelineEvent" not in block


# ── Test 6: Backend timestamp format matches client expectation ──


class TestBackendTimestampFormat:
    """Verify the server uses timezone-aware JST timestamps."""

    def test_jst_isoformat_no_z(self) -> None:
        """now_iso() (JST-aware) produces ISO without Z suffix."""
        from core.time_utils import now_iso
        ts = now_iso()
        assert not ts.endswith("Z")

    def test_jst_isoformat_pattern(self) -> None:
        """Format matches YYYY-MM-DDTHH:MM:SS.ffffff+09:00."""
        from core.time_utils import now_iso
        ts = now_iso()
        assert re.match(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+\+09:00", ts)

    def test_activity_logger_uses_now_iso(self) -> None:
        """ActivityLogger.log() must use now_iso() for timezone-aware timestamps."""
        from core.memory.activity import ActivityLogger
        import inspect
        source = inspect.getsource(ActivityLogger.log)
        assert "now_iso()" in source
        # Must not use raw datetime.now()
        assert "datetime.now()" not in source
