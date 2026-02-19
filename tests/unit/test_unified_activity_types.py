"""Unit tests for unified activity type mapping (TYPE_ICONS consolidation).

Verifies:
1. Shared JS module structure (activity-types.js)
2. Consumer JS files: local TYPE_ICONS removed, shared import added
3. Backend: activity.log() calls include summary parameter
"""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import re
from pathlib import Path

import pytest

# ── Constants ─────────────────────────────────────────────

_WORKTREE = Path(__file__).resolve().parents[2]
_STATIC = _WORKTREE / "server" / "static"
_SHARED_MODULE = _STATIC / "shared" / "activity-types.js"

_CONSUMER_FILES = [
    _STATIC / "pages" / "activity.js",
    _STATIC / "pages" / "home.js",
    _STATIC / "pages" / "chat.js",
    _STATIC / "modules" / "activity.js",
    _STATIC / "workspace" / "modules" / "timeline.js",
    _STATIC / "workspace" / "modules" / "app.js",
]


# ── Shared Module Tests ──────────────────────────────────

class TestSharedActivityTypesModule:
    """Verify server/static/shared/activity-types.js structure."""

    def test_module_exists(self) -> None:
        assert _SHARED_MODULE.exists(), "shared/activity-types.js must exist"

    def test_exports_type_icons(self) -> None:
        content = _SHARED_MODULE.read_text(encoding="utf-8")
        assert "export const TYPE_ICONS" in content

    def test_exports_get_icon(self) -> None:
        content = _SHARED_MODULE.read_text(encoding="utf-8")
        assert "export function getIcon" in content

    def test_exports_get_display_summary(self) -> None:
        content = _SHARED_MODULE.read_text(encoding="utf-8")
        assert "export function getDisplaySummary" in content

    def test_exports_normalize_event(self) -> None:
        content = _SHARED_MODULE.read_text(encoding="utf-8")
        assert "export function normalizeEvent" in content

    def test_exports_type_categories(self) -> None:
        content = _SHARED_MODULE.read_text(encoding="utf-8")
        assert "export const TYPE_CATEGORIES" in content

    def test_contains_all_api_types(self) -> None:
        """All 14 API event types must be present."""
        content = _SHARED_MODULE.read_text(encoding="utf-8")
        api_types = [
            "message_received", "response_sent", "channel_read", "channel_post",
            "dm_received", "dm_sent", "human_notify", "tool_use",
            "heartbeat_start", "heartbeat_end", "cron_executed", "memory_write",
            "error", "issue_resolved",
        ]
        for t in api_types:
            assert f"{t}:" in content, f"API type '{t}' missing from TYPE_ICONS"

    def test_contains_websocket_types(self) -> None:
        """WebSocket simplified types must be present."""
        content = _SHARED_MODULE.read_text(encoding="utf-8")
        ws_types = [
            "message:", "heartbeat:", "cron:", "chat:", "board:",
            "notification:", "status:", "system:", "session:",
        ]
        for t in ws_types:
            # Match as key in TYPE_ICONS (not as substring of API types)
            pattern = rf'^\s+{re.escape(t)}'
            assert re.search(pattern, content, re.MULTILINE), \
                f"WebSocket type '{t}' missing from TYPE_ICONS"

    def test_type_categories_count(self) -> None:
        """TYPE_CATEGORIES should have 9 entries (All + 8 categories)."""
        content = _SHARED_MODULE.read_text(encoding="utf-8")
        count = content.count("{ label:")
        assert count == 9, f"Expected 9 TYPE_CATEGORIES entries, got {count}"

    def test_fallback_icon_defined(self) -> None:
        content = _SHARED_MODULE.read_text(encoding="utf-8")
        assert "FALLBACK_ICON" in content

    def test_type_defaults_defined(self) -> None:
        content = _SHARED_MODULE.read_text(encoding="utf-8")
        assert "TYPE_DEFAULTS" in content


# ── Consumer File Tests ──────────────────────────────────

class TestConsumerFilesNoLocalTypeIcons:
    """Verify local TYPE_ICONS definitions removed from consumer files."""

    @pytest.mark.parametrize("filepath", _CONSUMER_FILES, ids=[p.name for p in _CONSUMER_FILES])
    def test_no_local_type_icons_declaration(self, filepath: Path) -> None:
        """Consumer files must NOT declare their own TYPE_ICONS constant."""
        content = filepath.read_text(encoding="utf-8")
        # Match 'const TYPE_ICONS = {' or 'const _TYPE_ICONS = {'
        pattern = r'^\s*const\s+_?TYPE_ICONS\s*='
        matches = re.findall(pattern, content, re.MULTILINE)
        assert len(matches) == 0, (
            f"{filepath.name} still has local TYPE_ICONS declaration: {matches}"
        )


class TestConsumerFilesImportShared:
    """Verify consumer files import from shared/activity-types.js."""

    @pytest.mark.parametrize("filepath", _CONSUMER_FILES, ids=[p.name for p in _CONSUMER_FILES])
    def test_imports_from_shared_module(self, filepath: Path) -> None:
        content = filepath.read_text(encoding="utf-8")
        assert "activity-types.js" in content, (
            f"{filepath.name} does not import from shared/activity-types.js"
        )


class TestConsumerFilesUseSharedFunctions:
    """Verify consumer files use getIcon() instead of TYPE_ICONS[x]."""

    _FILES_USING_GET_ICON = [
        _STATIC / "pages" / "activity.js",
        _STATIC / "pages" / "home.js",
        _STATIC / "pages" / "chat.js",
        _STATIC / "modules" / "activity.js",
        _STATIC / "workspace" / "modules" / "timeline.js",
        _STATIC / "workspace" / "modules" / "app.js",
    ]

    @pytest.mark.parametrize("filepath", _FILES_USING_GET_ICON, ids=[p.name for p in _FILES_USING_GET_ICON])
    def test_uses_get_icon(self, filepath: Path) -> None:
        content = filepath.read_text(encoding="utf-8")
        assert "getIcon(" in content, (
            f"{filepath.name} should use getIcon() function"
        )

    def test_activity_page_uses_get_display_summary(self) -> None:
        content = (_STATIC / "pages" / "activity.js").read_text(encoding="utf-8")
        assert "getDisplaySummary(" in content

    def test_home_page_uses_get_display_summary(self) -> None:
        content = (_STATIC / "pages" / "home.js").read_text(encoding="utf-8")
        assert "getDisplaySummary(" in content

    def test_chat_page_uses_get_display_summary(self) -> None:
        content = (_STATIC / "pages" / "chat.js").read_text(encoding="utf-8")
        assert "getDisplaySummary(" in content

    def test_timeline_uses_normalize_event(self) -> None:
        content = (_STATIC / "workspace" / "modules" / "timeline.js").read_text(encoding="utf-8")
        assert "normalizeEvent(" in content


# ── WebSocket Default Case Test ──────────────────────────

class TestWebSocketDefaultCase:
    """Verify websocket.js default case no longer uses JSON.stringify."""

    def test_no_json_stringify_in_default(self) -> None:
        content = (_STATIC / "modules" / "websocket.js").read_text(encoding="utf-8")
        # Find the default case section
        default_idx = content.rfind("default:")
        assert default_idx >= 0, "default: case not found"
        default_section = content[default_idx:default_idx + 300]
        assert "JSON.stringify" not in default_section, (
            "default case should not use JSON.stringify"
        )

    def test_uses_summary_fallback(self) -> None:
        content = (_STATIC / "modules" / "websocket.js").read_text(encoding="utf-8")
        default_idx = content.rfind("default:")
        default_section = content[default_idx:default_idx + 300]
        assert "data.summary" in default_section, (
            "default case should use data.summary fallback"
        )


# ── Backend Summary Tests ────────────────────────────────

class TestBackendActivityLogSummary:
    """Verify core/anima.py activity.log() calls include summary."""

    _ANIMA_PY = _WORKTREE / "core" / "anima.py"

    def test_heartbeat_start_has_summary(self) -> None:
        content = self._ANIMA_PY.read_text(encoding="utf-8")
        pattern = r'activity\.log\("heartbeat_start",\s*summary="定期巡回開始"\)'
        assert re.search(pattern, content), (
            "heartbeat_start activity.log() should include summary='定期巡回開始'"
        )

    def test_message_received_has_summary(self) -> None:
        content = self._ANIMA_PY.read_text(encoding="utf-8")
        pattern = r'activity\.log\("message_received",\s*content=content,\s*summary=content\[:100\]'
        matches = re.findall(pattern, content)
        assert len(matches) == 2, (
            f"Expected 2 message_received calls with summary, found {len(matches)}"
        )

    def test_dm_received_has_summary(self) -> None:
        content = self._ANIMA_PY.read_text(encoding="utf-8")
        pattern = r'activity\.log\("dm_received",\s*content=_m\.content,\s*summary=_m\.content\[:200\]'
        assert re.search(pattern, content), (
            "dm_received activity.log() should use full content and summary=_m.content[:200]"
        )


# ── Workspace Timeline Filter Tests ─────────────────────

class TestTimelineFilterStructure:
    """Verify timeline.js filter structure changed to types-based."""

    _TIMELINE = _STATIC / "workspace" / "modules" / "timeline.js"

    def test_filter_uses_types_array(self) -> None:
        """filterDefs should use 'types' arrays, not 'value' strings."""
        content = self._TIMELINE.read_text(encoding="utf-8")
        assert '"types":' in content or "types:" in content, (
            "filterDefs should use types arrays"
        )
        # Should NOT have the old value-based format
        assert 'value: "all"' not in content, (
            "Old value-based filter format should be removed"
        )

    def test_current_filter_is_array(self) -> None:
        """_currentFilter should be initialized as empty array."""
        content = self._TIMELINE.read_text(encoding="utf-8")
        assert "let _currentFilter = []" in content, (
            "_currentFilter should be initialized as empty array"
        )

    def test_filter_includes_detailed_types(self) -> None:
        """Filters should include detailed API types."""
        content = self._TIMELINE.read_text(encoding="utf-8")
        for t in ["message_received", "response_sent", "heartbeat_start", "cron_executed"]:
            assert t in content, f"Filter should include detailed type '{t}'"


# ── Workspace App.js Field Name Tests ────────────────────

class TestAppJsFieldNames:
    """Verify app.js addTimelineEvent calls use correct field names."""

    _APP_JS = _STATIC / "workspace" / "modules" / "app.js"

    def test_no_animas_array_in_timeline_events(self) -> None:
        """addTimelineEvent calls should use 'anima' not 'animas'."""
        content = self._APP_JS.read_text(encoding="utf-8")
        # Find addTimelineEvent calls and check for animas (should be absent)
        # Look for 'animas:' within addTimelineEvent context
        pattern = r'addTimelineEvent\(\{[^}]*animas:'
        matches = re.findall(pattern, content, re.DOTALL)
        assert len(matches) == 0, (
            f"Found {len(matches)} addTimelineEvent calls still using 'animas:'"
        )

    def test_uses_ts_not_timestamp(self) -> None:
        """addTimelineEvent calls should use 'ts' not 'timestamp'."""
        content = self._APP_JS.read_text(encoding="utf-8")
        pattern = r'addTimelineEvent\(\{[^}]*\btimestamp:'
        matches = re.findall(pattern, content, re.DOTALL)
        assert len(matches) == 0, (
            f"Found {len(matches)} addTimelineEvent calls still using 'timestamp:'"
        )

    def test_uses_meta_not_metadata(self) -> None:
        """addTimelineEvent calls should use 'meta' not 'metadata'."""
        content = self._APP_JS.read_text(encoding="utf-8")
        pattern = r'addTimelineEvent\(\{[^}]*\bmetadata:'
        matches = re.findall(pattern, content, re.DOTALL)
        assert len(matches) == 0, (
            f"Found {len(matches)} addTimelineEvent calls still using 'metadata:'"
        )
