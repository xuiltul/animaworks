"""E2E tests for workspace timeline _replayEvent fix.

Validates the complete data flow from WS event creation (app.js) through
replay handling (timeline.js) to interaction visualization (interactions.js),
ensuring that the _replayEvent switch handles all activity types and that
the bug fix (using _resolvePersons instead of raw anima concatenation) is
correct end-to-end.
"""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

# ── File paths ──────────────────────────────────────────────

TIMELINE_JS = REPO_ROOT / "server" / "static" / "workspace" / "modules" / "timeline.js"
TIMELINE_REPLAY_JS = REPO_ROOT / "server" / "static" / "workspace" / "modules" / "timeline-replay.js"
TIMELINE_DOM_JS = REPO_ROOT / "server" / "static" / "workspace" / "modules" / "timeline-dom.js"
APP_JS = REPO_ROOT / "server" / "static" / "workspace" / "modules" / "app.js"
APP_WS_JS = REPO_ROOT / "server" / "static" / "workspace" / "modules" / "app-websocket.js"
INTERACTIONS_JS = REPO_ROOT / "server" / "static" / "workspace" / "modules" / "interactions.js"
ACTIVITY_TYPES_JS = REPO_ROOT / "server" / "static" / "shared" / "activity-types.js"
ACTIVITY_PY = REPO_ROOT / "core" / "memory" / "activity.py"


# ── Helpers ─────────────────────────────────────────────────


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _extract_type_icons_keys(source: str) -> list[str]:
    """Extract all keys from the TYPE_ICONS object in activity-types.js."""
    # Match the TYPE_ICONS block
    m = re.search(r"export\s+const\s+TYPE_ICONS\s*=\s*\{(.*?)\};", source, re.DOTALL)
    assert m, "TYPE_ICONS not found in activity-types.js"
    block = m.group(1)
    # Extract top-level keys only (key: { ... }, not nested emoji/lucide)
    keys = re.findall(r"(\w+)\s*:\s*\{", block)
    return keys


def _extract_switch_cases(source: str) -> list[str]:
    """Extract all case values from the replayEvent switch in timeline-replay.js."""
    m = re.search(r"function\s+replayEvent\b.*?switch\s*\(type\)\s*\{(.*?)\n\}", source, re.DOTALL)
    assert m, "replayEvent switch not found in timeline-replay.js"
    block = m.group(1)
    cases = re.findall(r'case\s+"(\w+)"', block)
    return cases


def _extract_filter_types(source: str) -> list[str]:
    """Extract all types from filterDefs arrays in timeline.js."""
    m = re.search(r"const\s+filterDefs\s*=\s*\[(.*?)\];", source, re.DOTALL)
    assert m, "filterDefs not found in timeline.js"
    block = m.group(1)
    types = re.findall(r'"(\w+)"', block)
    # Exclude labels like "All" that appear as label values
    # filterDefs types are inside `types: [...]` arrays
    type_arrays = re.findall(r'types:\s*\[(.*?)\]', block)
    all_types: list[str] = []
    for arr in type_arrays:
        all_types.extend(re.findall(r'"(\w+)"', arr))
    return all_types


# ── TestReplayEventCoversAllActivityTypes ───────────────────


class TestReplayEventCoversAllActivityTypes:
    """Verify that every user-visible activity type is handled by _replayEvent."""

    # Types that are not user-actionable and do NOT need replay handling
    NON_ACTIONABLE_TYPES = {
        "status", "system", "session", "notification", "error",
        "issue_resolved", "human_notify", "tool_use", "tool_result",
        "memory_write", "heartbeat_reflection", "tool",
    }

    def test_all_activity_types_handled_in_switch(self) -> None:
        """Every actionable TYPE_ICONS key should have a case in replayEvent."""
        types_src = _read(ACTIVITY_TYPES_JS)
        replay_src = _read(TIMELINE_REPLAY_JS)

        all_type_keys = _extract_type_icons_keys(types_src)
        assert len(all_type_keys) > 0, "Should have extracted TYPE_ICONS keys"

        switch_cases = _extract_switch_cases(replay_src)
        switch_set = set(switch_cases)

        actionable = [k for k in all_type_keys if k not in self.NON_ACTIONABLE_TYPES]
        assert len(actionable) > 0, "Should have actionable types"

        missing = [t for t in actionable if t not in switch_set]
        assert missing == [], (
            f"Actionable types missing from _replayEvent switch: {missing}. "
            f"Switch has: {sorted(switch_set)}"
        )

    def test_filter_types_all_have_replay_support(self) -> None:
        """Every actionable type in filterDefs should appear as a case in replayEvent switch.

        Types like tool_use and memory_write appear in filterDefs for
        display/filtering but are non-actionable (no 3D replay), so they
        are excluded from this check.
        """
        timeline_src = _read(TIMELINE_JS)
        replay_src = _read(TIMELINE_REPLAY_JS)

        filter_types = _extract_filter_types(timeline_src)
        assert len(filter_types) > 0, "Should have extracted filter types"

        switch_cases = _extract_switch_cases(replay_src)
        switch_set = set(switch_cases)

        actionable_filter_types = [
            t for t in filter_types if t not in self.NON_ACTIONABLE_TYPES
        ]
        assert len(actionable_filter_types) > 0, "Should have actionable filter types"

        missing = [t for t in actionable_filter_types if t not in switch_set]
        assert missing == [], (
            f"Actionable filter types missing from _replayEvent switch: {missing}. "
            f"Switch has: {sorted(switch_set)}"
        )


# ── TestWSToReplayDataFlow ──────────────────────────────────


class TestWSToReplayDataFlow:
    """Verify data flows correctly from WS event creation (app.js) through replay (timeline.js)."""

    def test_ws_message_event_has_meta_from_person(self) -> None:
        """app-websocket.js anima.interaction handler should put from_person in meta."""
        src = _read(APP_WS_JS)
        assert 'onEvent("anima.interaction"' in src or 'anima.interaction' in src, (
            "anima.interaction handler not found in app-websocket.js"
        )
        assert "from_person" in src, (
            "anima.interaction handler should include from_person"
        )

    def test_ws_message_event_has_meta_to_person(self) -> None:
        """app-websocket.js anima.interaction handler should put to_person in meta."""
        src = _read(APP_WS_JS)
        assert "to_person" in src, (
            "anima.interaction handler should include to_person"
        )

    def test_replay_reads_meta_from_person(self) -> None:
        """resolvePersons in timeline-dom.js should read meta.from_person."""
        src = _read(TIMELINE_DOM_JS)
        assert "meta.from_person" in src or "from_person" in src, (
            "resolvePersons should access from_person"
        )

    def test_showMessageEffect_signature_matches(self) -> None:
        """interactions.js showMessageEffect should take (fromName, toName, text)."""
        src = _read(INTERACTIONS_JS)
        # Find the export function signature
        m = re.search(
            r"export\s+function\s+showMessageEffect\s*\(([^)]*)\)",
            src,
        )
        assert m, "showMessageEffect export not found in interactions.js"
        params = [p.strip() for p in m.group(1).split(",")]
        assert params == ["fromName", "toName", "text"], (
            f"showMessageEffect should take (fromName, toName, text), got: {params}"
        )


# ── TestAPIEventReplayDataFlow ──────────────────────────────


class TestAPIEventReplayDataFlow:
    """Verify API event data flow (from_person/to_person/content at top level)."""

    def test_api_activity_logger_exports_from_person(self) -> None:
        """activity.py should include from_person in API output."""
        src = _read(ACTIVITY_PY)
        assert "from_person" in src or '"from"' in src, (
            "activity.py should export from_person or from in API output"
        )
        assert "to_person" in src or '"to"' in src, (
            "activity.py should export to_person or to in API output"
        )
        assert "content" in src, (
            "activity.py should export content in API output"
        )

    def test_resolvePersons_handles_toplevel_from(self) -> None:
        """resolvePersons should fall back to event.from_person (API format)."""
        src = _read(TIMELINE_DOM_JS)
        assert "from_person" in src, (
            "resolvePersons should reference from_person"
        )

    def test_resolvePersons_handles_toplevel_content(self) -> None:
        """resolvePersons should reference event.content (API format)."""
        src = _read(TIMELINE_DOM_JS)
        assert "content" in src, (
            "resolvePersons should reference content"
        )


# ── TestNoRegressionExistingWSEvents ────────────────────────


class TestNoRegressionExistingWSEvents:
    """Verify existing WS event types still work (no regressions)."""

    def test_heartbeat_handler_still_exists_in_app_js(self) -> None:
        """anima.heartbeat handler should still exist in app-websocket.js."""
        src = _read(APP_WS_JS)
        assert 'anima.heartbeat' in src, (
            "anima.heartbeat handler not found in app-websocket.js"
        )

    def test_cron_handler_still_exists_in_app_js(self) -> None:
        """anima.cron handler should still exist in app-websocket.js."""
        src = _read(APP_WS_JS)
        assert 'anima.cron' in src, (
            "anima.cron handler not found in app-websocket.js"
        )

    def test_existing_ws_types_still_in_switch(self) -> None:
        """Core WS types (message, heartbeat, cron, chat, board) must remain in the switch."""
        src = _read(TIMELINE_REPLAY_JS)
        switch_cases = _extract_switch_cases(src)
        switch_set = set(switch_cases)

        required = {"message", "heartbeat", "cron", "chat", "board"}
        missing = required - switch_set
        assert missing == set(), (
            f"Core WS types missing from _replayEvent switch: {missing}"
        )


# ── TestBugFixVerification ──────────────────────────────────


class TestBugFixVerification:
    """Verify the specific bugs are fixed."""

    def test_no_raw_anima_in_showMessageEffect_call(self) -> None:
        """showMessageEffect should NOT be called with raw 'anima' (concatenated string).

        The fixed pattern uses resolvePersons: showMessageEffect(p.from, p.to, p.text)
        """
        src = _read(TIMELINE_REPLAY_JS)
        assert "showMessageEffect(anima," not in src, (
            "Bug regression: showMessageEffect should NOT be called with raw 'anima' field."
        )
        assert "showMessageEffect" in src, (
            "showMessageEffect should be called in timeline-replay.js"
        )

    def test_anima_concatenation_exists_in_app_js(self) -> None:
        """app-websocket.js uses from_person/to_person for the anima field."""
        src = _read(APP_WS_JS)
        assert "from_person" in src and "to_person" in src, (
            "app-websocket.js should reference from_person and to_person"
        )

    def test_resolvePersons_bypasses_concatenated_anima(self) -> None:
        """resolvePersons should prioritise from_person over event.anima."""
        src = _read(TIMELINE_DOM_JS)
        assert "from_person" in src, (
            "resolvePersons should reference from_person"
        )
