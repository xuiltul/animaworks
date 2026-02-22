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
APP_JS = REPO_ROOT / "server" / "static" / "workspace" / "modules" / "app.js"
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
    # Extract keys (word characters before the colon)
    keys = re.findall(r"(\w+)\s*:", block)
    return keys


def _extract_switch_cases(source: str) -> list[str]:
    """Extract all case values from the _replayEvent switch in timeline.js."""
    # Find the switch block inside _replayEvent
    m = re.search(r"function\s+_replayEvent\b.*?switch\s*\(type\)\s*\{(.*?)\n\}", source, re.DOTALL)
    assert m, "_replayEvent switch not found in timeline.js"
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
        "issue_resolved", "human_notify", "tool_use", "memory_write",
        "heartbeat_reflection",
    }

    def test_all_activity_types_handled_in_switch(self) -> None:
        """Every actionable TYPE_ICONS key should have a case in _replayEvent."""
        types_src = _read(ACTIVITY_TYPES_JS)
        timeline_src = _read(TIMELINE_JS)

        all_type_keys = _extract_type_icons_keys(types_src)
        assert len(all_type_keys) > 0, "Should have extracted TYPE_ICONS keys"

        switch_cases = _extract_switch_cases(timeline_src)
        switch_set = set(switch_cases)

        actionable = [k for k in all_type_keys if k not in self.NON_ACTIONABLE_TYPES]
        assert len(actionable) > 0, "Should have actionable types"

        missing = [t for t in actionable if t not in switch_set]
        assert missing == [], (
            f"Actionable types missing from _replayEvent switch: {missing}. "
            f"Switch has: {sorted(switch_set)}"
        )

    def test_filter_types_all_have_replay_support(self) -> None:
        """Every actionable type in filterDefs should appear as a case in _replayEvent switch.

        Types like tool_use and memory_write appear in filterDefs for
        display/filtering but are non-actionable (no 3D replay), so they
        are excluded from this check.
        """
        timeline_src = _read(TIMELINE_JS)

        filter_types = _extract_filter_types(timeline_src)
        assert len(filter_types) > 0, "Should have extracted filter types"

        switch_cases = _extract_switch_cases(timeline_src)
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
        """app.js anima.interaction handler should put from_person in meta."""
        src = _read(APP_JS)
        # Find the anima.interaction handler block
        assert 'onEvent("anima.interaction"' in src, (
            "anima.interaction handler not found in app.js"
        )
        # Verify meta includes from_person
        assert "from_person: data.from_person" in src or "from_person:" in src, (
            "anima.interaction handler should include from_person in meta"
        )
        # More specific: the addTimelineEvent call inside the interaction handler
        # should have meta.from_person
        interaction_block = src[src.index('onEvent("anima.interaction"'):]
        # Find the addTimelineEvent call within a reasonable scope
        next_on_event = interaction_block.find('onEvent(', 1)
        if next_on_event > 0:
            interaction_block = interaction_block[:next_on_event]
        assert "from_person: data.from_person" in interaction_block, (
            "addTimelineEvent in anima.interaction should set meta.from_person = data.from_person"
        )

    def test_ws_message_event_has_meta_to_person(self) -> None:
        """app.js anima.interaction handler should put to_person in meta."""
        src = _read(APP_JS)
        interaction_block = src[src.index('onEvent("anima.interaction"'):]
        next_on_event = interaction_block.find('onEvent(', 1)
        if next_on_event > 0:
            interaction_block = interaction_block[:next_on_event]
        assert "to_person: data.to_person" in interaction_block, (
            "addTimelineEvent in anima.interaction should set meta.to_person = data.to_person"
        )

    def test_replay_reads_meta_from_person(self) -> None:
        """_resolvePersons in timeline.js should read meta.from_person."""
        src = _read(TIMELINE_JS)
        assert "meta.from_person" in src, (
            "_resolvePersons should access meta.from_person"
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
        """to_api_dict in activity.py should include from_person in output."""
        src = _read(ACTIVITY_PY)
        # Find the to_api_dict method
        assert "def to_api_dict" in src, "to_api_dict method not found in activity.py"
        # Extract the method body
        m = re.search(r"def to_api_dict\(.*?\).*?:\n(.*?)(?=\n    def |\nclass |\Z)", src, re.DOTALL)
        assert m, "Could not extract to_api_dict body"
        body = m.group(1)
        assert '"from_person"' in body, (
            "to_api_dict should include 'from_person' in its output dict"
        )
        assert '"to_person"' in body, (
            "to_api_dict should include 'to_person' in its output dict"
        )
        assert '"content"' in body, (
            "to_api_dict should include 'content' in its output dict"
        )

    def test_resolvePersons_handles_toplevel_from(self) -> None:
        """_resolvePersons should fall back to event.from_person (API format)."""
        src = _read(TIMELINE_JS)
        # Find _resolvePersons function
        assert "function _resolvePersons" in src, (
            "_resolvePersons not found in timeline.js"
        )
        m = re.search(r"function _resolvePersons\(.*?\)\s*\{(.*?)\n\}", src, re.DOTALL)
        assert m, "Could not extract _resolvePersons body"
        body = m.group(1)
        # Should reference event.from_person as fallback (not just meta.from_person)
        assert "event.from_person" in body, (
            "_resolvePersons should fall back to event.from_person for API events"
        )

    def test_resolvePersons_handles_toplevel_content(self) -> None:
        """_resolvePersons should fall back to event.content (API format)."""
        src = _read(TIMELINE_JS)
        m = re.search(r"function _resolvePersons\(.*?\)\s*\{(.*?)\n\}", src, re.DOTALL)
        assert m, "Could not extract _resolvePersons body"
        body = m.group(1)
        assert "event.content" in body, (
            "_resolvePersons should fall back to event.content for API events"
        )


# ── TestNoRegressionExistingWSEvents ────────────────────────


class TestNoRegressionExistingWSEvents:
    """Verify existing WS event types still work (no regressions)."""

    def test_heartbeat_handler_still_exists_in_app_js(self) -> None:
        """anima.heartbeat handler should still exist in app.js."""
        src = _read(APP_JS)
        assert 'onEvent("anima.heartbeat"' in src, (
            "anima.heartbeat handler not found in app.js"
        )

    def test_cron_handler_still_exists_in_app_js(self) -> None:
        """anima.cron handler should still exist in app.js."""
        src = _read(APP_JS)
        assert 'onEvent("anima.cron"' in src, (
            "anima.cron handler not found in app.js"
        )

    def test_existing_ws_types_still_in_switch(self) -> None:
        """Core WS types (message, heartbeat, cron, chat, board) must remain in the switch."""
        src = _read(TIMELINE_JS)
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

        The old buggy pattern was: showMessageEffect(anima, ...)
        The fixed pattern uses _resolvePersons: showMessageEffect(p.from, p.to, p.text)
        """
        src = _read(TIMELINE_JS)
        # The old buggy pattern: showMessageEffect(anima,
        # This would pass the concatenated "from -> to" string as fromName
        assert "showMessageEffect(anima," not in src, (
            "Bug regression: showMessageEffect should NOT be called with raw 'anima' field. "
            "Use _resolvePersons(event) to extract p.from/p.to instead."
        )
        # Verify the correct pattern is present (using p.from, p.to, p.text)
        assert re.search(r"showMessageEffect\(\s*p\.from\s*,\s*p\.to\s*,\s*p\.text\s*\)", src), (
            "showMessageEffect should be called with (p.from, p.to, p.text) from _resolvePersons"
        )

    def test_anima_concatenation_exists_in_app_js(self) -> None:
        """app.js still uses 'from_person} -> ${' pattern for the anima field.

        This confirms that the anima field in WS events is a concatenated
        display string like "alice -> bob", and timeline.js must NOT pass
        it directly to showMessageEffect.
        """
        src = _read(APP_JS)
        # The interaction handler builds: anima: `${data.from_person} -> ${data.to_person}`
        # We check for the arrow pattern in the anima field construction
        assert re.search(r"from_person\}\s*.*?\s*\$\{.*?to_person\}", src), (
            "app.js should build anima as a concatenated 'from -> to' display string"
        )

    def test_resolvePersons_bypasses_concatenated_anima(self) -> None:
        """_resolvePersons should prioritise meta.from_person over event.anima.

        When meta.from_person is available, the concatenated anima string
        (e.g. "alice -> bob") is never used for showMessageEffect.
        """
        src = _read(TIMELINE_JS)
        m = re.search(r"function _resolvePersons\(.*?\)\s*\{(.*?)\n\}", src, re.DOTALL)
        assert m, "Could not extract _resolvePersons body"
        body = m.group(1)

        # The from field should check meta.from_person FIRST (before event.anima)
        # Pattern: meta.from_person || event.from_person || event.anima || ""
        from_line = [line for line in body.splitlines() if "from:" in line or "from :" in line]
        assert len(from_line) > 0, "_resolvePersons should have a 'from' assignment"
        from_expr = from_line[0]

        # meta.from_person should appear before event.anima in the fallback chain
        meta_pos = from_expr.find("meta.from_person")
        anima_pos = from_expr.find("event.anima")
        assert meta_pos >= 0, (
            "_resolvePersons 'from' should reference meta.from_person"
        )
        if anima_pos >= 0:
            assert meta_pos < anima_pos, (
                "meta.from_person must take priority over event.anima in the fallback chain"
            )
