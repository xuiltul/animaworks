"""Unit tests for workspace timeline _replayEvent fix — API detail type support."""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import re
from pathlib import Path

import pytest

# ── Paths ──────────────────────────────────────────

REPO_ROOT = Path(__file__).resolve().parents[2]
TIMELINE_REPLAY_JS = REPO_ROOT / "server" / "static" / "workspace" / "modules" / "timeline-replay.js"
TIMELINE_DOM_JS = REPO_ROOT / "server" / "static" / "workspace" / "modules" / "timeline-dom.js"


def _read_replay() -> str:
    return TIMELINE_REPLAY_JS.read_text(encoding="utf-8")


def _read_dom() -> str:
    return TIMELINE_DOM_JS.read_text(encoding="utf-8")


# ── TestResolvePersonsHelper ───────────────────────


class TestResolvePersonsHelper:
    """Verify the resolvePersons helper function exists and behaves correctly."""

    def test_resolvePersons_function_exists(self) -> None:
        """Verify `function resolvePersons(event)` exists in timeline-dom.js."""
        src = _read_dom()
        assert "function resolvePersons(event)" in src

    def test_resolvePersons_returns_from_to_text(self) -> None:
        """Verify the function body contains `from:`, `to:`, and `text:` properties."""
        src = _read_dom()
        match = re.search(
            r"function resolvePersons\(event\)\s*\{(.*?)\n\}",
            src,
            re.DOTALL,
        )
        assert match, "resolvePersons function body not found"
        body = match.group(1)
        assert "from:" in body, "Missing 'from:' property in resolvePersons return"
        assert "to:" in body, "Missing 'to:' property in resolvePersons return"
        assert "text:" in body, "Missing 'text:' property in resolvePersons return"

    def test_resolvePersons_meta_priority(self) -> None:
        """Verify `meta.from_person` appears before `event.from_person` in the
        fallback chain, ensuring meta takes priority over top-level fields."""
        src = _read_dom()
        match = re.search(
            r"function resolvePersons\(event\)\s*\{(.*?)\n\}",
            src,
            re.DOTALL,
        )
        assert match, "resolvePersons function body not found"
        body = match.group(1)
        idx_meta = body.index("meta.from_person")
        idx_event = body.index("event.from_person")
        assert idx_meta < idx_event, (
            "meta.from_person must appear before event.from_person "
            "to ensure meta takes priority"
        )

    def test_resolvePersons_imported_by_replay(self) -> None:
        """Verify timeline-replay.js imports resolvePersons from timeline-dom.js."""
        src = _read_replay()
        assert 'import' in src and 'resolvePersons' in src, (
            "timeline-replay.js must import resolvePersons from timeline-dom.js"
        )


# ── TestReplayEventDMCases ─────────────────────────


class TestReplayEventDMCases:
    """Verify _replayEvent handles DM event types and uses _resolvePersons."""

    def test_dm_received_case_exists(self) -> None:
        """Verify `case "dm_received"` exists in the switch."""
        src = _read_replay()
        assert 'case "dm_received"' in src

    def test_dm_sent_case_exists(self) -> None:
        """Verify `case "dm_sent"` exists in the switch."""
        src = _read_replay()
        assert 'case "dm_sent"' in src

    def test_message_case_uses_resolvePersons(self) -> None:
        """Verify the message/DM case block calls `resolvePersons(event)`."""
        src = _read_replay()
        match = re.search(
            r"function replayEvent\(event,\s*el,\s*ctx\)\s*\{(.*?)\n\}",
            src,
            re.DOTALL,
        )
        assert match, "replayEvent function body not found"
        body = match.group(1)
        assert "resolvePersons(event)" in body, (
            "message/DM case must call resolvePersons(event)"
        )

    def test_message_case_does_not_use_raw_anima(self) -> None:
        """Verify `showMessageEffect(anima,` (the old buggy pattern using raw
        anima as first arg) is NOT in the code.  Instead it should use
        `showMessageEffect(p.from,`."""
        src = _read_replay()
        assert "showMessageEffect(anima," not in src, (
            "showMessageEffect must not use raw 'anima' as first argument"
        )
        assert "showMessageEffect(p.from," in src, (
            "showMessageEffect should use 'p.from' as first argument"
        )


# ── TestReplayEventAPICases ────────────────────────


class TestReplayEventAPICases:
    """Verify _replayEvent handles all API-originated event types."""

    @pytest.mark.parametrize("case_label", [
        "message_received",
        "response_sent",
        "heartbeat_start",
        "heartbeat_end",
        "cron_executed",
        "channel_read",
        "channel_post",
    ])
    def test_api_case_exists(self, case_label: str) -> None:
        """Verify each API detail event type has a case in the switch."""
        src = _read_replay()
        assert f'case "{case_label}"' in src, (
            f'case "{case_label}" not found in _replayEvent switch'
        )

    # Also provide individual named tests for clarity in reports

    def test_message_received_case_exists(self) -> None:
        assert 'case "message_received"' in _read_replay()

    def test_response_sent_case_exists(self) -> None:
        assert 'case "response_sent"' in _read_replay()

    def test_heartbeat_start_case_exists(self) -> None:
        assert 'case "heartbeat_start"' in _read_replay()

    def test_heartbeat_end_case_exists(self) -> None:
        assert 'case "heartbeat_end"' in _read_replay()

    def test_cron_executed_case_exists(self) -> None:
        assert 'case "cron_executed"' in _read_replay()

    def test_channel_read_case_exists(self) -> None:
        assert 'case "channel_read"' in _read_replay()

    def test_channel_post_case_exists(self) -> None:
        assert 'case "channel_post"' in _read_replay()


# ── TestReplayEventExistingCasesPreserved ──────────


class TestReplayEventExistingCasesPreserved:
    """Verify pre-existing WS-originated case labels are still present."""

    @pytest.mark.parametrize("case_label", [
        "message",
        "chat",
        "board",
        "heartbeat",
        "cron",
    ])
    def test_existing_case_preserved(self, case_label: str) -> None:
        src = _read_replay()
        assert f'case "{case_label}"' in src, (
            f'Existing case "{case_label}" must be preserved in _replayEvent'
        )

    def test_existing_message_case_preserved(self) -> None:
        assert 'case "message"' in _read_replay()

    def test_existing_chat_case_preserved(self) -> None:
        assert 'case "chat"' in _read_replay()

    def test_existing_board_case_preserved(self) -> None:
        assert 'case "board"' in _read_replay()

    def test_existing_heartbeat_case_preserved(self) -> None:
        assert 'case "heartbeat"' in _read_replay()

    def test_existing_cron_case_preserved(self) -> None:
        assert 'case "cron"' in _read_replay()


# ── TestReplayEventDeskHighlightCoverage ───────────


class TestReplayEventDeskHighlightCoverage:
    """Verify desk highlight and clear-timeout patterns exist in _replayEvent."""

    def test_highlightDesk_called_for_chat_group(self) -> None:
        """Verify `_highlightDesk(anima)` appears in the code (for
        chat/board/heartbeat/cron cases)."""
        src = _read_replay()
        assert "_highlightDesk(anima)" in src, (
            "_highlightDesk(anima) must be called for desk-highlight cases"
        )

    def test_clearHighlight_timeout_present(self) -> None:
        """Verify `setTimeout` with `_clearHighlight` pattern exists."""
        src = _read_replay()
        # Pattern: setTimeout(() => { if (_clearHighlight) _clearHighlight(); }, ...)
        # The arrow-function body contains ')' chars, so we use DOTALL to
        # match across the full setTimeout(...) call.
        assert re.search(
            r"setTimeout\(.*?_clearHighlight.*?\)",
            src,
            re.DOTALL,
        ), "setTimeout + _clearHighlight pattern not found"


# ── TestResolvePersonsFallbackForDM ────────────────


class TestResolvePersonsFallbackForDM:
    """Verify the message/DM block has a fallback _highlightDesk when
    showMessageEffect cannot fire (e.g. only from_person available)."""

    def test_dm_fallback_to_desk_highlight(self) -> None:
        """In the message/dm block, verify there is a fallback
        `_highlightDesk(p.from || p.to)` for when showMessageEffect
        cannot fire."""
        src = _read_replay()
        # Extract the message/dm case block from _replayEvent
        # The block starts at 'case "message":' and ends at the 'break;'
        match = re.search(
            r'case "message":\s*\n\s*case "dm_received":\s*\n\s*case "dm_sent":\s*\{(.*?)\bbreak;',
            src,
            re.DOTALL,
        )
        assert match, "message/dm_received/dm_sent case block not found"
        block = match.group(1)
        assert "_highlightDesk(p.from || p.to)" in block, (
            "Fallback _highlightDesk(p.from || p.to) not found in message/DM block"
        )
