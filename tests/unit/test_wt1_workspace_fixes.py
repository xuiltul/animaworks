"""Unit tests for WT-1 workspace fixes — static analysis of JS/Python source files.

Covers four issues:
  A (P1): WS Event Mismatch — data.job→data.task, chat.response removal, dm_received to_person
  B (P1): Activity Sidebar Empty — isoTs param, loadActivityHistory, activateRightTab
  C (P2): chat.js Integration — stream imports, heartbeat relay, resumeConversationStream, chat.js deletion
  D (P3): Timeline Filter Expansion — human_notify/notification and error/issue_resolved filterDefs
"""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import re
from pathlib import Path

import pytest

# ── Paths ──────────────────────────────────────────────

REPO_ROOT = Path(__file__).resolve().parents[2]
APP_JS = REPO_ROOT / "server" / "static" / "workspace" / "modules" / "app-websocket.js"
ACTIVITY_JS = REPO_ROOT / "server" / "static" / "workspace" / "modules" / "activity.js"
SIDEBAR_JS = REPO_ROOT / "server" / "static" / "workspace" / "modules" / "sidebar.js"
CHAT_STREAMING_JS = REPO_ROOT / "server" / "static" / "workspace" / "modules" / "chat-streaming.js"
CHAT_CONTROLLER_JS = REPO_ROOT / "server" / "static" / "workspace" / "modules" / "chat-controller.js"
RENDER_UTILS_JS = REPO_ROOT / "server" / "static" / "shared" / "chat" / "render-utils.js"
WEBSOCKET_JS = REPO_ROOT / "server" / "static" / "modules" / "websocket.js"
TIMELINE_JS = REPO_ROOT / "server" / "static" / "workspace" / "modules" / "timeline.js"
CHAT_JS = REPO_ROOT / "server" / "static" / "workspace" / "modules" / "chat.js"
ANIMA_PY = REPO_ROOT / "core" / "anima.py"


# ── Issue A: WS Event Mismatch ────────────────────────


class TestIssueA_WSEventMismatch:
    """Verify WS event field renames and dead code removal."""

    def test_app_js_cron_handler_uses_data_task(self) -> None:
        """app.js cron handler must reference data.task, not data.job."""
        src = APP_JS.read_text(encoding="utf-8")
        # Find the anima.cron handler block
        match = re.search(
            r'onEvent\("anima\.cron".*?\)\)',
            src,
            re.DOTALL,
        )
        assert match, "anima.cron handler not found in app.js"
        block = match.group(0)
        assert "data.task" in block, "data.task not found in cron handler"
        assert "data.job" not in block, "data.job still present in cron handler"

    def test_app_js_cron_handler_data_task_count(self) -> None:
        """data.task should appear at least twice in the cron handler (addActivity + addTimelineEvent)."""
        src = APP_JS.read_text(encoding="utf-8")
        match = re.search(
            r'onEvent\("anima\.cron".*?\)\)',
            src,
            re.DOTALL,
        )
        assert match, "anima.cron handler not found in app.js"
        block = match.group(0)
        assert block.count("data.task") >= 2, (
            f"Expected >=2 data.task occurrences in cron handler, got {block.count('data.task')}"
        )

    def test_websocket_js_cron_handler_uses_data_task(self) -> None:
        """websocket.js cron handler must reference data.task, not data.job."""
        src = WEBSOCKET_JS.read_text(encoding="utf-8")
        # Find the anima.cron case block
        match = re.search(
            r'case\s+"anima\.cron".*?break;',
            src,
            re.DOTALL,
        )
        assert match, "anima.cron case not found in websocket.js"
        block = match.group(0)
        assert "data.task" in block, "data.task not found in websocket.js cron handler"
        assert "data.job" not in block, "data.job still present in websocket.js cron handler"

    def test_chat_response_handler_removed_from_app_js(self) -> None:
        """chat.response WS handler was dead code and must be removed."""
        src = APP_JS.read_text(encoding="utf-8")
        assert 'onEvent("chat.response"' not in src, (
            "chat.response handler still present in app.js (should be removed)"
        )

    def test_anima_py_dm_received_has_to_person(self) -> None:
        """message_received (from_type=anima) log call in anima modules must include to_person=self.name."""
        src = "\n".join(
            p.read_text(encoding="utf-8")
            for p in [ANIMA_PY, *(REPO_ROOT / "core").glob("_anima_*.py")]
        )
        # Find activity.log("message_received" ...) calls — may span multiple lines
        matches = re.findall(
            r'activity\.log\(\s*"message_received"[\s\S]*?\n\s*\)',
            src,
        )
        assert len(matches) > 0, "No activity.log('message_received') call found in anima.py"
        # At least one call must have from_type in meta and to_person=self.name
        anima_dm_calls = [
            m for m in matches
            if "from_type" in m and "to_person" in m
        ]
        assert len(anima_dm_calls) >= 1, (
            "No message_received log with from_type meta and to_person found"
        )
        for call in anima_dm_calls:
            assert "to_person=self.name" in call, (
                f"to_person=self.name missing from message_received log: {call[:150]}"
            )


# ── Issue B: Activity Sidebar Empty ───────────────────


class TestIssueB_ActivitySidebarEmpty:
    """Verify activity sidebar initialization and history loading."""

    def test_addActivity_accepts_isoTs_parameter(self) -> None:
        """addActivity function signature must include isoTs parameter."""
        src = ACTIVITY_JS.read_text(encoding="utf-8")
        match = re.search(r"function\s+addActivity\s*\([^)]*\)", src)
        assert match, "addActivity function not found in activity.js"
        sig = match.group(0)
        assert "isoTs" in sig, f"isoTs parameter missing from addActivity signature: {sig}"

    def test_addActivity_uses_isoTs_for_date(self) -> None:
        """addActivity must use isoTs when provided (not always new Date())."""
        src = ACTIVITY_JS.read_text(encoding="utf-8")
        idx = src.index("function addActivity")
        block = src[idx:idx + 300]
        assert "isoTs" in block, "isoTs not used in addActivity body"
        assert "new Date(isoTs)" in block, "addActivity does not parse isoTs into Date"

    def test_loadActivityHistory_exists(self) -> None:
        """loadActivityHistory function must exist in activity.js."""
        src = ACTIVITY_JS.read_text(encoding="utf-8")
        assert "function loadActivityHistory" in src or "async function loadActivityHistory" in src

    def test_loadActivityHistory_fetches_activity_recent(self) -> None:
        """loadActivityHistory must fetch from /api/activity/recent endpoint."""
        src = ACTIVITY_JS.read_text(encoding="utf-8")
        match = re.search(
            r"(?:async\s+)?function\s+loadActivityHistory.*?(?=\nfunction\s|\n(?:async\s+)?function\s|\nexport\s|\nclass\s|$)",
            src,
            re.DOTALL,
        )
        assert match, "loadActivityHistory function body not found"
        body = match.group(0)
        assert "/api/activity/recent" in body, (
            "loadActivityHistory does not fetch /api/activity/recent"
        )

    def test_activateRightTab_handles_activity(self) -> None:
        """activateRightTab must have an 'activity' case that calls loadActivityHistory."""
        src = SIDEBAR_JS.read_text(encoding="utf-8")
        match = re.search(
            r"function\s+activateRightTab\s*\([^)]*\).*?(?=\nexport\s+function\s|\nclass\s|$)",
            src,
            re.DOTALL,
        )
        assert match, "activateRightTab function not found"
        body = match.group(0)
        assert '"activity"' in body, 'No "activity" case in activateRightTab'
        assert "loadActivityHistory" in body, (
            "activateRightTab does not call loadActivityHistory for activity tab"
        )


# ── Issue C: chat.js Integration ──────────────────────


class TestIssueC_ChatJsIntegration:
    """Verify chat-stream imports, heartbeat relay, stream recovery, and chat.js deletion."""

    def test_fetchActiveStream_imported(self) -> None:
        """fetchActiveStream must be imported via chat-controller.js."""
        src = CHAT_CONTROLLER_JS.read_text(encoding="utf-8")
        assert "fetchActiveStream" in src
        import_match = re.search(r"import\s*\{[^}]*fetchActiveStream[^}]*\}.*?from.*?chat-stream", src)
        assert import_match, "fetchActiveStream not imported from chat-stream.js in chat-controller.js"

    def test_fetchStreamProgress_imported(self) -> None:
        """fetchStreamProgress must be imported via chat-controller.js."""
        src = CHAT_CONTROLLER_JS.read_text(encoding="utf-8")
        assert "fetchStreamProgress" in src
        import_match = re.search(r"import\s*\{[^}]*fetchStreamProgress[^}]*\}.*?from.*?chat-stream", src)
        assert import_match, "fetchStreamProgress not imported from chat-stream.js in chat-controller.js"

    def test_onHeartbeatRelayStart_handler(self) -> None:
        """onHeartbeatRelayStart callback must exist in streamChat call."""
        src = CHAT_STREAMING_JS.read_text(encoding="utf-8")
        assert "onHeartbeatRelayStart" in src

    def test_onHeartbeatRelay_handler(self) -> None:
        """onHeartbeatRelay callback must exist in streamChat call."""
        src = CHAT_STREAMING_JS.read_text(encoding="utf-8")
        matches = re.findall(r"onHeartbeatRelay(?!Start|Done)\b", src)
        assert len(matches) >= 1, "onHeartbeatRelay handler not found in chat-streaming.js"

    def test_onHeartbeatRelayDone_handler(self) -> None:
        """onHeartbeatRelayDone callback must exist in streamChat call."""
        src = CHAT_STREAMING_JS.read_text(encoding="utf-8")
        assert "onHeartbeatRelayDone" in src

    def test_heartbeatRelay_rendering_in_renderStreamingBubbleInner(self) -> None:
        """renderStreamingBubbleInner delegates to _renderTextZoneContent which handles heartbeatRelay."""
        src = RENDER_UTILS_JS.read_text(encoding="utf-8")
        match = re.search(
            r"function\s+_renderTextZoneContent.*?(?=\nfunction\s+_render|\nexport\s+function\s|$)",
            src,
            re.DOTALL,
        )
        assert match, "_renderTextZoneContent (streaming helper) not found"
        body = match.group(0)
        assert "heartbeatRelay" in body, "heartbeatRelay state not handled in streaming text zone"

    def test_afterHeartbeatRelay_rendering_in_renderStreamingBubbleInner(self) -> None:
        """renderStreamingBubbleInner delegates to _renderTextZoneContent which handles afterHeartbeatRelay."""
        src = RENDER_UTILS_JS.read_text(encoding="utf-8")
        match = re.search(
            r"function\s+_renderTextZoneContent.*?(?=\nfunction\s+_render|\nexport\s+function\s|$)",
            src,
            re.DOTALL,
        )
        assert match, "_renderTextZoneContent (streaming helper) not found"
        body = match.group(0)
        assert "afterHeartbeatRelay" in body, (
            "afterHeartbeatRelay state not handled in streaming text zone"
        )

    def test_resumeConversationStream_exists(self) -> None:
        """resumeConversationStream function must exist for page-reload stream recovery."""
        src = CHAT_STREAMING_JS.read_text(encoding="utf-8")
        assert (
            "function resumeConversationStream" in src
            or "async function resumeConversationStream" in src
        ), "resumeConversationStream function not found in chat-streaming.js"

    def test_resumeConversationStream_delegates_to_session_manager(self) -> None:
        """resumeConversationStream must delegate to ChatSessionManager."""
        src = CHAT_STREAMING_JS.read_text(encoding="utf-8")
        match = re.search(
            r"(?:async\s+)?function\s+resumeConversationStream.*?(?=\nfunction\s|\n(?:async\s+)?function\s|\nexport\s|\nclass\s|$)",
            src,
            re.DOTALL,
        )
        assert match, "resumeConversationStream function body not found"
        body = match.group(0)
        assert "_mgr()" in body, "resumeConversationStream does not delegate to ChatSessionManager"

    def test_chat_js_deleted(self) -> None:
        """workspace/modules/chat.js must be deleted (dead code removed)."""
        assert not CHAT_JS.exists(), (
            f"chat.js still exists at {CHAT_JS} — should have been deleted"
        )


# ── Issue D: Timeline Filter Expansion ────────────────


class TestIssueD_TimelineFilterExpansion:
    """Verify new filter entries in timeline.js filterDefs."""

    @pytest.fixture()
    def filter_defs_block(self) -> str:
        """Extract the filterDefs array definition from timeline.js."""
        src = TIMELINE_JS.read_text(encoding="utf-8")
        match = re.search(
            r"const\s+filterDefs\s*=\s*\[(.*?)\];",
            src,
            re.DOTALL,
        )
        assert match, "filterDefs array not found in timeline.js"
        return match.group(0)

    def test_human_notify_type_in_filter_defs(self, filter_defs_block: str) -> None:
        assert '"human_notify"' in filter_defs_block, (
            "human_notify type missing from filterDefs"
        )

    def test_notification_type_in_filter_defs(self, filter_defs_block: str) -> None:
        assert '"notification"' in filter_defs_block, (
            "notification type missing from filterDefs"
        )

    def test_error_type_in_filter_defs(self, filter_defs_block: str) -> None:
        assert '"error"' in filter_defs_block, (
            "error type missing from filterDefs"
        )

    def test_issue_resolved_type_in_filter_defs(self, filter_defs_block: str) -> None:
        assert '"issue_resolved"' in filter_defs_block, (
            "issue_resolved type missing from filterDefs"
        )

    def test_megaphone_label_in_filter_defs(self, filter_defs_block: str) -> None:
        """Megaphone emoji label groups human_notify and notification."""
        assert "\U0001f4e3" in filter_defs_block, (
            "Megaphone label missing from filterDefs"
        )

    def test_warning_label_in_filter_defs(self, filter_defs_block: str) -> None:
        """Warning emoji label groups error and issue_resolved."""
        assert "\u26a0\ufe0f" in filter_defs_block, (
            "Warning label missing from filterDefs"
        )

    def test_human_notify_and_notification_grouped(self, filter_defs_block: str) -> None:
        """human_notify and notification must be in the same filter entry."""
        # Find the entry containing human_notify
        match = re.search(
            r"\{[^}]*human_notify[^}]*\}",
            filter_defs_block,
        )
        assert match, "No filter entry with human_notify found"
        entry = match.group(0)
        assert '"notification"' in entry, (
            "notification not grouped with human_notify in the same filter entry"
        )

    def test_error_and_issue_resolved_grouped(self, filter_defs_block: str) -> None:
        """error and issue_resolved must be in the same filter entry."""
        match = re.search(
            r"\{[^}]*\"error\"[^}]*\}",
            filter_defs_block,
        )
        assert match, "No filter entry with error found"
        entry = match.group(0)
        assert '"issue_resolved"' in entry, (
            "issue_resolved not grouped with error in the same filter entry"
        )
