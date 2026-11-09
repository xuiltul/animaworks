"""Unit tests for WT-1 workspace fixes — static analysis of JS/Python source files.

Covers two issues:
  A (P1): WS Event Mismatch — data.job→data.task and chat.response removal
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
WEBSOCKET_JS = REPO_ROOT / "server" / "static" / "modules" / "websocket.js"
TIMELINE_JS = REPO_ROOT / "server" / "static" / "workspace" / "modules" / "timeline.js"


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
        assert 'onEvent("chat.response"' not in src, "chat.response handler still present in app.js (should be removed)"


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
        assert '"human_notify"' in filter_defs_block, "human_notify type missing from filterDefs"

    def test_notification_type_in_filter_defs(self, filter_defs_block: str) -> None:
        assert '"notification"' in filter_defs_block, "notification type missing from filterDefs"

    def test_error_type_in_filter_defs(self, filter_defs_block: str) -> None:
        assert '"error"' in filter_defs_block, "error type missing from filterDefs"

    def test_issue_resolved_type_in_filter_defs(self, filter_defs_block: str) -> None:
        assert '"issue_resolved"' in filter_defs_block, "issue_resolved type missing from filterDefs"

    def test_megaphone_label_in_filter_defs(self, filter_defs_block: str) -> None:
        """Megaphone emoji label groups human_notify and notification."""
        assert "\U0001f4e3" in filter_defs_block, "Megaphone label missing from filterDefs"

    def test_warning_label_in_filter_defs(self, filter_defs_block: str) -> None:
        """Warning emoji label groups error and issue_resolved."""
        assert "\u26a0\ufe0f" in filter_defs_block, "Warning label missing from filterDefs"

    def test_human_notify_and_notification_grouped(self, filter_defs_block: str) -> None:
        """human_notify and notification must be in the same filter entry."""
        # Find the entry containing human_notify
        match = re.search(
            r"\{[^}]*human_notify[^}]*\}",
            filter_defs_block,
        )
        assert match, "No filter entry with human_notify found"
        entry = match.group(0)
        assert '"notification"' in entry, "notification not grouped with human_notify in the same filter entry"

    def test_error_and_issue_resolved_grouped(self, filter_defs_block: str) -> None:
        """error and issue_resolved must be in the same filter entry."""
        match = re.search(
            r"\{[^}]*\"error\"[^}]*\}",
            filter_defs_block,
        )
        assert match, "No filter entry with error found"
        entry = match.group(0)
        assert '"issue_resolved"' in entry, "issue_resolved not grouped with error in the same filter entry"
