"""Unit tests for dashboard tool_use filtering — static analysis of JS/Python source.

Verifies:
- org-dashboard.js: _CARD_VISIBLE_TYPES, VISIBLE_TOOL_NAMES, _isCardVisible,
  _loadInitialStreams filter, updateCardActivity guards, _summarizeEvent ⚙ icon
- app-websocket.js: VISIBLE_TOOL_NAMES import, isStreamingTool check,
  conditional guard before updateCardActivity
- core/memory/activity.py: _LIVE_EVENT_TYPES excludes tool_use,
  _VISIBLE_TOOL_NAMES frozenset, log() conditional for tool_use
"""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
ORG_DASHBOARD_JS = REPO_ROOT / "server" / "static" / "workspace" / "modules" / "org-dashboard.js"
APP_WEBSOCKET_JS = REPO_ROOT / "server" / "static" / "workspace" / "modules" / "app-websocket.js"
ACTIVITY_PY = REPO_ROOT / "core" / "memory" / "activity.py"


# ── org-dashboard.js: Visibility Filter ──────────────────────


class TestOrgDashboardVisibilityFilter:
    """org-dashboard.js contains the visibility filter constants and helpers."""

    @pytest.fixture(autouse=True)
    def _load(self):
        self.src = ORG_DASHBOARD_JS.read_text(encoding="utf-8")

    def test_card_visible_types_set_defined(self):
        assert "_CARD_VISIBLE_TYPES" in self.src

    def test_card_visible_types_has_expected_event_types(self):
        assert "heartbeat_start" in self.src
        assert "heartbeat_end" in self.src
        assert "message_sent" in self.src
        assert "message_received" in self.src
        assert "response_sent" in self.src
        assert "channel_post" in self.src
        assert "task_created" in self.src
        assert "task_updated" in self.src

    def test_visible_tool_names_exported_not_prefixed(self):
        assert "export const VISIBLE_TOOL_NAMES" in self.src

    def test_visible_tool_names_has_expected_tools(self):
        assert "delegate_task" in self.src
        assert "update_task" in self.src
        assert "backlog_task" in self.src
        assert "submit_tasks" in self.src
        assert "call_human" in self.src
        assert "post_channel" in self.src
        assert "send_message" in self.src

    def test_is_card_visible_function_defined(self):
        assert "function _isCardVisible" in self.src

    def test_load_initial_streams_uses_filter_before_slice(self):
        assert ".filter(_isCardVisible)" in self.src
        assert ".slice(" in self.src
        # Filter must appear before slice in the chain
        filter_pos = self.src.find(".filter(_isCardVisible)")
        slice_pos = self.src.find(".slice(0, MAX_STREAM_ENTRIES)")
        assert filter_pos >= 0 and slice_pos >= 0
        assert filter_pos < slice_pos

    def test_update_card_activity_tool_start_guard(self):
        assert 'eventType === "tool_start"' in self.src
        assert "VISIBLE_TOOL_NAMES.has(toolName)" in self.src

    def test_update_card_activity_tool_end_guard(self):
        assert 'eventType === "tool_end"' in self.src or 'eventType === "tool_use"' in self.src

    def test_update_card_activity_tool_detail_guard(self):
        assert 'eventType === "tool_detail"' in self.src

    def test_summarize_event_uses_gear_icon_for_tool_use(self):
        assert "⚙" in self.src
        assert "tool_use" in self.src


class TestOrgDashboardUpdateCardActivityGuards:
    """updateCardActivity has VISIBLE_TOOL_NAMES.has(toolName) guards on tool branches."""

    @pytest.fixture(autouse=True)
    def _load(self):
        self.src = ORG_DASHBOARD_JS.read_text(encoding="utf-8")

    def test_tool_start_has_guard(self):
        # tool_start branch: if (!VISIBLE_TOOL_NAMES.has(toolName)) return;
        assert "tool_start" in self.src
        assert "VISIBLE_TOOL_NAMES.has(toolName)" in self.src
        assert "return" in self.src

    def test_tool_end_tool_use_has_guard(self):
        assert "tool_end" in self.src
        assert "tool_use" in self.src

    def test_tool_detail_has_guard(self):
        assert "tool_detail" in self.src


# ── app-websocket.js: Filtering ──────────────────────


class TestAppWebSocketFiltering:
    """app-websocket.js contains the filtering logic for tool activity."""

    @pytest.fixture(autouse=True)
    def _load(self):
        self.src = APP_WEBSOCKET_JS.read_text(encoding="utf-8")

    def test_imports_visible_tool_names_from_org_dashboard(self):
        assert "VISIBLE_TOOL_NAMES" in self.src
        assert "org-dashboard.js" in self.src

    def test_has_is_streaming_tool_variable(self):
        assert "isStreamingTool" in self.src

    def test_is_streaming_tool_checks_tool_start_end_detail(self):
        assert "tool_start" in self.src
        assert "tool_end" in self.src
        assert "tool_detail" in self.src

    def test_conditional_guard_before_update_card_activity(self):
        assert "updateCardActivity" in self.src
        assert "isStreamingTool" in self.src
        assert "VISIBLE_TOOL_NAMES.has(toolName)" in self.src
        # Guard and update must appear in same handler; guard precedes the call
        tool_activity_start = self.src.find("anima.tool_activity")
        assert tool_activity_start >= 0
        handler_block = self.src[tool_activity_start:]
        guard_pos = handler_block.find("isStreamingTool")
        update_pos = handler_block.find("updateCardActivity")
        assert guard_pos >= 0 and update_pos >= 0
        assert guard_pos < update_pos


# ── core/memory/activity.py: Backend Changes ──────────────────────


class TestActivityBackendLiveEventTypes:
    """_LIVE_EVENT_TYPES does NOT contain tool_use."""

    @pytest.fixture(autouse=True)
    def _load(self):
        self.src = ACTIVITY_PY.read_text(encoding="utf-8")

    def test_live_event_types_defined(self):
        assert "_LIVE_EVENT_TYPES" in self.src

    def test_live_event_types_excludes_tool_use(self):
        # _LIVE_EVENT_TYPES should not include "tool_use" (handled via _VISIBLE_TOOL_NAMES)
        live_pos = self.src.find("_LIVE_EVENT_TYPES")
        visible_pos = self.src.find("_VISIBLE_TOOL_NAMES")
        assert live_pos >= 0 and visible_pos >= 0
        block = self.src[live_pos:visible_pos]
        assert '"tool_use"' not in block
        assert "'tool_use'" not in block


class TestActivityBackendVisibleToolNames:
    """_VISIBLE_TOOL_NAMES frozenset is defined with expected tool names."""

    @pytest.fixture(autouse=True)
    def _load(self):
        self.src = ACTIVITY_PY.read_text(encoding="utf-8")

    def test_visible_tool_names_defined(self):
        assert "_VISIBLE_TOOL_NAMES" in self.src

    def test_visible_tool_names_is_frozenset(self):
        assert "frozenset" in self.src

    def test_visible_tool_names_has_expected_tools(self):
        assert "delegate_task" in self.src
        assert "update_task" in self.src
        assert "backlog_task" in self.src
        assert "submit_tasks" in self.src
        assert "call_human" in self.src
        assert "post_channel" in self.src
        assert "send_message" in self.src


class TestActivityLogMethodConditional:
    """log() method has conditional for tool_use and _VISIBLE_TOOL_NAMES."""

    @pytest.fixture(autouse=True)
    def _load(self):
        self.src = ACTIVITY_PY.read_text(encoding="utf-8")

    def test_log_has_tool_use_conditional(self):
        assert 'event_type == "tool_use"' in self.src

    def test_log_has_visible_tool_names_check(self):
        assert "entry.tool in self._VISIBLE_TOOL_NAMES" in self.src

    def test_log_emits_live_event_for_visible_tool_use(self):
        assert "_emit_live_event" in self.src
        assert "tool_use" in self.src
        assert "_VISIBLE_TOOL_NAMES" in self.src
