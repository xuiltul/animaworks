"""Unit tests for dashboard visualization enhancement — static analysis of JS/CSS source.

Verifies:
- New event types in updateCardActivity (message_sent, response_sent, etc.)
- Direction icons in card stream (↑/↓/⚙)
- External channel nodes (Slack, Chatwork, GitHub, Gmail, Web)
- Message line type variants (internal, external, human, delegation)
- showExternalLine export and _TOOL_TO_EXTERNAL mapping
- Enhanced app-websocket.js event handling
"""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
ORG_DASHBOARD_JS = (
    REPO_ROOT / "server" / "static" / "workspace" / "modules" / "org-dashboard.js"
)
APP_WEBSOCKET_JS = (
    REPO_ROOT / "server" / "static" / "workspace" / "modules" / "app-websocket.js"
)
STYLE_CSS = REPO_ROOT / "server" / "static" / "workspace" / "style.css"


# ── Card Stream: New Event Types ──────────────────────

class TestCardStreamNewEventTypes:
    """updateCardActivity handles message_sent, response_sent, etc."""

    @pytest.fixture(autouse=True)
    def _load(self):
        self.src = ORG_DASHBOARD_JS.read_text(encoding="utf-8")

    def test_handles_message_sent(self):
        assert 'eventType === "message_sent"' in self.src

    def test_handles_message_received(self):
        assert 'eventType === "message_received"' in self.src

    def test_handles_response_sent(self):
        assert 'eventType === "response_sent"' in self.src

    def test_handles_channel_post(self):
        assert 'eventType === "channel_post"' in self.src

    def test_handles_task_events(self):
        assert 'eventType === "task_created"' in self.src
        assert 'eventType === "task_updated"' in self.src

    def test_msg_out_type_for_sent(self):
        assert 'type: "msg_out"' in self.src

    def test_msg_in_type_for_received(self):
        assert 'type: "msg_in"' in self.src

    def test_task_type(self):
        assert 'type: "task"' in self.src


class TestCardStreamDirectionIcons:
    """Card stream uses direction arrows in entry text."""

    @pytest.fixture(autouse=True)
    def _load(self):
        self.src = ORG_DASHBOARD_JS.read_text(encoding="utf-8")

    def test_up_arrow_for_outgoing(self):
        assert "↑ " in self.src

    def test_down_arrow_for_incoming(self):
        assert "↓ " in self.src

    def test_gear_for_task(self):
        assert "⚙ " in self.src

    def test_render_stream_has_new_type_icons(self):
        assert "msg_out" in self.src
        assert "msg_in" in self.src


# ── External Channel Nodes ──────────────────────

class TestExternalChannelNodes:
    """Dashboard renders external platform nodes."""

    @pytest.fixture(autouse=True)
    def _load(self):
        self.src = ORG_DASHBOARD_JS.read_text(encoding="utf-8")

    def test_external_channels_defined(self):
        assert "_EXTERNAL_CHANNELS" in self.src

    def test_has_slack_node(self):
        assert '"Slack"' in self.src or "'Slack'" in self.src

    def test_has_chatwork_node(self):
        assert '"Chatwork"' in self.src or "'Chatwork'" in self.src

    def test_has_github_node(self):
        assert '"GitHub"' in self.src or "'GitHub'" in self.src

    def test_has_gmail_node(self):
        assert '"Gmail"' in self.src or "'Gmail'" in self.src

    def test_has_web_node(self):
        assert '"Web"' in self.src or "'Web'" in self.src

    def test_create_external_nodes_function(self):
        assert "_createExternalNodes" in self.src

    def test_external_bar_class(self):
        assert "org-external-bar" in self.src

    def test_external_node_class(self):
        assert "org-external-node" in self.src


# ── Message Line Types ──────────────────────

class TestMessageLineTypes:
    """showMessageLine supports type variants via options parameter."""

    @pytest.fixture(autouse=True)
    def _load(self):
        self.src = ORG_DASHBOARD_JS.read_text(encoding="utf-8")

    def test_accepts_options_parameter(self):
        assert "options" in self.src
        assert "lineType" in self.src

    def test_internal_line_type(self):
        assert "internal" in self.src

    def test_external_line_type(self):
        assert "external" in self.src

    def test_external_in_line_type(self):
        assert "external_in" in self.src


class TestShowExternalLine:
    """showExternalLine maps tool names to external node IDs."""

    @pytest.fixture(autouse=True)
    def _load(self):
        self.src = ORG_DASHBOARD_JS.read_text(encoding="utf-8")

    def test_exported(self):
        assert "export function showExternalLine" in self.src

    def test_tool_to_external_mapping(self):
        assert "_TOOL_TO_EXTERNAL" in self.src

    def test_maps_slack(self):
        assert "slack" in self.src and "ext_slack" in self.src

    def test_maps_github(self):
        assert "github" in self.src and "ext_github" in self.src


# ── app-websocket.js: Enhanced Event Handling ──────────────────────

class TestWebSocketEnhancedHandling:
    """app-websocket.js passes extended data to updateCardActivity."""

    @pytest.fixture(autouse=True)
    def _load(self):
        self.src = APP_WEBSOCKET_JS.read_text(encoding="utf-8")

    def test_passes_summary_to_card_activity(self):
        assert "summary:" in self.src
        assert "data.summary" in self.src

    def test_passes_from_person_to_card_activity(self):
        assert "from_person:" in self.src
        assert "data.from_person" in self.src

    def test_passes_to_person_to_card_activity(self):
        assert "to_person:" in self.src
        assert "data.to_person" in self.src

    def test_passes_channel_to_card_activity(self):
        assert 'channel:' in self.src
        assert "data.channel" in self.src

    def test_imports_show_external_line(self):
        assert "showExternalLine" in self.src

    def test_calls_show_external_line(self):
        assert "showExternalLine(" in self.src


# ── CSS: External Nodes & Line Types ──────────────────────

class TestDashboardVisualizationCSS:
    """CSS styles for external nodes and line type variants."""

    @pytest.fixture(autouse=True)
    def _load(self):
        self.src = STYLE_CSS.read_text(encoding="utf-8")

    def test_external_bar_style(self):
        assert ".org-external-bar" in self.src

    def test_external_node_style(self):
        assert ".org-external-node" in self.src

    def test_external_icon_style(self):
        assert ".org-external-icon" in self.src

    def test_external_label_style(self):
        assert ".org-external-label" in self.src

    def test_trail_internal_style(self):
        assert ".org-msg-trail--internal" in self.src

    def test_trail_external_style(self):
        assert ".org-msg-trail--external" in self.src

    def test_trail_external_in_style(self):
        assert ".org-msg-trail--external_in" in self.src

    def test_packet_internal_style(self):
        assert ".org-msg-packet--internal" in self.src

    def test_packet_external_style(self):
        assert ".org-msg-packet--external" in self.src

    def test_trail_human_style(self):
        assert ".org-msg-trail--human" in self.src

    def test_trail_delegation_style(self):
        assert ".org-msg-trail--delegation" in self.src
