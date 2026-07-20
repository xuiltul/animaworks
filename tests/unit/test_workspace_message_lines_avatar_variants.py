"""Unit tests for workspace message lines & avatar variants — static analysis of JS/CSS.

Verifies that org-dashboard.js has showMessageLine() for SVG animated
message lines and updateAvatarExpression() for state-based avatar switching,
that app-websocket.js integrates both, and that style.css defines the required
animation and filter classes.
"""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import re
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
AVATAR_RESOLVER_JS = (
    REPO_ROOT / "server" / "static" / "modules" / "avatar-resolver.js"
)


# ── Message Line Drawing ──────────────────────


class TestMessageLineFunctionality:
    """Verify showMessageLine is implemented in org-dashboard.js."""

    @pytest.fixture(autouse=True)
    def _load(self):
        self.src = ORG_DASHBOARD_JS.read_text(encoding="utf-8")

    def test_show_message_line_exported(self):
        assert "export function showMessageLine" in self.src

    def test_show_message_line_params(self):
        match = re.search(r"export function showMessageLine\(([^)]+)\)", self.src)
        assert match, "showMessageLine function not found"
        params = match.group(1)
        assert "fromName" in params
        assert "toName" in params

    def test_svg_path_creation(self):
        assert 'createElementNS("http://www.w3.org/2000/svg", "path")' in self.src
        assert "org-msg-trail" in self.src

    def test_svg_circle_packet(self):
        assert 'createElementNS("http://www.w3.org/2000/svg", "circle")' in self.src
        assert "org-msg-packet" in self.src

    def test_animate_motion_element(self):
        assert 'createElementNS("http://www.w3.org/2000/svg", "animateMotion")' in self.src

    def test_quadratic_bezier_path(self):
        assert re.search(r"[`'\"]M.*Q.*[`'\"]", self.src), (
            "Expected quadratic bezier path (M...Q...) in SVG path data"
        )

    def test_message_line_duration_constant(self):
        assert "MESSAGE_LINE_DURATION" in self.src

    def test_message_line_fade_constant(self):
        assert "MESSAGE_LINE_FADE" in self.src

    def test_fade_out_on_complete(self):
        assert "org-msg-line--fading" in self.src

    def test_msg_line_group_class(self):
        assert "org-msg-line-group" in self.src

    def test_positions_guard(self):
        assert "_positions.get(fromName)" in self.src or "_positions.get(toName)" in self.src

    def test_svg_layer_guard(self):
        assert "if (!_svgLayer" in self.src

    def test_connections_group_separation(self):
        assert "_connectionsGroup" in self.src
        assert "_msgLinesGroup" in self.src
        assert "org-connections-group" in self.src
        assert "org-msg-lines-group" in self.src

    def test_connections_group_clear_not_innerHTML(self):
        lines = self.src.split("\n")
        in_update_connections = False
        for line in lines:
            if "function _updateConnections" in line:
                in_update_connections = True
            if in_update_connections and "_svgLayer.innerHTML" in line:
                pytest.fail("_updateConnections should not use _svgLayer.innerHTML")
            if in_update_connections and line.startswith("function ") and "_updateConnections" not in line:
                break

    def test_offset_alternation(self):
        assert "_msgLineCounter" in self.src, (
            "Expected counter for alternating offset direction"
        )


# ── Avatar Variant Management ──────────────────────


class TestAvatarVariants:
    """Verify avatar expression switching in org-dashboard.js."""

    @pytest.fixture(autouse=True)
    def _load(self):
        self.src = ORG_DASHBOARD_JS.read_text(encoding="utf-8")

    def test_update_avatar_expression_exported(self):
        assert "export function updateAvatarExpression" in self.src

    def test_status_to_expression_mapping(self):
        assert "STATUS_TO_EXPRESSION" in self.src

    def test_idle_maps_to_neutral(self):
        assert re.search(r'idle\s*:\s*"neutral"', self.src)

    def test_thinking_maps_to_thinking(self):
        assert re.search(r'thinking\s*:\s*"thinking"', self.src)

    def test_error_maps_to_troubled(self):
        assert re.search(r'error\s*:\s*"troubled"', self.src)

    def test_chatting_maps_to_smile(self):
        assert re.search(r'chatting\s*:\s*"smile"', self.src)

    def test_expressions_array(self):
        assert "EXPRESSIONS" in self.src
        for expr in ["neutral", "smile", "laugh", "troubled", "surprised", "thinking", "embarrassed"]:
            assert f'"{expr}"' in self.src

    def test_preload_avatar_expressions(self):
        assert "_preloadAvatarExpressions" in self.src

    def test_bustup_expression_candidates_import(self):
        assert "bustupExpressionCandidates" in self.src

    def test_avatar_expressions_map(self):
        assert "_avatarExpressions" in self.src

    def test_css_filter_fallback(self):
        assert "dataset.expression" in self.src

    def test_avatar_transition_class(self):
        assert "org-avatar--transitioning" in self.src

    def test_debounce_with_raf(self):
        assert "_avatarUpdateRafPending" in self.src
        assert "requestAnimationFrame" in self.src

    def test_idle_callback_preload(self):
        assert "requestIdleCallback" in self.src

    def test_dispose_clears_avatar_state(self):
        assert "_avatarExpressions.clear()" in self.src
        assert "_avatarUpdateRafPending.clear()" in self.src


# ── app-websocket.js Integration ──────────────────────


class TestAppWebSocketIntegration:
    """Verify app-websocket.js imports and calls new functions."""

    @pytest.fixture(autouse=True)
    def _load(self):
        self.src = APP_WEBSOCKET_JS.read_text(encoding="utf-8")

    def test_imports_show_message_line(self):
        assert "showMessageLine" in self.src

    def test_imports_update_avatar_expression(self):
        assert "updateAvatarExpression" in self.src

    def test_interaction_handler_calls_show_message_line(self):
        assert re.search(
            r'showMessageLine\s*\(\s*data\.from_person\s*,\s*data\.to_person',
            self.src,
        )

    def test_interaction_org_view_guard(self):
        lines = self.src.split("\n")
        found_guard = False
        for i, line in enumerate(lines):
            if "showMessageLine" in line:
                context = "\n".join(lines[max(0, i - 5):i + 1])
                if 'getCurrentView() === "org"' in context:
                    found_guard = True
                    break
        assert found_guard, "showMessageLine should be guarded by org view check"

    def test_status_handler_calls_update_avatar_expression(self):
        assert re.search(
            r'updateAvatarExpression\s*\(\s*data\.name',
            self.src,
        )

    def test_status_extracts_state_string(self):
        assert "data.status.state" in self.src or "data.status.status" in self.src


# ── CSS Animation & Filter Classes ──────────────────────


class TestMessageLineCSS:
    """Verify CSS classes for message line animation."""

    @pytest.fixture(autouse=True)
    def _load(self):
        self.src = STYLE_CSS.read_text(encoding="utf-8")

    def test_msg_trail_class(self):
        assert ".org-msg-trail" in self.src

    def test_msg_trail_has_stroke(self):
        assert re.search(r"\.org-msg-trail\s*\{[^}]*stroke:", self.src)

    def test_msg_trail_has_glow(self):
        assert re.search(r"\.org-msg-trail\s*\{[^}]*drop-shadow", self.src)

    def test_msg_packet_class(self):
        assert ".org-msg-packet" in self.src

    def test_msg_packet_has_glow(self):
        assert re.search(r"\.org-msg-packet\s*\{[^}]*drop-shadow", self.src)

    def test_msg_line_group_transition(self):
        assert re.search(r"\.org-msg-line-group\s*\{[^}]*transition:", self.src)

    def test_msg_line_fading_class(self):
        assert ".org-msg-line--fading" in self.src
        assert re.search(r"\.org-msg-line--fading\s*\{[^}]*opacity:\s*0", self.src)


class TestAvatarVariantCSS:
    """Verify CSS classes for avatar expression switching."""

    @pytest.fixture(autouse=True)
    def _load(self):
        self.src = STYLE_CSS.read_text(encoding="utf-8")

    def test_avatar_img_transition(self):
        assert re.search(r"\.org-card-avatar\s+img\s*\{[^}]*transition:", self.src)

    def test_avatar_transitioning_class(self):
        assert ".org-avatar--transitioning" in self.src

    def test_troubled_filter(self):
        assert re.search(
            r'\[data-expression="troubled"\].*\.org-card-avatar\s+img\s*\{[^}]*filter:',
            self.src,
        )

    def test_thinking_filter(self):
        assert re.search(
            r'\[data-expression="thinking"\].*\.org-card-avatar\s+img\s*\{[^}]*filter:',
            self.src,
        )

    def test_smile_filter(self):
        assert re.search(
            r'\[data-expression="smile"\].*\.org-card-avatar\s+img\s*\{[^}]*filter:',
            self.src,
        )

    def test_reduced_motion_message_line(self):
        assert re.search(
            r"prefers-reduced-motion.*org-msg-trail.*stroke-dasharray",
            self.src,
            re.DOTALL,
        )

    def test_reduced_motion_avatar_transition(self):
        assert re.search(
            r"prefers-reduced-motion.*\.org-card-avatar\s+img.*transition:\s*none",
            self.src,
            re.DOTALL,
        )


# ── Avatar Resolver (no changes expected but verify expression support) ──


class TestAvatarResolverExpressionSupport:
    """Verify avatar-resolver.js has bustupExpressionCandidates."""

    @pytest.fixture(autouse=True)
    def _load(self):
        self.src = AVATAR_RESOLVER_JS.read_text(encoding="utf-8")

    def test_expression_candidates_function(self):
        assert "export function bustupExpressionCandidates" in self.src

    def test_expression_returns_expression_filename(self):
        assert "avatar_bustup_" in self.src
        assert "_realistic.png" in self.src

    def test_neutral_fallback(self):
        assert re.search(r'expression\s*===\s*"neutral"', self.src)

    def test_resolve_cached_avatar_appends_size_query(self):
        """Server-side thumbs: resolveCachedAvatar returns ?size= URL (no icon exclusion)."""
        assert "export async function resolveCachedAvatar" in self.src
        assert "size=${size}" in self.src or 'size=" + size' in self.src or "size=${size}" in self.src
        # Icon exclusion removed — all avatars get server-side thumbnails
        assert "endsWith(\"/icon.png\")" not in self.src
        assert "endsWith('/icon.png')" not in self.src


# ── Dispose Cleanup ──────────────────────


class TestOrgDashboardDispose:
    """Verify dispose cleans up message line and avatar state."""

    @pytest.fixture(autouse=True)
    def _load(self):
        self.src = ORG_DASHBOARD_JS.read_text(encoding="utf-8")

    def test_dispose_clears_connections_group(self):
        assert "_connectionsGroup = null" in self.src

    def test_dispose_clears_msg_lines_group(self):
        assert "_msgLinesGroup = null" in self.src

    def test_dispose_resets_msg_line_counter(self):
        assert "_msgLineCounter = 0" in self.src
