"""Unit tests for HTML theme support structure."""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import pytest


INDEX_HTML = (
    Path(__file__).resolve().parents[2]
    / "server"
    / "static"
    / "index.html"
)
WS_INDEX_HTML = (
    Path(__file__).resolve().parents[2]
    / "server"
    / "static"
    / "workspace"
    / "index.html"
)


class TestMainIndexHTML:
    """Tests for main index.html theme support."""

    def test_loads_tokens_css(self):
        content = INDEX_HTML.read_text()
        assert "tokens.css" in content

    def test_tokens_css_before_base_css(self):
        content = INDEX_HTML.read_text()
        tokens_pos = content.index("tokens.css")
        base_pos = content.index("base.css")
        assert tokens_pos < base_pos, "tokens.css must load before base.css"

    def test_has_theme_toggle(self):
        content = INDEX_HTML.read_text()
        assert "themeSwitch" in content or "themeToggle" in content

    def test_has_lucide_script(self):
        content = INDEX_HTML.read_text()
        assert "lucide" in content.lower()

    def test_nav_has_dual_icons(self):
        content = INDEX_HTML.read_text()
        assert "nav-emoji" in content
        assert "nav-lucide" in content
        assert "data-lucide" in content

    def test_nav_lucide_mappings(self):
        content = INDEX_HTML.read_text()
        expected_icons = [
            "message-circle",
            "clipboard-list",
            "layout-dashboard",
            "activity",
            "settings",
            "users",
            "bot",
            "cpu",
            "globe",
            "brain",
            "file-text",
            "palette",
        ]
        for icon in expected_icons:
            assert icon in content, f"Missing Lucide icon: {icon}"


class TestWorkspaceIndexHTML:
    """Tests for workspace index.html theme support."""

    def test_loads_tokens_css(self):
        content = WS_INDEX_HTML.read_text()
        assert "tokens.css" in content

    def test_has_lucide_script(self):
        content = WS_INDEX_HTML.read_text()
        assert "lucide" in content.lower()

    def test_has_view_toggle(self):
        content = WS_INDEX_HTML.read_text()
        assert "wsViewToggle" in content

    def test_has_org_panel(self):
        content = WS_INDEX_HTML.read_text()
        assert "wsOrgPanel" in content
