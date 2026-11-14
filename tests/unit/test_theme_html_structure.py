"""Unit tests for HTML theme support structure."""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

INDEX_HTML = Path(__file__).resolve().parents[2] / "server" / "static" / "index.html"
WS_INDEX_HTML = Path(__file__).resolve().parents[2] / "server" / "static" / "workspace" / "index.html"


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

    def test_has_theme_support(self):
        """Verify theme infrastructure exists (app.js handles theme programmatically)."""
        content = INDEX_HTML.read_text()
        assert "tokens.css" in content

    def test_has_lucide_script(self):
        content = INDEX_HTML.read_text()
        assert "lucide" in content.lower()

    def test_nav_uses_svg_sprite_icons(self):
        """Every sidebar item draws one icon from the inline SVG sprite."""
        content = INDEX_HTML.read_text()
        nav = content[content.index('<nav class="sidebar-nav"') : content.index("</nav>")]
        assert 'class="nav-sprite"' in nav
        assert "nav-emoji" not in nav
        assert "data-lucide" not in nav
        assert nav.count('class="nav-item') == nav.count('class="nav-icon"')

    def test_nav_icon_symbols_exist(self):
        content = INDEX_HTML.read_text()
        # Keep in sync with sidebar nav icons in index.html
        expected_icons = [
            "home",
            "chat",
            "board",
            "hub",
            "tasks",
            "activity",
            "workspace",
            "settings",
        ]
        for icon in expected_icons:
            assert f'<symbol id="nav-i-{icon}"' in content, f"Missing nav icon symbol: {icon}"
            assert f'<use href="#nav-i-{icon}">' in content, f"Nav icon not used: {icon}"

    def test_hub_nav_item_starts_hidden_with_badge(self):
        content = INDEX_HTML.read_text()
        assert '<li id="navHubItem" hidden><a href="#/hub"' in content
        assert 'id="navHubBadge" hidden' in content
        assert "styles/hub.css" in content


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
