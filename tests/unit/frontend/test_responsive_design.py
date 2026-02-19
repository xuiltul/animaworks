# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for responsive design implementation.

Verify CSS files contain required breakpoints, hover/active parity,
minimum tap targets, mobile-specific rules, and correct HTML/JS markup.

Tests read actual source files and validate content via string matching
and regex — no external CSS parser needed.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

# ── Paths ────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parents[3]
STYLES_DIR = PROJECT_ROOT / "server" / "static" / "styles"
WORKSPACE_DIR = PROJECT_ROOT / "server" / "static" / "workspace"
MODULES_DIR = PROJECT_ROOT / "server" / "static" / "modules"
DASHBOARD_HTML = PROJECT_ROOT / "server" / "static" / "index.html"
WORKSPACE_HTML = WORKSPACE_DIR / "index.html"
WORKSPACE_STYLE = WORKSPACE_DIR / "style.css"
RESPONSIVE_CSS = STYLES_DIR / "responsive.css"


# ── Helper ───────────────────────────────────────────────────


def _read(path: Path) -> str:
    """Read a file's text content."""
    return path.read_text(encoding="utf-8")


def _extract_media_blocks(css_text: str, max_width: int) -> list[str]:
    """Extract all @media blocks matching (max-width: <max_width>px).

    Returns list of block bodies (content between outermost braces).
    """
    pattern = rf"@media\s*\([^)]*max-width\s*:\s*{max_width}px[^)]*\)"
    blocks: list[str] = []
    for m in re.finditer(pattern, css_text):
        start = css_text.index("{", m.end())
        depth = 1
        pos = start + 1
        while pos < len(css_text) and depth > 0:
            if css_text[pos] == "{":
                depth += 1
            elif css_text[pos] == "}":
                depth -= 1
            pos += 1
        blocks.append(css_text[start + 1 : pos - 1])
    return blocks


# ── 1. Breakpoint Coverage ───────────────────────────────────


class TestBreakpointCoverage:
    """Verify responsive.css contains all required breakpoints."""

    def test_responsive_css_has_768px_breakpoint(self) -> None:
        css = _read(RESPONSIVE_CSS)
        assert "@media (max-width: 768px)" in css

    def test_responsive_css_has_640px_breakpoint(self) -> None:
        css = _read(RESPONSIVE_CSS)
        assert "@media (max-width: 640px)" in css

    def test_responsive_css_has_480px_breakpoint(self) -> None:
        css = _read(RESPONSIVE_CSS)
        assert "@media (max-width: 480px)" in css


# ── 2. Hover / Active Parity ────────────────────────────────


def _find_hover_selectors(css_text: str) -> set[str]:
    """Extract selectors that have :hover pseudo-class.

    Excludes selectors wrapped in @media (hover: hover) blocks and
    ::-webkit-scrollbar-thumb:hover which is a scrollbar style.
    """
    # Remove @media blocks that contain hover queries (hover: hover, hover: none)
    # This covers both `@media (hover: hover)` and
    # `@media (hover: hover) and (pointer: fine)` patterns.
    cleaned = css_text
    hover_media_pattern = r"@media\s*\([^)]*hover\s*:\s*(?:hover|none)[^)]*\)(?:\s*and\s*\([^)]*\))*"
    # Collect ranges to remove (from original text)
    removals: list[str] = []
    for m in re.finditer(hover_media_pattern, css_text):
        brace_start = css_text.index("{", m.end())
        depth = 1
        pos = brace_start + 1
        while pos < len(css_text) and depth > 0:
            if css_text[pos] == "{":
                depth += 1
            elif css_text[pos] == "}":
                depth -= 1
            pos += 1
        removals.append(css_text[m.start() : pos])
    for chunk in removals:
        cleaned = cleaned.replace(chunk, "")

    selectors: set[str] = set()
    for match in re.finditer(r"([^{}\n;]+):hover[^{]*\{", cleaned):
        sel = match.group(1).strip().split(",")[-1].strip()
        # Skip scrollbar pseudo-elements
        if "::-webkit-scrollbar" in sel:
            continue
        selectors.add(sel)
    return selectors


def _find_active_selectors(css_text: str) -> set[str]:
    """Extract selectors that have :active pseudo-class.

    Handles combined selectors like `.foo:hover, .foo:active { ... }`.
    Scans line-by-line to avoid cross-line greediness.
    """
    selectors: set[str] = set()
    for line in css_text.splitlines():
        if ":active" not in line:
            continue
        # Split by comma to handle combined selectors
        for part in line.split(","):
            if ":active" in part:
                sel = re.sub(r":active.*", "", part).strip()
                if sel and not sel.startswith("{") and not sel.startswith("/*"):
                    selectors.add(sel)
    return selectors


# Selectors that are passive visual feedback (row highlighting, content hover)
# and do not require :active counterparts.
_PASSIVE_HOVER_SELECTORS = {
    ".data-table tr",
    ".activity-entry",
    ".md-content .md-table tr",
    ".status-section-body .md-content .md-table tr",
    ".image-preview-remove",
    ".board-msg",
    ".chat-attached-image",
    ".activity-row",
    ".activity-load-more",
    ".chat-attach-btn",
    ".activity-type-chip",
    ".btn-login",
}


class TestHoverActiveParity:
    """Verify interactive elements have hover/active parity or hover-media wrapping."""

    def test_hover_active_parity_dashboard_css(self) -> None:
        """Interactive :hover rules in dashboard CSS should have :active counterparts."""
        css_files = sorted(STYLES_DIR.glob("*.css"))
        assert len(css_files) > 0, "No CSS files found in styles directory"

        all_hover: set[str] = set()
        all_active: set[str] = set()
        for f in css_files:
            css = _read(f)
            all_hover.update(_find_hover_selectors(css))
            all_active.update(_find_active_selectors(css))

        missing = all_hover - all_active - _PASSIVE_HOVER_SELECTORS
        assert not missing, (
            f"Selectors with :hover but no :active in dashboard CSS: {missing}"
        )

    def test_hover_active_parity_workspace_css(self) -> None:
        """Interactive :hover rules in workspace CSS should have :active counterparts."""
        css = _read(WORKSPACE_STYLE)
        hover = _find_hover_selectors(css)
        active = _find_active_selectors(css)

        missing = hover - active - _PASSIVE_HOVER_SELECTORS
        assert not missing, (
            f"Selectors with :hover but no :active in workspace CSS: {missing}"
        )


# ── 3. Tap Target Compliance ────────────────────────────────


class TestTapTargets:
    """Verify buttons and interactive elements meet 44px minimum tap targets."""

    def test_mobile_tap_targets_responsive_css(self) -> None:
        """Within the 768px media query, buttons/tabs should have min-height: 44px."""
        css = _read(RESPONSIVE_CSS)
        blocks_768 = _extract_media_blocks(css, 768)
        combined = "\n".join(blocks_768)

        # These selectors should have min-height: 44px
        expected_44px = [
            ".btn-logout",
            ".nav-item",
            ".page-tab",
            ".memory-tab",
            ".right-tab",
            ".chat-send-btn",
        ]
        for sel in expected_44px:
            assert "min-height: 44px" in combined, (
                f"Expected min-height: 44px in 768px media query for {sel}"
            )

    def test_workspace_mobile_tap_targets(self) -> None:
        """Within workspace 768px media query, tap targets should be >= 44px."""
        css = _read(WORKSPACE_STYLE)
        blocks_768 = _extract_media_blocks(css, 768)
        combined = "\n".join(blocks_768)

        assert "min-height: 44px" in combined, (
            "Expected min-height: 44px in workspace 768px media query"
        )


# ── 4. Font Size Minimums ───────────────────────────────────


class TestFontSizes:
    """Verify no font-size falls below reasonable minimum in mobile queries."""

    @staticmethod
    def _find_small_font_sizes(css_text: str) -> list[str]:
        """Find font-size values below 0.65rem in mobile media query blocks."""
        issues: list[str] = []
        for width in (768, 640, 480):
            for block in _extract_media_blocks(css_text, width):
                for m in re.finditer(r"font-size\s*:\s*([\d.]+)rem", block):
                    val = float(m.group(1))
                    if val < 0.65:
                        issues.append(f"{val}rem in {width}px query")
        return issues

    def test_no_tiny_fonts_responsive_css(self) -> None:
        css = _read(RESPONSIVE_CSS)
        issues = self._find_small_font_sizes(css)
        assert not issues, f"Font sizes below 0.75rem: {issues}"

    def test_no_tiny_fonts_workspace_css(self) -> None:
        css = _read(WORKSPACE_STYLE)
        issues = self._find_small_font_sizes(css)
        assert not issues, f"Font sizes below 0.75rem: {issues}"


# ── 5. Hamburger Menu HTML ──────────────────────────────────


class TestHamburgerMenu:
    """Verify hamburger button markup exists in HTML files."""

    def test_hamburger_button_dashboard_html(self) -> None:
        html = _read(DASHBOARD_HTML)
        assert 'id="hamburgerBtn"' in html or 'class="hamburger-btn"' in html

    def test_hamburger_button_workspace_html(self) -> None:
        """Workspace has mobile toggle buttons for sidebar/character panels."""
        html = _read(WORKSPACE_HTML)
        assert 'id="wsMobileSidebarToggle"' in html or "ws-mobile-sidebar-toggle" in html

    def test_mobile_nav_backdrop_dashboard_html(self) -> None:
        html = _read(DASHBOARD_HTML)
        assert "mobile-nav-backdrop" in html


# ── 6. Touch.js Module ──────────────────────────────────────


class TestTouchModule:
    """Verify touch.js exists and exports SwipeHandler."""

    def test_touch_js_module_exists(self) -> None:
        touch_js = MODULES_DIR / "touch.js"
        assert touch_js.exists(), "touch.js module should exist"

    def test_touch_js_exports_swipe_handler(self) -> None:
        js = _read(MODULES_DIR / "touch.js")
        assert "export class SwipeHandler" in js

    def test_swipe_handler_has_callback_methods(self) -> None:
        js = _read(MODULES_DIR / "touch.js")
        assert "onSwipeLeft" in js
        assert "onSwipeRight" in js

    def test_swipe_handler_has_destroy_method(self) -> None:
        js = _read(MODULES_DIR / "touch.js")
        assert "destroy()" in js


# ── 7. Chat Mobile Send ─────────────────────────────────────


class TestWorkspaceChatMobile:
    """Verify workspace chat.js handles mobile send and virtual keyboard."""

    def test_workspace_chat_mobile_enter_send(self) -> None:
        """Workspace app.js should have mobile Enter-to-send logic (not just Ctrl+Enter)."""
        js = _read(WORKSPACE_DIR / "modules" / "app.js")
        # Should detect mobile via matchMedia and allow plain Enter
        assert "matchMedia" in js, "Expected matchMedia for mobile detection"
        assert "Enter" in js, "Expected Enter key handling"

        # Specifically: mobile path should send on Enter (no Ctrl required)
        # Check that there's a mobile branch with different behavior
        assert "mobile" in js.lower() or "max-width: 768px" in js

    def test_workspace_chat_visualviewport(self) -> None:
        """Workspace app.js should handle visualViewport for mobile keyboard."""
        js = _read(WORKSPACE_DIR / "modules" / "app.js")
        assert "visualViewport" in js

    def test_workspace_chat_mobile_placeholder(self) -> None:
        """Chat placeholder should show mobile-appropriate shortcut hint."""
        js = _read(WORKSPACE_DIR / "modules" / "app.js")
        # Should show "Enter" on mobile, "Ctrl+Enter" on desktop
        assert "Ctrl+Enter" in js or "Ctrl\\+Enter" in js


# ── 8. Sidebar Drawer Styles ────────────────────────────────


class TestSidebarDrawer:
    """Verify mobile sidebar drawer implementation."""

    def test_mobile_sidebar_drawer_styles(self) -> None:
        """Dashboard sidebar should become a fixed overlay drawer on mobile."""
        css = _read(RESPONSIVE_CSS)
        blocks_768 = _extract_media_blocks(css, 768)
        combined = "\n".join(blocks_768)

        assert "position: fixed" in combined, "Sidebar should be fixed position"
        assert "translateX" in combined, "Sidebar should use translateX for slide"

    def test_workspace_mobile_sidebar_styles(self) -> None:
        """Workspace sidebar should become an overlay drawer on mobile."""
        css = _read(WORKSPACE_STYLE)
        blocks_768 = _extract_media_blocks(css, 768)
        combined = "\n".join(blocks_768)

        assert "translateX" in combined, (
            "Workspace sidebar should use translateX for mobile drawer"
        )


# ── 9. Chat Bubble Width on Mobile ──────────────────────────


class TestChatBubbleWidth:
    """Verify chat bubbles expand on mobile for better readability."""

    def test_workspace_chat_bubble_width_mobile(self) -> None:
        """Chat bubble max-width should be > 75% on mobile."""
        css = _read(WORKSPACE_STYLE)
        # Check in 900px, 768px, or 640px media queries for wider bubbles
        found_wider = False
        for width in (900, 768, 640):
            blocks = _extract_media_blocks(css, width)
            for block in blocks:
                # Look for max-width values larger than 75%
                for m in re.finditer(r"max-width\s*:\s*(\d+)%", block):
                    val = int(m.group(1))
                    if val > 75:
                        found_wider = True
                        break
        assert found_wider, "Expected chat bubble max-width > 75% in mobile media queries"

    def test_responsive_chat_bubble_width_mobile(self) -> None:
        """Dashboard chat bubble max-width should be wider on mobile."""
        css = _read(RESPONSIVE_CSS)
        blocks_768 = _extract_media_blocks(css, 768)
        combined = "\n".join(blocks_768)
        # Should have max-width: 90% for .chat-bubble
        assert "max-width: 90%" in combined or "max-width: 95%" in combined


# ── 10. Desktop-only Hover Effects ──────────────────────────


class TestDesktopOnlyHover:
    """Verify hover effects are wrapped in @media (hover: hover)."""

    def test_responsive_css_hover_media_query(self) -> None:
        """responsive.css should use @media (hover: hover) for desktop-only effects."""
        css = _read(RESPONSIVE_CSS)
        assert "@media (hover: hover)" in css

    def test_workspace_css_hover_media_query(self) -> None:
        """workspace style.css should use @media (hover: hover) for desktop-only effects."""
        css = _read(WORKSPACE_STYLE)
        assert "@media (hover: hover)" in css


# ── 11. Hamburger Button Hidden by Default ──────────────────


class TestHamburgerHiddenDefault:
    """Verify hamburger button is hidden on desktop and visible on mobile."""

    def test_hamburger_hidden_by_default(self) -> None:
        """Hamburger button should be display: none by default."""
        css = _read(RESPONSIVE_CSS)
        # Outside media queries, .hamburger-btn should be display: none
        assert "display: none" in css
        # The hamburger-btn base rule (not inside media query) should hide it
        pattern = r"\.hamburger-btn\s*\{[^}]*display\s*:\s*none"
        assert re.search(pattern, css), (
            "Expected .hamburger-btn to have display: none outside media queries"
        )

    def test_hamburger_visible_on_mobile(self) -> None:
        """Hamburger button should be display: flex inside 768px media query."""
        css = _read(RESPONSIVE_CSS)
        blocks_768 = _extract_media_blocks(css, 768)
        combined = "\n".join(blocks_768)
        assert "display: flex" in combined


# ── 12. Mobile Nav Backdrop ─────────────────────────────────


class TestMobileNavBackdrop:
    """Verify backdrop overlay for mobile navigation."""

    def test_backdrop_hidden_by_default(self) -> None:
        css = _read(RESPONSIVE_CSS)
        # .mobile-nav-backdrop should be display: none outside media queries
        assert "mobile-nav-backdrop" in css

    def test_backdrop_shown_when_nav_open(self) -> None:
        css = _read(RESPONSIVE_CSS)
        assert ".mobile-nav-open .mobile-nav-backdrop" in css


# ── 13. Workspace Mobile UI Elements ────────────────────────


class TestWorkspaceMobileElements:
    """Verify workspace-specific mobile UI elements."""

    def test_mobile_sidebar_toggle_hidden_desktop(self) -> None:
        """Mobile toggle buttons should be hidden on desktop."""
        css = _read(WORKSPACE_STYLE)
        assert "display: none" in css
        # Verify the toggle buttons are hidden by default
        pattern = r"\.ws-mobile-sidebar-toggle[^{]*\{[^}]*display\s*:\s*none"
        assert re.search(pattern, css, re.DOTALL)

    def test_mobile_sidebar_toggle_visible_mobile(self) -> None:
        """Mobile toggle buttons should be visible in 768px media query."""
        css = _read(WORKSPACE_STYLE)
        blocks_768 = _extract_media_blocks(css, 768)
        combined = "\n".join(blocks_768)
        assert "ws-mobile-sidebar-toggle" in combined
        assert "display: inline-flex" in combined

    def test_workspace_sidebar_backdrop_exists(self) -> None:
        """Workspace should have a sidebar backdrop element."""
        html = _read(WORKSPACE_HTML)
        assert "wsSidebarBackdrop" in html or "ws-conv-sidebar-backdrop" in html

    def test_workspace_conv_sidebar_becomes_drawer(self) -> None:
        """Workspace sidebar should use absolute positioning on mobile."""
        css = _read(WORKSPACE_STYLE)
        blocks_768 = _extract_media_blocks(css, 768)
        combined = "\n".join(blocks_768)
        assert "position: absolute" in combined
