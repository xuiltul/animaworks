# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for Anima birth reveal animation integration.

Validates that the workspace frontend files contain the required
reveal overlay elements, CSS classes, and JS module wiring.
"""
from __future__ import annotations

from pathlib import Path

import pytest

# ── Path to workspace static files ──────────────────────

WORKSPACE_DIR = Path(__file__).resolve().parents[3] / "server" / "static" / "workspace"


# ── HTML Structure Tests ────────────────────────────────


class TestRevealOverlayHTML:
    """Verify workspace/index.html contains all required reveal elements."""

    @pytest.fixture(autouse=True)
    def _load_html(self):
        self.html = (WORKSPACE_DIR / "index.html").read_text(encoding="utf-8")

    def test_overlay_container_exists(self):
        assert 'id="wsRevealOverlay"' in self.html

    def test_overlay_has_class(self):
        assert 'class="ws-reveal-overlay"' in self.html

    def test_overlay_has_aria_hidden(self):
        assert 'aria-hidden="true"' in self.html

    def test_flash_element_exists(self):
        assert 'class="ws-reveal-flash"' in self.html

    def test_content_element_exists(self):
        assert 'class="ws-reveal-content"' in self.html

    def test_avatar_element_exists(self):
        assert 'id="wsRevealAvatar"' in self.html
        assert 'class="ws-reveal-avatar"' in self.html

    def test_text_element_exists(self):
        assert 'id="wsRevealText"' in self.html
        assert 'class="ws-reveal-text"' in self.html

    def test_overlay_inside_dashboard(self):
        """Reveal overlay should be inside #wsDashboard so it's only visible when logged in."""
        dashboard_start = self.html.index('id="wsDashboard"')
        overlay_pos = self.html.index('id="wsRevealOverlay"')
        # Find closing </div> for dashboard — overlay must be before it
        assert overlay_pos > dashboard_start


# ── CSS Tests ───────────────────────────────────────────


class TestRevealCSS:
    """Verify workspace/style.css contains all required reveal CSS rules."""

    @pytest.fixture(autouse=True)
    def _load_css(self):
        self.css = (WORKSPACE_DIR / "style.css").read_text(encoding="utf-8")

    def test_overlay_base_class(self):
        assert ".ws-reveal-overlay" in self.css

    def test_overlay_active_class(self):
        assert ".ws-reveal-overlay.active" in self.css

    def test_flash_class(self):
        assert ".ws-reveal-flash" in self.css

    def test_content_class(self):
        assert ".ws-reveal-content" in self.css

    def test_avatar_class(self):
        assert ".ws-reveal-avatar" in self.css

    def test_text_class(self):
        assert ".ws-reveal-text" in self.css

    def test_flash_keyframes(self):
        assert "@keyframes ws-reveal-flash" in self.css

    def test_content_keyframes(self):
        assert "@keyframes ws-reveal-content" in self.css

    def test_z_index_high(self):
        """Reveal overlay must have very high z-index to cover everything."""
        assert "z-index: 10000" in self.css

    def test_fixed_positioning(self):
        assert "position: fixed" in self.css

    def test_prefers_reduced_motion(self):
        assert "prefers-reduced-motion" in self.css

    def test_gpu_optimized_will_change(self):
        assert "will-change:" in self.css


# ── JS Module Tests ─────────────────────────────────────


class TestRevealJSModule:
    """Verify reveal.js module exists and exports required functions."""

    @pytest.fixture(autouse=True)
    def _load_js(self):
        self.js = (WORKSPACE_DIR / "modules" / "reveal.js").read_text(encoding="utf-8")

    def test_exports_play_reveal(self):
        assert "export async function playReveal" in self.js

    def test_exports_preload_image(self):
        assert "export function preloadImage" in self.js

    def test_uses_document_get_element_by_id(self):
        """Should reference the overlay elements by their IDs."""
        assert 'getElementById("wsRevealOverlay")' in self.js
        assert 'getElementById("wsRevealAvatar")' in self.js
        assert 'getElementById("wsRevealText")' in self.js

    def test_sets_text_content(self):
        """Should set the birth announcement text."""
        assert "さんが生まれました" in self.js

    def test_handles_animation_end(self):
        assert "animationend" in self.js

    def test_has_fallback_timeout(self):
        """Must have a fallback timeout to prevent infinite waits."""
        assert "FALLBACK_TIMEOUT" in self.js
        assert "setTimeout" in self.js

    def test_adds_active_class(self):
        assert '"active"' in self.js

    def test_handles_image_load_failure(self):
        """preloadImage should resolve on error too."""
        assert "onerror" in self.js


# ── App.js Integration Tests ────────────────────────────


class TestAppJSRevealIntegration:
    """Verify app.js imports and uses the reveal module."""

    @pytest.fixture(autouse=True)
    def _load_app_js(self):
        self.app_js = (WORKSPACE_DIR / "modules" / "app.js").read_text(encoding="utf-8")

    def test_imports_play_reveal(self):
        assert 'import { playReveal } from "./reveal.js"' in self.app_js

    def test_calls_play_reveal_on_assets_updated(self):
        assert "playReveal" in self.app_js

    def test_checks_for_avatar_assets(self):
        """Should only trigger reveal when avatar assets are present."""
        assert 'startsWith("avatar_")' in self.app_js

    def test_constructs_avatar_url(self):
        assert "avatar_bustup.png" in self.app_js
