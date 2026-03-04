"""Unit tests for CSS design tokens file structure."""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import pytest


TOKENS_CSS = (
    Path(__file__).resolve().parents[2]
    / "server"
    / "static"
    / "styles"
    / "tokens.css"
)


class TestTokensCSS:
    """Tests for tokens.css file existence and structure."""

    def test_tokens_file_exists(self):
        assert TOKENS_CSS.exists(), f"tokens.css not found at {TOKENS_CSS}"

    def test_has_root_variables(self):
        content = TOKENS_CSS.read_text()
        assert ":root {" in content or ":root{" in content

    def test_has_business_theme(self):
        content = TOKENS_CSS.read_text()
        assert ".theme-business" in content

    def test_required_color_variables(self):
        content = TOKENS_CSS.read_text()
        required_vars = [
            "--aw-color-accent",
            "--aw-color-bg-primary",
            "--aw-color-bg-secondary",
            "--aw-color-text-primary",
            "--aw-color-text-secondary",
            "--aw-color-success",
            "--aw-color-warning",
            "--aw-color-error",
            "--aw-color-border",
        ]
        for var in required_vars:
            assert var in content, f"Missing CSS variable: {var}"

    def test_animation_control_variables(self):
        content = TOKENS_CSS.read_text()
        assert "--aw-animation-play-state" in content
        assert "--aw-avatar-display" in content
        assert "--aw-avatar-initial-display" in content
        assert "--aw-emoji-display" in content
        assert "--aw-icon-display" in content

    def test_realistic_mode_hides_emoji_and_shows_icons(self):
        content = TOKENS_CSS.read_text()
        mode_start = content.index(".mode-realistic {")
        mode_block = content[
            mode_start : content.index("}", mode_start) + 1
        ]
        assert "--aw-emoji-display: none" in mode_block
        assert "--aw-icon-display: inline-flex" in mode_block

    def test_realistic_mode_hides_emoji(self):
        content = TOKENS_CSS.read_text()
        mode_start = content.index(".mode-realistic {")
        depth = 0
        end = mode_start
        for i in range(mode_start, len(content)):
            if content[i] == "{":
                depth += 1
            elif content[i] == "}":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
        mode_block = content[mode_start:end]
        assert "--aw-emoji-display" in mode_block


class TestCSSFilesUseTokens:
    """Test that CSS files reference token variables instead of hardcoded colors."""

    STYLES_DIR = (
        Path(__file__).resolve().parents[2]
        / "server"
        / "static"
        / "styles"
    )

    @pytest.mark.parametrize(
        "css_file",
        [
            "base.css",
            "chat.css",
            "dashboard.css",
            "sidebar-nav.css",
            "sidebar.css",
            "layout.css",
            "activity.css",
            "board.css",
            "history.css",
            "memory.css",
            "avatar.css",
            "responsive.css",
        ],
    )
    def test_css_uses_variables(self, css_file: str):
        """Each CSS file should reference --aw-color-* variables."""
        path = self.STYLES_DIR / css_file
        assert path.exists(), f"{css_file} not found"
        content = path.read_text()
        assert (
            "var(--aw-color-" in content
        ), f"{css_file} does not use --aw-color-* CSS variables"


class TestLucideBundle:
    """Test that Lucide icon library is bundled."""

    VENDOR_DIR = (
        Path(__file__).resolve().parents[2]
        / "server"
        / "static"
        / "vendor"
        / "lucide"
    )

    def test_lucide_js_exists(self):
        js_file = self.VENDOR_DIR / "lucide.min.js"
        assert js_file.exists(), "lucide.min.js not bundled"

    def test_lucide_license_exists(self):
        license_file = self.VENDOR_DIR / "LICENSE"
        assert license_file.exists(), "Lucide LICENSE not included"

    def test_lucide_js_not_empty(self):
        js_file = self.VENDOR_DIR / "lucide.min.js"
        assert js_file.stat().st_size > 1000, "lucide.min.js seems too small"
