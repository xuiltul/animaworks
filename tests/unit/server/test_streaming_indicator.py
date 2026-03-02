# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for streaming loading indicator improvements.

Verifies that:
1. CSS files contain breathe-bg animation for .streaming bubbles
2. JS files clear activeTool on tool_end and done events
"""
from __future__ import annotations

from pathlib import Path

import pytest

# ── Paths ──────────────────────────────────────────────────

_PROJECT_ROOT = Path(__file__).resolve().parents[3]

_DASHBOARD_CSS = _PROJECT_ROOT / "server" / "static" / "styles" / "chat.css"
_WORKSPACE_CSS = _PROJECT_ROOT / "server" / "static" / "workspace" / "style.css"
_DASHBOARD_JS = _PROJECT_ROOT / "server" / "static" / "pages" / "chat" / "streaming-controller.js"


# ── CSS: Breathing Animation ──────────────────────────────


class TestBreathingAnimationCSS:
    """Verify breathe-bg keyframes and animation are defined in both CSS files."""

    @pytest.mark.parametrize("css_path", [_DASHBOARD_CSS, _WORKSPACE_CSS])
    def test_breathe_bg_keyframes_defined(self, css_path: Path):
        content = css_path.read_text()
        assert "@keyframes breathe-bg" in content, (
            f"breathe-bg keyframes not found in {css_path.name}"
        )

    @pytest.mark.parametrize("css_path", [_DASHBOARD_CSS, _WORKSPACE_CSS])
    def test_streaming_class_has_animation(self, css_path: Path):
        content = css_path.read_text()
        assert "animation: breathe-bg" in content or "animation:breathe-bg" in content, (
            f".streaming class missing breathe-bg animation in {css_path.name}"
        )

    @pytest.mark.parametrize("css_path", [_DASHBOARD_CSS, _WORKSPACE_CSS])
    def test_breathe_bg_has_two_colors(self, css_path: Path):
        """Keyframes should transition between two background-color values."""
        content = css_path.read_text()
        start = content.find("@keyframes breathe-bg")
        assert start != -1
        depth = 0
        end = start
        for i, ch in enumerate(content[start:], start=start):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
        keyframes_block = content[start:end]
        assert keyframes_block.count("background-color") >= 2, (
            "breathe-bg should define at least 2 background-color values"
        )


# ── JS: tool_end Behavior ─────────────────────────────────


class TestToolEndBehaviorJS:
    """Verify onToolEnd handler clears activeTool in JS files."""

    @pytest.mark.parametrize("js_path", [_DASHBOARD_JS])
    def test_tool_end_clears_active_tool(self, js_path: Path):
        """After onToolEnd, activeTool should be reset to null."""
        content = js_path.read_text()

        tool_end_idx = content.find("onToolEnd")
        assert tool_end_idx != -1, f"onToolEnd handler not found in {js_path.name}"

        after_tool_end = content[tool_end_idx:tool_end_idx + 300]
        assert "activeTool = null" in after_tool_end, (
            f"onToolEnd handler should clear activeTool in {js_path.name}"
        )


class TestDoneHandlerClearsActiveTool:
    """Verify done (onDone) handler clears activeTool in JS files."""

    @pytest.mark.parametrize("js_path", [_DASHBOARD_JS])
    def test_done_clears_active_tool(self, js_path: Path):
        """The onDone event handler should reset activeTool to null."""
        content = js_path.read_text()

        done_idx = content.find("onDone")
        assert done_idx != -1, f"onDone handler not found in {js_path.name}"

        after_done = content[done_idx:done_idx + 500]
        assert "activeTool = null" in after_done, (
            f"onDone handler should clear activeTool in {js_path.name}"
        )
