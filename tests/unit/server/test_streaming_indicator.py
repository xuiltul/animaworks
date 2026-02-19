# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for streaming loading indicator improvements.

Verifies that:
1. CSS files contain breathe-bg animation for .streaming bubbles
2. JS files keep activeTool visible after tool_end (not cleared)
3. JS files clear activeTool only on done event
"""
from __future__ import annotations

from pathlib import Path

import pytest

# ── Paths ──────────────────────────────────────────────────

_PROJECT_ROOT = Path(__file__).resolve().parents[3]

_DASHBOARD_CSS = _PROJECT_ROOT / "server" / "static" / "styles" / "chat.css"
_WORKSPACE_CSS = _PROJECT_ROOT / "server" / "static" / "workspace" / "style.css"
_DASHBOARD_JS = _PROJECT_ROOT / "server" / "static" / "modules" / "chat.js"


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
        # The .streaming rule must reference breathe-bg animation
        assert "animation: breathe-bg" in content or "animation:breathe-bg" in content, (
            f".streaming class missing breathe-bg animation in {css_path.name}"
        )

    @pytest.mark.parametrize("css_path", [_DASHBOARD_CSS, _WORKSPACE_CSS])
    def test_breathe_bg_has_two_colors(self, css_path: Path):
        """Keyframes should transition between two background-color values."""
        content = css_path.read_text()
        # Extract the keyframes block
        start = content.find("@keyframes breathe-bg")
        assert start != -1
        # Find the matching closing brace (nested braces)
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
    """Verify tool_end (onToolEnd) handler does NOT clear activeTool in JS files.

    The JS uses callback-style handlers (onToolEnd) rather than switch/case.
    The onToolEnd callback should keep the tool indicator visible (not null it).
    """

    @pytest.mark.parametrize("js_path", [_DASHBOARD_JS])
    def test_tool_end_does_not_clear_active_tool(self, js_path: Path):
        """After onToolEnd, activeTool should remain set (not nulled)."""
        content = js_path.read_text()

        # Find the onToolEnd callback
        tool_end_idx = content.find("onToolEnd")
        assert tool_end_idx != -1, f"onToolEnd handler not found in {js_path.name}"

        # Extract a reasonable chunk around onToolEnd
        after_tool_end = content[tool_end_idx:tool_end_idx + 300]

        # Find the callback body (between the first { and matching })
        brace_start = after_tool_end.find("{")
        assert brace_start != -1, f"onToolEnd callback body not found in {js_path.name}"
        # Find the closing brace of the callback
        depth = 0
        end = brace_start
        for i, ch in enumerate(after_tool_end[brace_start:], start=brace_start):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
        tool_end_block = after_tool_end[brace_start:end]

        # activeTool = null should NOT appear in the onToolEnd block
        assert "activeTool = null" not in tool_end_block, (
            f"onToolEnd handler should NOT clear activeTool in {js_path.name}"
        )


class TestDoneHandlerClearsActiveTool:
    """Verify done (onDone) handler DOES clear activeTool in JS files.

    The JS uses callback-style handlers (onDone) rather than switch/case.
    """

    @pytest.mark.parametrize("js_path", [_DASHBOARD_JS])
    def test_done_clears_active_tool(self, js_path: Path):
        """The onDone event handler should reset activeTool to null."""
        content = js_path.read_text()

        # Find the onDone callback
        done_idx = content.find("onDone")
        assert done_idx != -1, f"onDone handler not found in {js_path.name}"

        # Extract a reasonable chunk after "onDone"
        after_done = content[done_idx:done_idx + 500]

        # activeTool = null should appear in the onDone block
        assert "activeTool = null" in after_done, (
            f"onDone handler should clear activeTool in {js_path.name}"
        )
