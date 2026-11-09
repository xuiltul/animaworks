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
_WORKSPACE_HTML = _PROJECT_ROOT / "server" / "static" / "workspace" / "index.html"
_DASHBOARD_JS = _PROJECT_ROOT / "server" / "static" / "pages" / "chat" / "streaming-controller.js"


# ── CSS: Breathing Animation ──────────────────────────────


class TestStreamingIndicatorCSS:
    """Verify the streaming cursor style exists and reaches the Workspace dock."""

    def test_streaming_cursor_class_defined(self):
        content = _DASHBOARD_CSS.read_text()
        assert ".streaming-cursor" in content, f".streaming-cursor class not found in {_DASHBOARD_CSS.name}"

    def test_streaming_cursor_has_animation(self):
        content = _DASHBOARD_CSS.read_text()
        cursor_start = content.index(".streaming-cursor")
        assert "animation" in content[cursor_start:], f".streaming-cursor missing animation in {_DASHBOARD_CSS.name}"

    def test_workspace_dock_loads_the_shared_chat_styles(self):
        assert "/styles/chat.css" in _WORKSPACE_HTML.read_text()


# ── JS: tool_end Behavior ─────────────────────────────────


class TestToolEndBehaviorJS:
    """Verify onToolEnd handler clears activeTool in JS files."""

    @pytest.mark.parametrize("js_path", [_DASHBOARD_JS])
    def test_tool_end_clears_active_tool(self, js_path: Path):
        """After onToolEnd, activeTool should be reset to null."""
        content = js_path.read_text()

        tool_end_idx = content.find("onToolEnd")
        assert tool_end_idx != -1, f"onToolEnd handler not found in {js_path.name}"

        after_tool_end = content[tool_end_idx : tool_end_idx + 300]
        assert "activeTool = null" in after_tool_end, f"onToolEnd handler should clear activeTool in {js_path.name}"


class TestDoneHandlerClearsActiveTool:
    """Verify done (onDone) handler clears activeTool in JS files."""

    @pytest.mark.parametrize("js_path", [_DASHBOARD_JS])
    def test_done_clears_active_tool(self, js_path: Path):
        """The onDone event handler should reset activeTool to null."""
        content = js_path.read_text()

        done_idx = content.find("onDone")
        assert done_idx != -1, f"onDone handler not found in {js_path.name}"

        after_done = content[done_idx : done_idx + 1200]
        assert "activeTool = null" in after_done, f"onDone handler should clear activeTool in {js_path.name}"
