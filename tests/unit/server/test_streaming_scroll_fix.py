"""Unit tests for streaming bubble scroll fix.

Verifies that the JavaScript source files contain the correct patterns
for rAF-based scrolling and throttled streaming updates.
"""
from __future__ import annotations

from pathlib import Path

import pytest

# Path to project root (worktree)
PROJECT_ROOT = Path(__file__).resolve().parents[3]


class TestAppJsScrollFix:
    """Verify app.js contains correct scroll and throttle patterns.

    app.js has two streaming contexts:
    - The conversation panel (sendConversationMessage) uses scheduleStreamingUpdate
      with rAF throttling and updateStreamingBubble for rendering.
    - renderConvMessages uses synchronous scrollTop (non-streaming render).
    """

    @pytest.fixture()
    def source(self) -> str:
        return (PROJECT_ROOT / "server/static/workspace/modules/app.js").read_text()

    def test_update_streaming_bubble_exists(self, source: str):
        """updateStreamingBubble function should exist in app.js."""
        assert "function updateStreamingBubble" in source

    def test_update_streaming_bubble_updates_inner_html(self, source: str):
        """updateStreamingBubble should update bubble.innerHTML."""
        idx = source.index("function updateStreamingBubble")
        end_marker = source.find("\n// ──", idx + 1)
        if end_marker == -1:
            end_marker = source.find("\nfunction ", idx + 100)
        func_body = source[idx:end_marker]
        assert "bubble.innerHTML" in func_body

    def test_schedule_streaming_update_exists(self, source: str):
        """scheduleStreamingUpdate function with rAF throttle guard must exist."""
        assert "function scheduleStreamingUpdate" in source
        assert "_convRafPending" in source

    def test_text_delta_uses_throttled_update(self, source: str):
        """onTextDelta callback should call scheduleStreamingUpdate, not updateStreamingBubble directly."""
        # Find the onTextDelta callback in the conversation streaming code
        idx = source.index("onTextDelta")
        # Get surrounding context (next 400 chars to capture the full callback body)
        context = source[idx:idx + 400]
        assert "scheduleStreamingUpdate" in context

    def test_render_conv_messages_renders_chat(self, source: str):
        """renderConvMessages should render chat messages into the DOM."""
        idx = source.index("function renderConvMessages")
        end_idx = source.find("\nasync function", idx + 1)
        if end_idx == -1:
            end_idx = source.find("\nfunction ", idx + 100)
        func_body = source[idx:end_idx]
        assert "innerHTML" in func_body


class TestChatJsScrollFix:
    """Verify chat.js contains correct scroll and throttle patterns."""

    @pytest.fixture()
    def source(self) -> str:
        return (PROJECT_ROOT / "server/static/workspace/modules/chat.js").read_text()

    def test_update_streaming_bubble_uses_raf_scroll(self, source: str):
        """updateStreamingBubble should use rAF + scrollIntoView."""
        assert "bubble.scrollIntoView(" in source
        assert "requestAnimationFrame" in source

    def test_no_synchronous_scroll_in_update_streaming_bubble(self, source: str):
        """updateStreamingBubble must not set scrollTop synchronously after innerHTML."""
        idx = source.index("function updateStreamingBubble")
        end_marker = source.find("\nfunction ", idx + 100)
        if end_marker == -1:
            end_marker = len(source)
        func_body = source[idx:end_marker]
        assert "scrollTop" not in func_body

    def test_schedule_streaming_update_exists(self, source: str):
        """scheduleStreamingUpdate function with rAF throttle guard must exist."""
        assert "function scheduleStreamingUpdate" in source
        assert "_chatRafPending" in source

    def test_text_delta_uses_throttled_update(self, source: str):
        """onTextDelta handler should call scheduleStreamingUpdate."""
        idx = source.index("onTextDelta")
        context = source[idx:idx + 200]
        assert "scheduleStreamingUpdate" in context

    def test_render_all_messages_uses_raf_scroll(self, source: str):
        """renderAllMessages should use rAF + scrollIntoView."""
        idx = source.index("function renderAllMessages")
        end_idx = source.find("\n// ──", idx + 1)
        if end_idx == -1:
            end_idx = source.find("\nfunction ", idx + 100)
        func_body = source[idx:end_idx]
        assert "requestAnimationFrame" in func_body
        assert "scrollIntoView" in func_body
        assert "scrollTop" not in func_body
