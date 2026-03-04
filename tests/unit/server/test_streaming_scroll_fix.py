# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
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
    """Verify chat-streaming.js contains correct scroll and throttle patterns.

    chat-streaming.js handles workspace streaming with scheduleStreamingUpdate
    (rAF throttling) and updateStreamingBubble for rendering.
    renderConvMessages lives in chat-history.js for synchronous scrollTop render.
    """

    @pytest.fixture()
    def streaming_source(self) -> str:
        return (PROJECT_ROOT / "server/static/workspace/modules/chat-streaming.js").read_text()

    @pytest.fixture()
    def history_source(self) -> str:
        return (PROJECT_ROOT / "server/static/workspace/modules/chat-history.js").read_text()

    def test_update_streaming_bubble_exists(self, streaming_source: str):
        """updateStreamingBubble function should exist in chat-streaming.js."""
        assert "function updateStreamingBubble" in streaming_source

    def test_update_streaming_bubble_updates_inner_html(self, streaming_source: str):
        """updateStreamingBubble delegates to updateStreamingZone which updates bubble.innerHTML."""
        # bubble.innerHTML is in render-utils.js updateStreamingZone (imported by chat-streaming)
        render_utils = (PROJECT_ROOT / "server/static/shared/chat/render-utils.js").read_text()
        assert "bubble.innerHTML" in render_utils

    def test_schedule_streaming_update_exists(self, streaming_source: str):
        """scheduleStreamingUpdate function with rAF throttle guard must exist."""
        assert "function scheduleStreamingUpdate" in streaming_source
        assert "_convRafPending" in streaming_source

    def test_text_delta_uses_throttled_update(self, streaming_source: str):
        """onTextDelta callback should call scheduleStreamingUpdate, not updateStreamingBubble directly."""
        idx = streaming_source.index("onTextDelta")
        context = streaming_source[idx:idx + 400]
        assert "scheduleStreamingUpdate" in context

    def test_render_conv_messages_renders_chat(self, history_source: str):
        """renderConvMessages should render chat messages into the DOM."""
        idx = history_source.index("function renderConvMessages")
        end_idx = history_source.find("\nasync function", idx + 1)
        if end_idx == -1:
            end_idx = history_source.find("\nfunction ", idx + 100)
            if end_idx == -1:
                end_idx = history_source.find("\nexport function ", idx + 100)
        func_body = history_source[idx:end_idx]
        assert "innerHTML" in func_body


class TestWorkspaceAppJsScrollFix:
    """Verify workspace chat-streaming.js contains correct scroll and throttle patterns.

    After workspace chat.js deletion, chat-streaming.js handles all workspace streaming.
    Uses scrollTop-based scrolling (not scrollIntoView).
    """

    @pytest.fixture()
    def streaming_source(self) -> str:
        return (PROJECT_ROOT / "server/static/workspace/modules/chat-streaming.js").read_text()

    @pytest.fixture()
    def history_source(self) -> str:
        return (PROJECT_ROOT / "server/static/workspace/modules/chat-history.js").read_text()

    def test_update_streaming_bubble_scrolls(self, streaming_source: str):
        """updateStreamingBubble should scroll the messages container."""
        idx = streaming_source.index("function updateStreamingBubble")
        end_marker = streaming_source.find("\n// ──", idx + 1)
        if end_marker == -1:
            end_marker = streaming_source.find("\nfunction ", idx + 100)
            if end_marker == -1:
                end_marker = streaming_source.find("\nexport function ", idx + 100)
        func_body = streaming_source[idx:end_marker]
        assert "scrollTop" in func_body or "scrollIntoView" in func_body

    def test_schedule_streaming_update_exists(self, streaming_source: str):
        """scheduleStreamingUpdate function with rAF throttle guard must exist."""
        assert "function scheduleStreamingUpdate" in streaming_source
        assert "_convRafPending" in streaming_source

    def test_text_delta_uses_throttled_update(self, streaming_source: str):
        """onTextDelta handler should call scheduleStreamingUpdate."""
        idx = streaming_source.index("onTextDelta")
        context = streaming_source[idx:idx + 400]
        assert "scheduleStreamingUpdate" in context

    def test_render_conv_messages_scrolls(self, history_source: str):
        """renderConvMessages should scroll messages container."""
        idx = history_source.index("function renderConvMessages")
        end_idx = history_source.find("\nasync function", idx + 1)
        if end_idx == -1:
            end_idx = history_source.find("\nfunction ", idx + 100)
            if end_idx == -1:
                end_idx = history_source.find("\nexport function ", idx + 100)
        func_body = history_source[idx:end_idx]
        assert "scrollTop" in func_body
