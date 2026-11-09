# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for streaming bubble scroll and throttling.

The Workspace now embeds the Chat pane, so streaming lives in
``pages/chat/streaming-controller.js`` and ``pages/chat/chat-renderer.js``.
"""
from __future__ import annotations

from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
STREAMING_JS = PROJECT_ROOT / "server" / "static" / "pages" / "chat" / "streaming-controller.js"
RENDERER_JS = PROJECT_ROOT / "server" / "static" / "pages" / "chat" / "chat-renderer.js"
RENDER_UTILS_JS = PROJECT_ROOT / "server" / "static" / "shared" / "chat" / "render-utils.js"


class TestStreamingScrollFix:
    """Verify Chat streaming keeps the message list pinned to the bottom."""

    @pytest.fixture()
    def streaming_source(self) -> str:
        return STREAMING_JS.read_text(encoding="utf-8")

    @pytest.fixture()
    def renderer_source(self) -> str:
        return RENDERER_JS.read_text(encoding="utf-8")

    def test_streaming_renders_are_throttled(self, streaming_source: str) -> None:
        assert "_scheduleRender" in streaming_source
        assert "requestAnimationFrame" in streaming_source

    def test_streaming_scrolls_to_bottom(self, streaming_source: str) -> None:
        assert "msgs.scrollTop = msgs.scrollHeight" in streaming_source

    def test_streaming_uses_text_animator(self, streaming_source: str) -> None:
        assert "TextAnimator" in streaming_source
        assert "_textAnimator.push(" in streaming_source

    def test_renderer_scroll_helper_uses_scroll_top(self, renderer_source: str) -> None:
        idx = renderer_source.index("function scrollToBottom")
        end = renderer_source.index("function ", idx + 20)
        block = renderer_source[idx:end]
        assert "scrollTop = el.scrollHeight" in block

    def test_render_utils_updates_streaming_bubble(self) -> None:
        source = RENDER_UTILS_JS.read_text(encoding="utf-8")
        assert "function updateStreamingZone" in source
        assert "bubble.innerHTML" in source
