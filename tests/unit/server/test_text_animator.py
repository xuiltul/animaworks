# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for TextAnimator FE streaming smoothing.

Verifies that:
1. TextAnimator class is exported from render-utils.js
2. TextAnimator has required API (start, push, flush, stop, displayText)
3. _renderTextZoneContent uses _displayText for streaming
4. updateStreamingZone fast path uses _displayText
6. streaming-controller.js (Chat Page) imports and integrates TextAnimator
7. All terminal callbacks (onDone, onError, onAbort, onFinally) flush/stop the animator
"""
from __future__ import annotations

from pathlib import Path

import pytest

# ── Paths ──────────────────────────────────────────────────

_PROJECT_ROOT = Path(__file__).resolve().parents[3]

_RENDER_UTILS = _PROJECT_ROOT / "server" / "static" / "shared" / "chat" / "render-utils.js"
_CHAT_STREAMING = _PROJECT_ROOT / "server" / "static" / "pages" / "chat" / "streaming-controller.js"


@pytest.fixture()
def render_utils_src() -> str:
    return _RENDER_UTILS.read_text()



@pytest.fixture()
def chat_streaming_src() -> str:
    return _CHAT_STREAMING.read_text()


# ── TextAnimator Class Definition ──────────────────────────


class TestTextAnimatorClass:
    """TextAnimator class is properly defined and exported."""

    def test_class_exported(self, render_utils_src: str):
        assert "export class TextAnimator" in render_utils_src

    def test_has_start_method(self, render_utils_src: str):
        assert "start()" in render_utils_src

    def test_has_push_method(self, render_utils_src: str):
        assert "push(delta)" in render_utils_src

    def test_has_flush_method(self, render_utils_src: str):
        assert "flush()" in render_utils_src

    def test_has_stop_method(self, render_utils_src: str):
        assert "stop()" in render_utils_src

    def test_has_display_text_getter(self, render_utils_src: str):
        assert "get displayText()" in render_utils_src

    def test_has_is_animating_getter(self, render_utils_src: str):
        assert "get isAnimating()" in render_utils_src

    def test_uses_request_animation_frame(self, render_utils_src: str):
        assert "requestAnimationFrame" in render_utils_src

    def test_has_catchup_acceleration(self, render_utils_src: str):
        assert "_CATCHUP_THRESHOLD_FAST" in render_utils_src
        assert "_CATCHUP_THRESHOLD_MED" in render_utils_src

    def test_has_adaptive_rate(self, render_utils_src: str):
        assert "_adaptRate" in render_utils_src
        assert "_pushHistory" in render_utils_src
        assert "_RATE_WINDOW_SIZE" in render_utils_src
        assert "_RATE_DRAIN_FACTOR" in render_utils_src

    def test_push_records_history(self, render_utils_src: str):
        assert "this._pushHistory.push(" in render_utils_src

    def test_rate_has_bounds(self, render_utils_src: str):
        assert "_MIN_CHAR_INTERVAL_MS" in render_utils_src
        assert "_MAX_CHAR_INTERVAL_MS" in render_utils_src


# ── Render Utils Integration ───────────────────────────────


class TestRenderUtilsThinkingDisplayText:
    """_renderThinkingZoneContent uses _displayThinkingText."""

    def test_thinking_zone_uses_display_thinking_text(self, render_utils_src: str):
        assert "msg._displayThinkingText || msg.thinkingText" in render_utils_src


class TestRenderUtilsDisplayText:
    """_renderTextZoneContent and fast path use _displayText."""

    def test_text_zone_uses_display_text(self, render_utils_src: str):
        assert "msg._displayText || msg.text" in render_utils_src

    def test_fast_path_uses_display_text(self, render_utils_src: str):
        assert "const visibleText = msg._displayText || msg.text" in render_utils_src

    def test_fast_path_checks_visible_text(self, render_utils_src: str):
        assert "visibleText.length - c.len" in render_utils_src
        assert "visibleText.slice(c.len)" in render_utils_src


# ── Chat Page Integration ──────────────────────────────────


class TestChatPageTextAnimatorIntegration:
    """streaming-controller.js imports and uses TextAnimator correctly."""

    def test_imports_text_animator(self, chat_streaming_src: str):
        assert "TextAnimator" in chat_streaming_src
        assert "import" in chat_streaming_src.split("TextAnimator")[0].split("\n")[-1]

    def test_creates_animator_on_stream_start(self, chat_streaming_src: str):
        assert "new TextAnimator(" in chat_streaming_src

    def test_pushes_delta_to_animator(self, chat_streaming_src: str):
        assert "_textAnimator.push(text)" in chat_streaming_src or "_textAnimator) _textAnimator.push(text)" in chat_streaming_src

    def test_sets_display_text_on_update(self, chat_streaming_src: str):
        assert "streamingMsg._displayText = displayText" in chat_streaming_src

    def test_flushes_on_done(self, chat_streaming_src: str):
        lines = chat_streaming_src.split("\n")
        for i, line in enumerate(lines):
            if "onDone:" in line:
                block = "\n".join(lines[i:i + 10])
                if "flush()" in block:
                    return
        pytest.fail("onDone callback does not call flush()")

    def test_flushes_on_abort(self, chat_streaming_src: str):
        lines = chat_streaming_src.split("\n")
        for i, line in enumerate(lines):
            if "onAbort:" in line:
                block = "\n".join(lines[i:i + 5])
                if "flush()" in block:
                    return
        pytest.fail("onAbort callback does not call flush()")

    def test_stops_on_finally(self, chat_streaming_src: str):
        assert "_textAnimator.stop()" in chat_streaming_src or "_textAnimator) { _textAnimator.stop()" in chat_streaming_src

    def test_cleans_display_text_on_done(self, chat_streaming_src: str):
        assert "delete streamingMsg._displayText" in chat_streaming_src

    def test_finalizes_error_flushes_animator(self, chat_streaming_src: str):
        lines = chat_streaming_src.split("\n")
        for i, line in enumerate(lines):
            if "finalizeStreamError" in line and "const" in line:
                block = "\n".join(lines[i:i + 15])
                if "flush()" in block:
                    return
        pytest.fail("finalizeStreamError does not call flush()")

    def test_resume_also_uses_animator(self, chat_streaming_src: str):
        assert "_resumeAnimator" in chat_streaming_src

    def test_resume_composites_base_text(self, chat_streaming_src: str):
        """Resume streams must composite recovered text + new animator output."""
        assert "resumeBase" in chat_streaming_src
        assert "resumeBase + displayText" in chat_streaming_src

    def test_thinking_animator_created(self, chat_streaming_src: str):
        assert "_thinkingAnimator" in chat_streaming_src

    def test_thinking_delta_pushes_to_animator(self, chat_streaming_src: str):
        assert "_thinkingAnimator.push(text)" in chat_streaming_src or "_thinkingAnimator) _thinkingAnimator.push(text)" in chat_streaming_src

    def test_thinking_end_flushes_animator(self, chat_streaming_src: str):
        assert "_thinkingAnimator.flush()" in chat_streaming_src

    def test_thinking_sets_display_text(self, chat_streaming_src: str):
        assert "_displayThinkingText" in chat_streaming_src

    def test_cleans_display_thinking_text_on_done(self, chat_streaming_src: str):
        assert "delete streamingMsg._displayThinkingText" in chat_streaming_src
