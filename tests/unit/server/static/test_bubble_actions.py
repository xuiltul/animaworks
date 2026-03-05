# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for bubble action bar (copy/download) in render-utils.js and chat.css.

Validates that:
- render-utils.js contains the action bar HTML generation helpers
- render-utils.js exports bindBubbleActionHandlers
- CSS includes bubble-actions styling
- Consumer files import and call the handler
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

_STATIC_DIR = Path(__file__).resolve().parents[4] / "server" / "static"
_RENDER_UTILS_JS = _STATIC_DIR / "shared" / "chat" / "render-utils.js"
_CHAT_CSS = _STATIC_DIR / "styles" / "chat.css"
_CHAT_RENDERER_JS = _STATIC_DIR / "pages" / "chat" / "chat-renderer.js"
_WS_CHAT_HISTORY_JS = _STATIC_DIR / "workspace" / "modules" / "chat-history.js"


class TestRenderUtilsHelpers:
    """Verify render-utils.js has the action bar helpers."""

    @pytest.fixture(autouse=True)
    def _load_source(self) -> None:
        assert _RENDER_UTILS_JS.exists()
        self.source = _RENDER_UTILS_JS.read_text(encoding="utf-8")

    def test_escape_attr_function_exists(self) -> None:
        assert "function _escapeAttr(" in self.source

    def test_escape_attr_handles_ampersand(self) -> None:
        assert '"&amp;"' in self.source

    def test_escape_attr_handles_quotes(self) -> None:
        assert '"&quot;"' in self.source

    def test_escape_attr_handles_lt_gt(self) -> None:
        assert '"&lt;"' in self.source
        assert '"&gt;"' in self.source

    def test_bubble_actions_html_function_exists(self) -> None:
        assert "function _bubbleActionsHtml(" in self.source

    def test_bubble_actions_html_returns_empty_for_no_text(self) -> None:
        assert 'if (!rawText) return ""' in self.source

    def test_bubble_actions_html_has_copy_button(self) -> None:
        assert 'data-action="copy"' in self.source

    def test_bubble_actions_html_has_download_button(self) -> None:
        assert 'data-action="download"' in self.source

    def test_bubble_actions_uses_lucide_icons(self) -> None:
        assert 'data-lucide="copy"' in self.source
        assert 'data-lucide="download"' in self.source


class TestRenderUtilsHistoryMessage:
    """Verify renderHistoryMessage embeds data-raw-text and action bar."""

    @pytest.fixture(autouse=True)
    def _load_source(self) -> None:
        self.source = _RENDER_UTILS_JS.read_text(encoding="utf-8")

    def test_history_message_has_data_raw_text(self) -> None:
        assert "data-raw-text" in self.source

    def test_history_message_uses_escape_attr(self) -> None:
        assert "_escapeAttr(rawText)" in self.source

    def test_history_message_calls_bubble_actions_html(self) -> None:
        assert "_bubbleActionsHtml(rawText)" in self.source


class TestRenderUtilsLiveBubble:
    """Verify renderLiveBubble hides action bar during streaming."""

    @pytest.fixture(autouse=True)
    def _load_source(self) -> None:
        self.source = _RENDER_UTILS_JS.read_text(encoding="utf-8")

    def test_live_bubble_hides_actions_during_streaming(self) -> None:
        assert 'msg.streaming ? "" : _bubbleActionsHtml(rawText)' in self.source

    def test_live_bubble_conditional_data_attr(self) -> None:
        assert "!msg.streaming" in self.source


class TestBindBubbleActionHandlers:
    """Verify bindBubbleActionHandlers is exported and implements required features."""

    @pytest.fixture(autouse=True)
    def _load_source(self) -> None:
        self.source = _RENDER_UTILS_JS.read_text(encoding="utf-8")

    def test_function_is_exported(self) -> None:
        assert "export function bindBubbleActionHandlers(" in self.source

    def test_uses_event_delegation(self) -> None:
        assert "container.addEventListener" in self.source

    def test_has_copy_to_clipboard(self) -> None:
        assert "navigator.clipboard.writeText" in self.source

    def test_has_clipboard_fallback(self) -> None:
        assert 'document.execCommand("copy")' in self.source

    def test_has_download_function(self) -> None:
        assert "function _downloadAsText(" in self.source

    def test_download_creates_blob(self) -> None:
        assert "new Blob([text]" in self.source

    def test_download_uses_create_object_url(self) -> None:
        assert "URL.createObjectURL" in self.source

    def test_has_copy_visual_feedback(self) -> None:
        assert 'data-lucide", "check"' in self.source

    def test_copy_feedback_reverts(self) -> None:
        assert "setTimeout" in self.source
        assert "1500" in self.source

    def test_has_timestamp_extraction(self) -> None:
        assert "function _extractBubbleTimestamp(" in self.source

    def test_has_timestamp_formatting(self) -> None:
        assert "function _formatTimestamp(" in self.source

    def test_mobile_toggle_support(self) -> None:
        assert ".bubble-actions.visible" in self.source or 'classList.toggle("visible")' in self.source

    def test_prevents_duplicate_binding(self) -> None:
        assert "bubbleActionsBound" in self.source

    def test_excludes_text_artifact_card(self) -> None:
        assert ".text-artifact-card" in self.source


class TestChatCSSBubbleActions:
    """Verify chat.css includes bubble action bar styles."""

    @pytest.fixture(autouse=True)
    def _load_source(self) -> None:
        assert _CHAT_CSS.exists()
        self.source = _CHAT_CSS.read_text(encoding="utf-8")

    def test_bubble_actions_class_exists(self) -> None:
        assert ".bubble-actions" in self.source

    def test_bubble_actions_is_absolute(self) -> None:
        actions_match = re.search(
            r"\.bubble-actions\s*\{([^}]+)\}", self.source,
        )
        assert actions_match, ".bubble-actions rule not found"
        block = actions_match.group(1)
        assert "position: absolute" in block

    def test_assistant_bubble_has_relative_position(self) -> None:
        assert "position: relative" in self.source

    def test_hover_shows_actions(self) -> None:
        assert ".chat-bubble.assistant:hover .bubble-actions" in self.source

    def test_streaming_hides_actions(self) -> None:
        assert ".chat-bubble.assistant.streaming .bubble-actions" in self.source

    def test_bubble_action_btn_styled(self) -> None:
        assert ".bubble-action-btn" in self.source

    def test_mobile_override_exists(self) -> None:
        assert ".bubble-actions.visible" in self.source

    def test_uses_design_tokens(self) -> None:
        assert "var(--aw-color-" in self.source


class TestConsumerImports:
    """Verify consumer files import and use bindBubbleActionHandlers."""

    def test_chat_renderer_imports_handler(self) -> None:
        source = _CHAT_RENDERER_JS.read_text(encoding="utf-8")
        assert "bindBubbleActionHandlers" in source

    def test_chat_renderer_calls_handler(self) -> None:
        source = _CHAT_RENDERER_JS.read_text(encoding="utf-8")
        assert "_sharedBindBubbleActionHandlers(messagesEl)" in source

    def test_chat_renderer_initializes_lucide(self) -> None:
        source = _CHAT_RENDERER_JS.read_text(encoding="utf-8")
        assert "lucide.createIcons" in source

    def test_ws_chat_history_imports_handler(self) -> None:
        source = _WS_CHAT_HISTORY_JS.read_text(encoding="utf-8")
        assert "bindBubbleActionHandlers" in source

    def test_ws_chat_history_calls_handler(self) -> None:
        source = _WS_CHAT_HISTORY_JS.read_text(encoding="utf-8")
        assert "bindBubbleActionHandlers(dom.convMessages)" in source

    def test_ws_chat_history_initializes_lucide(self) -> None:
        source = _WS_CHAT_HISTORY_JS.read_text(encoding="utf-8")
        assert "lucide.createIcons" in source
