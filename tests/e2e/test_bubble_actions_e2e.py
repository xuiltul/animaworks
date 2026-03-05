# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
"""E2E tests for bubble action bar feature.

Validates the complete integration: render-utils.js generates correct HTML
with data-raw-text attributes and action bar markup, CSS rules are
consistent with the HTML structure, and consumer files wire everything up.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

_STATIC_DIR = Path(__file__).resolve().parents[2] / "server" / "static"
_RENDER_UTILS_JS = _STATIC_DIR / "shared" / "chat" / "render-utils.js"
_CHAT_CSS = _STATIC_DIR / "styles" / "chat.css"
_CHAT_RENDERER_JS = _STATIC_DIR / "pages" / "chat" / "chat-renderer.js"
_WS_CHAT_HISTORY_JS = _STATIC_DIR / "workspace" / "modules" / "chat-history.js"


class TestActionBarHTMLConsistency:
    """Verify HTML structure matches CSS selectors."""

    @pytest.fixture(autouse=True)
    def _load(self) -> None:
        self.js = _RENDER_UTILS_JS.read_text(encoding="utf-8")
        self.css = _CHAT_CSS.read_text(encoding="utf-8")

    def test_bubble_actions_class_in_both(self) -> None:
        assert "bubble-actions" in self.js
        assert ".bubble-actions" in self.css

    def test_bubble_action_btn_class_in_both(self) -> None:
        assert "bubble-action-btn" in self.js
        assert ".bubble-action-btn" in self.css

    def test_visible_toggle_class_in_both(self) -> None:
        assert '"visible"' in self.js
        assert ".bubble-actions.visible" in self.css

    def test_streaming_class_hiding_in_both(self) -> None:
        assert "renderStreamingBubbleInner" in self.js
        assert "_bubbleActionsHtml" in self.js
        assert ".chat-bubble.assistant.streaming .bubble-actions" in self.css

    def test_data_action_attributes_match(self) -> None:
        js_actions = set(re.findall(r'data-action="(\w+)"', self.js))
        assert "copy" in js_actions
        assert "download" in js_actions

    def test_data_raw_text_attribute_consistency(self) -> None:
        assert "data-raw-text" in self.js
        assert "dataset.rawText" in self.js


class TestActionBarScopeRestriction:
    """Verify action bar only appears on assistant bubbles."""

    @pytest.fixture(autouse=True)
    def _load(self) -> None:
        self.js = _RENDER_UTILS_JS.read_text(encoding="utf-8")

    def test_action_bar_only_in_assistant_role(self) -> None:
        lines = self.js.splitlines()
        in_system = False
        in_user = False
        for line in lines:
            stripped = line.strip()
            if 'msg.role === "system"' in stripped:
                in_system = True
                in_user = False
            elif 'msg.role === "user"' in stripped or 'role === "user"' in stripped:
                in_user = True
                in_system = False
            elif 'msg.role === "assistant"' in stripped or "// assistant" in stripped:
                in_system = False
                in_user = False

            if (in_system or in_user) and "_bubbleActionsHtml" in stripped:
                pytest.fail(f"_bubbleActionsHtml found in non-assistant context: {stripped}")

    def test_bind_handler_targets_assistant_class(self) -> None:
        assert '.closest(".chat-bubble.assistant")' in self.js


class TestClipboardImplementation:
    """Verify clipboard copy has proper fallback chain."""

    @pytest.fixture(autouse=True)
    def _load(self) -> None:
        self.js = _RENDER_UTILS_JS.read_text(encoding="utf-8")

    def test_primary_clipboard_api(self) -> None:
        assert "navigator.clipboard.writeText(text)" in self.js

    def test_fallback_textarea_approach(self) -> None:
        assert 'document.createElement("textarea")' in self.js
        assert 'document.execCommand("copy")' in self.js

    def test_fallback_cleanup(self) -> None:
        assert "ta.remove()" in self.js


class TestDownloadImplementation:
    """Verify text download creates proper file."""

    @pytest.fixture(autouse=True)
    def _load(self) -> None:
        self.js = _RENDER_UTILS_JS.read_text(encoding="utf-8")

    def test_blob_creation(self) -> None:
        assert "new Blob([text]" in self.js
        assert "text/plain" in self.js

    def test_filename_format(self) -> None:
        assert "response_" in self.js
        assert ".txt" in self.js

    def test_url_cleanup(self) -> None:
        assert "URL.revokeObjectURL" in self.js

    def test_anchor_cleanup(self) -> None:
        assert "a.remove()" in self.js


class TestCopyFeedbackCycle:
    """Verify visual feedback for copy action."""

    @pytest.fixture(autouse=True)
    def _load(self) -> None:
        self.js = _RENDER_UTILS_JS.read_text(encoding="utf-8")

    def test_changes_to_check_icon(self) -> None:
        assert '"check"' in self.js

    def test_reverts_after_delay(self) -> None:
        lines = self.js.splitlines()
        found_revert = False
        for line in lines:
            if "setTimeout" in line and "1500" in line:
                found_revert = True
                break
            if "1500" in line and "setTimeout" in self.js:
                found_revert = True
                break
        assert found_revert, "Copy feedback must revert after 1500ms"

    def test_lucide_re_render(self) -> None:
        assert "lucide.createIcons" in self.js


class TestConsumerIntegrationFlow:
    """Verify consumers wire up handlers and icons correctly."""

    def test_dashboard_bind_order(self) -> None:
        source = _CHAT_RENDERER_JS.read_text(encoding="utf-8")
        tool_pos = source.index("bindToolCallHandlers(messagesEl)")
        bubble_pos = source.index("_sharedBindBubbleActionHandlers(messagesEl)")
        lucide_pos = source.index("lucide.createIcons")
        assert tool_pos < bubble_pos < lucide_pos, (
            "Dashboard must bind tool handlers, then bubble actions, then init lucide icons"
        )

    def test_workspace_bind_order(self) -> None:
        source = _WS_CHAT_HISTORY_JS.read_text(encoding="utf-8")
        tool_pos = source.index("bindToolCallHandlers(dom.convMessages)")
        bubble_pos = source.index("bindBubbleActionHandlers(dom.convMessages)")
        lucide_pos = source.index("lucide.createIcons")
        assert tool_pos < bubble_pos < lucide_pos, (
            "Workspace must bind tool handlers, then bubble actions, then init lucide icons"
        )


class TestDuplicateBindingPrevention:
    """Verify handler is only bound once per container."""

    @pytest.fixture(autouse=True)
    def _load(self) -> None:
        self.js = _RENDER_UTILS_JS.read_text(encoding="utf-8")

    def test_guard_attribute_checked(self) -> None:
        assert "container.dataset.bubbleActionsBound" in self.js

    def test_guard_attribute_set(self) -> None:
        assert 'container.dataset.bubbleActionsBound = "1"' in self.js


class TestInteractionExclusions:
    """Verify mobile tap toggle doesn't interfere with other interactive elements."""

    @pytest.fixture(autouse=True)
    def _load(self) -> None:
        self.js = _RENDER_UTILS_JS.read_text(encoding="utf-8")

    def test_excludes_tool_call_rows(self) -> None:
        assert '.closest(".tool-call-row")' in self.js

    def test_excludes_tool_call_headers(self) -> None:
        assert '.closest(".tool-call-group-header")' in self.js

    def test_excludes_links(self) -> None:
        assert '.closest("a")' in self.js

    def test_excludes_text_artifact_cards(self) -> None:
        assert '.closest(".text-artifact-card")' in self.js


class TestEscapeAttrSafety:
    """Verify _escapeAttr handles all HTML-special characters."""

    @pytest.fixture(autouse=True)
    def _load(self) -> None:
        self.js = _RENDER_UTILS_JS.read_text(encoding="utf-8")

    def test_escapes_ampersand(self) -> None:
        assert "/&/g" in self.js

    def test_escapes_double_quote(self) -> None:
        assert '/"/g' in self.js

    def test_escapes_less_than(self) -> None:
        assert "/</g" in self.js

    def test_escapes_greater_than(self) -> None:
        assert "/>/g" in self.js

    def test_handles_null_input(self) -> None:
        assert 'if (!str) return ""' in self.js
