# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for multi-thread chat frontend implementation.

Static analysis tests: read JS/CSS files and verify patterns for
thread tabs, thread_id passthrough, and refactored chat history structure.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

# ── Paths ────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CHAT_JS = PROJECT_ROOT / "server" / "static" / "pages" / "chat.js"
WORKSPACE_APP_JS = PROJECT_ROOT / "server" / "static" / "workspace" / "modules" / "app.js"
WORKSPACE_STATE_JS = PROJECT_ROOT / "server" / "static" / "workspace" / "modules" / "state.js"
CHAT_CSS = PROJECT_ROOT / "server" / "static" / "styles" / "chat.css"
WORKSPACE_STYLE = PROJECT_ROOT / "server" / "static" / "workspace" / "style.css"


def _read(path: Path) -> str:
    """Read a file's text content."""
    return path.read_text(encoding="utf-8")


# ── TestChatJsThreadTabs ────────────────────────────────────


@pytest.mark.unit
class TestChatJsThreadTabs:
    """Verify chat.js HTML structure for thread tabs."""

    def test_chat_js_contains_thread_tabs_div_with_id(self) -> None:
        """chat.js contains thread-tabs div with chatThreadTabs id."""
        js = _read(CHAT_JS)
        assert 'class="thread-tabs"' in js
        assert 'id="chatThreadTabs"' in js

    def test_chat_js_contains_chat_new_thread_btn(self) -> None:
        """chat.js contains chatNewThreadBtn button."""
        js = _read(CHAT_JS)
        assert "chatNewThreadBtn" in js

    def test_chat_js_selected_thread_id_initialized_default(self) -> None:
        """_selectedThreadId variable is initialized to 'default'."""
        js = _read(CHAT_JS)
        assert "_selectedThreadId = \"default\"" in js

    def test_chat_js_threads_initialized_empty_object(self) -> None:
        """_threads variable is initialized as empty object {}."""
        js = _read(CHAT_JS)
        assert "_threads = {}" in js


# ── TestChatJsThreadIdInSendChat ─────────────────────────────


@pytest.mark.unit
class TestChatJsThreadIdInSendChat:
    """Verify thread_id is included in send and fetch calls."""

    def test_send_chat_includes_thread_id_in_body(self) -> None:
        """_sendChat function includes thread_id: tid in the body JSON."""
        js = _read(CHAT_JS)
        assert "thread_id: tid" in js
        assert "bodyObj" in js or "body" in js
        # Body is built as JSON with thread_id
        assert re.search(r"thread_id\s*:\s*tid", js)

    def test_fetch_conversation_history_includes_thread_id_param(self) -> None:
        """_fetchConversationHistory includes thread_id parameter in URL."""
        js = _read(CHAT_JS)
        assert "thread_id" in js
        # Template literal: `&thread_id=${encodeURIComponent(threadId)}`
        assert "&thread_id=" in js
        assert "encodeURIComponent" in js and "threadId" in js


# ── TestChatJsChatHistoryRefactored ─────────────────────────


@pytest.mark.unit
class TestChatJsChatHistoryRefactored:
    """Verify _chatHistories and _historyState use per-thread structure."""

    def test_render_chat_accesses_chat_histories_by_name_and_tid(self) -> None:
        """_renderChat accesses _chatHistories[name]?.[tid]."""
        js = _read(CHAT_JS)
        assert "_chatHistories[name]" in js
        # Refactored: per-thread access via [tid] or ?.[tid]
        assert "?.[tid]" in js or "_chatHistories[name][tid]" in js

    def test_render_chat_accesses_history_state_by_name_and_tid(self) -> None:
        """History state is accessed as _historyState[name]?.[tid]."""
        js = _read(CHAT_JS)
        assert "_historyState[name]" in js
        # Refactored: per-thread access
        assert "?.[tid]" in js or "?.[threadId]" in js or "_historyState[name][" in js


# ── TestWorkspaceStateFields ────────────────────────────────


@pytest.mark.unit
class TestWorkspaceStateFields:
    """Verify state.js contains thread-related state fields."""

    def test_state_js_contains_active_thread_id(self) -> None:
        """state.js contains activeThreadId field."""
        js = _read(WORKSPACE_STATE_JS)
        assert "activeThreadId" in js

    def test_state_js_contains_threads(self) -> None:
        """state.js contains threads field."""
        js = _read(WORKSPACE_STATE_JS)
        assert "threads:" in js or "threads :" in js

    def test_state_js_contains_chat_messages_by_thread(self) -> None:
        """state.js contains chatMessagesByThread field."""
        js = _read(WORKSPACE_STATE_JS)
        assert "chatMessagesByThread" in js


# ── TestWorkspaceAppJsThreadTabs ─────────────────────────────


@pytest.mark.unit
class TestWorkspaceAppJsThreadTabs:
    """Verify workspace app.js thread tab rendering and thread_id in send."""

    def test_app_js_contains_thread_tab_rendering(self) -> None:
        """app.js contains thread tab rendering function/logic."""
        js = _read(WORKSPACE_APP_JS)
        assert "thread-tab" in js
        assert "wsThreadTabs" in js or "threadTabs" in js

    def test_send_conversation_message_includes_thread_id_in_body(self) -> None:
        """sendConversationMessage includes thread_id in the body."""
        js = _read(WORKSPACE_APP_JS)
        assert "thread_id" in js
        assert re.search(r"thread_id\s*:\s*threadId", js)


# ── TestChatCssThreadStyles ──────────────────────────────────


@pytest.mark.unit
class TestChatCssThreadStyles:
    """Verify chat.css contains thread tab CSS classes."""

    def test_chat_css_thread_tabs_class(self) -> None:
        """chat.css contains .thread-tabs class."""
        css = _read(CHAT_CSS)
        assert ".thread-tabs" in css

    def test_chat_css_thread_tab_class(self) -> None:
        """chat.css contains .thread-tab class."""
        css = _read(CHAT_CSS)
        assert ".thread-tab" in css

    def test_chat_css_thread_tab_active_class(self) -> None:
        """chat.css contains .thread-tab.active class."""
        css = _read(CHAT_CSS)
        assert ".thread-tab.active" in css

    def test_chat_css_thread_tab_new_class(self) -> None:
        """chat.css contains .thread-tab-new class."""
        css = _read(CHAT_CSS)
        assert ".thread-tab-new" in css


# ── TestWorkspaceCssThreadStyles ─────────────────────────────


@pytest.mark.unit
class TestWorkspaceCssThreadStyles:
    """Verify workspace style.css contains thread tab styles."""

    def test_workspace_css_thread_tabs_exists(self) -> None:
        """Workspace style.css contains thread-tabs / ws-thread-tabs."""
        css = _read(WORKSPACE_STYLE)
        assert "thread-tabs" in css or "ws-thread-tabs" in css

    def test_workspace_css_thread_tab_class(self) -> None:
        """Workspace has .thread-tab styling."""
        css = _read(WORKSPACE_STYLE)
        assert ".thread-tab" in css

    def test_workspace_css_thread_tab_new_class(self) -> None:
        """Workspace has .thread-tab-new styling."""
        css = _read(WORKSPACE_STYLE)
        assert ".thread-tab-new" in css
