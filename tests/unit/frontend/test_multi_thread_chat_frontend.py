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
CHAT_CTX_JS = PROJECT_ROOT / "server" / "static" / "pages" / "chat" / "ctx.js"
CHAT_STREAMING_JS = PROJECT_ROOT / "server" / "static" / "pages" / "chat" / "streaming-controller.js"
CHAT_RENDERER_JS = PROJECT_ROOT / "server" / "static" / "pages" / "chat" / "chat-renderer.js"
SESSION_MANAGER_JS = PROJECT_ROOT / "server" / "static" / "shared" / "chat" / "session-manager.js"
CHAT_CSS = PROJECT_ROOT / "server" / "static" / "styles" / "chat.css"


def _read(path: Path) -> str:
    """Read a file's text content."""
    return path.read_text(encoding="utf-8")


# ── TestChatJsThreadTabs ────────────────────────────────────


@pytest.mark.unit
class TestChatJsThreadTabs:
    """Verify chat.js HTML structure for thread tabs."""

    def test_chat_js_contains_thread_tabs_div_with_id(self) -> None:
        """Pane host contains thread-tabs div with chatThreadTabs data-chat-id."""
        pane_host = PROJECT_ROOT / "server" / "static" / "pages" / "chat" / "pane-host.js"
        js = _read(pane_host)
        assert 'class="thread-tabs"' in js
        assert 'data-chat-id="chatThreadTabs"' in js

    def test_chat_js_contains_chat_new_thread_btn(self) -> None:
        """Pane host contains chatNewThreadBtn button."""
        pane_host = PROJECT_ROOT / "server" / "static" / "pages" / "chat" / "pane-host.js"
        js = _read(pane_host)
        assert "chatNewThreadBtn" in js

    def test_chat_js_selected_thread_id_initialized_default(self) -> None:
        """selectedThreadId is initialized to 'default' in chat context."""
        js = _read(CHAT_CTX_JS)
        assert 'selectedThreadId: "default"' in js

    def test_chat_js_threads_initialized_empty_object(self) -> None:
        """threads is initialized as empty object {} in chat context."""
        js = _read(CHAT_CTX_JS)
        assert "threads: {}" in js


# ── TestChatJsThreadIdInSendChat ─────────────────────────────


@pytest.mark.unit
class TestChatJsThreadIdInSendChat:
    """Verify thread_id is included in send and fetch calls."""

    def test_send_chat_includes_thread_id_in_body(self) -> None:
        """Session manager includes thread_id: thread in the body JSON."""
        js = _read(SESSION_MANAGER_JS)
        assert "thread_id: thread" in js
        assert "bodyObj" in js or "body" in js
        assert re.search(r"thread_id\s*:\s*thread", js)

    def test_fetch_conversation_history_includes_thread_id_param(self) -> None:
        """Chat feed fetch includes thread_id parameter in URL."""
        js = _read(CHAT_CTX_JS)
        assert "thread_id" in js
        assert "&thread_id=" in js
        assert "encodeURIComponent" in js and "threadId" in js


# ── TestChatJsChatHistoryRefactored ─────────────────────────


@pytest.mark.unit
class TestChatJsChatHistoryRefactored:
    """Verify _chatHistories and _historyState use per-thread structure."""

    def test_render_chat_accesses_chat_histories_by_name_and_tid(self) -> None:
        """Chat renderer accesses messages via mgr.getMessages(name, tid)."""
        js = _read(CHAT_RENDERER_JS)
        assert "mgr.getMessages(name, tid)" in js

    def test_render_chat_accesses_history_state_by_name_and_tid(self) -> None:
        """History state is accessed via mgr.getHistoryState(name, tid)."""
        js = _read(CHAT_RENDERER_JS)
        assert "mgr.getHistoryState(name, tid)" in js


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
