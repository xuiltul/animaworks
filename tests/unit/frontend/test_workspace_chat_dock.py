"""Static checks for the Workspace embedded Chat dock."""

from __future__ import annotations

from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
STATIC = PROJECT_ROOT / "server" / "static"
WORKSPACE = STATIC / "workspace"
WORKSPACE_MODULES = WORKSPACE / "modules"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


@pytest.mark.unit
class TestWorkspaceLayout:
    """Verify the Workspace shell embeds the Chat pane beside the stage."""

    def test_index_loads_what_the_chat_modules_need_at_import_time(self) -> None:
        html = _read(WORKSPACE / "index.html")
        # modules/state.js calls marked.use() while it is imported; without this
        # script the whole Workspace module graph fails and the page stays blank.
        assert "marked/marked.min.js" in html
        assert html.index("marked/marked.min.js") < html.index('type="module"')

    @pytest.mark.parametrize("sheet", ["chat", "chat-memory", "activity", "memory", "history"])
    def test_index_loads_chat_popover_stylesheets(self, sheet: str) -> None:
        assert f"/styles/{sheet}.css" in _read(WORKSPACE / "index.html")

    def test_index_has_dock_and_stage_but_no_legacy_overlay(self) -> None:
        html = _read(WORKSPACE / "index.html")
        assert 'id="wsChatDock"' in html
        assert 'id="wsStage"' in html
        assert "wsConvOverlay" not in html
        assert "ws-conv" not in html

    def test_index_loads_chat_styles(self) -> None:
        html = _read(WORKSPACE / "index.html")
        for css in (
            "styles/chat.css",
            "styles/chat-line.css",
            "styles/chat-call.css",
            "styles/chat-presence.css",
            "styles/chat-thinking.css",
            "styles/chat-memory.css",
            "styles/chat-mobile.css",
        ):
            assert css in html, f"{css} is not loaded by the Workspace page"

    def test_old_workspace_chat_modules_are_gone(self) -> None:
        for name in (
            "chat-controller.js",
            "chat-history.js",
            "chat-streaming.js",
            "chat-thread.js",
            "chat-mobile.js",
        ):
            assert not (WORKSPACE_MODULES / name).exists(), f"{name} should be deleted"

    def test_workspace_does_not_greet_on_open(self) -> None:
        sources = "\n".join(p.read_text(encoding="utf-8") for p in WORKSPACE_MODULES.glob("*.js"))
        assert "greetAnima" not in sources
        assert "/greet" not in sources


@pytest.mark.unit
class TestChatDockModule:
    """Verify chat-dock.js reuses the Chat pane with the Workspace socket."""

    def test_uses_create_pane_host_embedded(self) -> None:
        src = _read(WORKSPACE_MODULES / "chat-dock.js")
        assert "createPaneHost" in src
        assert "embedded: true" in src

    def test_uses_workspace_websocket(self) -> None:
        src = _read(WORKSPACE_MODULES / "chat-dock.js")
        assert 'from "./websocket.js"' in src

    def test_open_close_lifecycle(self) -> None:
        src = _read(WORKSPACE_MODULES / "chat-dock.js")
        assert "openConversation" in src
        assert "closeConversation" in src
        assert "setVisible" in src
        assert "highlightAnima" in _read(WORKSPACE_MODULES / "app.js")
        assert "destroyAllForAnima" not in src

    def test_app_initializes_dock_with_close_callback(self) -> None:
        src = _read(WORKSPACE_MODULES / "app.js")
        assert "initChatDock(dom.chatDock" in src
        assert "onClosed: () => highlightAnima(null)" in src
        assert "openConversation(name)" in src


@pytest.mark.unit
class TestPaneHostEmbeddedSupport:
    """Verify the shared Chat pane host keeps the Chat screen behavior intact."""

    def test_on_event_is_an_option(self) -> None:
        src = _read(STATIC / "pages" / "chat" / "pane-host.js")
        assert "options.onEvent" in src
        assert "initPresence({ api, onEvent })" in src

    def test_embedded_skips_rail_and_layout(self) -> None:
        src = _read(STATIC / "pages" / "chat" / "pane-host.js")
        assert "if (embedded) hostEl.classList.add" in src
        assert "if (embedded) return;" in src
        assert "if (embedded || _isMobile()) return null;" in src

    def test_embedded_does_not_save_ui_state(self) -> None:
        src = _read(STATIC / "pages" / "chat" / "ctx.js")
        assert "if (ctx.state.embedded) return;" in src

    def test_embedded_close_button_is_marked(self) -> None:
        src = _read(STATIC / "pages" / "chat" / "pane-host.js")
        assert 'data-chat-id="chatDockClose"' in src
        assert 't("common.aria_close")' in src


@pytest.mark.unit
class TestWorkspaceStyles:
    """Verify the dock layout and embedded control visibility rules."""

    def test_stage_and_dock_layout(self) -> None:
        css = _read(WORKSPACE / "style.css")
        assert ".ws-stage" in css
        assert ".ws-chat-dock" in css
        assert "width: clamp(380px, 38vw, 560px)" in css
        assert ".ws-chat-dock[hidden]" in css

    def test_embedded_controls_are_hidden(self) -> None:
        css = _read(WORKSPACE / "style.css")
        for selector in (
            ".chat-unified-hamburger",
            ".anima-tabs",
            ".chat-rail-toggle",
            ".chat-call-btn",
            ".chat-add-conversation",
            ".chat-split-pane-btn",
            ".chat-close-pane-btn",
            ".chat-unified-user-btn",
            ".chat-unified-user-menu",
        ):
            assert selector in css, f"{selector} should be hidden in the dock"

    def test_old_chat_css_does_not_collide(self) -> None:
        css = _read(WORKSPACE / "style.css")
        for selector in (
            ".chat-bubble",
            ".chat-msg-row",
            ".chat-msg-avatar",
            ".chat-input-wrap",
            ".thread-tab",
            ".tool-call-row",
            ".session-divider",
        ):
            assert selector not in css, f"{selector} should come from Chat CSS only"

    def test_dark_theme_is_forwarded(self) -> None:
        src = _read(WORKSPACE_MODULES / "app.js")
        assert 'setAttribute("data-theme", "dark")' in src
        assert 'removeAttribute("data-theme")' in src
