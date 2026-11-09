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
        assert "var(--ws-dock-width, clamp(380px, 38vw, 560px))" in css
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


@pytest.mark.unit
class TestPresenceMount:
    """The presence/bust-up panel moves into the stage while the dock is open."""

    def test_index_has_a_presence_slot_inside_the_stage(self) -> None:
        html = _read(WORKSPACE / "index.html")
        assert 'id="wsPresenceSlot"' in html
        stage_start = html.index('id="wsStage"')
        stage_end = html.index("</div>", html.index('id="wsPresenceSlot"'))
        assert stage_start < html.index('id="wsPresenceSlot"') < stage_end

    def test_dock_passes_a_presence_mount_to_the_pane_host(self) -> None:
        src = _read(WORKSPACE_MODULES / "chat-dock.js")
        assert "presenceMount" in src
        assert 'document.getElementById("wsPresenceSlot")' in src

    def test_pane_host_accepts_presence_mount_as_an_option(self) -> None:
        src = _read(STATIC / "pages" / "chat" / "pane-host.js")
        assert "options.presenceMount" in src
        assert "ctx.state.presenceMount" in src

    def test_presence_panel_uses_the_mount_and_embedded_visibility(self) -> None:
        src = _read(STATIC / "pages" / "chat" / "presence-panel.js")
        assert "ctx.state.presenceMount" in src
        assert "areaIsMessages" in src
        assert "state.embeddedVisible === false" in src

    def test_slot_hugs_the_timeline_and_is_hidden_away_from_the_office(self) -> None:
        css = _read(WORKSPACE / "style.css")
        assert ".ws-presence-slot" in css
        assert "left: 14px" in css
        assert "var(--ws-timeline-bar-height, 44px)" in css
        assert "pointer-events: none" in css
        assert ".ws-chat-dock[hidden] ~ .ws-stage .ws-presence-slot" in css
        assert ".ws-office-panel.hidden" in css


@pytest.mark.unit
class TestReplyWaitingBubble:
    """Unread replies become non-expiring office bubbles."""

    def test_dock_subscribes_to_presence_and_sets_reply_waiting(self) -> None:
        src = _read(WORKSPACE_MODULES / "chat-dock.js")
        assert "presence.subscribe" in src
        assert "presence.addUnread" in src
        assert "setReplyWaiting" in src
        assert "unread > 0" in src

    def test_office_renderer_exposes_optional_reply_waiting(self) -> None:
        src = _read(WORKSPACE_MODULES / "office-renderer.js")
        assert "export function setReplyWaiting" in src
        assert 'typeof _activeInstance?.setReplyWaiting === "function"' in src
        assert "_replyWaitingNames" in src

    def test_pixel_renderer_and_actor_implement_the_bubble(self) -> None:
        renderer = _read(WORKSPACE_MODULES / "renderers" / "office-pixel-renderer.js")
        actors = _read(WORKSPACE / "pixel" / "modules" / "actors.js")
        assert "function setReplyWaiting" in renderer
        assert "actor.replyWaiting" in renderer
        assert "this.replyWaiting" in actors
        assert 't("ws.reply_waiting")' in actors
        assert "this.replyWaiting || this.dynamicLabel()" in actors

    @pytest.mark.parametrize(
        ("lang", "value"),
        [("ja", "返事あり"), ("en", "Reply waiting"), ("ko", "답장 있음")],
    )
    def test_reply_waiting_is_translated(self, lang: str, value: str) -> None:
        i18n = _read(STATIC / "i18n" / f"{lang}.json")
        assert f'"ws.reply_waiting": "{value}"' in i18n

    def test_hidden_dock_keeps_a_completed_reply_unread(self) -> None:
        renderer = _read(STATIC / "pages" / "chat" / "chat-renderer.js")
        pane_host = _read(STATIC / "pages" / "chat" / "pane-host.js")
        assert "state.embeddedVisible !== false" in renderer
        assert "embedded ? Math.max(current, n) : n" in pane_host


@pytest.mark.unit
class TestDockResizer:
    """The conversation/stage boundary is draggable, keyboard-friendly and saved."""

    def test_index_has_the_resizer_between_dock_and_stage(self) -> None:
        html = _read(WORKSPACE / "index.html")
        assert 'id="wsDockResizer"' in html
        assert 'role="separator"' in html
        assert 'aria-orientation="vertical"' in html
        assert 'tabindex="0"' in html
        assert 'id="wsChatDock"' in html
        assert html.index('id="wsChatDock"') < html.index('id="wsDockResizer"') < html.index('id="wsStage"')
        assert "hidden" in html[html.index('id="wsDockResizer"') : html.index('id="wsStage"')]

    def test_resizer_module_handles_drag_keys_reset_and_storage(self) -> None:
        src = _read(WORKSPACE_MODULES / "chat-dock-resizer.js")
        assert "setPointerCapture" in src
        assert "releasePointerCapture" in src
        assert "pointermove" in src
        assert "ArrowLeft" in src and "ArrowRight" in src
        assert "KEYBOARD_STEP = 20" in src
        assert "dblclick" in src
        assert '"aw-workspace-dock-width"' in src
        assert "localStorage.setItem" in src
        assert "document.body.style.userSelect" in src
        assert "MIN_DOCK_WIDTH = 340" in src
        assert "MAX_DOCK_WIDTH = 760" in src
        assert "MIN_STAGE_WIDTH = 320" in src
        assert "window.innerWidth <= 768" in src
        assert "resizer.hidden = narrow || dock.hidden" in src

    def test_dock_resizer_is_initialized_and_refreshed(self) -> None:
        src = _read(WORKSPACE_MODULES / "chat-dock.js")
        assert "initChatDockResizer" in src
        assert 'document.getElementById("wsDockResizer")' in src
        assert "_resizer?.refresh()" in src

    def test_css_uses_the_width_variable_and_keeps_the_stage_clipped(self) -> None:
        css = _read(WORKSPACE / "style.css")
        assert "--ws-dock-width" in css
        assert ".ws-dock-resizer" in css
        assert ".ws-dock-resizer:active" in css
        assert "cursor: col-resize" in css
        assert "overflow: hidden" in css


@pytest.mark.unit
class TestMobileDockLayout:
    """Phones stack the stage over the dock and allow a full-screen conversation."""

    def test_mobile_stack_and_fullscreen_rules_exist(self) -> None:
        css = _read(WORKSPACE / "style.css")
        mobile = css[css.index("@media (max-width: 768px)") :]
        assert "flex-direction: column" in mobile
        assert "order: 1" in mobile
        assert "order: 2" in mobile
        assert "30dvh" in mobile
        assert "is-dock-full" in mobile
        assert ".ws-dock-handle" in mobile

    def test_dock_handle_toggles_fullscreen(self) -> None:
        src = _read(WORKSPACE_MODULES / "chat-dock.js")
        assert "wsDockHandle" in src
        assert "is-dock-full" in src
        assert "ws.dock_expand" in src

    def test_mobile_chat_column_keeps_the_input_at_the_bottom(self) -> None:
        css = _read(WORKSPACE / "style.css")
        mobile = css[css.index("@media (max-width: 768px)") :]
        assert ".ws-chat-dock .chat-page-main" in mobile
        assert "flex-direction: column" in mobile
        assert ".chat-messages-area" in mobile
        assert ".chat-input-form" in mobile

    def test_shared_mobile_stylesheet_does_not_override_the_dock(self) -> None:
        css = _read(STATIC / "styles" / "chat-mobile.css")
        assert ".ws-chat-dock" not in css
        assert ".ws-stage" not in css


@pytest.mark.unit
class TestMemoryPopoverBesideDock:
    """The embedded memory popover opens to the right of the whole dock."""

    def test_embedded_branch_uses_the_dock_right_edge(self) -> None:
        src = _read(STATIC / "pages" / "chat" / "memory-popover.js")
        assert "ctx.state.embedded" in src
        assert 'closest?.(".ws-chat-dock")' in src
        assert "dockRect.right + 14 + 480 <= width" in src
        assert "anchorRight" in src
        assert "dialog.style.left = (anchorRight + 14)" in src

    def test_non_embedded_anchor_logic_is_preserved(self) -> None:
        src = _read(STATIC / "pages" / "chat" / "memory-popover.js")
        assert "rect.right < width * 0.3" in src
        assert "Math.min(popWidth, 480)" in src
