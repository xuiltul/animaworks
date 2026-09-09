# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path

from rich.text import Text
from textual.actions import SkipAction
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.widget import Widget
from textual.widgets import Static

from cli.tui.client import AnimaWorksClient, AnimaWorksClientError
from cli.tui.commands import get_command, is_command, iter_commands
from cli.tui.keybindings import app_keymap, load_keybindings
from cli.tui.session import SessionInfo, new_session, save_session
from cli.tui.sse import SseEvent
from cli.tui.state import (
    AppState,
    PaletteItem,
    apply_ws_event,
    filter_palette,
    is_valid_thread_id,
    new_thread_id,
)
from cli.tui.widgets import (
    CallHumanOption,
    ChatInputContainer,
    ChatSubmitted,
    InteractionCard,
    Palette,
    Sidebar,
    StatusBar,
    ToolCard,
    Transcript,
    format_input_summary,
)
from cli.tui.widgets.response_status import ResponseStatus
from cli.tui.widgets.sidebar import AnimaChosen
from cli.tui.widgets.thinking import ThinkingBlock
from cli.tui.widgets.transcript import AssistantBlock, HumanTurn, SystemNote, strip_html_comments


def _system_label(msg: dict) -> str:
    """Name a system entry by where it came from: heartbeat, cron, a notification."""
    source = str(msg.get("source_key") or msg.get("type") or "").strip()
    label = f"system · {source}" if source else "system"
    subject = str(msg.get("subject") or "").strip()
    return f"{label} — {subject}" if subject else label


# WS event types that are fed into the shared sidebar state.
_WS_STATE_TYPES = {
    "anima.status",
    "anima.tool_activity",
    "anima.heartbeat",
    "anima.cron",
    "anima.bootstrap",
    "anima.proactive_message",
    "anima.notification",
    "anima.interaction",
    "board.post",
}


class AnimaChatApp(App):
    """The interactive terminal chat UI for talking to an anima."""

    # No colours are picked anywhere in the UI: backgrounds stay
    # transparent and text/borders use `ansi_default`, so everything is
    # drawn with the terminal's own background and foreground colours.
    CSS = """
    Screen {
        layout: vertical;
        background: transparent;
    }
    /* Scrollbars: the track stays transparent. The thumb has to differ
       from the background to be visible at all, so it uses the
       terminal's own grey (ansi_bright_black) rather than a theme hue. */
    Screen, Screen * {
        scrollbar-color: ansi_bright_black;
        scrollbar-color-hover: ansi_bright_black;
        scrollbar-color-active: ansi_bright_black;
        scrollbar-background: ansi_default;
        scrollbar-background-hover: ansi_default;
        scrollbar-background-active: ansi_default;
        scrollbar-corner-color: ansi_default;
    }
    #body {
        height: 1fr;
        layout: horizontal;
    }
    Sidebar {
        width: 32;
    }
    #right {
        width: 1fr;
        height: 1fr;
        layout: vertical;
    }
    #transcript {
        height: 1fr;
        border: none;
        background: transparent;
        padding: 0;
        scrollbar-size-vertical: 1;
    }
    #palette {
        height: auto;
        max-height: 15;
        display: none;
    }
    /* The theme maps every colour to ansi_default, so neutralise the
       OptionList's own highlight/hover tints; Palette.render_line paints
       the selected row in reverse video instead. */
    #palette > .option-list--option-highlighted,
    #palette > .option-list--option-hover {
        color: ansi_default;
        background: ansi_default;
    }
    #input-container {
        height: auto;
        background: transparent;
        border-top: solid ansi_default;
        border-bottom: solid ansi_default;
        padding-top: 0;
    }
    #response-status {
        display: none;
        height: 1;
        background: transparent;
        padding: 0 1;
    }
    /* The blank line above belongs to the container: padding it onto the
       label alone pushed `>` one row below the text being typed. */
    #input-container .input-prompt {
        padding: 0 0 0 1;
        text-style: bold;
    }
    ChatInput {
        height: auto;
        border: none;
        padding: 0 1;
        background: transparent;
    }
    #status {
        height: 1;
        dock: bottom;
        background: transparent;
        padding: 0 1;
    }
    /* The one place a colour is picked: the reader has to be able to tell
       their own turns from the anima's at a glance. Both are ANSI slots,
       so they still come from the terminal's palette, and blue/bright
       white is legible on light and dark schemes alike. */
    #transcript HumanTurn {
        margin-top: 1;
        padding: 0 1;
        background: ansi_blue;
        color: ansi_bright_white;
    }
    #transcript .human-label, #transcript .human-message {
        background: ansi_blue;
        color: ansi_bright_white;
    }
    #transcript .assistant-label {
        margin-top: 1;
        color: ansi_green;
    }
    #transcript SystemNote {
        margin-top: 1;
    }
    #transcript .system-message {
        width: 100%;
        padding: 0 1;
    }
    #transcript .assistant-text, #transcript .assistant-error, #transcript .human-message {
        width: 100%;
    }
    #transcript .assistant-text, #transcript .assistant-error {
        padding: 0 1;
    }
    #transcript .transient {
        height: auto;
        width: 100%;
        padding: 0 1;
    }
    """

    BINDINGS = [
        # Priority so it beats the input's own Enter; the action skips
        # itself when nothing is selected, and the key then submits.
        Binding(
            "enter",
            "copy_selection",
            "Copy selection",
            show=False,
            priority=True,
            id="copy_selection",
        ),
        # Also priority: Textual's own ``Screen._key_escape`` clears the
        # selection before a non-priority binding would see it, which made
        # Escape-to-deselect interrupt the answer as well.
        Binding("escape", "maybe_interrupt", "Interrupt", id="interrupt", priority=True),
        Binding("ctrl+c", "quit_or_confirm", "Quit"),
        Binding("ctrl+d", "quit_now", "Quit", id="quit", priority=True),
        Binding("ctrl+b", "toggle_sidebar", "Toggle sidebar", show=False, id="toggle_sidebar"),
        Binding("ctrl+t", "toggle_thinking", "Toggle thinking", show=False, id="toggle_thinking"),
        Binding("ctrl+l", "focus_input", "Focus input", show=False, id="focus_input"),
        Binding("pageup", "scroll_up", "Scroll up", show=False, id="scroll_up"),
        Binding("pagedown", "scroll_down", "Scroll down", show=False, id="scroll_down"),
    ]

    def __init__(
        self,
        client: AnimaWorksClient,
        anima_name: str,
        thread_id: str = "default",
        *,
        session: SessionInfo | None = None,
        session_dir: Path | None = None,
        no_reattach: bool = False,
        keymap: dict[str, str] | None = None,
    ) -> None:
        super().__init__()
        self.client = client
        self.anima_name = anima_name
        self.thread_id = thread_id

        self.busy = False
        self.current: AssistantBlock | None = None
        self.tool_cards: dict[str, ToolCard] = {}
        self.show_thinking = False
        self._last_ctrlc = 0.0

        self.state = AppState()
        self.state.current = anima_name
        self._skills: list[dict] = []
        self._pending_cards: dict[str, dict] = {}
        self._suppress_palette = False
        self._sidebar_open = True

        # Model picker state.
        self._models: list[dict] = []
        self._palette_mode: str | None = None
        self._model_palette_items: list[PaletteItem] = []

        # Phase 3: session + resume + keybindings.
        self.session_dir = session_dir
        if session is None:
            session = new_session(
                anima=anima_name,
                thread_id=thread_id,
                gateway_url=getattr(client, "base_url", "") or "http://localhost:18500",
                from_person=getattr(client, "from_person", "human") or "human",
            )
        self.session: SessionInfo = session
        self.chat_model: str = self.session.model or ""
        self.no_reattach = no_reattach
        self._keymap, self._key_warnings = (keymap, []) if keymap is not None else load_keybindings()
        self._key_warnings = list(self._key_warnings)

        # History lazy-loading cursor state.
        self._history_cursor: str | None = None
        self._history_end = False
        self._history_loading = False

        # Stream reconnect state.
        self._last_event_id: str | None = None
        self._last_response_id: str | None = None

        # Session save throttling.
        self._session_save_pending = False

    # ── Lifecycle ──────────────────────────────────────────
    def compose(self) -> ComposeResult:
        with Horizontal(id="body"):
            yield Sidebar(id="sidebar")
            with Vertical(id="right"):
                yield Transcript(id="transcript")
                yield Palette(id="palette")
                yield ResponseStatus(id="response-status")
                yield ChatInputContainer(id="input-container")
        yield StatusBar(self.anima_name, self.thread_id, id="status")

    def on_mount(self) -> None:
        self.title = f"AnimaWorks — {self.anima_name}"
        self._apply_theme()
        try:
            self._bindings.apply_keymap(app_keymap(self._keymap))
        except Exception:
            pass

        self.body = self.query_one("#body", Horizontal)
        self.sidebar = self.query_one("#sidebar", Sidebar)
        self.transcript = self.query_one("#transcript", Transcript)
        self.input_container = self.query_one("#input-container", ChatInputContainer)
        self.status_bar = self.query_one("#status", StatusBar)
        self.palette = self.query_one("#palette", Palette)
        self.response_status = self.query_one("#response-status", ResponseStatus)
        self.input_container.input.set_controller(self)
        # Show the model restored from the saved session, if any.
        self.status_bar.set_model(self.chat_model or None)

        # Default the sidebar closed on narrow terminals.
        width = getattr(self, "size", None)
        term_w = getattr(width, "width", None) if width else None
        if term_w is not None and term_w < 100:
            self._set_sidebar_open(False)

        self.call_after_refresh(self.focus_input)

        self.run_worker(self._bootstrap(), group="init", exit_on_error=False)

    def _apply_theme(self) -> None:
        """Use the terminal's own colours instead of a painted theme.

        The ``ansi-*`` themes map every colour variable to the terminal
        palette (background and foreground become ``ansi_default``), so
        the UI inherits whatever the user's terminal is set to. Override
        with ``ANIMAWORKS_TUI_THEME`` (any built-in Textual theme name).
        """
        wanted = os.environ.get("ANIMAWORKS_TUI_THEME") or "ansi-light"
        try:
            self.theme = wanted
        except Exception:
            self.theme = "ansi-light"

    def focus_input(self) -> None:
        if self.input_container.input.has_focus is not True:
            self.input_container.focus_input()

    def focus_sidebar(self) -> None:
        self.sidebar.animas.focus()

    def _refresh_default_model(self) -> None:
        """Tell the status bar which model the current anima falls back to."""
        row = self.state.animas.get(self.anima_name)
        self.status_bar.set_default_model(row.model if row else None)

    async def _bootstrap(self) -> None:
        try:
            animas = await self.client.list_animas()
        except AnimaWorksClientError as exc:
            self.status_bar.set_state(status="error", right_hint=f"connection error: {exc}")
            self.show_transient(f"Connection error: {exc}")
            return
        self.state.set_animas(animas)
        self.sidebar.update_state(self.state)
        self._refresh_default_model()
        names = {a.get("name") for a in animas}
        if self.anima_name not in names:
            self.status_bar.set_state(
                status="error",
                right_hint=f"Anima '{self.anima_name}' not found",
            )
            self.show_transient(f"Anima '{self.anima_name}' not found.")
            self.set_timer(0.5, lambda: self.exit(return_code=1))
            return
        entry = next(
            (a for a in animas if a.get("name") == self.anima_name),
            None,
        )
        busy = (entry or {}).get("busy")
        if isinstance(busy, dict) and busy.get("is_busy"):
            status = "busy"
        else:
            status = "idle"
        self.status_bar.set_state(status=status)
        self.run_worker(self._load_history_then_reattach(), group="init", exit_on_error=False)
        self.run_worker(self._load_skills(), group="init", exit_on_error=False)
        self.run_worker(self._ws_loop(), group="ws", exit_on_error=False)

    async def _load_history_then_reattach(self) -> None:
        # History first so a resumed in-flight response is appended after it.
        await self._load_history()
        await self._check_reattach()

    # ── History ───────────────────────────────────────────
    async def _load_history(self) -> None:
        history = await self._get_history()
        if history is None:
            return
        self._history_cursor = history.get("next_before")
        if not history.get("has_more"):
            self._history_end = True
        await self.render_history(history)
        # Open on the latest message, not the oldest of the first page.
        self.call_after_refresh(self.transcript.jump_to_end)
        # In the background: reattaching to an in-flight response must not
        # wait on however many pages this takes.
        self.run_worker(self._fill_viewport(), group="history", exit_on_error=False)

    async def _await_refresh(self) -> None:
        """Wait for the next screen refresh, so layout figures are current."""
        done = asyncio.Event()
        self.call_after_refresh(done.set)
        try:
            await asyncio.wait_for(done.wait(), timeout=2.0)
        except TimeoutError:
            pass

    async def _fill_viewport(self, pages: int = 8) -> None:
        """Pull older pages until the transcript can actually scroll.

        A page can render to less than a screenful — a thread whose recent
        history is all background traffic, say. Then there is no scrollbar
        and no way to reach the conversation behind it, because loading
        older history is triggered by scrolling to the top.
        """
        for _ in range(pages):
            await self._await_refresh()
            if self._history_end or self.transcript.max_scroll_y > 0:
                return
            cursor = self._history_cursor
            await self.load_older_history()
            if self._history_cursor == cursor:
                return

    async def _get_history(self, *, before: str | None = None) -> dict | None:
        try:
            return await self.client.get_history(
                self.anima_name,
                thread_id=self.thread_id,
                limit=50,
                before=before,
            )
        except AnimaWorksClientError:
            self.status_bar.set_state(right_hint="history unavailable")
            return None

    def _clear_history_cursor(self) -> None:
        self._history_cursor = None
        self._history_end = False
        self._history_loading = False

    async def render_history(self, history: dict) -> None:
        for session in history.get("sessions", []):
            for msg in session.get("messages", []):
                await self._render_history_msg(msg, prepend=False)
        self.current = None

    def _history_widget(self, msg: dict) -> Widget | None:
        """Build the transcript row for one stored message, if it has one."""
        content = msg.get("content")
        if not content:
            return None
        role = msg.get("role")
        if role == "human":
            return HumanTurn("You", str(content))
        if role == "assistant":
            block = self.transcript.new_assistant(self.anima_name)
            block.set_final(str(content))
            return block
        if role == "system":
            return SystemNote(_system_label(msg), str(content))
        return None

    async def _render_history_msg(self, msg: dict, *, prepend: bool) -> None:
        widget = self._history_widget(msg)
        if widget is not None:
            await self.transcript.mount(widget, before=0 if prepend else None)

    async def _show_beginning_marker(self) -> None:
        if any(
            isinstance(c, Static) and getattr(c, "classes", None) and "beginning" in c.classes
            for c in self.transcript.children
        ):
            return
        marker = Static("— beginning of history —", classes="beginning")
        await self.transcript.mount(marker, before=0)

    async def load_older_history(self) -> None:
        """Load and prepend older history; called when the user scrolls to the top."""
        if self._history_loading:
            return
        if self._history_end:
            await self._show_beginning_marker()
            return
        if not self._history_cursor:
            self._history_end = True
            await self._show_beginning_marker()
            return
        self._history_loading = True
        history = None
        try:
            history = await self._get_history(before=self._history_cursor)
        finally:
            self._history_loading = False
        if history is None:
            return
        self._history_cursor = history.get("next_before")
        if not history.get("has_more"):
            self._history_end = True
        await self.render_history_at_top(history)
        if self._history_end:
            await self._show_beginning_marker()

    async def render_history_at_top(self, history: dict) -> None:
        """Render older history above the current transcript, keeping scroll position."""
        prev_scroll = self.transcript.scroll_y
        old_height = self.transcript.virtual_size.height
        old_first = self.transcript.children[0] if self.transcript.children else None
        added = 0
        for session in history.get("sessions", []):
            for msg in session.get("messages", []):
                widget = self._history_widget(msg)
                if widget is None:
                    continue
                added += 1
                await self.transcript.mount(widget, before=old_first)
        if not added:
            return

        def _keep_view() -> None:
            # Once the new rows are laid out, shift the viewport by exactly the
            # height they added so the reader stays on the same line. This also
            # moves scroll_y off 0 so scrolling up again can load the next page.
            delta = self.transcript.virtual_size.height - old_height
            self.transcript.scroll_to(y=prev_scroll + max(delta, 0), animate=False, force=True)

        self.call_after_refresh(_keep_view)

    def load_more_history(self) -> None:
        """Slash-command handler: load 50 more (older) messages."""
        self.run_worker(self.load_older_history(), group="history", exit_on_error=False)

    async def reload_history(self, limit: int = 50) -> None:
        self._clear_history_cursor()
        history = await self._get_history()
        if history is None:
            return
        self.transcript.clear_all()
        self.current = None
        self._history_cursor = history.get("next_before")
        if not history.get("has_more"):
            self._history_end = True
        await self.render_history(history)

    def on_transcript_scrolled_to_top(self, _message) -> None:
        self.run_worker(self.load_older_history(), group="history", exit_on_error=False)

    # ── Skills ───────────────────────────────────────────
    async def _load_skills(self) -> None:
        list_skills = getattr(self.client, "list_skills", None)
        if list_skills is None:
            self._skills = []
            self._update_skill_count()
            return
        try:
            resp = await list_skills(self.anima_name, self.thread_id)
        except Exception:
            self._skills = []
            self._update_skill_count()
            return
        self._skills = resp.get("skills") or []
        self._update_skill_count()

    def _update_skill_count(self) -> None:
        count = sum(1 for s in self._skills if s.get("active"))
        self.status_bar.set_state(skill_count=count)

    # ── WebSocket ─────────────────────────────────────────
    async def _ws_loop(self) -> None:
        async for msg in self.client.ws_events():
            self.handle_ws(msg)

    def handle_ws(self, msg: dict) -> None:
        event_type = msg.get("type")
        data = msg.get("data") or {}
        if event_type == "_ws_status":
            self.status_bar.set_state(connected=bool(data.get("connected")))
            return

        if event_type in _WS_STATE_TYPES:
            eff = apply_ws_event(self.state, msg)
            self._apply_ws_effect(eff, data)
            self.sidebar.update_state(self.state)

        # Per-anima (self) status bar handling stays unchanged.
        if event_type == "anima.status" and data.get("name") == self.anima_name:
            status = data.get("status") or "idle"
            self.status_bar.set_state(status=status)
        elif event_type == "anima.tool_activity" and data.get("name") == self.anima_name:
            evt = data.get("event")
            kind = data.get("kind")
            tool_name = data.get("tool_name") or data.get("tool")
            if (evt == "tool_start" or kind == "tool_use") and self.busy:
                # Only while a response is in flight: a late tool_start after
                # ``done`` must not leave a stale tool name in the status bar.
                self.status_bar.set_state(active_tool=tool_name)
            elif evt == "tool_end" or kind == "tool_result":
                self.status_bar.set_state(active_tool=None)

    def _apply_ws_effect(self, eff, data: dict) -> None:
        for _title, _body in eff.toasts:
            self.notify(_body or _title, timeout=6)
        for pmsg in eff.proactive:
            name = pmsg.get("anima") or pmsg.get("name")
            if name == self.anima_name:
                self._thread_proactive(pmsg)
        for card_data in eff.cards:
            self._thread_interaction_card(card_data)

    def _thread_proactive(self, data: dict) -> None:
        subject = data.get("subject") or ""
        body = strip_html_comments(data.get("body") or "")
        text = (f"**{subject}**\n{body}" if subject else body).replace("**", "")
        block = self.transcript.new_assistant(self.anima_name)
        block.set_final(text)
        self.run_worker(self.transcript.mount_assistant(block), group="ui", exit_on_error=False)

    def _thread_interaction_card(self, data: dict) -> None:
        name = data.get("anima") or data.get("name") or self.anima_name
        card = InteractionCard(name, data)
        self._pending_cards[card.callback_id] = {"anima": name, "options": card.options}
        self.run_worker(self.transcript.mount(card), group="ui", exit_on_error=False)

    # ── Input / palette (called by ChatInput) ────────────
    def palette_is_open(self) -> bool:
        return self.palette.is_open

    def on_input_text_changed(self, text: str) -> None:
        if self._suppress_palette:
            self._suppress_palette = False
            return
        if self._palette_mode == "model":
            query = text if text.startswith("/") else "/" + text
            items = filter_palette(self._model_palette_items, query)
            if items:
                self.palette.set_items(items)
            else:
                self.palette.close()
            return
        if text.startswith("/") and not self.busy:
            items = filter_palette(self._palette_items(), text)
            if items:
                self.palette.set_items(items)
            else:
                self.palette.close()
        else:
            self.palette.close()

    def _clear_input(self) -> None:
        self._suppress_palette = True
        self._palette_mode = None
        self.input_container.input.text = ""
        self.palette.close()

    def _set_input_text(self, text: str) -> None:
        """Replace the input buffer without reopening the palette, leaving
        the caret at the end (``TextArea.text`` resets it to the start)."""
        self._suppress_palette = True
        widget = self.input_container.input
        widget.text = text
        widget.move_cursor(widget.document.end)

    def palette_move(self, direction: str) -> None:
        if self.palette.is_open:
            self.palette.move(direction)

    def palette_close(self) -> None:
        self._palette_mode = None
        self.palette.close()

    def palette_complete(self) -> None:
        """Tab: complete with the selected candidate only (never execute)."""
        if not self.palette.is_open:
            return
        item = self.palette.selected()
        self.palette.close()
        if item is None:
            return
        self._set_input_text(item.value)
        self.focus_input()

    def palette_confirm(self) -> None:
        """Enter: run a fully-typed command, otherwise complete (running where
        the candidate needs no further input).

        Mirrors Claude Code: an input that is exactly a registered command
        name runs immediately; a partial input completes with the selected
        candidate (running it if it takes no arguments, otherwise keeping the
        user in the input to continue).
        """
        if not self.palette.is_open:
            return
        raw = self.input_container.input.text
        if self._matches_exact_command(raw):
            self._clear_input()
            self.run_worker(
                self._handle_message(raw.strip()),
                group="submit",
                exit_on_error=False,
            )
            return
        item = self.palette.selected()
        self.palette.close()
        self._palette_mode = None
        if item is None:
            return
        if item.on_confirm is not None:
            self._clear_input()
            item.on_confirm(self)
            return
        self._set_input_text(item.value)
        self.palette.close()
        if not item.takes_args:
            self._clear_input()
            self.run_worker(
                self._handle_message(item.value.strip()),
                group="submit",
                exit_on_error=False,
            )
        else:
            self.focus_input()

    def _matches_exact_command(self, text: str) -> bool:
        """True when ``text`` is exactly ``/name`` (no arguments) and ``name``
        is a registered command or an existing skill.

        Built-in commands win over a skill with the same name. Trailing
        whitespace is ignored.
        """
        stripped = text.strip()
        if not stripped.startswith("/"):
            return False
        name = stripped[1:].strip()
        if not name or " " in name:
            return False
        if get_command(name) is not None:
            return True
        return self._find_skill_ref(name) is not None

    def _palette_label(self, head: str, tail: str) -> Text:
        """One-line palette row: bold ``head`` + dim ``tail``, cut to width.

        ``OptionList`` wraps long prompts regardless of Rich's ``no_wrap``,
        so the text is truncated here to the palette's usable width (the
        input column minus the border, padding and scrollbar). Newlines inside skill
        descriptions are collapsed so every candidate takes exactly one row.
        """
        tail = " ".join(tail.split())
        label = Text.assemble(Text(head, style="bold"), Text(f"  {tail}", style="dim"), no_wrap=True)
        width = self.input_container.size.width if self.input_container.size.width > 0 else 80
        label.truncate(max(width - 8, 16), overflow="ellipsis")
        return label

    def _palette_items(self) -> list[PaletteItem]:
        items: list[PaletteItem] = []
        # Skills first (active ones first, then by name), so ``/`` shows
        # the anima's skills as the main content. Built-ins come after.
        ordered_skills = sorted(
            self._skills,
            key=lambda s: (not bool(s.get("active")), (s.get("name") or "").lower()),
        )
        for skill in ordered_skills:
            name = skill.get("name") or ""
            if not name:
                continue
            desc = skill.get("description") or ""
            marks = []
            if skill.get("active"):
                marks.append("● ")
            if skill.get("is_common"):
                marks.append("(common) ")
            if skill.get("is_procedure"):
                marks.append("(procedure) ")
            label = self._palette_label(f"/{name}", f"{''.join(marks)}{desc}")
            low = name.lower()
            items.append(
                PaletteItem(
                    value=f"/{name} ",
                    label=label,
                    takes_args=True,
                    search=f"/{low} {low} /skill {low}",
                )
            )
        for cmd in iter_commands():
            label = self._palette_label(f"/{cmd.name}", cmd.description)
            if cmd.takes_args:
                items.append(
                    PaletteItem(
                        value=f"/{cmd.name} ",
                        label=label,
                        takes_args=True,
                        search=f"/{cmd.name}",
                    )
                )
            else:
                items.append(
                    PaletteItem(
                        value=f"/{cmd.name}",
                        label=label,
                        takes_args=False,
                        search=f"/{cmd.name}",
                        on_confirm=_make_runner(cmd.handler),
                    )
                )
        return items

    # ── Input ────────────────────────────────────────────
    def on_chat_submitted(self, message: ChatSubmitted) -> None:
        self.run_worker(
            self._handle_message(message.text.strip()),
            group="submit",
            exit_on_error=False,
        )

    async def _handle_message(self, text: str) -> None:
        if not text:
            return
        if is_command(text):
            await self.handle_command(text)
        elif self.busy:
            self.status_bar.set_state(right_hint="Busy — wait for the response to finish")
        else:
            self._palette_mode = None
            await self.send_message(text)

    # ── Sending / streaming ──────────────────────────────
    async def send_message(self, text: str) -> None:
        self.busy = True
        self.response_status.start()
        self.status_bar.set_state(status="thinking", right_hint="responding…")
        await self.transcript.add_human("You", text)
        self.current = self.transcript.new_assistant(self.anima_name)
        await self.transcript.mount_assistant(self.current)
        self.tool_cards = {}
        self._last_response_id = None
        self._last_event_id = None
        self.session.in_flight = True
        self.session.last_response_id = None
        self.session.last_event_id = None
        self.run_worker(
            self._chat_worker(text),
            group="chat",
            exclusive=True,
            exit_on_error=False,
        )

    def _handle_stream_event_state(self, sse: SseEvent) -> bool:
        """Track resume state from an event.

        Returns True when an ``error`` event with ``STREAM_NOT_FOUND`` is
        seen (the caller should stop reconnecting).
        """
        if sse.event == "stream_start":
            rid = sse.data.get("response_id")
            if rid:
                self._last_response_id = rid
                self.session.last_response_id = rid
                self.session.in_flight = True
                self._schedule_session_save(force=True)
        if sse.id:
            self._last_event_id = sse.id
            self.session.last_event_id = sse.id
            self._schedule_session_save()
        return sse.event == "error" and sse.data.get("code") == "STREAM_NOT_FOUND"

    async def _chat_worker(
        self,
        text: str,
        *,
        resume: str | None = None,
        last_event_id: str | None = None,
    ) -> None:
        delay = 1.0
        attempts = 0
        while True:
            try:
                async for sse in self.client.chat_stream(
                    self.anima_name,
                    text,
                    thread_id=self.thread_id,
                    resume=resume,
                    last_event_id=last_event_id,
                    model=self.chat_model or None,
                ):
                    self._handle_stream_event_state(sse)
                    await self.handle_sse(sse)
                break
            except AnimaWorksClientError as exc:
                if self._last_response_id is None:
                    await self._show_error(str(exc))
                    break
                if attempts >= 3:
                    self.show_transient("stream lost")
                    break
                attempts += 1
                await asyncio.sleep(delay)
                delay = min(delay * 2, 4.0)
                resume = self._last_response_id
                last_event_id = self._last_event_id
        self._stop_thinking_blink()
        self.response_status.stop()
        self.busy = False
        self.session.in_flight = False
        self._schedule_session_save(force=True)
        self.status_bar.set_state(active_tool=None)
        self.call_after_refresh(self.focus_input)

    def _schedule_session_save(self, *, force: bool = False) -> None:
        if force:
            self._write_session()
            self._session_save_pending = False
            return
        if not self._session_save_pending:
            self._session_save_pending = True
            self.set_timer(1.0, self._flush_session_save)

    def _flush_session_save(self) -> None:
        if not self._session_save_pending:
            return
        self._session_save_pending = False
        self._write_session()

    def _write_session(self) -> None:
        try:
            save_session(self.session, base_dir=self.session_dir)
        except Exception:
            pass

    async def handle_sse(self, sse: SseEvent) -> None:
        name = sse.event
        data = sse.data
        if name == "stream_start":
            self.status_bar.set_state(status="thinking")
        elif name == "text_delta":
            if self.current is not None:
                self.current.append_text(data.get("text", ""))
        elif name == "thinking_start":
            if self.current is not None:
                block = self.current.ensure_thinking()
                block.set_visible(self.show_thinking)
                block.start()
        elif name == "thinking_delta":
            if self.current is not None:
                block = self.current.ensure_thinking()
                block.set_visible(self.show_thinking)
                block.start()
                block.add_delta(data.get("text", ""))
        elif name == "thinking_end":
            self._stop_thinking_blink()
        elif name == "tool_start":
            tool_id = data.get("tool_id", "")
            card = ToolCard(data.get("tool_name", "tool"), tool_id)
            if data.get("input_summary"):
                card.set_preview(format_input_summary(str(data.get("input_summary"))))
            self.tool_cards[tool_id] = card
            if self.current is not None:
                await self.current.add_tool(card)
        elif name == "tool_detail":
            card = self.tool_cards.get(data.get("tool_id", ""))
            if card is not None:
                card.add_detail(str(data.get("detail", "")))
        elif name == "tool_end":
            card = self.tool_cards.get(data.get("tool_id", ""))
            if card is not None:
                card.finish(
                    result_summary=data.get("result_summary"),
                    is_error=bool(data.get("is_error")),
                    input_summary=data.get("input_summary"),
                )
            if data.get("tool_name") == self.status_bar.active_tool:
                self.status_bar.set_state(active_tool=None)
        elif name == "chain_start":
            pass
        elif name == "done":
            summary = data.get("summary") or ""
            self._stop_thinking_blink()
            self.response_status.stop()
            if self.current is not None:
                self.current.set_final(summary)
            self.busy = False
            self.session.in_flight = False
            self._schedule_session_save(force=True)
            self.status_bar.set_state(status="idle", active_tool=None, right_hint="Esc: interrupt")
        elif name == "error":
            self._stop_thinking_blink()
            self.response_status.stop()
            await self._show_error(data.get("message", "Stream error"))
            self.busy = False
            self.session.in_flight = False
            self._schedule_session_save(force=True)
            self.status_bar.set_state(status="error", right_hint="Esc: interrupt")
        elif name == "bootstrap":
            self.show_transient(f"bootstrap: {data.get('status')} {data.get('message') or ''}")
        elif name == "context_update":
            ratio = data.get("context_usage_ratio")
            if ratio is not None:
                self.status_bar.set_context_usage(
                    ratio,
                    input_tokens=data.get("input_tokens"),
                    context_window=data.get("context_window"),
                )
        elif name == "heartbeat_relay":
            pass

    def _stop_thinking_blink(self) -> None:
        if self.current is not None and self.current.thinking is not None:
            self.current.thinking.stop()

    async def _show_error(self, message: str) -> None:
        if self.current is not None:
            await self.current.add_error(message)

    # ── Slash commands ───────────────────────────────────
    async def handle_command(self, text: str) -> None:
        self._palette_mode = None
        parts = text.split()
        name = parts[0][1:]
        args = parts[1:]
        cmd = get_command(name)
        if cmd is not None:
            cmd.handler(args, self)
            self.call_after_refresh(self.focus_input)
            return
        # A skill can be invoked directly as ``/<skill name>``.
        if self._find_skill_ref(name) is not None:
            self.run_worker(
                self._invoke_skill_worker(name, args),
                group="submit",
                exit_on_error=False,
            )
            return
        self.show_transient(f"Unknown command: /{name}   (try /help)")

    def show_transient(self, text: str) -> None:
        self.run_worker(self._show_transient_async(text), group="ui", exit_on_error=False)

    async def _show_transient_async(self, text: str) -> None:
        widget = Static(Text(str(text), style="italic dim"), classes="transient")
        await self.transcript.mount(widget)
        self.transcript.jump_to_end()

    def clear_transcript(self) -> None:
        self.transcript.clear_all()
        self.current = None
        self.tool_cards = {}

    def toggle_thinking(self) -> None:
        self.show_thinking = not self.show_thinking
        self.session.show_thinking = self.show_thinking
        self._write_session()
        for block in self.query(ThinkingBlock):
            block.set_visible(self.show_thinking)
        self.show_transient(f"Thinking display: {'on' if self.show_thinking else 'off'}")

    def request_interrupt(self) -> None:
        self.action_maybe_interrupt()

    # ── Model selection (called by commands.py) ────────
    def choose_model(self, args: list[str]) -> None:
        if not args:
            self.run_worker(
                self._model_palette_worker(),
                group="model",
                exit_on_error=False,
            )
            return
        raw = " ".join(args)
        if raw in ("default", "off", "-"):
            self.set_chat_model("")
            return
        resolved = None
        for m in self._models:
            if m.get("id") == raw:
                resolved = m.get("id")
                break
        if resolved is None:
            for m in self._models:
                if (m.get("model") or "") == raw:
                    resolved = m.get("id")
                    break
        if resolved is None:
            low = raw.lower()
            for m in self._models:
                if low in (m.get("label") or "").lower():
                    resolved = m.get("id")
                    break
        self.set_chat_model(resolved or raw)

    async def _model_palette_worker(self) -> None:
        self.show_transient("Loading models…")
        try:
            models = await self.client.list_available_models()
        except (AnimaWorksClientError, AttributeError) as exc:
            self.show_transient(f"Failed to load models: {exc}")
            self._palette_mode = None
            return
        if not models:
            self.show_transient("No models available.")
            self._palette_mode = None
            return
        self._models = models
        items: list[PaletteItem] = []
        items.append(
            PaletteItem(
                value="/model default",
                label=self._palette_label(
                    "(anima default)",
                    "use the anima's configured model",
                ),
                takes_args=False,
                search="default anima",
                on_confirm=lambda app: app.set_chat_model(""),
            )
        )
        selected = self.chat_model
        for m in models:
            mid = m.get("id") or ""
            group = m.get("group") or ""
            label = m.get("label") or mid
            note = m.get("note") or ""
            mark = "● " if mid == selected else ""
            items.append(
                PaletteItem(
                    value=mid,
                    label=self._palette_label(f"{mark}{group}  {label}", note),
                    takes_args=False,
                    search=f"{mid} {m.get('model') or ''} {label} {group}".lower(),
                    on_confirm=lambda app, mid=mid: app.set_chat_model(mid),
                )
            )
        self._palette_mode = "model"
        self._model_palette_items = items
        self.palette.set_items(items)
        self.focus_input()

    def set_chat_model(self, model_id: str) -> None:
        self.chat_model = model_id
        self.session.model = model_id
        self._write_session()
        self._palette_mode = None
        if self.palette.is_open:
            self.palette.close()
        self.status_bar.set_model(model_id or None)
        if model_id:
            self.show_transient(f"Model: {model_id}")
        else:
            self.show_transient("Model: anima default")

    # ── Anima switching / sidebar ───────────────────────
    def switch_anima(self, name: str) -> None:
        self.run_worker(
            self._switch_anima_worker(name),
            group="switch",
            exclusive=True,
            exit_on_error=False,
        )

    async def _switch_anima_worker(self, name: str) -> None:
        if name == self.anima_name:
            return
        if self.busy:
            self.show_transient("Finish or Esc-interrupt the current response before switching.")
            return
        if name not in self.state.animas:
            self.show_transient(f"Unknown anima: {name}")
            return
        self.anima_name = name
        self.state.current = name
        self.state.clear_unread(name)
        self.title = f"AnimaWorks — {name}"
        self.status_bar.set_anima(name, self.thread_id)
        self.busy = False
        self.clear_transcript()
        self.sidebar.update_state(self.state)
        await self.reload_history(limit=50)
        self.show_transient(f"-- switched to {name} --")
        if len(self.transcript.children) == 0:
            self.show_transient("no conversation history yet")
        await self._load_skills()
        if name not in self.session.recent_animas:
            self.session.recent_animas.append(name)
        self.session.anima = name
        # Model validity is anima-specific: reset quietly (no transient) so
        # the previous anima's model isn't carried over unchanged.
        self.chat_model = ""
        self.session.model = ""
        self._palette_mode = None
        self.palette.close()
        self.status_bar.set_model(None)
        self._refresh_default_model()
        self._write_session()

    # ── Thread switching ────────────────────────────────
    def switch_thread(self, thread_id: str) -> None:
        self.run_worker(
            self._switch_thread_worker(thread_id),
            group="switch",
            exclusive=True,
            exit_on_error=False,
        )

    async def _switch_thread_worker(self, thread_id: str, announce: str | None = None) -> None:
        if self.busy:
            self.show_transient("Finish or Esc-interrupt the current response before switching threads.")
            return
        if not is_valid_thread_id(thread_id):
            self.show_transient(f"Invalid thread id: {thread_id}")
            return
        if thread_id == self.thread_id:
            self.show_transient(f"Already on thread {thread_id}")
            return
        self.thread_id = thread_id
        self.session.thread_id = thread_id
        self.status_bar.set_anima(self.anima_name, thread_id)
        self.clear_transcript()
        await self.reload_history(limit=50)
        self.show_transient(announce if announce is not None else f"-- thread {thread_id} --")
        if len(self.transcript.children) == 0:
            self.show_transient("no conversation history yet")
        await self._load_skills()
        # Reset stream-resume state: it is per-thread.
        self.session.last_response_id = None
        self.session.last_event_id = None
        self.session.in_flight = False
        self._last_response_id = None
        self._last_event_id = None
        self._write_session()

    def new_thread(self) -> None:
        tid = new_thread_id()
        self.run_worker(
            self._switch_thread_worker(tid, announce=f"-- new thread {tid} --"),
            group="switch",
            exclusive=True,
            exit_on_error=False,
        )

    def show_threads(self) -> None:
        self.run_worker(
            self._show_threads_worker(),
            group="threads",
            exit_on_error=False,
        )

    async def _show_threads_worker(self) -> None:
        list_threads = getattr(self.client, "list_threads", None)
        if list_threads is None:
            self.show_transient("Thread listing not supported by this client.")
            return
        try:
            data = await list_threads(self.anima_name)
        except AnimaWorksClientError as exc:
            self.show_transient(f"Could not list threads: {exc}")
            return

        def _fmt(ts: str | None) -> str:
            if not ts:
                return ""
            return (ts or "")[:16].replace("T", " ")

        lines = [f"Threads for {self.anima_name}:"]
        # The default thread always comes first (from active_conversation).
        active = (data or {}).get("active_conversation") or {}
        default_turns = int((active or {}).get("total_turn_count") or 0)
        default_ts = _fmt(active.get("last_timestamp") or "")
        if self.thread_id == "default":
            lines.append(f"* default {default_turns:>3} turns   last {default_ts}")
        else:
            lines.append(f"  default {default_turns:>3} turns   last {default_ts}")
        threads = sorted(
            (data or {}).get("threads") or [],
            key=lambda t: t.get("last_timestamp") or "",
            reverse=True,
        )
        current = self.thread_id
        seen_current = current == "default"
        for t in threads:
            tid = t.get("thread_id") or "?"
            if tid == current:
                seen_current = True
            turns = int(t.get("total_turn_count") or t.get("turn_count") or 0)
            ts = _fmt(t.get("last_timestamp") or "")
            mark = "*" if tid == current else " "
            line = f"{mark} {tid:<8} {turns:>3} turns"
            if ts:
                line += f"   last {ts}"
            lines.append(line)
        # The current thread may not exist on the server yet (a brand-new id).
        if not seen_current:
            lines.append(f"* {current:<8}   0 turns")
        lines.append("(/thread <id> to switch, /thread to start a new one)")
        self.show_transient("\n".join(lines))

    def _set_sidebar_open(self, open_state: bool) -> None:
        self._sidebar_open = open_state
        self.sidebar.display = "block" if open_state else "none"

    def toggle_sidebar(self) -> None:
        self._set_sidebar_open(not self._sidebar_open)
        self.session.sidebar_open = self._sidebar_open
        self._write_session()

    def on_anima_chosen(self, message: AnimaChosen) -> None:
        self.switch_anima(message.name)

    def action_toggle_sidebar(self) -> None:
        self.toggle_sidebar()

    # ── Reattach (Phase 3) ──────────────────────────────
    async def _check_reattach(self) -> None:
        if self.no_reattach:
            return
        active = None
        try:
            active = await self.client.get_active_stream(
                self.anima_name,
                thread_id=self.thread_id,
            )
        except AnimaWorksClientError:
            return
        action = decide_reattach(active, self.session.in_flight)
        if action == "reattach":
            await self._render_reattach(active)
        elif action == "render_final":
            await self._render_final(active)
            self.session.in_flight = False
            self._write_session()
        elif action == "lost":
            self.show_transient("The previous response could not be resumed. Use /history to review the conversation.")

    async def _render_reattach(self, active: dict) -> None:
        if self.current is not None and self.current._body:
            # Already rendering part of this response; only resume the tail.
            pass
        else:
            self.current = self.transcript.new_assistant(self.anima_name)
            full = active.get("full_text") or ""
            self.current.set_final(full)
            await self.transcript.mount_assistant(self.current)
            for tool in active.get("tool_history") or []:
                try:
                    card = ToolCard(tool.get("tool_name", "tool"), tool.get("tool_id", ""))
                    preview = tool.get("detail") or tool.get("input_summary")
                    if preview:
                        card.set_preview(format_input_summary(str(preview)))
                    if tool.get("result_summary") or tool.get("completed"):
                        card.finish(
                            result_summary=tool.get("result_summary"),
                            is_error=bool(tool.get("is_error")),
                        )
                    self.tool_cards[tool.get("tool_id", "")] = card
                    await self.current.add_tool(card)
                except Exception:
                    continue
        self.busy = True
        self.session.in_flight = True
        self.response_status.start()
        self.status_bar.set_state(status="streaming", right_hint="resuming stream…")
        self.run_worker(
            self._chat_worker(
                "",
                resume=active.get("response_id"),
                last_event_id=active.get("last_event_id"),
            ),
            group="chat",
            exclusive=True,
            exit_on_error=False,
        )

    async def _render_final(self, active: dict) -> None:
        self.current = self.transcript.new_assistant(self.anima_name)
        self.current.set_final(active.get("full_text") or "")
        await self.transcript.mount_assistant(self.current)

    # ── Command implementations (called by commands.py) ─
    def show_sessions(self) -> None:
        from cli.tui.session import list_sessions

        infos = list_sessions(base_dir=self.session_dir)
        if not infos:
            self.show_transient("No saved sessions.")
            return
        lines = [f"Sessions ({self.session_dir or 'default'}):"]
        for s in infos:
            ts = (s.updated_at or "")[:16].replace("T", " ")
            in_flight = " *" if s.in_flight else ""
            lines.append(f"  {s.session_id}  {ts}  {s.anima}/{s.thread_id}{in_flight}")
        self.show_transient("\n".join(lines))

    def show_keys(self) -> None:
        lines = ["Key bindings:"]
        for key, value in self._keymap.items():
            lines.append(f"  {key:<16} {value}")
        if self._key_warnings:
            for w in self._key_warnings:
                lines.append(f"  (warn) {w}")
        self.show_transient("\n".join(lines))

    def compact_session(self) -> None:
        """Manually compact the current thread's context (``/compact``)."""
        if self.busy:
            self.show_transient("Wait for the response to finish before /compact")
            return
        compact = getattr(self.client, "compact_session", None)
        if compact is None:
            self.show_transient("Compaction not supported by this client.")
            return
        self.run_worker(
            self._compact_session_worker(compact),
            group="compact",
            exit_on_error=False,
        )

    async def _compact_session_worker(self, compact) -> None:
        self.show_transient("Compacting…")
        try:
            result = await compact(self.anima_name, self.thread_id)
        except AnimaWorksClientError as exc:
            self.show_transient(f"Compaction failed: {exc}")
            return
        status = (result or {}).get("status")
        mode = (result or {}).get("mode") or "?"
        if status == "ok":
            self.show_transient(f"Context compacted (mode {mode}). The next message starts a fresh session.")
        elif status == "skipped":
            self.show_transient("Compaction skipped: the thread is busy")
        else:
            self.show_transient("Compaction failed: unexpected response")

    def show_animas(self) -> None:
        lines = ["Animas:"]
        for name, row in sorted(self.state.animas.items()):
            mark = "●" if row.busy or row.status in ("busy", "thinking", "streaming") else "○"
            tool = f"  {row.active_tool}" if row.active_tool else ""
            unread = f"  ({row.unread})" if row.unread else ""
            lines.append(f"{mark} {name}  {row.status}{tool}{unread}")
        self.show_transient("\n".join(lines))

    def show_skills(self) -> None:
        self.run_worker(self._show_skills_worker(), group="skills", exit_on_error=False)

    async def _show_skills_worker(self) -> None:
        await self.reload_skills_catalog()
        lines = [f"Skills for {self.anima_name} (thread {self.thread_id}):"]
        for skill in self._skills:
            name = skill.get("name") or skill.get("ref") or "?"
            active = "●" if skill.get("active") else "○"
            marks = []
            if skill.get("is_common"):
                marks.append("common")
            if skill.get("is_procedure"):
                marks.append("procedure")
            mark_s = f" ({', '.join(marks)})" if marks else ""
            desc = (skill.get("description") or "")[:60]
            lines.append(f"{active} {name}{mark_s}: {desc}")
        if not self._skills:
            lines.append("(no skills listed)")
        self.show_transient("\n".join(lines))

    async def reload_skills_catalog(self) -> None:
        """Re-fetch the current anima's skill catalog from the server."""
        list_skills = getattr(self.client, "list_skills", None)
        if list_skills is None:
            return
        try:
            resp = await list_skills(self.anima_name, self.thread_id)
        except Exception:
            return
        self._skills = resp.get("skills") or []
        self._update_skill_count()

    def activate_skill(self, args: list[str]) -> None:
        self.run_worker(
            self._activate_skill_worker(args),
            group="skill",
            exit_on_error=False,
        )

    async def _activate_skill_worker(self, args: list[str]) -> bool:
        """Apply a skill activation/deactivation.

        Returns ``True`` when the requested state was reached (skill is now
        active, or was already active), so callers can proceed to send a
        follow-up message. Returns ``False`` on usage error, unknown
        skill, I/O failure, or when the activation was rejected.
        """
        confirm = "--confirm" in args
        off = "--off" in args
        wanted = next((a for a in args if not a.startswith("--")), "")
        if not wanted:
            self.show_transient("usage: /skill <name> [--confirm] [--off]")
            return False
        ref = self._find_skill_ref(wanted)
        if ref is None:
            self.show_transient(f"Unknown skill: {wanted}")
            return False
        try:
            active = await self.client.get_active_skills(self.anima_name, thread_id=self.thread_id)
            current_refs = [item.get("ref") for item in active.get("accepted", [])]
        except (AnimaWorksClientError, AttributeError) as exc:
            self.show_transient(f"Could not read active skills: {exc}")
            return False
        new_refs = list(current_refs)
        if off:
            if ref in new_refs:
                new_refs.remove(ref)
            else:
                self.show_transient(f"Skill not active: {wanted}")
                return False
        else:
            if ref in new_refs:
                self.show_transient(f"Skill already active: {wanted}")
                return True
            new_refs.append(ref)
        try:
            result = await self.client.set_active_skills(self.anima_name, self.thread_id, new_refs, confirm)
        except (AnimaWorksClientError, AttributeError) as exc:
            self.show_transient(f"Activation failed: {exc}")
            return False
        self._render_skill_result(result, off)
        await self._load_skills()
        return not bool(result.get("rejections"))

    async def _invoke_skill_worker(self, name: str, args: list[str]) -> None:
        """Handle ``/<skill name>`` direct invocation with optional message.

        ``--off`` deactivates the skill. Otherwise the skill is activated
        (a no-op when already active) and then any non-flag words are sent
        to the anima as a follow-up message.
        """
        if "--off" in args:
            await self._activate_skill_worker([name, "--off"])
            return
        flags = [a for a in args if a.startswith("--")]
        rest = " ".join(a for a in args if not a.startswith("--")).strip()
        ok = await self._activate_skill_worker([name] + flags)
        if not ok:
            return
        if rest:
            if self.busy:
                self.status_bar.set_state(right_hint="Busy — wait for the response to finish")
            else:
                await self.send_message(rest)

    def _find_skill_ref(self, name: str) -> str | None:
        for skill in self._skills:
            if skill.get("name") == name or skill.get("ref") == name:
                return skill.get("ref")
        return None

    def _render_skill_result(self, result: dict, off: bool) -> None:
        accepted = result.get("accepted") or []
        rejections = result.get("rejections") or []
        warnings = result.get("warnings") or []
        lines: list[str] = []
        if accepted:
            names = ", ".join(i.get("name") or i.get("ref") or "?" for i in accepted)
            lines.append(f"Accepted: {names}")
        for rej in rejections:
            reason = rej.get("reason") or ""
            hint = " (add --confirm to override)" if "confirm" in reason.lower() else ""
            lines.append(f"Rejected: {rej.get('ref')}: {reason}{hint}")
        for warn in warnings:
            lines.append(f"Warning: {warn.get('name')}: {warn.get('reason')}")
        if not lines:
            lines = ["Skill request processed (no changes)."]
        self.show_transient("\n".join(lines))

    def show_board(self, args: list[str]) -> None:
        self.run_worker(self._board_worker(args), group="board", exit_on_error=False)

    async def _board_worker(self, args: list[str]) -> None:
        if not args:
            try:
                channels = await self.client.list_channels()
            except (AnimaWorksClientError, AttributeError) as exc:
                self.show_transient(f"Could not list channels: {exc}")
                return
            if not channels:
                self.show_transient("No channels.")
                return
            lines = ["Channels:"]
            for ch in channels:
                lines.append(f"  #{ch.get('name')}  ({ch.get('message_count', 0)} messages)")
            self.show_transient("\n".join(lines))
            return
        channel = args[0]
        limit = 20
        if len(args) > 1:
            try:
                limit = int(args[1])
            except ValueError:
                limit = 20
        try:
            resp = await self.client.read_channel(channel, limit=limit)
        except (AnimaWorksClientError, AttributeError) as exc:
            self.show_transient(f"Could not read #{channel}: {exc}")
            return
        messages = resp.get("messages") or []
        if not messages:
            self.show_transient(f"#{channel}: no messages.")
            return
        lines = [f"#{channel} (last {len(messages)}):"]
        for msg in reversed(messages):
            frm = msg.get("from") or "?"
            ts = (msg.get("ts") or "")[11:16]
            text = (msg.get("text") or "")[:120]
            lines.append(f"  {ts} {frm}: {text}")
        self.show_transient("\n".join(lines))

    def post_to_channel(self, args: list[str]) -> None:
        if len(args) < 2:
            self.show_transient("usage: /post <channel> <text>")
            return
        channel = args[0]
        text = " ".join(args[1:])
        self.run_worker(self._post_worker(channel, text), group="post", exit_on_error=False)

    async def _post_worker(self, channel: str, text: str) -> None:
        try:
            await self.client.post_channel(channel, text)
        except (AnimaWorksClientError, AttributeError) as exc:
            self.show_transient(f"Post failed: {exc}")
            return
        self.show_transient(f"Posted to #{channel}.")

    def show_tasks(self, args: list[str]) -> None:
        assignee = args[0] if args else self.anima_name
        self.run_worker(self._tasks_worker(assignee), group="tasks", exit_on_error=False)

    async def _tasks_worker(self, assignee: str) -> None:
        try:
            resp = await self.client.list_tasks(assignee=assignee)
        except (AnimaWorksClientError, AttributeError) as exc:
            self.show_transient(f"Could not load tasks: {exc}")
            return
        tasks = resp.get("tasks") or []
        if not tasks:
            self.show_transient(f"No tasks for {assignee}.")
            return
        lines = [f"Tasks ({assignee}):"]
        for task in tasks:
            title = task.get("title") or task.get("summary") or "?"
            status = task.get("status") or task.get("column") or "?"
            lines.append(f"  [{status}] {title}")
        self.show_transient("\n".join(lines))

    def resolve_interaction(self, args: list[str]) -> None:
        if not args:
            self.show_transient("usage: /approve <callback_id> [option]")
            return
        callback_id = args[0]
        option = args[1] if len(args) > 1 else None
        meta = self._pending_cards.get(callback_id)
        if option is None:
            options = (meta or {}).get("options") or []
            if "approve" in options:
                option = "approve"
            elif options:
                option = options[0]
            else:
                option = "approve"
        anima = meta.get("anima", self.anima_name) if meta else self.anima_name
        self.run_worker(
            self._resolve_worker(anima, callback_id, option),
            group="interact",
            exit_on_error=False,
        )

    async def _resolve_worker(self, anima: str, callback_id: str, option: str) -> None:
        try:
            await self.client.resolve_interaction(anima, callback_id, option)
        except AnimaWorksClientError as exc:
            if "resolved" in str(exc).lower():
                self.show_transient("Already resolved or expired.")
            else:
                self.show_transient(f"Resolve failed: {exc}")
            return
        self._pending_cards.pop(callback_id, None)
        self.show_transient(f"Resolved {callback_id} → {option}.")

    def on_call_human_option(self, message: CallHumanOption) -> None:
        self.resolve_interaction([message.callback_id, message.option])

    # ── Actions ──────────────────────────────────────────
    def action_copy_selection(self) -> None:
        """``Enter`` copies the current mouse selection, tmux copy-mode style.

        The clipboard is written with OSC 52, which reaches the outer
        terminal through tmux (``set-clipboard on``). With nothing
        selected the action skips itself so the key still submits.
        """
        text = self._selected_text()
        if not text:
            self._restore_input_focus()
            raise SkipAction()
        self.copy_to_clipboard(text)
        self.clear_selection()
        self._restore_input_focus()
        # Kept short: the hint sits at the right edge and is clipped there.
        lines = text.count("\n") + 1
        self.status_bar.flash(f"Copied {len(text)} chars" if lines == 1 else f"Copied {lines} lines")

    def _restore_input_focus(self) -> None:
        """Put focus back in the input box after a click in the transcript.

        Clicking (which is also how a selection starts) moves focus to the
        scrollable transcript — or drops it entirely — and every keystroke
        after that goes nowhere. Those two cases are repaired; a deliberate
        Tab into the sidebar must survive.
        """
        if self.focused is None or self.focused is self.transcript:
            # ``Widget.focus()`` defers through ``call_later``, which is too
            # late for the key being handled right now: Textual reads
            # ``app.focused`` again to pick the forward target as soon as
            # this action returns. Setting it on the screen is immediate.
            self.screen.set_focus(self.input_container.input)

    def _selected_text(self) -> str:
        """Return the transcript selection, ignoring the input box.

        ``TextArea`` keeps its own selection and its own copy binding, so
        text highlighted while typing must not hijack Enter.
        """
        screen = self.screen
        if not screen.selections:
            return ""
        chunks: list[str] = []
        for widget, selection in screen.selections.items():
            if widget is self.input_container.input or not widget.is_attached:
                continue
            selected = widget.get_selection(selection)
            if selected is not None:
                chunks.extend(selected)
        return "".join(chunks).rstrip("\n")

    def action_maybe_interrupt(self) -> None:
        # The slash palette owns Escape while it is open (the input closes
        # it), and a live selection is what Escape drops next: the reader
        # is picking text out of the transcript, not stopping the anima.
        if self.palette_is_open():
            raise SkipAction()
        if self.screen.selections:
            self.clear_selection()
            self._restore_input_focus()
            return
        if self.busy:
            self.run_worker(self._do_interrupt(), group="interrupt", exit_on_error=False)

    async def _do_interrupt(self) -> None:
        try:
            await self.client.interrupt(self.anima_name, thread_id=self.thread_id)
        except AnimaWorksClientError as exc:
            self.show_transient(f"Interrupt failed: {exc}")
            return
        self._stop_thinking_blink()
        self.busy = False
        self.session.in_flight = False
        self._write_session()
        self.status_bar.set_state(
            status="idle",
            active_tool=None,
            right_hint="Esc: interrupt",
        )
        self.show_transient("Interrupted.")

    def _save_on_exit(self) -> None:
        self.session.updated_at = ""
        self._write_session()

    def action_quit_or_confirm(self) -> None:
        now = time.monotonic()
        if now - self._last_ctrlc < 2.0:
            self._save_on_exit()
            self.exit()
        else:
            self._last_ctrlc = now
            self.status_bar.set_state(right_hint="Press Ctrl+C again to quit")

    def action_quit_now(self) -> None:
        self._save_on_exit()
        self.exit()

    def action_scroll_up(self) -> None:
        self.transcript.scroll_up()

    def action_scroll_down(self) -> None:
        self.transcript.scroll_down()


def decide_reattach(active: dict | None, session_in_flight: bool) -> str:
    """Decide how to handle a possibly in-flight stream (pure function).

    ``active`` is the ``/stream/active`` response (``None`` or
    ``{"active": false}`` when there is nothing). Returns one of:

    - ``reattach``      there is an actively streaming response → resume the tail
    - ``render_final``  the stream is complete but we never saw it finish → render once
    - ``lost``          the session says in-flight but the server has nothing
    - ``nothing``       nothing to do
    """
    if not active or not active.get("active"):
        return "lost" if session_in_flight else "nothing"
    status = active.get("status")
    if status == "streaming":
        return "reattach"
    # complete / finished etc.
    return "render_final" if session_in_flight else "nothing"


def _make_runner(handler):
    def run(app) -> None:
        handler([], app)

    return run
