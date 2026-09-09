# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from textual import events
from textual.containers import Horizontal
from textual.message import Message
from textual.widgets import Label, TextArea


class ChatSubmitted(Message):
    """Posted when the user submits the chat input via Enter."""

    def __init__(self, text: str) -> None:
        super().__init__()
        self.text = text


class ChatInputChanged(Message):
    """Posted whenever the input buffer changes (used to drive the palette)."""

    def __init__(self, text: str) -> None:
        super().__init__()
        self.text = text


class ChatInput(TextArea):
    """A multi-line chat input.

    ``Enter`` submits the current **entire** buffer. A newline is inserted
    by any of ``Shift+Enter`` (needs a terminal / tmux that forwards the
    kitty keyboard protocol), ``Ctrl+J`` (works everywhere: it is the raw
    ``\\n`` byte) or a trailing backslash before ``Enter`` (Claude Code's
    ``\\``+``Enter``, which swaps the backslash for the newline).
    Submissions are posted as a :class:`ChatSubmitted` message. Grows up
    to ``max_lines`` rows as content is added and returns to a single row
    after submission.

    A ``controller`` (the running app) may be attached so that arrow /
    tab keys can drive the open slash-command :class:`~cli.tui.widgets.palette.Palette`
    while the input keeps focus.
    """

    max_lines = 6
    NEWLINE_KEYS = ("shift+enter", "ctrl+j")

    def __init__(self, placeholder: str = "Say something…", *, controller=None, **kwargs) -> None:
        super().__init__(placeholder=placeholder, **kwargs)
        self._controller = controller

    def set_controller(self, controller) -> None:
        self._controller = controller

    def _palette_open(self) -> bool:
        return bool(self._controller is not None and self._controller.palette_is_open())

    def on_mount(self) -> None:
        self._update_height()

    def on_text_area_changed(self, _event) -> None:
        self._update_height()
        if self._controller is not None:
            self._controller.on_input_text_changed(self.text)

    def _on_resize(self) -> None:
        # The base handler re-wraps the document for the new width; the row
        # count that falls out of it is what the box has to be sized to.
        super()._on_resize()
        self._update_height()

    def _update_height(self) -> None:
        # Visual rows, not document lines: soft wrap is on, so one long
        # line occupies several rows. Sizing by `document.line_count` kept
        # the box one row tall and scrolled the earlier rows out of sight,
        # which read as a line running off to the right.
        lines = max(self.wrapped_document.height, self.document.line_count, 1)
        self.styles.height = min(lines, self.max_lines)

    async def _on_key(self, event: events.Key) -> None:
        if event.key == "enter":
            if self._palette_open():
                event.stop()
                event.prevent_default()
                self._controller.palette_confirm()
                return
            event.stop()
            event.prevent_default()
            if self._consume_backslash_continuation():
                return
            text = self.text
            if text.strip():
                self.post_message(ChatSubmitted(text))
            self.clear()
            self._update_height()
            return
        if event.key == "tab" and self._palette_open():
            event.stop()
            event.prevent_default()
            self._controller.palette_complete()
            return
        if event.key == "escape" and self._palette_open():
            event.stop()
            event.prevent_default()
            self._controller.palette_close()
            return
        if event.key in ("up", "down", "pageup", "pagedown", "home", "end") and self._palette_open():
            event.stop()
            event.prevent_default()
            self._controller.palette_move(event.key)
            return
        if event.key in self.NEWLINE_KEYS:
            event.stop()
            event.prevent_default()
            self.insert("\n")
            self._update_height()
            return
        await super()._on_key(event)

    def _consume_backslash_continuation(self) -> bool:
        """``\\`` immediately before the cursor + Enter → newline.

        Returns True when the backslash was replaced by a newline (the
        caller must then *not* submit).
        """
        row, col = self.cursor_location
        if col == 0:
            return False
        line = self.document.get_line(row)
        if line[col - 1] != "\\":
            return False
        self.delete((row, col - 1), (row, col))
        self.insert("\n")
        self._update_height()
        return True


class ChatInputContainer(Horizontal):
    """Prompt label + :class:`ChatInput`, auto-sizing up to 6 rows."""

    def __init__(self, placeholder: str = "Say something…", **kwargs) -> None:
        super().__init__(**kwargs)
        self._placeholder = placeholder
        self.input = ChatInput(placeholder=placeholder)

    def compose(self):
        yield Label(">", classes="input-prompt")
        yield self.input

    def focus_input(self) -> None:
        self.input.focus()
