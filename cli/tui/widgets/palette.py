# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Slash-command completion palette.

An overlay shown just above the chat input when the user types ``/``. It
lists matching built-in commands and skills, lets the user navigate with
arrow keys and confirm with Tab/Enter. Focus stays on the chat input so
the user can keep typing; the app drives navigation here.
"""

from __future__ import annotations

from rich.style import Style
from textual.strip import Strip
from textual.widgets import OptionList
from textual.widgets._option_list import Option

from cli.tui.state import PaletteItem


class Palette(OptionList):
    """A compact completion overlay (hidden until opened).

    The selected row is painted in reverse video (SGR 7) directly on the
    rendered strip: the ansi theme maps every colour to ``ansi_default``
    so the stock highlight is invisible, and ``text-style: reverse`` in
    CSS never reaches the row (Textual's ``VisualStyle`` drops it).
    """

    HIGHLIGHT_STYLE = Style(reverse=True)

    DEFAULT_CSS = """
    Palette {
        height: auto;
        max-height: 12;
        display: none;
        border: round ansi_default;
        background: transparent;
    }
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._items: list[PaletteItem] = []
        self._open = False

    # ── state helpers ─────────────────────────────────
    @property
    def is_open(self) -> bool:
        return self._open

    def open(self) -> None:
        self._open = True
        self.display = "block"

    def close(self) -> None:
        self._open = False
        self.display = "none"

    def set_items(self, items: list[PaletteItem]) -> None:
        self._items = items
        self.clear_options()
        if items:
            self.add_options([Option(item.label, id=str(index)) for index, item in enumerate(items)])
            self.highlighted = None
            self._focus_first()
            self.open()
        else:
            self.close()

    def _line_option_index(self, y: int) -> int | None:
        """Option index painted on widget line ``y`` (None for blank rows)."""
        try:
            option_index, _ = self._lines[self.scroll_offset.y + y]
        except IndexError:
            return None
        return option_index

    def render_line(self, y: int) -> Strip:
        strip = super().render_line(y)
        if self.highlighted is not None and self._line_option_index(y) == self.highlighted:
            strip = strip.apply_style(self.HIGHLIGHT_STYLE)
        return strip

    def _focus_first(self) -> None:
        if self._items:
            self.highlighted = 0

    def selected(self) -> PaletteItem | None:
        if not self._items:
            return None
        idx = self.highlighted
        if idx is None:
            idx = 0
        if idx < 0 or idx >= len(self._items):
            return None
        return self._items[idx]

    def move(self, direction: str) -> None:
        if not self._items:
            return
        if direction == "up":
            self.action_cursor_up()
        elif direction == "down":
            self.action_cursor_down()
        elif direction == "first":
            self.action_first()
        elif direction == "last":
            self.action_last()
        elif direction == "pageup":
            self.action_page_up()
        elif direction == "pagedown":
            self.action_page_down()
