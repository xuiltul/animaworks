# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from rich.style import Style
from rich.text import Text
from textual.containers import Vertical
from textual.widgets import Static


class ThinkingBlock(Vertical):
    """A block showing ``thinking`` deltas in dim style.

    The body always buffers whatever is streamed. Whether it is shown is
    controlled via :meth:`set_visible`, which just toggles the block's
    ``display`` (show/hide) — content is never discarded.

    While the anima is actually thinking (:meth:`start` … :meth:`stop`)
    the ``▸`` marker in the header blinks so the reader can tell a live
    stream from a stalled one.
    """

    BLINK_INTERVAL = 0.5

    DEFAULT_CSS = """
    ThinkingBlock {
        height: auto;
        width: 100%;
    }
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._body = ""
        self._active = False
        self._mark_on = True
        self._timer = None
        self._ready = False  # on_mount seen (``is_mounted`` flips only after it)
        self.header = Static("", classes="thinking-header")
        self.body = Static("", classes="thinking-body")

    def compose(self):
        yield self.header
        yield self.body

    def on_mount(self) -> None:
        self._ready = True
        if self._active:
            self._ensure_timer()
        self._refresh()

    def _refresh(self) -> None:
        mark = "▸" if (self._mark_on or not self._active) else " "
        header = Text(mark + " ", style=Style.parse("bold"))
        header.append("thinking ", style=Style.parse("dim"))
        self.header.update(header)
        self.body.update(Text(self._body, style="dim"))

    # ── blinking ──────────────────────────────────────
    @property
    def is_active(self) -> bool:
        return self._active

    def _ensure_timer(self) -> None:
        if self._timer is None and self._ready:
            self._timer = self.set_interval(self.BLINK_INTERVAL, self._tick)

    def _tick(self) -> None:
        self._mark_on = not self._mark_on
        self._refresh()

    def start(self) -> None:
        """Begin blinking (idempotent; safe before mount)."""
        if self._active:
            return
        self._active = True
        self._mark_on = True
        self._ensure_timer()
        self._refresh()

    def stop(self) -> None:
        """Stop blinking and leave the marker on."""
        self._active = False
        self._mark_on = True
        if self._timer is not None:
            self._timer.stop()
            self._timer = None
        self._refresh()

    def add_delta(self, text: str) -> None:
        self._body += text
        self._refresh()

    def set_visible(self, visible: bool) -> None:
        self.display = visible
        self._refresh()
