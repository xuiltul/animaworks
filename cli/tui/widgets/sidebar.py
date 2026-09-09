# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Sidebar widget: list of animas + activity feed.

The widget is a thin view over :class:`cli.tui.state.AppState`. It has
no knowledge of the server; it just re-renders whatever state it is
given via :meth:`Sidebar.refresh`.
"""

from __future__ import annotations

from datetime import datetime

from rich.text import Text
from textual.binding import Binding
from textual.containers import Vertical, VerticalScroll
from textual.message import Message
from textual.widgets import Static

from cli.tui.state import AppState


class AnimaChosen(Message):
    """Posted when the user picks an anima from the sidebar."""

    def __init__(self, name: str) -> None:
        super().__init__()
        self.name = name


class _AnimaRow(Static, can_focus=True):
    """A single, focusable row for one anima in the sidebar."""

    def __init__(self, name: str, **kwargs) -> None:
        super().__init__(**kwargs)
        self.anima_name = name

    def on_click(self) -> None:
        self.post_message(AnimaChosen(self.anima_name))

    def on_enter(self) -> None:
        self.post_message(AnimaChosen(self.anima_name))


class AnimaList(VerticalScroll):
    """The upper portion of the sidebar: one row per anima."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._rows: dict[str, _AnimaRow] = {}
        self._current: str | None = None
        self._rendered: dict[str, str] = {}

    def refresh_animas(self, state: AppState) -> None:
        names = list(state.animas)
        rows_here = set(self._rows)
        for name in set(rows_here) - set(names):
            self._rows[name].remove()
            del self._rows[name]
            self._rendered.pop(name, None)
        for name in names:
            row = self._rows.get(name)
            if row is None:
                row = _AnimaRow(name)
                self._rows[name] = row
                row.set_class(name == self._current, "current")
                self.mount(row)
            self._update_row(row, state)

    def set_current(self, name: str | None) -> None:
        if name != self._current:
            # The ▶ marker lives in the row text: force both rows to re-render.
            self._rendered.pop(self._current, None)
            self._rendered.pop(name, None)
        self._current = name
        for row_name, row in self._rows.items():
            row.set_class(row_name == name, "current")

    def _update_row(self, row: _AnimaRow, state: AppState) -> None:
        info = state.animas.get(row.anima_name)
        if info is None:
            return
        parts: list = []
        # Current chat partner marker: ▶ for the selected anima, blank otherwise.
        parts.append(Text("▶ " if row.anima_name == self._current else "  ", style="bold"))
        if info.busy or info.status in ("busy", "thinking", "streaming"):
            parts.append(Text("● ", style="bold"))
        else:
            parts.append(Text("○ ", style="dim"))
        parts.append(Text(row.anima_name, style="bold"))
        if info.unread:
            parts.append(Text(f" ({info.unread})", style="bold"))
        if info.active_tool:
            parts.append(Text(f"  {info.active_tool}", style="dim"))
        text = Text.assemble(*parts)
        key = text.plain + "".join(str(span) for span in text.spans)
        if self._rendered.get(row.anima_name) == key:
            return
        self._rendered[row.anima_name] = key
        row.update(text)


# Max visible text width for a single feed line (sidebar is ~32 cols).
_MAX_FEED_TEXT = 34


class _FeedLine(Static):
    """A single, non-wrapping activity feed row (truncated with `…`)."""

    DEFAULT_CSS = """
    _FeedLine {
        height: 1;
        width: 100%;
    }
    """


def _truncate(text: str, width: int = _MAX_FEED_TEXT) -> str:
    """Truncate ``text`` to ``width`` chars, appending `…` when cut."""
    if len(text) <= width:
        return text
    return text[: width - 1] + "…"


class ActivityFeed(VerticalScroll):
    """The lower portion of the sidebar: rolling activity feed.

    Each row is fixed at height 1 and never wraps; the feed itself
    scrolls internally so it can never overflow the sidebar.

    Updates are incremental: only entries added since the last call are
    mounted and the oldest rows beyond ``MAX_ROWS`` are dropped. A full
    rebuild per websocket event (remove + mount up to 60 widgets) cost
    ~75ms CPU each and made the UI unresponsive under event bursts.
    """

    MAX_ROWS = 60

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._seen_seq = 0

    @staticmethod
    def _make_line(entry) -> _FeedLine:
        now = datetime.now().strftime("%H:%M")
        label = entry.anima or "?"
        text_parts: list = []
        if entry.kind == "board":
            text_parts.append(Text(f"{now} ", style="dim"))
            text_parts.append(Text(entry.text, style="bold"))
        elif entry.kind == "interaction":
            text_parts.append(Text(f"{now} ", style="dim"))
            text_parts.append(Text(entry.text, style="italic"))
        else:
            text_parts.append(Text(f"{now} {label} ", style="dim"))
            if entry.kind == "tool":
                # Tool rows: keep the width for the tool name itself.
                text_parts.append(Text(entry.text or "tool"))
            else:
                text_parts.append(Text(entry.kind))
                if entry.text:
                    text_parts.append(Text(f" {entry.text}", style="dim"))
        return _FeedLine(_truncate(str(Text.assemble(*text_parts))))

    def update_state(self, state: AppState) -> None:
        seq = getattr(state, "activity_seq", None)
        if seq is None:
            self._rebuild(state)
            return
        new = seq - self._seen_seq
        self._seen_seq = seq
        if new <= 0:
            return
        if new >= self.MAX_ROWS or not self.children:
            self._rebuild(state)
            return
        for entry in state.activity[-new:]:
            self.mount(self._make_line(entry))
        rows = list(self.children)
        for row in rows[: max(0, len(rows) - self.MAX_ROWS)]:
            row.remove()
        self.scroll_end(animate=False)

    def _rebuild(self, state: AppState) -> None:
        self.remove_children()
        for entry in state.activity[-self.MAX_ROWS :]:
            self.mount(self._make_line(entry))
        if state.activity:
            self.scroll_end(animate=False)


class Sidebar(Vertical):
    """The full sidebar: anima list on top, activity feed below."""

    BINDINGS = [
        Binding("enter", "choose", "Choose selected anima", show=False),
    ]

    DEFAULT_CSS = """
    Sidebar {
        width: 32;
        height: 1fr;
        background: transparent;
        border-right: round ansi_default;
        padding: 0 1;
    }
    Sidebar > .sidebar-header {
        text-style: bold;
        margin-top: 1;
    }
    AnimaList {
        height: 1fr;
        width: 100%;
    }
    AnimaList > _AnimaRow.current {
        text-style: reverse;
    }
    ActivityFeed {
        height: 1fr;
        width: 100%;
        border-top: round ansi_default;
        margin-top: 1;
    }
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.animas = AnimaList()
        self.feed = ActivityFeed()

    def compose(self):
        yield Static("ANIMAS", classes="sidebar-header")
        yield self.animas
        yield Static("ACTIVITY", classes="sidebar-header")
        yield self.feed

    def update_state(self, state: AppState) -> None:
        # Current first: the row text embeds the ▶ marker.
        self.animas.set_current(state.current)
        self.animas.refresh_animas(state)
        self.feed.update_state(state)

    def action_choose(self) -> None:
        focused = self.focused
        if isinstance(focused, _AnimaRow):
            self.post_message(AnimaChosen(focused.anima_name))
