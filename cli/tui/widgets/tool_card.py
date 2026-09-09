# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import ast

from rich.text import Text
from textual.containers import Vertical
from textual.content import Content
from textual.widgets import Static

_MAX_PREVIEW = 200
_MAX_DETAIL = 8000


def _one_line(text: str) -> str:
    """Collapse *text* into a single whitespace-normalised line."""
    return " ".join(str(text).split())[:_MAX_PREVIEW]


def format_input_summary(raw: str) -> str:
    """Turn a raw ``input_summary`` (a ``dict`` repr) into a one-liner.

    Falls back to the raw text when it cannot be parsed — the server
    truncates long summaries, which leaves the repr unparsable.
    """
    text = str(raw).strip()
    if text.startswith("{"):
        try:
            parsed = ast.literal_eval(text)
        except (ValueError, SyntaxError):
            parsed = None
        if isinstance(parsed, dict):
            return _one_line(", ".join(f"{key}={value}" for key, value in parsed.items()))
    return _one_line(text)


class ToolCard(Vertical):
    """A one-line tool card, expandable to show detail.

    Shows ``▸ Name ✓ args`` on success or ``✗ error`` on failure — the
    arguments are the ``tool_detail`` one-liner, dimmed and clipped to
    the width of the card. Toggling expands the full detail below.
    """

    DEFAULT_CSS = """
    ToolCard {
        height: auto;
        width: 100%;
    }
    /* One row exactly: the argument preview is ellipsised rather than
       wrapped, so a long command never pushes the card open. */
    ToolCard > .tool-header {
        height: 1;
        width: 100%;
        text-wrap: nowrap;
        text-overflow: ellipsis;
    }
    """

    def __init__(self, tool_name: str, tool_id: str, **kwargs) -> None:
        super().__init__(**kwargs)
        self.tool_name = tool_name
        self.tool_id = tool_id
        self._detail = ""
        self._preview = ""
        self._result_summary: str | None = None
        self._is_error = False
        self._finished = False
        self.expanded = False
        self.header = Static("", classes="tool-header")
        self.detail = Static("", classes="tool-detail")
        # Hidden until expanded: an empty Static still occupies a row, which
        # left a blank line between consecutive tool cards.
        self.detail.display = False

    def compose(self):
        yield self.header
        yield self.detail

    def set_preview(self, text: str) -> None:
        """Set the dimmed argument preview shown on the header line."""
        preview = _one_line(text)
        if not preview or preview == self._preview:
            return
        self._preview = preview
        self._refresh()

    def add_detail(self, text: str) -> None:
        if len(self._detail) < _MAX_DETAIL:
            self._detail += text
        if self.expanded:
            self.detail.update(Text(self._detail, style="dim"))
        self.set_preview(self._detail)

    def toggle(self) -> None:
        self.expanded = not self.expanded
        self.detail.display = self.expanded
        if self.expanded:
            self.detail.update(Text(self._detail, style="dim"))
        self._refresh()

    def finish(
        self,
        result_summary: str | None = None,
        is_error: bool = False,
        input_summary: str | None = None,
    ) -> None:
        self._finished = True
        self._is_error = self._is_error or is_error
        self._result_summary = result_summary or self._result_summary
        if input_summary and not self._preview:
            self._preview = format_input_summary(input_summary)
        self._refresh()

    def _refresh(self) -> None:
        if self._is_error:
            marker = "✗"
            tail = " error"
        elif self._finished:
            marker = "✓"
            tail = f"  {self._result_summary}" if self._result_summary else ""
        else:
            marker = "…"
            tail = ""

        preview = f"  {self._preview}" if self._preview else ""
        # Content (not rich Text) so the ``text-overflow`` rule above can
        # clip the preview to the card width with an ellipsis.
        line = Content.assemble(
            "▸ ",
            (self.tool_name, "bold"),
            "  ",
            (marker, "bold"),
            (preview, "dim"),
            (tail, "dim"),
        )
        self.header.update(line)
