# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""A call_human notification card rendered in the transcript.

Shows the subject/body plus a list of clickable decision options. Picking
an option resolves the pending interaction via the client. Keyboard users
can use ``/approve <callback_id> [option]`` instead.
"""

from __future__ import annotations

from rich.text import Text
from textual.containers import Vertical
from textual.message import Message
from textual.widgets import Static

from cli.tui.widgets.transcript import strip_html_comments


class CallHumanOption(Message):
    """Posted when the user picks an option on an interaction card."""

    def __init__(self, callback_id: str, option: str) -> None:
        super().__init__()
        self.callback_id = callback_id
        self.option = option


class InteractionCard(Vertical):
    """A transcript block presenting a call_human decision request."""

    DEFAULT_CSS = """
    InteractionCard {
        height: auto;
        width: 100%;
        border: round ansi_default;
        margin: 1 0;
        padding: 0 1;
    }
    """

    def __init__(self, anima_name: str, data: dict, **kwargs) -> None:
        super().__init__(**kwargs)
        self.anima_name = anima_name
        self.callback_id = data.get("callback_id") or ""
        self.options = list(data.get("options") or [])
        self._subject = data.get("subject") or "Request"
        self._body = strip_html_comments(data.get("body") or "")
        self._option_widgets: list[Static] = []

    def compose(self):
        yield Static("", classes="ch-header")
        yield Static("", classes="ch-body")
        for _ in self.options:
            widget = Static("", classes="ch-option")
            self._option_widgets.append(widget)
            yield widget

    def on_mount(self) -> None:
        children = self.children
        children[0].update(
            Text.assemble(
                Text("■ ", style="bold"),
                Text(f"{self.anima_name}: ", style="bold"),
                Text(self._subject, style="bold"),
            )
        )
        children[1].update(Text(self._body, style="dim"))
        for option, widget in zip(self.options, self._option_widgets, strict=False):
            widget.update(
                Text.assemble(
                    Text("  [", style="dim"),
                    Text(option, style="bold"),
                    Text("]", style="dim"),
                )
            )

    def on_click(self, event) -> None:
        control = event.control
        if control in self._option_widgets:
            index = self._option_widgets.index(control)
            if index < len(self.options):
                self.post_message(CallHumanOption(self.callback_id, self.options[index]))
