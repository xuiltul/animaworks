# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from rich.text import Text
from textual.widget import Widget


def _clip(model: str, limit: int = 40) -> str:
    return model if len(model) <= limit else model[: limit - 1] + "…"


class StatusBar(Widget):
    """A single status line showing anima, state, tool, connection, thread."""

    def __init__(
        self,
        anima_name: str,
        thread_id: str,
        *,
        initial_status: str = "starting",
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.anima_name = anima_name
        self.thread_id = thread_id
        self.status = initial_status
        self.active_tool: str | None = None
        self.connected = False
        self.right_hint = "Esc: interrupt"
        self.skill_count = 0
        self.model: str | None = None
        self.context_usage_ratio: float | None = None
        self.context_input_tokens: int | None = None
        self.context_window: int | None = None
        # What "no override" resolves to: the anima's own configured model.
        self.default_model: str | None = None
        self._flash_restore: str | None = None
        self._flash_timer = None

    def set_anima(self, anima_name: str, thread_id: str | None = None) -> None:
        self.anima_name = anima_name
        if thread_id is not None:
            self.thread_id = thread_id
        self.refresh()

    def set_state(
        self,
        *,
        status: str | None = None,
        active_tool: str | None = None,
        connected: bool | None = None,
        right_hint: str | None = None,
        skill_count: int | None = None,
    ) -> None:
        if status is not None:
            self.status = status
        if active_tool is not None:
            self.active_tool = active_tool or None
        if connected is not None:
            self.connected = connected
        if right_hint is not None:
            self._cancel_flash()
            self.right_hint = right_hint
        if skill_count is not None:
            self.skill_count = skill_count
        self.refresh()

    def flash(self, message: str, duration: float = 3.0) -> None:
        """Show *message* in the hint slot, then restore the hint.

        Used for one-shot feedback (a copy, say) that would only clutter
        the transcript if it were written there.
        """
        if self._flash_restore is None:
            self._flash_restore = self.right_hint
        if self._flash_timer is not None:
            self._flash_timer.stop()
        self.right_hint = message
        self.refresh()
        self._flash_timer = self.set_timer(duration, self._end_flash)

    def _end_flash(self) -> None:
        if self._flash_restore is not None:
            self.right_hint = self._flash_restore
        self._cancel_flash()
        self.refresh()

    def _cancel_flash(self) -> None:
        if self._flash_timer is not None:
            self._flash_timer.stop()
        self._flash_timer = None
        self._flash_restore = None

    def set_model(self, model: str | None) -> None:
        """Set the model to show (``None`` leaves it untouched, "" clears it)."""
        self.model = model or None
        self.refresh()

    def set_default_model(self, model: str | None) -> None:
        """Name the anima's configured model, shown when nothing overrides it."""
        self.default_model = model or None
        self.refresh()

    def set_context_usage(
        self,
        ratio: float | int | str | None,
        *,
        input_tokens: int | str | None = None,
        context_window: int | str | None = None,
    ) -> None:
        """Persist the current conversation's context position."""
        self.context_usage_ratio = None if ratio is None else min(max(float(ratio), 0.0), 1.0)
        self.context_input_tokens = int(input_tokens) if input_tokens else None
        self.context_window = int(context_window) if context_window else None
        self.refresh()

    def render(self) -> Text:
        parts = [
            Text(self.anima_name),
            Text(" "),
            # The status word follows the dot, so no colour is needed.
            Text("●", style="bold"),
            Text(f" {self.status}"),
        ]

        if self.active_tool:
            parts.append(Text(f" | {self.active_tool}", style="bold"))

        if self.skill_count:
            parts.append(Text(f" | skills:{self.skill_count}", style="dim"))

        if self.context_usage_ratio is not None:
            percent = round(self.context_usage_ratio * 100)
            parts.append(Text(f" | ctx:{percent}%", style="dim"))

        # The model in use gets the permanent slot the connection used to
        # hold: `ws: connected` is the normal case and said nothing, so the
        # connection only speaks up now when it is actually down. With no
        # override the anima's own model is named, so the line always says
        # which model is about to answer rather than just "default".
        if self.model:
            shown = _clip(self.model)
        elif self.default_model:
            shown = f"{_clip(self.default_model)} (default)"
        else:
            shown = "default"
        parts.append(Text(f" | model:{shown}", style="dim"))

        if not self.connected:
            parts.append(Text(" | ws: disconnected", style="bold"))

        parts.append(Text(f" | thread:{self.thread_id}", style="dim"))

        line = Text.assemble(*parts)
        if self.right_hint:
            line.append_text(Text(f"  {self.right_hint}", style="dim"))
        return line
