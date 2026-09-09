# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import time

from rich.text import Text
from textual.widgets import Static


def format_elapsed(seconds: float) -> str:
    """Format a response duration without dropping elapsed hours."""
    total = max(0, int(seconds))
    minutes, secs = divmod(total, 60)
    return f"{minutes}m {secs}s"


class ResponseStatus(Static):
    """Blinking response indicator shown immediately above the prompt."""

    BLINK_INTERVAL = 0.5

    def __init__(self, **kwargs) -> None:
        super().__init__("", **kwargs)
        self._started_at: float | None = None
        self._mark_on = True
        self._timer = None

    @property
    def is_active(self) -> bool:
        return self._started_at is not None

    def start(self, *, started_at: float | None = None) -> None:
        """Show the indicator and start (or preserve) its elapsed clock."""
        if self.is_active:
            return
        self._started_at = time.monotonic() if started_at is None else started_at
        self._mark_on = True
        self.display = True
        self._timer = self.set_interval(self.BLINK_INTERVAL, self._tick)
        self._refresh_status()

    def stop(self) -> None:
        """Hide the indicator and release its timer."""
        self._started_at = None
        self._mark_on = True
        if self._timer is not None:
            self._timer.stop()
            self._timer = None
        self.display = False

    def _tick(self) -> None:
        self._mark_on = not self._mark_on
        self._refresh_status()

    def _refresh_status(self) -> None:
        if self._started_at is None:
            return
        mark = "●" if self._mark_on else " "
        elapsed = format_elapsed(time.monotonic() - self._started_at)
        self.update(Text.assemble(Text(mark, style="bold"), f" Thinking... ({elapsed})"))
