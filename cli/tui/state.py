# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Pure client-side UI state for the TUI sidebar, activity feed and palette.

Kept free of Textual so the state-transition and filtering logic can be
unit tested directly as pure functions.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from typing import Any

MAX_ACTIVITY = 200

# Same rule as core.skills.activation_state.validate_thread_id (kept here so
# the TUI does not import from core): alphanumerics, underscores and hyphens,
# 1 to 36 characters.
_THREAD_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,36}$")


def is_valid_thread_id(value: str) -> bool:
    """True when ``value`` is a valid thread id (see ``_THREAD_ID_RE``)."""
    return bool(_THREAD_ID_RE.match(value or ""))


def new_thread_id() -> str:
    """Return a fresh thread id: 8 hex chars, matching the Web UI's format."""
    return uuid.uuid4().hex[:8]


@dataclass
class AnimaRow:
    """Per-anima sidebar row state."""

    name: str
    status: str = "idle"
    busy: bool = False
    active_tool: str | None = None
    unread: int = 0
    needs_user_input: bool = False
    # The anima's own configured model — what "no override" resolves to.
    model: str = ""


@dataclass
class ActivityEntry:
    """A single line in the activity feed."""

    anima: str = ""
    kind: str = ""
    text: str = ""


@dataclass
class WsEffect:
    """Side effects produced by applying a websocket event."""

    feed: list[ActivityEntry] = field(default_factory=list)
    toasts: list[tuple[str, str]] = field(default_factory=list)
    proactive: list[dict] = field(default_factory=list)
    cards: list[dict] = field(default_factory=list)


class AppState:
    """Holds all anima rows, the active feed, and the current anima."""

    def __init__(self) -> None:
        self.animas: dict[str, AnimaRow] = {}
        self.activity: list[ActivityEntry] = []
        # Total number of entries ever added; lets the feed widget render
        # only what is new instead of rebuilding every row per event.
        self.activity_seq = 0
        self.current: str | None = None

    def set_animas(self, animas: list[dict]) -> None:
        """(Re)seed the state from a ``/api/animas`` response."""
        self.animas = {}
        for entry in animas:
            name = entry.get("name")
            if not name:
                continue
            busy = entry.get("busy")
            is_busy = isinstance(busy, dict) and bool(busy.get("is_busy"))
            self.animas[name] = AnimaRow(
                name=name,
                status="busy" if is_busy else "idle",
                busy=is_busy,
                needs_user_input=bool(entry.get("needs_user_input")),
                model=str(entry.get("model") or ""),
            )
        if self.current is not None and self.current not in self.animas:
            self.current = None

    def add_activity(self, entry: ActivityEntry | None) -> None:
        if entry is None:
            return
        self.activity.append(entry)
        self.activity_seq += 1
        if len(self.activity) > MAX_ACTIVITY:
            del self.activity[: len(self.activity) - MAX_ACTIVITY]

    def clear_unread(self, name: str) -> None:
        row = self.animas.get(name)
        if row is not None:
            row.unread = 0


def _anima_name(data: dict) -> str | None:
    return data.get("name") or data.get("anima")


def apply_ws_event(state: AppState, event: dict) -> WsEffect:
    """Apply one normalized websocket event to ``state`` (pure).

    Returns a :class:`WsEffect` describing the side effects (feed lines,
    toasts, transcript insertions, interaction cards) the UI should
    render. Events for animas we have never seen are ignored.
    """
    etype = event.get("type")
    data = event.get("data") or {}
    eff = WsEffect()

    if etype == "anima.status":
        name = _anima_name(data)
        row = state.animas.get(name) if name else None
        if row is not None:
            row.status = data.get("status") or row.status

    elif etype == "anima.tool_activity":
        name = _anima_name(data)
        row = state.animas.get(name) if name else None
        if row is None:
            return eff
        evt = data.get("event")
        tool = data.get("tool_name") or data.get("tool") or "tool"
        if evt == "tool_start":
            row.active_tool = tool
            eff.feed.append(ActivityEntry(name, "tool", tool))
        elif evt == "tool_end":
            row.active_tool = None
            suffix = " ✗" if bool(data.get("is_error")) else ""
            eff.feed.append(ActivityEntry(name, "tool", f"{tool}{suffix}"))
        elif evt == "tool_detail":
            # High volume — only update the side bar's current-tool column.
            row.active_tool = tool
        elif evt is None and data.get("type"):
            # Activity-log shape (no ``event`` key): type/kind/tool/summary.
            atype = data.get("type") or ""
            kind = data.get("kind") or ""
            summary = data.get("summary") or ""
            if kind == "tool_use":
                row.active_tool = tool
                if data.get("tool"):
                    eff.feed.append(ActivityEntry(name, "tool", str(data.get("tool"))))
            elif kind == "tool_result":
                row.active_tool = None
                mark = "" if not bool(data.get("is_error")) else " ✗"
                if mark:
                    eff.feed.append(ActivityEntry(name, "tool", f"{tool}{mark}"))
            else:
                text = atype if not summary else f"{atype} {str(summary)[:60]}"
                eff.feed.append(ActivityEntry(name, atype, text))
            # Mirror start/end phases into the row state when no anima.status
            # is pushed (background lanes).
            if atype.endswith("_start"):
                row.status = "busy"
                row.busy = True
            elif atype.endswith("_end"):
                row.status = "idle"
                row.busy = False
        elif evt:
            eff.feed.append(ActivityEntry(name, "tool", str(data.get("summary") or evt)))

    elif etype == "anima.proactive_message":
        name = _anima_name(data)
        if name and name in state.animas and name != state.current:
            state.animas[name].unread += 1
        eff.proactive.append(data)

    elif etype == "anima.notification":
        name = _anima_name(data) or "?"
        subject = data.get("subject") or ""
        body = data.get("body") or ""
        eff.feed.append(ActivityEntry(name, "notification", subject or body[:100]))
        eff.toasts.append((subject, body))
        if data.get("callback_id") and data.get("options"):
            eff.cards.append(data)

    elif etype == "anima.heartbeat":
        name = _anima_name(data) or "?"
        result = data.get("result") or {}
        text = "heartbeat"
        for key in ("summary", "result"):
            val = result.get(key)
            if val:
                text = str(val)[:80]
                break
        eff.feed.append(ActivityEntry(name, "heartbeat", text))

    elif etype == "anima.cron":
        name = _anima_name(data) or "?"
        eff.feed.append(ActivityEntry(name, "cron", str(data.get("summary") or data.get("task") or "cron")))

    elif etype == "anima.bootstrap":
        name = _anima_name(data) or "?"
        eff.feed.append(ActivityEntry(name, "bootstrap", str(data.get("status") or "")))

    elif etype == "anima.interaction":
        frm = data.get("from_person") or ""
        to = data.get("to_person") or ""
        label = frm or to or "?"
        eff.feed.append(ActivityEntry(label, "interaction", f"{frm} → {to}: {data.get('summary') or ''}"))

    elif etype == "board.post":
        channel = data.get("channel") or ""
        frm = data.get("from") or "unknown"
        eff.feed.append(ActivityEntry(frm, "board", f"#{channel} {frm}: {str(data.get('text') or '')}"))

    # Feed entries are also appended to the rolling activity ring buffer.
    for entry in eff.feed:
        state.add_activity(entry)

    return eff


# ── Palette candidate filtering (pure) ─────────────────────


@dataclass
class PaletteItem:
    """A completion candidate for the slash-command palette."""

    value: str  # text to insert into the input
    label: Any  # rich Text or str shown in the list
    takes_args: bool  # whether to keep typing after selecting
    search: str = ""  # space separated search tokens (lowercase)
    on_confirm: Any | None = None  # callable(app) used instead of inserting

    def matches(self, query: str) -> bool:
        if not query:
            return True
        tokens = self.search.split()
        return any(tok.startswith(query) or query in tok for tok in tokens)


def filter_palette(items: list[PaletteItem], text: str) -> list[PaletteItem]:
    """Return ``items`` ordered for the palette given the current input.

    ``text`` is the full current input (with a leading ``/``). Prefix
    matches are ranked ahead of partial matches; an empty query returns
    every item. Case-insensitive. Both the slashed (``/bar``) and bare
    (``bar``) forms of the query are tested against each search token so
    that ``/pr-review`` matches the ``/skill pr-review`` entry.
    """
    raw = text.lower()
    q = raw
    q_bare = raw[1:] if raw.startswith("/") else raw
    if not q_bare:
        return list(items)

    def is_prefix(tok: str) -> bool:
        return tok.startswith(q) or (raw.startswith("/") and tok.startswith(q_bare))

    def contains(tok: str) -> bool:
        return q in tok or (q_bare and q_bare in tok)

    prefix: list[PaletteItem] = []
    partial: list[PaletteItem] = []
    for item in items:
        tokens = item.search.split()
        if any(is_prefix(t) for t in tokens):
            prefix.append(item)
        elif any(contains(t) for t in tokens):
            partial.append(item)
    return prefix + partial
