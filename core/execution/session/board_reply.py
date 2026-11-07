from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Channels and inbox messages that triggered the current run.

The inbox sets the channels of the ``board_mention`` messages it is answering.
Only posts to those channels skip mention delivery; every other channel keeps
its notifications.  It also records the inbox messages themselves so Board
posts and DMs can carry the id they are replying to.  Tool subprocesses (the
Board CLI run from Bash, MCP servers) cannot see the context variables, so
engines export the same values through :data:`BOARD_REPLY_CHANNELS_ENV` and
:data:`RUN_TRIGGERS_ENV`.
"""

import contextvars
import json

from core.platform.env import get_env

BOARD_REPLY_CHANNELS_ENV = "ANIMAWORKS_BOARD_REPLY_CHANNELS"
RUN_TRIGGERS_ENV = "ANIMAWORKS_RUN_TRIGGERS"

suppress_board_fanout_channels: contextvars.ContextVar[frozenset[str]] = contextvars.ContextVar(
    "suppress_board_fanout_channels",
    default=frozenset(),
)

run_triggers: contextvars.ContextVar[tuple[dict[str, str], ...]] = contextvars.ContextVar(
    "run_triggers",
    default=(),
)


def _normalize_trigger(raw: object) -> dict[str, str] | None:
    """Return a well-formed trigger, or ``None`` when the entry is unusable."""
    if not isinstance(raw, dict):
        return None
    message_id = raw.get("id")
    sender = raw.get("from")
    if not isinstance(message_id, str) or not message_id:
        return None
    if not isinstance(sender, str) or not sender:
        return None
    trigger = {
        "id": message_id,
        "from": sender,
        "type": str(raw.get("type") or "message"),
    }
    for key in ("thread_id", "board_channel", "board_post_id"):
        value = raw.get(key)
        if isinstance(value, str) and value:
            trigger[key] = value
    return trigger


def board_reply_channels_env() -> dict[str, str]:
    """Return the environment entry that carries the suppressed channels."""
    return {BOARD_REPLY_CHANNELS_ENV: ",".join(sorted(suppress_board_fanout_channels.get()))}


def board_reply_channels() -> frozenset[str]:
    """Return the channels whose posts must not fan out mentions in this run."""
    channels = set(suppress_board_fanout_channels.get())
    raw = get_env(BOARD_REPLY_CHANNELS_ENV, "") or ""
    channels.update(part.strip() for part in raw.split(",") if part.strip())
    return frozenset(channels)


def run_triggers_env() -> dict[str, str]:
    """Return the environment entry that carries the run's inbox triggers."""
    return {RUN_TRIGGERS_ENV: json.dumps(list(run_triggers.get()), ensure_ascii=False)}


def current_run_triggers() -> tuple[dict[str, str], ...]:
    """Return the inbox messages that triggered this run, from context and env."""
    candidates: list[object] = list(run_triggers.get())
    raw = get_env(RUN_TRIGGERS_ENV, "") or ""
    if raw:
        try:
            parsed = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            parsed = None
        if isinstance(parsed, list):
            candidates.extend(parsed)

    triggers: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for candidate in candidates:
        trigger = _normalize_trigger(candidate)
        if trigger is None:
            continue
        key = (trigger["type"], trigger["id"])
        if key in seen:
            continue
        seen.add(key)
        triggers.append(trigger)
    return tuple(triggers)
