from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Channels whose Board posts must not fan out mentions in the current run.

The inbox sets the channels of the ``board_mention`` messages it is answering.
Only posts to those channels skip mention delivery; every other channel keeps
its notifications.  Tool subprocesses (the Board CLI run from Bash, MCP
servers) cannot see the context variable, so engines export the same set
through :data:`BOARD_REPLY_CHANNELS_ENV`.
"""

import contextvars

from core.platform.env import get_env

BOARD_REPLY_CHANNELS_ENV = "ANIMAWORKS_BOARD_REPLY_CHANNELS"

suppress_board_fanout_channels: contextvars.ContextVar[frozenset[str]] = contextvars.ContextVar(
    "suppress_board_fanout_channels",
    default=frozenset(),
)


def board_reply_channels_env() -> dict[str, str]:
    """Return the environment entry that carries the suppressed channels."""
    return {BOARD_REPLY_CHANNELS_ENV: ",".join(sorted(suppress_board_fanout_channels.get()))}


def board_reply_channels() -> frozenset[str]:
    """Return the channels whose posts must not fan out mentions in this run."""
    channels = set(suppress_board_fanout_channels.get())
    raw = get_env(BOARD_REPLY_CHANNELS_ENV, "") or ""
    channels.update(part.strip() for part in raw.split(",") if part.strip())
    return frozenset(channels)
