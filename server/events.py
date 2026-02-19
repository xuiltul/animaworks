from __future__ import annotations
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from fastapi import Request


# ── WebSocket Event Helper ────────────────────────────────────

async def emit(request: Request, event_type: str, data: dict[str, Any]) -> None:
    """Broadcast a WebSocket event if the manager is available."""
    ws = getattr(request.app.state, "ws_manager", None)
    if ws:
        await ws.broadcast({"type": event_type, "data": data})


async def emit_notification(request: Request, data: dict[str, Any]) -> None:
    """Broadcast an anima notification with queue support.

    Uses ``broadcast_notification`` which queues the event when no
    WebSocket clients are connected, flushing when a client reconnects.
    """
    ws = getattr(request.app.state, "ws_manager", None)
    if ws:
        await ws.broadcast_notification(data)
