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


async def emit_direct(
    ws_manager: Any, event_type: str, data: dict[str, Any],
) -> None:
    """Broadcast a WebSocket event using ws_manager directly.

    Unlike :func:`emit`, this does not require a ``Request`` object,
    making it suitable for background producer tasks that outlive the
    HTTP connection.
    """
    if ws_manager:
        await ws_manager.broadcast({"type": event_type, "data": data})


async def emit_notification_direct(
    ws_manager: Any, data: dict[str, Any],
) -> None:
    """Broadcast a notification using ws_manager directly.

    Unlike :func:`emit_notification`, this does not require a ``Request``
    object, making it suitable for background producer tasks.
    """
    if ws_manager:
        await ws_manager.broadcast_notification(data)
