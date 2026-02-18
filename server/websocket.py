from __future__ import annotations
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.


import asyncio
import json
import logging
import time
from contextlib import suppress

from fastapi import WebSocket

logger = logging.getLogger("animaworks.websocket")

# ── Heartbeat Constants ─────────────────────────────────
_HEARTBEAT_INTERVAL = 30  # seconds between pings
_HEARTBEAT_TIMEOUT = 60   # seconds before considering client dead (2 missed pongs)


class WebSocketManager:
    """Manages WebSocket connections, broadcasts, and application-level heartbeat."""

    _MAX_QUEUE_SIZE = 50  # prevent unbounded growth

    def __init__(self) -> None:
        self.active_connections: list[WebSocket] = []
        self._notification_queue: list[dict] = []
        self._heartbeat_task: asyncio.Task | None = None
        self._last_pong: dict[int, float] = {}

    # ── Connection Management ───────────────────────────────

    async def connect(self, websocket: WebSocket) -> None:
        """Accept a new WebSocket connection and register it."""
        await websocket.accept()
        self.active_connections.append(websocket)
        self._last_pong[id(websocket)] = time.time()
        logger.info("WebSocket connected. Total: %d", len(self.active_connections))
        # Flush any queued notifications to the new client
        await self.flush_notification_queue(websocket)

    def disconnect(self, websocket: WebSocket) -> None:
        """Remove a WebSocket connection and clean up tracking state."""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        self._last_pong.pop(id(websocket), None)
        logger.info(
            "WebSocket disconnected. Total: %d", len(self.active_connections)
        )

    # ── Client Message Handling ─────────────────────────────

    async def handle_client_message(self, websocket: WebSocket, data: str) -> None:
        """Process an incoming message from a WebSocket client.

        Handles application-level protocol messages (e.g. pong responses).
        Non-JSON messages are silently ignored.

        Args:
            websocket: The client WebSocket connection.
            data: Raw text data received from the client.
        """
        try:
            msg = json.loads(data)
            if isinstance(msg, dict) and msg.get("type") == "pong":
                self._last_pong[id(websocket)] = time.time()
        except (json.JSONDecodeError, ValueError):
            pass  # Non-JSON messages are OK

    # ── Heartbeat ───────────────────────────────────────────

    async def start_heartbeat(self) -> None:
        """Start the application-level heartbeat loop."""
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        logger.info("WebSocket heartbeat started (interval=%ds, timeout=%ds)",
                     _HEARTBEAT_INTERVAL, _HEARTBEAT_TIMEOUT)

    async def stop_heartbeat(self) -> None:
        """Stop the heartbeat loop if running."""
        if self._heartbeat_task is not None:
            self._heartbeat_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._heartbeat_task
            self._heartbeat_task = None
            logger.info("WebSocket heartbeat stopped")

    async def _heartbeat_loop(self) -> None:
        """Periodically ping clients and disconnect stale connections."""
        while True:
            await asyncio.sleep(_HEARTBEAT_INTERVAL)

            if not self.active_connections:
                continue

            now = time.time()
            ping_message = json.dumps({"type": "ping", "ts": now})

            # ── Send ping to all active connections ─────────
            stale: list[WebSocket] = []
            for conn in list(self.active_connections):
                # Check for stale connection first
                last_pong = self._last_pong.get(id(conn), now)
                if now - last_pong > _HEARTBEAT_TIMEOUT:
                    logger.warning(
                        "WebSocket client stale (no pong for %.0fs), disconnecting",
                        now - last_pong,
                    )
                    stale.append(conn)
                    continue

                # Send ping
                try:
                    await conn.send_text(ping_message)
                except Exception:
                    logger.warning("Failed to send ping, removing broken connection")
                    stale.append(conn)

            # ── Remove stale/broken connections ─────────────
            for conn in stale:
                self.disconnect(conn)

    # ── Notifications ───────────────────────────────────────

    async def broadcast_notification(self, data: dict) -> None:
        """Broadcast a call_human event as both proactive_message and notification.

        The proactive_message event is displayed in the chat conversation,
        while the notification event triggers toast/activity log in the UI.
        Both are queued when no clients are connected.
        """
        proactive_event = {"type": "anima.proactive_message", "data": data}
        notification_event = {"type": "anima.notification", "data": data}
        if self.active_connections:
            await self.broadcast(proactive_event)
            await self.broadcast(notification_event)
        else:
            self._notification_queue.append(proactive_event)
            self._notification_queue.append(notification_event)
            while len(self._notification_queue) > self._MAX_QUEUE_SIZE:
                self._notification_queue.pop(0)  # drop oldest

    async def flush_notification_queue(self, websocket: WebSocket) -> None:
        """Send queued notifications to a newly connected client."""
        while self._notification_queue:
            event = self._notification_queue.pop(0)
            try:
                await websocket.send_text(
                    json.dumps(event, ensure_ascii=False, default=str)
                )
            except Exception:
                break

    async def broadcast(self, data: dict) -> None:
        """Broadcast a message to all active WebSocket connections."""
        if not self.active_connections:
            return
        message = json.dumps(data, ensure_ascii=False, default=str)
        disconnected: list[WebSocket] = []
        for conn in self.active_connections:
            try:
                await conn.send_text(message)
            except Exception:
                logger.warning("broadcast_failed", exc_info=True)
                disconnected.append(conn)
        for conn in disconnected:
            self.disconnect(conn)
