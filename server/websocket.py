from __future__ import annotations

import json
import logging

from fastapi import WebSocket

logger = logging.getLogger("animaworks.websocket")


class WebSocketManager:
    """Manages WebSocket connections and broadcasts."""

    def __init__(self) -> None:
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info("WebSocket connected. Total: %d", len(self.active_connections))

    def disconnect(self, websocket: WebSocket) -> None:
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logger.info(
            "WebSocket disconnected. Total: %d", len(self.active_connections)
        )

    async def broadcast(self, data: dict) -> None:
        if not self.active_connections:
            return
        message = json.dumps(data, ensure_ascii=False, default=str)
        disconnected: list[WebSocket] = []
        for conn in self.active_connections:
            try:
                await conn.send_text(message)
            except Exception:
                disconnected.append(conn)
        for conn in disconnected:
            self.disconnect(conn)
