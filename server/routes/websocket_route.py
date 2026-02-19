from __future__ import annotations
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from core.auth.manager import load_auth, validate_session
from server.localhost import _is_safe_localhost_request

logger = logging.getLogger("animaworks.routes.websocket")


def create_websocket_router() -> APIRouter:
    """Create the WebSocket router with heartbeat-aware endpoint."""
    router = APIRouter()

    @router.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket) -> None:
        # Auth check for non-local_trust modes
        auth_config = load_auth()
        if auth_config.auth_mode != "local_trust":
            # Localhost trust bypass with CSRF check
            if not (auth_config.trust_localhost and _is_safe_localhost_request(websocket)):
                token = websocket.cookies.get("session_token")
                session = validate_session(token) if token else None
                if not session:
                    await websocket.close(code=4001, reason="Unauthorized")
                    return

        ws_manager = websocket.app.state.ws_manager
        await ws_manager.connect(websocket)
        try:
            while True:
                data = await websocket.receive_text()
                await ws_manager.handle_client_message(websocket, data)
        except WebSocketDisconnect:
            logger.info("WebSocket client disconnected normally")
        except Exception:
            logger.warning("WebSocket connection lost unexpectedly", exc_info=True)
        finally:
            ws_manager.disconnect(websocket)

    return router
