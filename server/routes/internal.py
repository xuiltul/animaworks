from __future__ import annotations
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

import json
import logging
from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from server.events import emit

logger = logging.getLogger("animaworks.routes.internal")


class MessageSentNotification(BaseModel):
    from_person: str
    to_person: str
    content: str = ""
    message_id: str = ""


def create_internal_router() -> APIRouter:
    router = APIRouter()

    @router.post("/internal/message-sent")
    async def internal_message_sent(
        body: MessageSentNotification, request: Request
    ):
        """Notify the server that a message was sent via CLI.

        Triggers WebSocket broadcast and updates reply tracking so that
        selective archival (Fix 2) works for CLI-sent messages too.
        """
        await emit(request, "anima.interaction", {
            "from_person": body.from_person,
            "to_person": body.to_person,
            "type": "message",
            "summary": body.content[:200],
            "message_id": body.message_id,
        })

        # Note: replied_to tracking is now managed by each Anima process.
        # The server no longer holds live DigitalAnima objects.

        return {"status": "ok"}

    @router.get("/messages/{message_id}")
    async def get_message(message_id: str, request: Request):
        """Return the full JSON of a stored message by its ID."""
        # Sanitize to prevent path traversal
        if "/" in message_id or "\\" in message_id or ".." in message_id:
            return JSONResponse(
                status_code=400,
                content={"detail": "Invalid message_id"},
            )

        shared_dir: Path = request.app.state.shared_dir
        inbox_root = shared_dir / "inbox"
        if not inbox_root.is_dir():
            return JSONResponse(
                status_code=404,
                content={"detail": "Message not found"},
            )

        filename = f"{message_id}.json"
        for anima_inbox in sorted(inbox_root.iterdir()):
            if not anima_inbox.is_dir():
                continue
            # Check processed first, then inbox root
            for candidate in [
                anima_inbox / "processed" / filename,
                anima_inbox / filename,
            ]:
                if candidate.is_file():
                    data = json.loads(candidate.read_text(encoding="utf-8"))
                    return data

        return JSONResponse(
            status_code=404,
            content={"detail": "Message not found"},
        )

    return router
