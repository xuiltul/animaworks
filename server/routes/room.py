# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
"""Meeting room API routes with SSE streaming."""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, field_validator

from core.config import load_config
from core.exceptions import AnimaNotFoundError
from core.exceptions import IPCConnectionError as IPCConnError
from core.execution.base import strip_thinking_tags
from core.paths import get_common_knowledge_dir
from server.room_manager import MeetingRoom, RoomManager
from server.routes.chat_chunk_handler import _chunk_to_event, _format_sse
from server.routes.chat_emotion import extract_emotion

logger = logging.getLogger(__name__)

# ── Pydantic Models ──────────────────────────────────────────


class CreateRoomRequest(BaseModel):
    """Request body for creating a meeting room."""

    participants: list[str]
    chair: str
    title: str = ""

    @field_validator("participants")
    @classmethod
    def validate_participants(cls, v: list[str]) -> list[str]:
        if len(v) > 5:
            raise ValueError("Maximum 5 participants")
        if len(v) < 1:
            raise ValueError("At least 1 participant required")
        return v


class MeetingChatRequest(BaseModel):
    """Request body for meeting chat stream."""

    message: str
    from_person: str = "human"


class AddParticipantRequest(BaseModel):
    """Request body for adding a participant."""

    name: str


# ── Helpers ─────────────────────────────────────────────────


def _get_room_manager(request: Request) -> RoomManager:
    """Get RoomManager from app state. Raises 503 if not available."""
    room_manager = getattr(request.app.state, "room_manager", None)
    if room_manager is None:
        raise HTTPException(
            status_code=503,
            detail="Meeting room feature not available",
        )
    return room_manager


def _get_created_by(request: Request) -> str:
    """Get created_by from authenticated user or default to 'human'."""
    if hasattr(request.state, "user"):
        return request.state.user.username
    return "human"


async def _build_message_for_anima(
    room_manager: RoomManager,
    room: MeetingRoom,
    target_name: str,
    human_message: str,
    *,
    is_chair: bool,
) -> str:
    """Build the message content sent to an Anima in the meeting context.

    Includes meeting context (conversation history) and role-specific prompt.
    """
    context = await room_manager.get_summarized_context(room.room_id)
    if is_chair:
        chair_prompt = room_manager.build_chair_prompt(room)
        return f"""{chair_prompt}

## 会議の流れ

{context or "(まだ発言なし)"}

上記の会議の流れを踏まえて、議長として応答してください。"""
    else:
        return f"""あなたは会議に参加しています。以下の会議の流れを踏まえて意見を述べてください。

## 会議の流れ

{context or "(まだ発言なし)"}

上記の会議の流れに対して、あなたの意見を述べてください。"""


async def _meeting_stream(
    room_id: str,
    message: str,
    from_person: str,
    room_manager: RoomManager,
    supervisor: Any,
) -> AsyncIterator[str]:
    """Async generator yielding SSE events for meeting chat round."""
    room = room_manager.get_room(room_id)
    if room is None:
        yield _format_sse("error", {"code": "ROOM_NOT_FOUND", "message": "Room not found"})
        return
    if room.closed:
        yield _format_sse("error", {"code": "ROOM_CLOSED", "message": "Room is closed"})
        return

    # Append human message to conversation
    room_manager.append_message(room_id, from_person, "human", message)

    # Determine targets: @mentions or chair
    mentions = room_manager.extract_mentions(message, room.participants)
    if mentions:
        targets = [t for t in mentions if t in room.participants]
    else:
        targets = [room.chair] if room.chair in room.participants else []

    if not targets:
        yield _format_sse("done", {"summary": "No targets to respond"})
        return

    # Verify all targets exist in supervisor
    for t in targets:
        if t not in supervisor.processes:
            yield _format_sse(
                "error",
                {"code": "ANIMA_NOT_FOUND", "message": f"Anima not found: {t}"},
            )
            return

    try:
        _config = load_config()
        _timeout = float(_config.server.ipc_stream_timeout)
    except Exception:
        _timeout = 60.0

    queue: list[str] = list(targets)
    prev_speaker = from_person

    while queue:
        target_name = queue.pop(0)
        is_chair = target_name == room.chair

        # Build message for this target
        msg_content = await _build_message_for_anima(
            room_manager,
            room,
            target_name,
            message,
            is_chair=is_chair,
        )

        params: dict[str, Any] = {
            "message": msg_content,
            "from_person": prev_speaker,
            "stream": True,
            "source": "meeting",
            "thread_id": f"meeting-{room_id}",
        }

        # Yield speaker_start
        role = "chair" if is_chair else "participant"
        yield _format_sse("speaker_start", {"speaker": target_name, "role": role})

        full_response = ""

        try:
            async for ipc_response in supervisor.send_request_stream(
                anima_name=target_name,
                method="process_message",
                params=params,
                timeout=_timeout,
            ):
                if ipc_response.done:
                    result = ipc_response.result or {}
                    full_response = result.get("response", "")
                    cycle_result = result.get("cycle_result", {})
                    if cycle_result and not full_response:
                        full_response = cycle_result.get("summary", "")
                    break

                if ipc_response.chunk:
                    try:
                        chunk_data = json.loads(ipc_response.chunk)
                        if chunk_data.get("type") == "keepalive":
                            continue
                        result = _chunk_to_event(chunk_data)
                        if result:
                            evt_name, evt_payload = result
                            # Don't yield "done" from cycle_done — we yield our own at the end
                            if evt_name == "done":
                                full_response = evt_payload.get("summary", full_response) or full_response
                                continue
                            evt_payload = dict(evt_payload)
                            evt_payload["speaker"] = target_name
                            yield _format_sse(evt_name, evt_payload)
                            if evt_name == "text_delta":
                                full_response += evt_payload.get("text", "")
                    except json.JSONDecodeError:
                        yield _format_sse(
                            "text_delta",
                            {"text": ipc_response.chunk, "speaker": target_name},
                        )
                        full_response += ipc_response.chunk

                if ipc_response.result and not full_response:
                    full_response = ipc_response.result.get("response", "")

        except (AnimaNotFoundError, IPCConnError, TimeoutError) as e:
            logger.warning("Meeting stream error for %s: %s", target_name, e)
            yield _format_sse(
                "error",
                {"code": "STREAM_ERROR", "message": str(e), "speaker": target_name},
            )
            full_response = ""

        # Clean response for storage
        clean_text, _ = extract_emotion(full_response)
        leaked, clean_text = strip_thinking_tags(clean_text)
        if leaked:
            clean_text = clean_text.strip()

        # Append to conversation
        room_manager.append_message(
            room_id,
            target_name,
            "chair" if is_chair else "participant",
            clean_text,
        )

        # Yield speaker_end
        yield _format_sse("speaker_end", {"speaker": target_name})

        prev_speaker = target_name

        # After chair responds: extract @mentions and add as next targets
        if is_chair and clean_text:
            new_mentions = room_manager.extract_mentions(clean_text, room.participants)
            for m in new_mentions:
                if m != target_name and m in room.participants and m not in queue:
                    queue.append(m)

    yield _format_sse("done", {"summary": "Meeting round complete"})


# ── Router ───────────────────────────────────────────────────


def create_room_router() -> APIRouter:
    """Create the meeting room API router."""
    router = APIRouter(prefix="/rooms", tags=["rooms"])

    @router.post("")
    async def create_room(body: CreateRoomRequest, request: Request):
        """Create a new meeting room."""
        room_manager = _get_room_manager(request)
        created_by = _get_created_by(request)
        try:
            room = room_manager.create_room(
                participants=body.participants,
                chair=body.chair,
                created_by=created_by,
                title=body.title,
            )
            return {
                "room_id": room.room_id,
                "participants": room.participants,
                "chair": room.chair,
                "title": room.title,
                "created_at": room.created_at.isoformat(),
                "closed": room.closed,
            }
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from None

    @router.get("")
    async def list_rooms(request: Request, include_closed: bool = False):
        """List meeting rooms."""
        room_manager = _get_room_manager(request)
        rooms = room_manager.list_rooms(include_closed=include_closed)
        return [
            {
                "room_id": r.room_id,
                "participants": r.participants,
                "chair": r.chair,
                "title": r.title,
                "created_at": r.created_at.isoformat(),
                "closed": r.closed,
                "closed_at": r.closed_at.isoformat() if r.closed_at else None,
            }
            for r in rooms
        ]

    @router.get("/{room_id}")
    async def get_room(room_id: str, request: Request):
        """Get room details including conversation."""
        room_manager = _get_room_manager(request)
        room = room_manager.get_room(room_id)
        if room is None:
            raise HTTPException(status_code=404, detail="Room not found")
        return {
            "room_id": room.room_id,
            "participants": room.participants,
            "chair": room.chair,
            "title": room.title,
            "created_at": room.created_at.isoformat(),
            "closed": room.closed,
            "closed_at": room.closed_at.isoformat() if room.closed_at else None,
            "conversation": room.conversation,
        }

    @router.post("/{room_id}/participants")
    async def add_participant(room_id: str, body: AddParticipantRequest, request: Request):
        """Add a participant to the room."""
        room_manager = _get_room_manager(request)
        try:
            room_manager.add_participant(room_id, body.name)
            return {"ok": True}
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from None

    @router.delete("/{room_id}/participants/{name}")
    async def remove_participant(room_id: str, name: str, request: Request):
        """Remove a participant from the room."""
        room_manager = _get_room_manager(request)
        try:
            room_manager.remove_participant(room_id, name)
            return {"ok": True}
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from None

    @router.post("/{room_id}/close")
    async def close_room(room_id: str, request: Request):
        """Close room and generate minutes."""
        room_manager = _get_room_manager(request)
        try:
            room_manager.close_room(room_id)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from None

        # Generate minutes
        try:
            common_knowledge_dir = get_common_knowledge_dir()
            minutes_path = await room_manager.generate_minutes(room_id, common_knowledge_dir)
            return {
                "ok": True,
                "minutes_path": str(minutes_path) if minutes_path else None,
            }
        except Exception as e:
            logger.warning("Failed to generate minutes for room %s: %s", room_id, e)
            return {"ok": True, "minutes_path": None}

    @router.post("/{room_id}/chat/stream")
    async def meeting_chat_stream(
        room_id: str,
        body: MeetingChatRequest,
        request: Request,
    ):
        """Main SSE streaming endpoint for meeting chat."""
        room_manager = _get_room_manager(request)
        supervisor = request.app.state.supervisor

        from_person = body.from_person
        if hasattr(request.state, "user"):
            from_person = request.state.user.username

        return StreamingResponse(
            _meeting_stream(
                room_id,
                body.message,
                from_person,
                room_manager,
                supervisor,
            ),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    return router
