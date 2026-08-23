from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
"""Chat routes facade. Imports from submodules and re-exports for tests."""
import asyncio
import logging
from collections.abc import AsyncIterator

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse

from core.exceptions import AnimaNotFoundError  # noqa: F401
from core.exceptions import IPCConnectionError as IPCConnError
from core.i18n import t
from server.events import emit, emit_notification
from server.routes.chat_chunk_handler import _chunk_to_event, _format_sse, _handle_chunk
from server.routes.chat_emotion import _EMOTION_PATTERN, extract_emotion
from server.routes.chat_images import _validate_images, build_content_blocks, save_images
from server.routes.chat_models import (
    MAX_CHAT_MESSAGE_SIZE,
    MAX_IMAGE_PAYLOAD_SIZE,
    MIME_TO_EXT,
    SUPPORTED_IMAGE_TYPES,
    ChatRequest,
    ChatResponse,
    ImageAttachment,
    _to_image_data,
)
from server.routes.chat_producer import _run_producer, _sse_tail
from server.routes.chat_resume import _handle_resume
from server.routes.chat_ws_effects import _emit_ws_side_effects
from server.stream_registry import StreamRegistry

logger = logging.getLogger("animaworks.routes.chat")

# Re-exports for tests and external consumers
__all__ = [
    "AnimaNotFoundError",
    "ChatRequest",
    "ChatResponse",
    "ImageAttachment",
    "MAX_CHAT_MESSAGE_SIZE",
    "MAX_IMAGE_PAYLOAD_SIZE",
    "MIME_TO_EXT",
    "SUPPORTED_IMAGE_TYPES",
    "_EMOTION_PATTERN",
    "_chunk_to_event",
    "_format_sse",
    "_handle_chunk",
    "_handle_resume",
    "_to_image_data",
    "_validate_images",
    "_emit_ws_side_effects",
    "_run_producer",
    "build_content_blocks",
    "create_chat_router",
    "extract_emotion",
    "save_images",
]


def create_chat_router() -> APIRouter:
    router = APIRouter()

    @router.post("/animas/{name}/chat")
    async def chat(name: str, body: ChatRequest, request: Request):
        # Override from_person with authenticated user
        if hasattr(request.state, "user"):
            body.from_person = request.state.user.username
        logger.info("chat_request anima=%s user=%s msg_len=%d", name, body.from_person, len(body.message))
        supervisor = request.app.state.supervisor

        # Guard: reject if anima is bootstrapping
        if supervisor.is_bootstrapping(name):
            return JSONResponse(
                {"error": t("chat.bootstrap_error")},
                status_code=503,
            )

        # Guard: reject oversized messages
        message_size = len(body.message.encode("utf-8"))
        if message_size > MAX_CHAT_MESSAGE_SIZE:
            return JSONResponse(
                {"error": t("chat.message_too_large", size_mb=message_size // 1024 // 1024)},
                status_code=413,
            )

        # Guard: validate image attachments
        if body.images:
            img_error = _validate_images(body.images)
            if img_error:
                return JSONResponse({"error": img_error}, status_code=413)

        # Save images to disk and build IPC params
        saved_paths = save_images(name, body.images) if body.images else []

        await emit(request, "anima.status", {"name": name, "status": "thinking"})

        try:
            # Send IPC request to Anima process
            result = await supervisor.send_request(
                anima_name=name,
                method="process_message",
                params={
                    "message": body.message,
                    "from_person": body.from_person,
                    "intent": body.intent,
                    "images": _to_image_data(body.images),
                    "attachment_paths": saved_paths,
                    "thread_id": body.thread_id,
                    "model": body.model,
                },
                timeout=float(
                    __import__("core.config", fromlist=["load_config"]).load_config().server.ipc_stream_timeout
                ),
            )

            response = result.get("response", "")
            clean_response, _ = extract_emotion(response)

            # Broadcast any queued notifications from this cycle
            for notif in result.get("notifications", []):
                await emit_notification(request, notif)

            await emit(request, "anima.status", {"name": name, "status": "idle"})

            logger.info("chat_response anima=%s response_len=%d", name, len(clean_response))
            images = result.get("images") or []
            cycle_result = result.get("cycle_result") or {}
            if not images and isinstance(cycle_result, dict):
                images = cycle_result.get("images") or cycle_result.get("artifacts") or []
            return ChatResponse(response=clean_response, anima=name, images=images)

        except (KeyError, AnimaNotFoundError):
            from fastapi import HTTPException

            raise HTTPException(status_code=404, detail=f"Anima not found: {name}") from None
        except (ValueError, IPCConnError) as e:
            from fastapi import HTTPException

            raise HTTPException(status_code=500, detail=str(e)) from None
        except TimeoutError:
            logger.error("Timeout waiting for chat response from anima=%s", name)
            return JSONResponse(
                {"error": "Request timed out"},
                status_code=504,
            )
        except RuntimeError as e:
            logger.exception("Runtime error in chat for anima=%s", name)
            return JSONResponse(
                {"error": f"Internal server error: {e}"},
                status_code=500,
            )

    @router.post("/animas/{name}/greet")
    async def greet(name: str, request: Request):
        """Generate a greeting when user clicks the character.

        Returns cached response if called within the 1-hour cooldown.
        Non-streaming, returns a single JSON response.
        """
        supervisor = request.app.state.supervisor

        # Guard: reject if anima is bootstrapping
        if supervisor.is_bootstrapping(name):
            return JSONResponse(
                {"error": t("chat.bootstrap_error")},
                status_code=503,
            )

        try:
            result = await supervisor.send_request(
                anima_name=name,
                method="greet",
                params={},
                timeout=60.0,
            )

            return {
                "response": result.get("response", ""),
                "emotion": result.get("emotion", "neutral"),
                "cached": result.get("cached", False),
                "anima": name,
            }

        except (KeyError, AnimaNotFoundError):
            from fastapi import HTTPException

            raise HTTPException(status_code=404, detail=f"Anima not found: {name}") from None
        except (ValueError, IPCConnError) as e:
            from fastapi import HTTPException

            raise HTTPException(status_code=500, detail=str(e)) from None
        except TimeoutError:
            logger.error("Timeout waiting for greet response from anima=%s", name)
            return JSONResponse(
                {"error": "Request timed out"},
                status_code=504,
            )
        except RuntimeError as e:
            logger.exception("Runtime error in greet for anima=%s", name)
            return JSONResponse(
                {"error": f"Internal server error: {e}"},
                status_code=500,
            )

    @router.post("/animas/{name}/chat/stream")
    async def chat_stream(name: str, body: ChatRequest, request: Request):
        """Stream chat response via SSE over IPC."""
        # Override from_person with authenticated user
        if hasattr(request.state, "user"):
            body.from_person = request.state.user.username
        logger.info("chat_stream_request anima=%s user=%s msg_len=%d", name, body.from_person, len(body.message))
        supervisor = request.app.state.supervisor

        # Verify anima exists before starting the stream
        if name not in supervisor.processes:
            from fastapi import HTTPException

            raise HTTPException(status_code=404, detail=f"Anima not found: {name}")

        # Guard: reject oversized messages
        message_size = len(body.message.encode("utf-8"))
        if message_size > MAX_CHAT_MESSAGE_SIZE:
            from fastapi import HTTPException

            raise HTTPException(
                status_code=413,
                detail=t("chat.message_too_large", size_mb=message_size // 1024 // 1024),
            )

        # Guard: validate image attachments
        if body.images:
            img_error = _validate_images(body.images)
            if img_error:
                from fastapi import HTTPException

                raise HTTPException(status_code=413, detail=img_error)

        # Save images to disk
        saved_paths = save_images(name, body.images) if body.images else []

        # Guard: return immediately if anima is bootstrapping
        if supervisor.is_bootstrapping(name):

            async def _bootstrap_busy() -> AsyncIterator[str]:
                yield _format_sse(
                    "bootstrap",
                    {
                        "status": "busy",
                        "message": t("chat.bootstrap_error"),
                    },
                )

            return StreamingResponse(
                _bootstrap_busy(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )

        # ── StreamRegistry integration ────────────────────
        registry: StreamRegistry = request.app.state.stream_registry

        # Handle resume request
        if body.resume:
            last_event_id = body.last_event_id or request.headers.get("Last-Event-ID", "")
            return _handle_resume(registry, body.resume, last_event_id, name, from_person=body.from_person)

        stream = registry.register(
            name,
            from_person=body.from_person,
            thread_id=body.thread_id,
        )

        # Launch producer task (runs in background, independent of SSE)
        ws_manager = getattr(request.app.state, "ws_manager", None)
        task = asyncio.create_task(
            _run_producer(
                stream,
                registry,
                supervisor,
                ws_manager,
                name=name,
                body=body,
                saved_paths=saved_paths,
            ),
            name=f"producer-{stream.response_id}",
        )
        registry.set_producer_task(stream.response_id, task)

        return StreamingResponse(
            _sse_tail(stream, request),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    @router.get("/animas/{name}/stream/active")
    async def get_active_stream(name: str, request: Request):
        """Return the active (or most recent) stream for an anima."""
        registry: StreamRegistry = request.app.state.stream_registry
        thread_id = request.query_params.get("thread_id")
        stream = registry.get_active(name, thread_id=thread_id)
        if stream is None:
            return JSONResponse({"active": False}, status_code=200)
        return {
            "active": True,
            "response_id": stream.response_id,
            "status": stream.status,
            "full_text": stream.full_text,
            "active_tool": stream.active_tool,
            "tool_history": stream.tool_history,
            "last_event_id": stream.last_event_id,
            "event_count": stream.event_count,
            "emotion": stream.emotion,
        }

    @router.get("/animas/{name}/stream/{response_id}/progress")
    async def get_stream_progress(name: str, response_id: str, request: Request):
        """Return progress of a specific stream."""
        registry: StreamRegistry = request.app.state.stream_registry
        stream = registry.get(response_id)
        if stream is None or stream.anima_name != name:
            return JSONResponse(
                {"error": "Stream not found"},
                status_code=404,
            )
        return {
            "response_id": stream.response_id,
            "anima_name": stream.anima_name,
            "status": stream.status,
            "full_text": stream.full_text,
            "active_tool": stream.active_tool,
            "tool_history": stream.tool_history,
            "last_event_id": stream.last_event_id,
            "event_count": stream.event_count,
            "emotion": stream.emotion,
        }

    return router
