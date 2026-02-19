from __future__ import annotations
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

import asyncio
import base64
import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any, AsyncIterator

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from server.dependencies import get_anima
from server.events import emit, emit_notification
from server.stream_registry import StreamRegistry

logger = logging.getLogger("animaworks.routes.chat")

MAX_CHAT_MESSAGE_SIZE = 10 * 1024 * 1024  # 10MB
MAX_IMAGE_PAYLOAD_SIZE = 20 * 1024 * 1024  # 20MB total base64

SUPPORTED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/gif", "image/webp"}
MIME_TO_EXT = {
    "image/jpeg": "jpeg",
    "image/png": "png",
    "image/gif": "gif",
    "image/webp": "webp",
}


class ImageAttachment(BaseModel):
    """A single base64-encoded image attachment."""

    data: str  # Base64 encoded string (no data: prefix)
    media_type: str  # "image/jpeg", "image/png", "image/gif", "image/webp"


class ChatRequest(BaseModel):
    message: str
    from_person: str = "human"
    images: list[ImageAttachment] = []
    resume: str | None = None
    last_event_id: str | None = None


class ChatResponse(BaseModel):
    response: str
    anima: str


# ── Image Helpers ─────────────────────────────────────────────

def _validate_images(images: list[ImageAttachment]) -> str | None:
    """Validate image attachments. Returns error message or None."""
    if not images:
        return None
    total_size = sum(len(img.data) for img in images)
    if total_size > MAX_IMAGE_PAYLOAD_SIZE:
        return f"画像データが大きすぎます（{total_size // 1024 // 1024}MB / 上限20MB）"
    for img in images:
        if img.media_type not in SUPPORTED_IMAGE_TYPES:
            return f"未対応の画像形式です: {img.media_type}"
    return None


def save_images(anima_name: str, images: list[ImageAttachment]) -> list[str]:
    """Save base64 images to ~/.animaworks/animas/{name}/attachments/.

    Returns:
        List of relative paths (e.g. ``attachments/20260217_120000_0.jpeg``).
    """
    if not images:
        return []
    from core.paths import get_data_dir

    attachments_dir = get_data_dir() / "animas" / anima_name / "attachments"
    attachments_dir.mkdir(parents=True, exist_ok=True)

    paths: list[str] = []
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    for i, img in enumerate(images):
        ext = MIME_TO_EXT.get(img.media_type, "png")
        filename = f"{ts}_{i}.{ext}"
        filepath = attachments_dir / filename
        filepath.write_bytes(base64.b64decode(img.data))
        paths.append(f"attachments/{filename}")
    return paths


def build_content_blocks(
    message: str, images: list[ImageAttachment],
) -> str | list[dict[str, Any]]:
    """Convert text + images to LLM content blocks.

    Returns plain string if no images are present.
    """
    if not images:
        return message
    blocks: list[dict[str, Any]] = []
    for img in images:
        blocks.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": img.media_type,
                "data": img.data,
            },
        })
    if message.strip():
        blocks.append({"type": "text", "text": message})
    return blocks


# ── SSE Helpers ───────────────────────────────────────────────

def _format_sse(event: str, payload: dict[str, Any]) -> str:
    """Format a single SSE frame."""
    data = json.dumps(payload, ensure_ascii=False, default=str)
    return f"event: {event}\ndata: {data}\n\n"



# ── Emotion Extraction ────────────────────────────────────

from core.schemas import VALID_EMOTIONS

_EMOTION_PATTERN = re.compile(
    r'<!--\s*emotion:\s*(\{.*?\})\s*-->', re.DOTALL,
)


def extract_emotion(response_text: str) -> tuple[str, str]:
    """Extract emotion metadata from LLM response text.

    The LLM appends ``<!-- emotion: {"emotion": "smile"} -->`` to its
    response.  This function strips the tag and returns the clean text
    plus the emotion name.

    Returns:
        (clean_text, emotion) where *emotion* falls back to ``"neutral"``
        when missing or invalid.
    """
    match = _EMOTION_PATTERN.search(response_text)
    if not match:
        return response_text, "neutral"

    clean_text = _EMOTION_PATTERN.sub("", response_text).rstrip()

    try:
        meta = json.loads(match.group(1))
        emotion = meta.get("emotion", "neutral")
        if emotion not in VALID_EMOTIONS:
            emotion = "neutral"
        return clean_text, emotion
    except (json.JSONDecodeError, AttributeError):
        return clean_text, "neutral"


def _handle_chunk(
    chunk: dict[str, Any],
    *,
    request: Request | None = None,
    anima_name: str | None = None,
) -> tuple[str | None, str]:
    """Map a stream chunk to an SSE event name and extract response text.

    Args:
        chunk: Stream chunk dictionary.
        request: FastAPI Request (optional, for emitting WebSocket events).
        anima_name: Anima name (optional, for WebSocket event data).

    Returns:
        Tuple of (sse_frame_or_None, accumulated_response_text).
    """
    event_type = chunk.get("type", "unknown")

    if event_type == "text_delta":
        return _format_sse("text_delta", {"text": chunk["text"]}), ""

    if event_type == "tool_start":
        return _format_sse("tool_start", {
            "tool_name": chunk["tool_name"],
            "tool_id": chunk["tool_id"],
        }), ""

    if event_type == "tool_end":
        return _format_sse("tool_end", {
            "tool_id": chunk["tool_id"],
            "tool_name": chunk.get("tool_name", ""),
        }), ""

    if event_type == "chain_start":
        return _format_sse("chain_start", {"chain": chunk["chain"]}), ""

    if event_type == "bootstrap_start":
        if request and anima_name:
            import asyncio
            asyncio.ensure_future(emit(
                request, "anima.bootstrap",
                {"name": anima_name, "status": "started"},
            ))
        return _format_sse("bootstrap", {"status": "started"}), ""

    if event_type == "bootstrap_complete":
        if request and anima_name:
            import asyncio
            asyncio.ensure_future(emit(
                request, "anima.bootstrap",
                {"name": anima_name, "status": "completed"},
            ))
        return _format_sse("bootstrap", {"status": "completed"}), ""

    if event_type == "bootstrap_busy":
        return _format_sse("bootstrap", {
            "status": "busy",
            "message": chunk.get("message", "初期化中です"),
        }), ""

    if event_type == "heartbeat_relay_start":
        return _format_sse("heartbeat_relay_start", {
            "message": chunk.get("message", "処理中です"),
        }), ""

    if event_type == "heartbeat_relay":
        return _format_sse("heartbeat_relay", {
            "text": chunk.get("text", ""),
        }), chunk.get("text", "")

    if event_type == "heartbeat_relay_done":
        return _format_sse("heartbeat_relay_done", {}), ""

    if event_type == "notification_sent":
        # Broadcast notification to all WebSocket clients (with queue support)
        if request:
            import asyncio
            notif_data = chunk.get("data", {})
            asyncio.ensure_future(
                emit_notification(request, notif_data)
            )
        return None, ""

    if event_type == "cycle_done":
        cycle_result = chunk.get("cycle_result", {})
        response_text = cycle_result.get("summary", "")
        # Extract emotion from response and include in done event
        clean_text, emotion = extract_emotion(response_text)
        cycle_result["summary"] = clean_text
        cycle_result["emotion"] = emotion
        return _format_sse("done", cycle_result), clean_text

    if event_type == "error":
        error_payload: dict[str, Any] = {
            "message": chunk.get("message", "Unknown error"),
        }
        if "code" in chunk:
            error_payload["code"] = chunk["code"]
        return _format_sse("error", error_payload), ""

    return None, ""


def _chunk_to_event(chunk: dict[str, Any]) -> tuple[str, dict[str, Any]] | None:
    """Extract SSE event name and payload from a stream chunk."""
    event_type = chunk.get("type", "unknown")
    if event_type == "text_delta":
        return "text_delta", {"text": chunk["text"]}
    if event_type == "tool_start":
        return "tool_start", {"tool_name": chunk["tool_name"], "tool_id": chunk["tool_id"]}
    if event_type == "tool_end":
        return "tool_end", {"tool_id": chunk["tool_id"], "tool_name": chunk.get("tool_name", "")}
    if event_type == "chain_start":
        return "chain_start", {"chain": chunk["chain"]}
    if event_type == "bootstrap_start":
        return "bootstrap", {"status": "started"}
    if event_type == "bootstrap_complete":
        return "bootstrap", {"status": "completed"}
    if event_type == "bootstrap_busy":
        return "bootstrap", {"status": "busy", "message": chunk.get("message", "初期化中です")}
    if event_type == "heartbeat_relay_start":
        return "heartbeat_relay_start", {"message": chunk.get("message", "処理中です")}
    if event_type == "heartbeat_relay":
        return "heartbeat_relay", {"text": chunk.get("text", "")}
    if event_type == "heartbeat_relay_done":
        return "heartbeat_relay_done", {}
    if event_type == "cycle_done":
        cycle_result = chunk.get("cycle_result", {})
        response_text = cycle_result.get("summary", "")
        clean_text, emotion = extract_emotion(response_text)
        cycle_result["summary"] = clean_text
        cycle_result["emotion"] = emotion
        return "done", cycle_result
    if event_type == "error":
        payload = {"message": chunk.get("message", "Unknown error")}
        if "code" in chunk:
            payload["code"] = chunk["code"]
        return "error", payload
    return None


async def _stream_events(
    anima: Any,
    name: str,
    body: ChatRequest,
    request: Request,
    *,
    images: list[dict[str, Any]] | None = None,
    attachment_paths: list[str] | None = None,
) -> AsyncIterator[str]:
    """Async generator that yields SSE frames for a streaming chat session."""
    full_response = ""
    try:
        await emit(request, "anima.status", {"name": name, "status": "thinking"})

        async for chunk in anima.process_message_stream(
            body.message, from_person=body.from_person,
            images=images, attachment_paths=attachment_paths,
        ):
            frame, response_text = _handle_chunk(
                chunk, request=request, anima_name=name,
            )
            if response_text:
                full_response = response_text
            if frame:
                yield frame

    except Exception:
        logger.exception("SSE stream error for anima=%s", name)
        yield _format_sse("error", {"code": "STREAM_ERROR", "message": "Internal server error"})

    finally:
        await emit(request, "anima.status", {"name": name, "status": "idle"})


# ── Router ────────────────────────────────────────────────────

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
                {"error": "現在キャラクターを作成中です。完了までお待ちください。"},
                status_code=503,
            )

        # Guard: reject oversized messages
        message_size = len(body.message.encode("utf-8"))
        if message_size > MAX_CHAT_MESSAGE_SIZE:
            return JSONResponse(
                {"error": f"メッセージが大きすぎます（{message_size // 1024 // 1024}MB / 上限10MB）"},
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
                    "images": [img.model_dump() for img in body.images] if body.images else [],
                    "attachment_paths": saved_paths,
                },
                timeout=60.0
            )

            response = result.get("response", "")
            clean_response, _ = extract_emotion(response)

            # Broadcast any queued notifications from this cycle
            for notif in result.get("notifications", []):
                await emit_notification(request, notif)

            await emit(request, "anima.status", {"name": name, "status": "idle"})

            logger.info("chat_response anima=%s response_len=%d", name, len(clean_response))
            return ChatResponse(response=clean_response, anima=name)

        except KeyError:
            from fastapi import HTTPException
            raise HTTPException(status_code=404, detail=f"Anima not found: {name}")
        except ValueError as e:
            from fastapi import HTTPException
            raise HTTPException(status_code=500, detail=str(e))
        except asyncio.TimeoutError:
            logger.error("Timeout waiting for chat response from anima=%s", name)
            return JSONResponse(
                {"error": "Request timed out"}, status_code=504,
            )
        except RuntimeError as e:
            logger.exception("Runtime error in chat for anima=%s", name)
            return JSONResponse(
                {"error": f"Internal server error: {e}"}, status_code=500,
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
                {"error": "現在キャラクターを作成中です。完了までお待ちください。"},
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

        except KeyError:
            from fastapi import HTTPException
            raise HTTPException(status_code=404, detail=f"Anima not found: {name}")
        except ValueError as e:
            from fastapi import HTTPException
            raise HTTPException(status_code=500, detail=str(e))
        except asyncio.TimeoutError:
            logger.error("Timeout waiting for greet response from anima=%s", name)
            return JSONResponse(
                {"error": "Request timed out"}, status_code=504,
            )
        except RuntimeError as e:
            logger.exception("Runtime error in greet for anima=%s", name)
            return JSONResponse(
                {"error": f"Internal server error: {e}"}, status_code=500,
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
                detail=f"メッセージが大きすぎます（{message_size // 1024 // 1024}MB / 上限10MB）",
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
                yield _format_sse("bootstrap", {
                    "status": "busy",
                    "message": "現在キャラクターを作成中です。完了までお待ちください。",
                })

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

        stream = registry.register(name, from_person=body.from_person)

        async def _ipc_stream_events() -> AsyncIterator[str]:
            """Async generator that converts IPC stream to SSE frames."""
            full_response = ""
            stream_done = False
            ipc_chunk_count = 0
            sse_frame_count = 0
            keepalive_count = 0
            import time as _time
            _stream_start_time = _time.monotonic()
            try:
                logger.info(
                    "[SSE-STREAM] start anima=%s stream=%s user=%s",
                    name, stream.response_id, body.from_person,
                )
                await emit(request, "anima.status", {"name": name, "status": "thinking"})

                # Emit stream_start with response_id for client tracking
                sse_frame_count += 1
                yield registry.format_sse(stream, "stream_start", {"response_id": stream.response_id})

                from core.config import load_config
                _config = load_config()
                _timeout = float(_config.server.ipc_stream_timeout)

                logger.info(
                    "[SSE-STREAM] starting IPC stream anima=%s timeout=%.1f",
                    name, _timeout,
                )
                async for ipc_response in supervisor.send_request_stream(
                    anima_name=name,
                    method="process_message",
                    params={
                        "message": body.message,
                        "from_person": body.from_person,
                        "stream": True,
                        "images": [img.model_dump() for img in body.images] if body.images else [],
                        "attachment_paths": saved_paths,
                    },
                    timeout=_timeout,
                ):
                    ipc_chunk_count += 1
                    if ipc_response.done:
                        # Final response with full result
                        result = ipc_response.result or {}
                        full_response = result.get("response", full_response)
                        cycle_result = result.get("cycle_result", {})
                        # Extract emotion from response
                        summary = cycle_result.get("summary", full_response)
                        clean_text, emotion = extract_emotion(summary)
                        cycle_result["summary"] = clean_text
                        cycle_result["emotion"] = emotion
                        full_response = clean_text
                        sse_frame_count += 1
                        elapsed = _time.monotonic() - _stream_start_time
                        logger.info(
                            "[SSE-STREAM] IPC done anima=%s stream=%s ipc_chunks=%d sse_frames=%d keepalives=%d elapsed=%.1fs response_len=%d",
                            name, stream.response_id, ipc_chunk_count, sse_frame_count,
                            keepalive_count, elapsed, len(clean_text),
                        )
                        yield registry.format_sse(stream, "done", cycle_result or {"summary": clean_text, "emotion": emotion})
                        stream_done = True
                        break

                    if ipc_response.chunk:
                        # Parse the chunk JSON from the IPC layer
                        try:
                            chunk_data = json.loads(ipc_response.chunk)

                            # Keep-alive chunks → SSE comment (invisible to client parser)
                            if chunk_data.get("type") == "keepalive":
                                keepalive_count += 1
                                elapsed = _time.monotonic() - _stream_start_time
                                logger.info(
                                    "[SSE-KEEPALIVE] anima=%s stream=%s keepalive#%d elapsed=%.1fs",
                                    name, stream.response_id, keepalive_count, elapsed,
                                )
                                yield ": keepalive\n\n"
                                continue

                            # Side effects (WebSocket emissions)
                            _, response_text = _handle_chunk(chunk_data, request=request, anima_name=name)
                            if response_text:
                                full_response = response_text
                            # Format SSE with event ID via registry
                            result = _chunk_to_event(chunk_data)
                            if result:
                                evt_name, evt_payload = result
                                if evt_name == "done":
                                    full_response = evt_payload.get("summary", full_response)
                                    stream_done = True
                                sse_frame_count += 1
                                if evt_name != "text_delta":
                                    logger.info(
                                        "[SSE-STREAM] yield event=%s anima=%s stream=%s frame#%d",
                                        evt_name, name, stream.response_id, sse_frame_count,
                                    )
                                yield registry.format_sse(stream, evt_name, evt_payload)
                        except json.JSONDecodeError:
                            # Raw text chunk fallback
                            full_response += ipc_response.chunk
                            yield registry.format_sse(stream, "text_delta", {"text": ipc_response.chunk})
                        continue

                    # Fallback: non-streaming IPC response (result without done flag)
                    if ipc_response.result:
                        result = ipc_response.result
                        full_response = result.get("response", "")
                        cycle_result = result.get("cycle_result", {})
                        summary = cycle_result.get("summary", full_response)
                        clean_text, emotion = extract_emotion(summary)
                        cycle_result["summary"] = clean_text
                        cycle_result["emotion"] = emotion
                        full_response = clean_text
                        yield registry.format_sse(stream, "done", cycle_result or {"summary": clean_text, "emotion": emotion})
                        stream_done = True
                        break

                # Fallback: stream ended without done event
                if not stream_done:
                    elapsed = _time.monotonic() - _stream_start_time
                    logger.warning(
                        "[SSE-STREAM] INCOMPLETE anima=%s stream=%s ipc_chunks=%d sse_frames=%d elapsed=%.1fs",
                        name, stream.response_id, ipc_chunk_count, sse_frame_count, elapsed,
                    )
                    yield registry.format_sse(stream, "error", {
                        "code": "STREAM_INCOMPLETE",
                        "message": "ストリームが予期せず終了しました。再試行してください。",
                    })

            except RuntimeError as e:
                if "IPC protocol error" in str(e):
                    elapsed = _time.monotonic() - _stream_start_time
                    logger.error(
                        "[SSE-STREAM] IPC_PROTOCOL_ERROR anima=%s stream=%s elapsed=%.1fs error=%s",
                        name, stream.response_id, elapsed, e,
                    )
                    yield registry.format_sse(stream, "error", {
                        "code": "IPC_PROTOCOL_ERROR",
                        "message": "通信エラーが発生しました。再試行してください。",
                    })
                else:
                    elapsed = _time.monotonic() - _stream_start_time
                    logger.exception(
                        "[SSE-STREAM] RUNTIME_ERROR anima=%s stream=%s elapsed=%.1fs error=%s",
                        name, stream.response_id, elapsed, e,
                    )
                    yield registry.format_sse(stream, "error", {"code": "STREAM_ERROR", "message": "Internal server error"})
            except ValueError as e:
                elapsed = _time.monotonic() - _stream_start_time
                logger.error(
                    "[SSE-STREAM] IPC_ERROR anima=%s stream=%s elapsed=%.1fs error=%s",
                    name, stream.response_id, elapsed, e,
                )
                yield registry.format_sse(stream, "error", {"code": "IPC_ERROR", "message": str(e)})
            except KeyError:
                elapsed = _time.monotonic() - _stream_start_time
                logger.error(
                    "[SSE-STREAM] ANIMA_NOT_FOUND anima=%s stream=%s elapsed=%.1fs",
                    name, stream.response_id, elapsed,
                )
                yield registry.format_sse(stream, "error", {"code": "ANIMA_NOT_FOUND", "message": f"Anima not found: {name}"})
            except TimeoutError:
                elapsed = _time.monotonic() - _stream_start_time
                logger.error(
                    "[SSE-STREAM] IPC_TIMEOUT anima=%s stream=%s elapsed=%.1fs ipc_chunks=%d",
                    name, stream.response_id, elapsed, ipc_chunk_count,
                )
                yield registry.format_sse(stream, "error", {"code": "IPC_TIMEOUT", "message": "応答がタイムアウトしました"})
            except Exception:
                elapsed = _time.monotonic() - _stream_start_time
                logger.exception(
                    "[SSE-STREAM] STREAM_ERROR anima=%s stream=%s elapsed=%.1fs ipc_chunks=%d",
                    name, stream.response_id, elapsed, ipc_chunk_count,
                )
                yield registry.format_sse(stream, "error", {"code": "STREAM_ERROR", "message": "Internal server error"})
            finally:
                elapsed = _time.monotonic() - _stream_start_time
                logger.info(
                    "[SSE-STREAM] finalize anima=%s stream=%s ipc_chunks=%d sse_frames=%d keepalives=%d elapsed=%.1fs done=%s",
                    name, stream.response_id, ipc_chunk_count, sse_frame_count,
                    keepalive_count, elapsed, stream_done,
                )
                registry.mark_complete(stream.response_id)
                await emit(request, "anima.status", {"name": name, "status": "idle"})

        return StreamingResponse(
            _ipc_stream_events(),
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
        stream = registry.get_active(name)
        if stream is None:
            return JSONResponse({"active": False}, status_code=200)
        return {
            "active": True,
            "response_id": stream.response_id,
            "status": stream.status,
            "full_text": stream.full_text,
            "active_tool": stream.active_tool,
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
                {"error": "Stream not found"}, status_code=404,
            )
        return {
            "response_id": stream.response_id,
            "anima_name": stream.anima_name,
            "status": stream.status,
            "full_text": stream.full_text,
            "active_tool": stream.active_tool,
            "last_event_id": stream.last_event_id,
            "event_count": stream.event_count,
            "emotion": stream.emotion,
        }

    return router


def _handle_resume(
    registry: StreamRegistry,
    resume_id: str,
    last_event_id: str,
    anima_name: str,
    *,
    from_person: str = "human",
):
    """Handle SSE stream resume request."""
    logger.info(
        "[SSE-RESUME] request stream=%s anima=%s last_event_id=%s from=%s",
        resume_id, anima_name, last_event_id, from_person,
    )
    stream = registry.get(resume_id)
    if stream is None or stream.anima_name != anima_name or stream.from_person != from_person:
        logger.info(
            "[SSE-RESUME] NOT_FOUND stream=%s (exists=%s anima_match=%s from_match=%s)",
            resume_id,
            stream is not None,
            stream.anima_name == anima_name if stream else "N/A",
            stream.from_person == from_person if stream else "N/A",
        )
        async def _not_found():
            yield _format_sse("error", {"code": "STREAM_NOT_FOUND", "message": "Stream not found or access denied"})
        return StreamingResponse(
            _not_found(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
        )

    # Parse last event ID to get sequence number
    after_seq = -1
    if last_event_id and ":" in last_event_id:
        try:
            after_seq = int(last_event_id.rsplit(":", 1)[1])
        except (ValueError, IndexError):
            pass

    logger.info(
        "[SSE-RESUME] replaying stream=%s after_seq=%d complete=%s total_events=%d",
        resume_id, after_seq, stream.complete, stream.event_count,
    )

    async def _replay_events():
        from server.stream_registry import format_sse_with_id
        current_seq = after_seq
        replay_count = 0
        for event in stream.events_after(after_seq):
            yield format_sse_with_id(event.event, event.payload, event.event_id)
            current_seq = event.seq
            replay_count += 1

        logger.info(
            "[SSE-RESUME] replayed stream=%s count=%d current_seq=%d complete=%s",
            resume_id, replay_count, current_seq, stream.complete,
        )

        if not stream.complete:
            wait_count = 0
            while not stream.complete:
                got_event = await stream.wait_new_event(timeout=30.0)
                if not got_event:
                    wait_count += 1
                    logger.info(
                        "[SSE-RESUME] keepalive stream=%s wait#%d current_seq=%d",
                        resume_id, wait_count, current_seq,
                    )
                    yield ": keepalive\n\n"
                    continue
                new_events = stream.events_after(current_seq)
                for event in new_events:
                    yield format_sse_with_id(event.event, event.payload, event.event_id)
                    current_seq = event.seq
                logger.info(
                    "[SSE-RESUME] new_events stream=%s count=%d current_seq=%d",
                    resume_id, len(new_events), current_seq,
                )
        logger.info("[SSE-RESUME] done stream=%s final_seq=%d", resume_id, current_seq)

    return StreamingResponse(
        _replay_events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )
