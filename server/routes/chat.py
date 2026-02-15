from __future__ import annotations
# AnimaWorks - Digital Person Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: AGPL-3.0-or-later

import asyncio
import json
import logging
import re
from typing import Any, AsyncIterator

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from server.dependencies import get_person
from server.events import emit

logger = logging.getLogger("animaworks.routes.chat")


class ChatRequest(BaseModel):
    message: str
    from_person: str = "human"


class ChatResponse(BaseModel):
    response: str
    person: str


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
    person_name: str | None = None,
) -> tuple[str | None, str]:
    """Map a stream chunk to an SSE event name and extract response text.

    Args:
        chunk: Stream chunk dictionary.
        request: FastAPI Request (optional, for emitting WebSocket events).
        person_name: Person name (optional, for WebSocket event data).

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
        if request and person_name:
            import asyncio
            asyncio.ensure_future(emit(
                request, "person.bootstrap",
                {"name": person_name, "status": "started"},
            ))
        return _format_sse("bootstrap", {"status": "started"}), ""

    if event_type == "bootstrap_complete":
        if request and person_name:
            import asyncio
            asyncio.ensure_future(emit(
                request, "person.bootstrap",
                {"name": person_name, "status": "completed"},
            ))
        return _format_sse("bootstrap", {"status": "completed"}), ""

    if event_type == "bootstrap_busy":
        return _format_sse("bootstrap", {
            "status": "busy",
            "message": chunk.get("message", "初期化中です"),
        }), ""

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


async def _stream_events(
    person: Any,
    name: str,
    body: ChatRequest,
    request: Request,
) -> AsyncIterator[str]:
    """Async generator that yields SSE frames for a streaming chat session."""
    full_response = ""
    try:
        await emit(request, "person.status", {"name": name, "status": "thinking"})

        async for chunk in person.process_message_stream(
            body.message, from_person=body.from_person
        ):
            frame, response_text = _handle_chunk(
                chunk, request=request, person_name=name,
            )
            if response_text:
                full_response = response_text
            if frame:
                yield frame

    except Exception:
        logger.exception("SSE stream error for person=%s", name)
        yield _format_sse("error", {"code": "STREAM_ERROR", "message": "Internal server error"})

    finally:
        await emit(request, "person.status", {"name": name, "status": "idle"})


# ── Router ────────────────────────────────────────────────────

def create_chat_router() -> APIRouter:
    router = APIRouter()

    @router.post("/persons/{name}/chat")
    async def chat(name: str, body: ChatRequest, request: Request):
        supervisor = request.app.state.supervisor

        # Guard: reject if person is bootstrapping
        if supervisor.is_bootstrapping(name):
            return JSONResponse(
                {"error": "現在キャラクターを作成中です。完了までお待ちください。"},
                status_code=503,
            )

        await emit(request, "person.status", {"name": name, "status": "thinking"})

        try:
            # Send IPC request to Person process
            result = await supervisor.send_request(
                person_name=name,
                method="process_message",
                params={
                    "message": body.message,
                    "from_person": body.from_person
                },
                timeout=60.0
            )

            response = result.get("response", "")
            clean_response, _ = extract_emotion(response)

            await emit(request, "person.status", {"name": name, "status": "idle"})

            return ChatResponse(response=clean_response, person=name)

        except KeyError:
            from fastapi import HTTPException
            raise HTTPException(status_code=404, detail=f"Person not found: {name}")
        except ValueError as e:
            from fastapi import HTTPException
            raise HTTPException(status_code=500, detail=str(e))
        except asyncio.TimeoutError:
            logger.error("Timeout waiting for chat response from person=%s", name)
            return JSONResponse(
                {"error": "Request timed out"}, status_code=504,
            )
        except RuntimeError as e:
            logger.exception("Runtime error in chat for person=%s", name)
            return JSONResponse(
                {"error": f"Internal server error: {e}"}, status_code=500,
            )

    @router.post("/persons/{name}/greet")
    async def greet(name: str, request: Request):
        """Generate a greeting when user clicks the character.

        Returns cached response if called within the 5-minute cooldown.
        Non-streaming, returns a single JSON response.
        """
        supervisor = request.app.state.supervisor

        # Guard: reject if person is bootstrapping
        if supervisor.is_bootstrapping(name):
            return JSONResponse(
                {"error": "現在キャラクターを作成中です。完了までお待ちください。"},
                status_code=503,
            )

        try:
            result = await supervisor.send_request(
                person_name=name,
                method="greet",
                params={},
                timeout=60.0,
            )

            return {
                "response": result.get("response", ""),
                "emotion": result.get("emotion", "neutral"),
                "cached": result.get("cached", False),
                "person": name,
            }

        except KeyError:
            from fastapi import HTTPException
            raise HTTPException(status_code=404, detail=f"Person not found: {name}")
        except ValueError as e:
            from fastapi import HTTPException
            raise HTTPException(status_code=500, detail=str(e))
        except asyncio.TimeoutError:
            logger.error("Timeout waiting for greet response from person=%s", name)
            return JSONResponse(
                {"error": "Request timed out"}, status_code=504,
            )
        except RuntimeError as e:
            logger.exception("Runtime error in greet for person=%s", name)
            return JSONResponse(
                {"error": f"Internal server error: {e}"}, status_code=500,
            )

    @router.post("/persons/{name}/chat/stream")
    async def chat_stream(name: str, body: ChatRequest, request: Request):
        """Stream chat response via SSE over IPC."""
        supervisor = request.app.state.supervisor

        # Verify person exists before starting the stream
        if name not in supervisor.processes:
            from fastapi import HTTPException
            raise HTTPException(status_code=404, detail=f"Person not found: {name}")

        # Guard: return immediately if person is bootstrapping
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

        async def _ipc_stream_events() -> AsyncIterator[str]:
            """Async generator that converts IPC stream to SSE frames."""
            full_response = ""
            try:
                await emit(request, "person.status", {"name": name, "status": "thinking"})

                async for ipc_response in supervisor.send_request_stream(
                    person_name=name,
                    method="process_message",
                    params={
                        "message": body.message,
                        "from_person": body.from_person,
                        "stream": True,
                    },
                    timeout=120.0,
                ):
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
                        yield _format_sse("done", cycle_result or {"summary": clean_text, "emotion": emotion})
                        break

                    if ipc_response.chunk:
                        # Parse the chunk JSON from the IPC layer
                        try:
                            chunk_data = json.loads(ipc_response.chunk)
                            frame, response_text = _handle_chunk(
                                chunk_data,
                                request=request,
                                person_name=name,
                            )
                            if response_text:
                                full_response = response_text
                            if frame:
                                yield frame
                        except json.JSONDecodeError:
                            # Raw text chunk fallback
                            full_response += ipc_response.chunk
                            yield _format_sse("text_delta", {"text": ipc_response.chunk})

            except KeyError:
                logger.error("Person not found during stream: %s", name)
                yield _format_sse("error", {"code": "PERSON_NOT_FOUND", "message": f"Person not found: {name}"})
            except TimeoutError:
                logger.error("IPC stream timeout for person=%s", name)
                yield _format_sse("error", {"code": "IPC_TIMEOUT", "message": "応答がタイムアウトしました"})
            except Exception:
                logger.exception("IPC stream error for person=%s", name)
                yield _format_sse("error", {"code": "STREAM_ERROR", "message": "Internal server error"})
            finally:
                await emit(request, "person.status", {"name": name, "status": "idle"})

        return StreamingResponse(
            _ipc_stream_events(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    return router
