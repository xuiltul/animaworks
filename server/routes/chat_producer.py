from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
import asyncio
import json
import logging
import time
from collections.abc import AsyncIterator
from typing import Any

from fastapi import Request

from core.exceptions import AnimaNotFoundError
from core.exceptions import IPCConnectionError as IPCConnError
from core.i18n import t
from server.events import emit, emit_direct
from server.routes.chat_chunk_handler import _chunk_to_event, _handle_chunk
from server.routes.chat_emotion import extract_emotion
from server.routes.chat_models import ChatRequest, _to_image_data
from server.routes.chat_ws_effects import _emit_ws_side_effects
from server.stream_registry import StreamRegistry, format_sse_with_id

logger = logging.getLogger("animaworks.routes.chat")


async def _run_producer(
    stream: Any,
    registry: StreamRegistry,
    supervisor: Any,
    ws_manager: Any,
    *,
    name: str,
    body: ChatRequest,
    saved_paths: list[str],
) -> None:
    """Background task: consume IPC stream and write events to StreamRegistry.

    Runs independently of the SSE connection — client disconnect does NOT
    kill this task, allowing the Anima to finish processing.
    """
    stream_done = False
    ipc_chunk_count = 0
    keepalive_count = 0
    _start = time.monotonic()

    try:
        await emit_direct(ws_manager, "anima.status", {"name": name, "status": "thinking", "thread_id": body.thread_id})

        # Emit stream_start event
        stream.add_event("stream_start", {"response_id": stream.response_id})

        from core.config import load_config

        _config = load_config()
        _timeout = float(_config.server.ipc_stream_timeout)

        logger.info(
            "[PRODUCER] start anima=%s stream=%s user=%s timeout=%.1f",
            name,
            stream.response_id,
            body.from_person,
            _timeout,
        )

        async for ipc_response in supervisor.send_request_stream(
            anima_name=name,
            method="process_message",
            params={
                "message": body.message,
                "from_person": body.from_person,
                "intent": body.intent,
                "stream": True,
                "images": _to_image_data(body.images),
                "attachment_paths": saved_paths,
                "thread_id": body.thread_id,
                "model": body.model,
            },
            timeout=_timeout,
        ):
            ipc_chunk_count += 1

            if ipc_response.done:
                # Final response with full result
                result = ipc_response.result or {}
                full_response = result.get("response", "")
                cycle_result = result.get("cycle_result", {})
                summary = cycle_result.get("summary", full_response)
                clean_text, emotion = extract_emotion(summary)
                cycle_result["summary"] = clean_text
                cycle_result["emotion"] = emotion
                elapsed = time.monotonic() - _start
                logger.info(
                    "[PRODUCER] IPC done anima=%s stream=%s ipc_chunks=%d keepalives=%d elapsed=%.1fs response_len=%d",
                    name,
                    stream.response_id,
                    ipc_chunk_count,
                    keepalive_count,
                    elapsed,
                    len(clean_text),
                )
                stream.add_event("done", cycle_result or {"summary": clean_text, "emotion": emotion})
                stream_done = True
                break

            if ipc_response.chunk:
                try:
                    chunk_data = json.loads(ipc_response.chunk)

                    # Keep-alive chunks — don't add to registry, tail sends
                    # SSE keepalive comments on wait_new_event timeout
                    if chunk_data.get("type") == "keepalive":
                        keepalive_count += 1
                        elapsed = time.monotonic() - _start
                        logger.debug(
                            "[PRODUCER] keepalive anima=%s stream=%s keepalive#%d elapsed=%.1fs",
                            name,
                            stream.response_id,
                            keepalive_count,
                            elapsed,
                        )
                        continue

                    # WebSocket side effects (no request dependency)
                    await _emit_ws_side_effects(chunk_data, ws_manager, name, body.thread_id)

                    # Convert to SSE event and buffer in registry
                    result = _chunk_to_event(chunk_data)
                    if result:
                        evt_name, evt_payload = result
                        stream.add_event(evt_name, evt_payload)
                        if evt_name == "done":
                            stream_done = True
                except json.JSONDecodeError:
                    # Raw text chunk fallback
                    stream.add_event("text_delta", {"text": ipc_response.chunk})
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
                stream.add_event("done", cycle_result or {"summary": clean_text, "emotion": emotion})
                stream_done = True
                break

        # Stream ended without done event
        if not stream_done:
            elapsed = time.monotonic() - _start
            logger.warning(
                "[PRODUCER] INCOMPLETE anima=%s stream=%s ipc_chunks=%d elapsed=%.1fs",
                name,
                stream.response_id,
                ipc_chunk_count,
                elapsed,
            )
            stream.add_event(
                "error",
                {
                    "code": "STREAM_INCOMPLETE",
                    "message": t("chat.stream_incomplete"),
                },
            )

        registry.mark_complete(stream.response_id, done=stream_done)

    except asyncio.CancelledError:
        elapsed = time.monotonic() - _start
        logger.info(
            "[PRODUCER] cancelled anima=%s stream=%s elapsed=%.1fs ipc_chunks=%d",
            name,
            stream.response_id,
            elapsed,
            ipc_chunk_count,
        )
        registry.mark_complete(stream.response_id, done=False)
        raise

    except RuntimeError as e:
        elapsed = time.monotonic() - _start
        error_str = str(e)
        if "Process restarting" in error_str:
            logger.warning(
                "[PRODUCER] ANIMA_RESTARTING anima=%s stream=%s elapsed=%.1fs", name, stream.response_id, elapsed
            )
            stream.add_event("error", {"code": "ANIMA_RESTARTING", "message": t("chat.anima_restarting")})
        elif "Not connected" in error_str:
            logger.error(
                "[PRODUCER] ANIMA_UNAVAILABLE anima=%s stream=%s elapsed=%.1fs", name, stream.response_id, elapsed
            )
            stream.add_event("error", {"code": "ANIMA_UNAVAILABLE", "message": t("chat.anima_unavailable")})
        elif "Connection closed during stream" in error_str:
            logger.error(
                "[PRODUCER] CONNECTION_LOST anima=%s stream=%s elapsed=%.1fs ipc_chunks=%d",
                name,
                stream.response_id,
                elapsed,
                ipc_chunk_count,
            )
            stream.add_event("error", {"code": "CONNECTION_LOST", "message": t("chat.connection_lost")})
        elif "IPC protocol error" in error_str:
            logger.error(
                "[PRODUCER] IPC_PROTOCOL_ERROR anima=%s stream=%s elapsed=%.1fs", name, stream.response_id, elapsed
            )
            stream.add_event("error", {"code": "IPC_PROTOCOL_ERROR", "message": t("chat.communication_error")})
        else:
            logger.exception(
                "[PRODUCER] RUNTIME_ERROR anima=%s stream=%s elapsed=%.1fs", name, stream.response_id, elapsed
            )
            stream.add_event("error", {"code": "STREAM_ERROR", "message": t("chat.internal_error")})
        registry.mark_complete(stream.response_id, done=False)

    except (ValueError, IPCConnError) as e:
        elapsed = time.monotonic() - _start
        logger.error(
            "[PRODUCER] IPC_ERROR anima=%s stream=%s elapsed=%.1fs error=%s", name, stream.response_id, elapsed, e
        )
        stream.add_event("error", {"code": "IPC_ERROR", "message": str(e)})
        registry.mark_complete(stream.response_id, done=False)

    except (KeyError, AnimaNotFoundError):
        elapsed = time.monotonic() - _start
        logger.error("[PRODUCER] ANIMA_NOT_FOUND anima=%s stream=%s elapsed=%.1fs", name, stream.response_id, elapsed)
        stream.add_event("error", {"code": "ANIMA_NOT_FOUND", "message": f"Anima not found: {name}"})
        registry.mark_complete(stream.response_id, done=False)

    except TimeoutError:
        elapsed = time.monotonic() - _start
        logger.error(
            "[PRODUCER] IPC_TIMEOUT anima=%s stream=%s elapsed=%.1fs ipc_chunks=%d",
            name,
            stream.response_id,
            elapsed,
            ipc_chunk_count,
        )
        stream.add_event("error", {"code": "IPC_TIMEOUT", "message": t("chat.timeout")})
        registry.mark_complete(stream.response_id, done=False)

    except Exception:
        elapsed = time.monotonic() - _start
        logger.exception(
            "[PRODUCER] STREAM_ERROR anima=%s stream=%s elapsed=%.1fs ipc_chunks=%d",
            name,
            stream.response_id,
            elapsed,
            ipc_chunk_count,
        )
        stream.add_event("error", {"code": "STREAM_ERROR", "message": "Internal server error"})
        registry.mark_complete(stream.response_id, done=False)

    finally:
        elapsed = time.monotonic() - _start
        logger.info(
            "[PRODUCER] finalize anima=%s stream=%s ipc_chunks=%d keepalives=%d elapsed=%.1fs done=%s",
            name,
            stream.response_id,
            ipc_chunk_count,
            keepalive_count,
            elapsed,
            stream_done,
        )
        try:
            remaining = registry.count_active(name)
            status = "idle" if remaining == 0 else "streaming"
            await emit_direct(ws_manager, "anima.status", {"name": name, "status": status, "thread_id": body.thread_id})
        except Exception:
            logger.debug("[PRODUCER] failed to emit status during cleanup")


async def _sse_tail(
    stream: Any,
    request: Request,
) -> AsyncIterator[str]:
    """Tail StreamRegistry and yield SSE frames.

    If the client disconnects, this generator exits but the producer
    task continues running in the background.  Uses ``wait_new_event()``
    for efficient event notification instead of busy polling.
    """
    seq = -1
    logger.info("[SSE-TAIL] start stream=%s", stream.response_id)

    while True:
        # Check client disconnect
        if await request.is_disconnected():
            logger.info(
                "[SSE-TAIL] client disconnected stream=%s seq=%d",
                stream.response_id,
                seq,
            )
            break

        # Fetch new events from buffer
        events = stream.events_after(seq)
        if events:
            for event in events:
                yield format_sse_with_id(event.event, event.payload, event.event_id)
                seq = event.seq
            # Yield control so ASGI server can flush chunks to the client
            await asyncio.sleep(0)
            # Loop back immediately to check for more events before waiting.
            # This prevents a race where events added during the yield above
            # would be missed by wait_new_event (their _notify_seq is already
            # captured), causing up to 30s delay.
            continue

        # Check if stream is complete and fully drained
        if stream.complete:
            # Drain any remaining events that arrived after our last check
            remaining = stream.events_after(seq)
            for event in remaining:
                yield format_sse_with_id(event.event, event.payload, event.event_id)
            logger.info(
                "[SSE-TAIL] complete stream=%s seq=%d done=%s",
                stream.response_id,
                seq,
                stream.done,
            )
            break

        # Wait efficiently for new events (no busy polling).
        # Pass after_seq so that events buffered between events_after()
        # and this call are detected immediately without waiting.
        got_event = await stream.wait_new_event(timeout=30.0, after_seq=seq)
        if not got_event and not stream.complete:
            # Timeout with no events — send keepalive to prevent connection drop
            yield ": keepalive\n\n"

    logger.info("[SSE-TAIL] end stream=%s seq=%d", stream.response_id, seq)


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
    from server.routes.chat_chunk_handler import _format_sse

    _full_response = ""
    try:
        await emit(request, "anima.status", {"name": name, "status": "thinking"})

        async for chunk in anima.process_message_stream(
            body.message,
            from_person=body.from_person,
            intent=body.intent,
            images=images,
            attachment_paths=attachment_paths,
        ):
            frame, response_text = _handle_chunk(
                chunk,
                request=request,
                anima_name=name,
            )
            if response_text:
                _full_response = response_text
            if frame:
                yield frame

    except Exception:
        logger.exception("SSE stream error for anima=%s", name)
        yield _format_sse("error", {"code": "STREAM_ERROR", "message": "Internal server error"})

    finally:
        await emit(request, "anima.status", {"name": name, "status": "idle"})
