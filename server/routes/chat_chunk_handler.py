from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
import json
from typing import Any

from fastapi import Request

from core.execution._tool_summary import summarize_tool_args
from core.execution.base import resolve_streamed_leaked_thinking
from core.i18n import t
from server.events import emit, emit_notification
from server.routes.chat_emotion import extract_emotion


def _format_sse(event: str, payload: dict[str, Any]) -> str:
    """Format a single SSE frame."""
    data = json.dumps(payload, ensure_ascii=False, default=str)
    return f"event: {event}\ndata: {data}\n\n"


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
    import asyncio

    event_type = chunk.get("type", "unknown")

    if event_type == "text_delta":
        return _format_sse("text_delta", {"text": chunk["text"]}), ""

    if event_type == "tool_start":
        return _format_sse(
            "tool_start",
            {
                "tool_name": chunk["tool_name"],
                "tool_id": chunk["tool_id"],
            },
        ), ""

    if event_type == "tool_end":
        return _format_sse(
            "tool_end",
            {
                "tool_id": chunk["tool_id"],
                "tool_name": chunk.get("tool_name", ""),
            },
        ), ""

    if event_type == "chain_start":
        return _format_sse("chain_start", {"chain": chunk["chain"]}), ""

    if event_type == "bootstrap_start":
        if request and anima_name:
            asyncio.ensure_future(
                emit(
                    request,
                    "anima.bootstrap",
                    {"name": anima_name, "status": "started"},
                )
            )
        return _format_sse("bootstrap", {"status": "started"}), ""

    if event_type == "bootstrap_complete":
        if request and anima_name:
            asyncio.ensure_future(
                emit(
                    request,
                    "anima.bootstrap",
                    {"name": anima_name, "status": "completed"},
                )
            )
        return _format_sse("bootstrap", {"status": "completed"}), ""

    if event_type == "bootstrap_busy":
        return _format_sse(
            "bootstrap",
            {
                "status": "busy",
                "message": chunk.get("message", t("chat.bootstrap_busy")),
            },
        ), ""

    if event_type == "compression_start":
        return _format_sse("compression_start", {}), ""

    if event_type == "compression_end":
        return _format_sse("compression_end", {}), ""

    if event_type == "heartbeat_relay_start":
        return _format_sse(
            "heartbeat_relay_start",
            {
                "message": chunk.get("message", t("chat.heartbeat_processing")),
            },
        ), ""

    if event_type == "heartbeat_relay":
        return _format_sse(
            "heartbeat_relay",
            {
                "text": chunk.get("text", ""),
            },
        ), chunk.get("text", "")

    if event_type == "heartbeat_relay_done":
        return _format_sse("heartbeat_relay_done", {}), ""

    if event_type == "thinking_start":
        return _format_sse("thinking_start", {}), ""

    if event_type == "thinking_delta":
        return _format_sse("thinking_delta", {"text": chunk.get("text", "")}), ""

    if event_type == "thinking_end":
        return _format_sse("thinking_end", {}), ""

    if event_type == "context_update":
        return _format_sse(
            "context_update",
            {
                "context_usage_ratio": chunk.get("context_usage_ratio", 0),
                "input_tokens": chunk.get("input_tokens", 0),
                "context_window": chunk.get("context_window", 0),
                "threshold": chunk.get("threshold", 0),
            },
        ), ""

    if event_type == "notification_sent":
        # Broadcast notification to all WebSocket clients (with queue support)
        if request:
            notif_data = chunk.get("data", {})
            asyncio.ensure_future(emit_notification(request, notif_data))
        return None, ""

    if event_type == "meeting_redirect":
        return _format_sse(
            "meeting_redirect",
            {
                "room_id": chunk.get("room_id", ""),
                "redirect_id": chunk.get("redirect_id", ""),
                "from": chunk.get("from", ""),
                "to": chunk.get("to", ""),
                "content": chunk.get("content", ""),
                "intent": chunk.get("intent", ""),
                "ts": chunk.get("ts", ""),
            },
        ), ""

    if event_type == "cycle_done":
        cycle_result = chunk.get("cycle_result", {})
        response_text = cycle_result.get("summary", "")
        # Extract emotion from response and include in done event
        clean_text, emotion = extract_emotion(response_text)
        # Defensive strip: catch any residual <think> tags the streaming
        # filter / safety net missed (e.g. across-chunk missing-<think>).
        leaked, clean_text = resolve_streamed_leaked_thinking(clean_text)
        thinking_raw = cycle_result.pop("thinking_text", "") or ""
        if leaked and not thinking_raw:
            thinking_raw = leaked
        cycle_result["summary"] = clean_text
        cycle_result["emotion"] = emotion
        cycle_result.pop("tool_call_records", None)
        cycle_result["thinking_summary"] = thinking_raw[:5000] if thinking_raw else None
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
        start_payload: dict[str, Any] = {
            "tool_name": chunk["tool_name"],
            "tool_id": chunk["tool_id"],
        }
        # Engines that already know the arguments at start time (Gemini CLI)
        # pass them along, so the UI can show them without waiting for
        # ``tool_detail``.
        raw_input = chunk.get("input")
        if isinstance(raw_input, dict) and raw_input:
            summary = summarize_tool_args(chunk["tool_name"] or "", raw_input)
            if summary:
                start_payload["input_summary"] = summary[:200]
        return "tool_start", start_payload
    if event_type == "tool_detail":
        return "tool_detail", {
            "tool_id": chunk.get("tool_id", ""),
            "tool_name": chunk.get("tool_name", ""),
            "detail": chunk.get("detail", ""),
        }
    if event_type == "tool_end":
        payload: dict[str, Any] = {"tool_id": chunk["tool_id"], "tool_name": chunk.get("tool_name", "")}
        record = chunk.get("record")
        if isinstance(record, dict):
            if record.get("result_summary"):
                payload["result_summary"] = record["result_summary"][:200]
            if record.get("input_summary"):
                payload["input_summary"] = record["input_summary"][:200]
            if record.get("is_error"):
                payload["is_error"] = True
        return "tool_end", payload
    if event_type == "chain_start":
        return "chain_start", {"chain": chunk["chain"]}
    if event_type == "bootstrap_start":
        return "bootstrap", {"status": "started"}
    if event_type == "bootstrap_complete":
        return "bootstrap", {"status": "completed"}
    if event_type == "bootstrap_busy":
        return "bootstrap", {"status": "busy", "message": chunk.get("message", t("chat.bootstrap_busy"))}
    if event_type == "compression_start":
        return "compression_start", {}
    if event_type == "compression_end":
        return "compression_end", {}
    if event_type == "heartbeat_relay_start":
        return "heartbeat_relay_start", {"message": chunk.get("message", t("chat.heartbeat_processing"))}
    if event_type == "heartbeat_relay":
        return "heartbeat_relay", {"text": chunk.get("text", "")}
    if event_type == "heartbeat_relay_done":
        return "heartbeat_relay_done", {}
    if event_type == "thinking_start":
        return "thinking_start", {}
    if event_type == "thinking_delta":
        return "thinking_delta", {"text": chunk.get("text", "")}
    if event_type == "thinking_end":
        return "thinking_end", {}
    if event_type == "meeting_redirect":
        return "meeting_redirect", {
            "room_id": chunk.get("room_id", ""),
            "redirect_id": chunk.get("redirect_id", ""),
            "from": chunk.get("from", ""),
            "to": chunk.get("to", ""),
            "content": chunk.get("content", ""),
            "intent": chunk.get("intent", ""),
            "ts": chunk.get("ts", ""),
        }
    if event_type == "context_update":
        return "context_update", {
            "context_usage_ratio": chunk.get("context_usage_ratio", 0),
            "input_tokens": chunk.get("input_tokens", 0),
            "context_window": chunk.get("context_window", 0),
            "threshold": chunk.get("threshold", 0),
        }
    if event_type == "cycle_done":
        cycle_result = chunk.get("cycle_result", {})
        response_text = cycle_result.get("summary", "")
        clean_text, emotion = extract_emotion(response_text)
        leaked, clean_text = resolve_streamed_leaked_thinking(clean_text)
        thinking_raw = cycle_result.pop("thinking_text", "") or ""
        if leaked and not thinking_raw:
            thinking_raw = leaked
        cycle_result["summary"] = clean_text
        cycle_result["emotion"] = emotion
        cycle_result.pop("tool_call_records", None)
        cycle_result["thinking_summary"] = thinking_raw[:5000] if thinking_raw else None
        return "done", cycle_result
    if event_type == "error":
        payload = {"message": chunk.get("message", "Unknown error")}
        if "code" in chunk:
            payload["code"] = chunk["code"]
        return "error", payload
    return None
