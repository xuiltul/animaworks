from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
import asyncio
import json
import logging
import math
import threading
from datetime import datetime, timedelta
from typing import Any, Literal

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, Field

from core.activity.conversation import _summarize_tool_call
from core.activity.logger import ActivityLogger
from core.memory.conversation.memory import ConversationMemory
from core.memory.conversation.shortterm import ShortTermMemory
from core.memory.manager import MemoryManager
from core.skills.activation import validate_thread_id
from core.time_utils import now_local

logger = logging.getLogger("animaworks.routes.sessions")

_AWAITING_CACHE_LOCK = threading.RLock()
_AWAITING_CACHE_BY_ROOT: dict[str, dict[str, Any]] = {}


class NotificationReplyRequest(BaseModel):
    notification_id: str = Field(min_length=1, max_length=300)
    text: str = Field(min_length=1, max_length=4000)


def _find_notification(activity: ActivityLogger, notification_id: str) -> dict | None:
    """Find a call_human notification by id (its timestamp) in that day's activity log."""
    try:
        day = datetime.fromisoformat(str(notification_id).replace("Z", "+00:00")).date()
    except ValueError:
        day = None
    days = [day] if day else []
    for target_day in days or [None]:
        entries = activity._load_conversation_entries(
            before=None, limit=5000, thread_id="default", _specific_day=target_day
        )
        for message in activity._entries_to_messages(entries):
            if message.get("type") == "human_notify" and notification_id in {
                str(message.get("notification_id") or ""),
                str(message.get("ts") or ""),
            }:
                return message
    return None


def _settle_older_notifications(activity: ActivityLogger, view: dict, thread_id: str) -> None:
    """Mark notifications on an older page as replied when the reply sits on a newer page."""
    pending = [
        message
        for session in view.get("sessions", [])
        for message in session.get("messages", [])
        if message.get("type") == "human_notify" and not message.get("replied")
    ]
    if not pending:
        return
    latest = activity.get_conversation_view(limit=500, thread_id=thread_id)
    replied_ts = {
        str(message.get("ts") or "")
        for session in latest.get("sessions", [])
        for message in session.get("messages", [])
        if message.get("type") == "human_notify" and message.get("replied")
    }
    for message in pending:
        if str(message.get("ts") or "") in replied_ts:
            message["replied"] = True


def _thread_file_conversation_view(anima_dir, thread_id: str) -> dict | None:
    """Build a conversation view from the thread's conversation file.

    Threads persist their turns in ``state/conversations/{thread_id}.json``
    while the activity log only holds entries within its retention window
    (and older threads predate thread_id tagging entirely). When the
    activity-based view is missing turns, this file is the durable source.
    Returns None when the file is absent or unreadable.
    """
    conv_file = anima_dir / "state" / "conversations" / f"{thread_id}.json"
    if not conv_file.is_file():
        return None
    try:
        data = json.loads(conv_file.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError, TypeError):
        return None

    turns = data.get("turns", [])
    if not isinstance(turns, list):
        return None

    messages = []
    for turn in turns:
        if not isinstance(turn, dict):
            continue
        role = turn.get("role", "")
        if role not in ("human", "assistant"):
            continue
        tool_calls = []
        for rec in turn.get("tool_records", []) or []:
            if not isinstance(rec, dict):
                continue
            tool_calls.append(
                {
                    "tool_use_id": rec.get("tool_id", ""),
                    "tool_name": rec.get("tool_name", ""),
                    "input": rec.get("input_summary", ""),
                    "result": rec.get("result_summary", ""),
                    "is_error": bool(rec.get("is_error", False)),
                }
            )
        messages.append(
            {
                "ts": turn.get("timestamp", ""),
                "role": role,
                "content": turn.get("content", ""),
                "from_person": "",
                "to_person": "",
                "source_key": "chat",
                "tool_calls": tool_calls,
            }
        )

    if not messages:
        return None

    sessions = ActivityLogger._group_into_sessions(messages, 10)
    return {"sessions": sessions, "has_more": False, "next_before": None}


def _count_view_messages(view: dict) -> int:
    return sum(len(s.get("messages", [])) for s in view.get("sessions", []))


def _summarize_view_tool_calls(view: dict) -> dict:
    """Strip tool payloads from a conversation-file fallback view."""
    for session in view.get("sessions", []):
        for message in session.get("messages", []):
            calls = message.get("tool_calls")
            if message.get("role") == "assistant" and calls:
                message["tool_calls"] = [_summarize_tool_call(call) for call in calls]
                message["tool_calls_detail"] = "lazy"
    return view


def create_sessions_router() -> APIRouter:
    router = APIRouter()

    @router.get("/animas/{name}/sessions")
    def list_sessions(name: str, request: Request):
        """List all available sessions: active conversation, archives, episodes."""
        animas_dir = request.app.state.animas_dir
        anima_dir = animas_dir / name
        if not anima_dir.exists():
            raise HTTPException(status_code=404, detail=f"Anima not found: {name}")

        from core.config.models import load_model_config

        model_config = load_model_config(anima_dir)

        # Active conversation
        conv = ConversationMemory(anima_dir, model_config)
        conv_state = conv.load()
        active_conv = None
        if conv_state.turns or conv_state.compressed_summary:
            active_conv = {
                "exists": True,
                "turn_count": len(conv_state.turns),
                "total_turn_count": conv_state.total_turn_count,
                "last_timestamp": (conv_state.turns[-1].timestamp if conv_state.turns else ""),
                "first_timestamp": (conv_state.turns[0].timestamp if conv_state.turns else ""),
                "has_summary": bool(conv_state.compressed_summary),
            }

        # Thread conversations
        threads = []
        conv_dir = anima_dir / "state" / "conversations"
        if conv_dir.is_dir():
            for conv_file in sorted(conv_dir.glob("*.json"), reverse=True):
                try:
                    data = json.loads(conv_file.read_text(encoding="utf-8"))
                    turns = data.get("turns", [])
                    threads.append(
                        {
                            "thread_id": conv_file.stem,
                            "turn_count": len(turns),
                            "total_turn_count": len(turns) + data.get("compressed_turn_count", 0),
                            "last_timestamp": turns[-1].get("timestamp", "") if turns else "",
                            "has_summary": bool(data.get("compressed_summary", "")),
                        }
                    )
                except (json.JSONDecodeError, TypeError):
                    pass

        # Archived sessions — metadata only (no full content read)
        stm = ShortTermMemory(anima_dir, read_only=True)
        archived = []
        archive_dir = stm._archive_dir
        if archive_dir.exists():
            for json_file in sorted(archive_dir.glob("*.json"), reverse=True):
                try:
                    data = json.loads(json_file.read_text(encoding="utf-8"))
                    ts_str = json_file.stem
                    archived.append(
                        {
                            "id": ts_str,
                            "timestamp": data.get("timestamp", ts_str),
                            "trigger": data.get("trigger", ""),
                            "turn_count": data.get("turn_count", 0),
                            "context_usage_ratio": data.get(
                                "context_usage_ratio",
                                0,
                            ),
                            "original_prompt_preview": data.get(
                                "original_prompt",
                                "",
                            )[:200],
                            "has_markdown": (archive_dir / f"{ts_str}.md").exists(),
                        }
                    )
                except (json.JSONDecodeError, TypeError):
                    pass

        # Episodes — partial read (first 200 chars only, no full file load)
        memory = MemoryManager(anima_dir, read_only=True)
        episodes = []
        for stem in memory.list_episode_files():
            ep_path = memory.episodes_dir / f"{stem}.md"
            preview = ""
            if ep_path.exists():
                with ep_path.open(encoding="utf-8") as f:
                    preview = f.read(200)
            episodes.append({"date": stem, "preview": preview})

        # Transcripts — count lines without loading full content
        transcripts = []
        transcript_dir = anima_dir / "transcripts"
        if transcript_dir.exists():
            for tf in sorted(transcript_dir.glob("*.jsonl"), reverse=True):
                line_count = sum(1 for line in tf.read_text(encoding="utf-8").splitlines() if line.strip())
                transcripts.append({"date": tf.stem, "message_count": line_count})

        return {
            "anima": name,
            "active_conversation": active_conv,
            "threads": threads,
            "archived_sessions": archived,
            "episodes": episodes,
            "transcripts": transcripts,
        }

    # ── Conversation history (activity_log-based) ──────────

    @router.get("/animas/{name}/conversation/history")
    async def get_conversation_history(
        name: str,
        limit: int = 50,
        before: str | None = None,
        thread_id: str = "default",
        strict_thread: bool = False,
        tool_detail: Literal["full", "summary"] = "full",
        request: Request = None,
    ):
        """Get conversation history from activity log.

        Returns sessions with messages, tool calls nested into assistant
        messages, and cursor-based pagination.
        """
        animas_dir = request.app.state.animas_dir
        anima_dir = animas_dir / name
        if not anima_dir.exists():
            raise HTTPException(status_code=404, detail=f"Anima not found: {name}")

        limit = max(1, min(limit, 200))
        activity = ActivityLogger(anima_dir)
        # Log scanning is sync file I/O + JSON parsing; keep it off the
        # event loop (UI polls this endpoint every few seconds).
        view = await asyncio.to_thread(
            activity.get_conversation_view,
            before=before,
            limit=limit,
            thread_id=thread_id,
            strict_thread=strict_thread,
            tool_detail=tool_detail,
        )
        if before is not None:
            await asyncio.to_thread(_settle_older_notifications, activity, view, thread_id)

        # Named threads: the activity log may miss turns (retention window,
        # pre-thread_id-tagging history, or a response whose activity entry
        # was lost). Fall back to the thread's own conversation file when it
        # holds more messages than the activity-based view.
        if thread_id not in ("default", "inbox") and before is None:
            try:
                safe_thread_id = validate_thread_id(thread_id)
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid thread_id: {thread_id}") from None
            file_view = _thread_file_conversation_view(anima_dir, safe_thread_id)
            if file_view and _count_view_messages(file_view) > _count_view_messages(view):
                call_messages = [
                    message
                    for session in view.get("sessions", [])
                    for message in session.get("messages", [])
                    if message.get("type") == "voice_call"
                ]
                if call_messages:
                    file_messages = [
                        message for session in file_view.get("sessions", []) for message in session.get("messages", [])
                    ]
                    present_calls = {(message.get("ts", ""), message.get("type", "")) for message in file_messages}
                    file_messages.extend(
                        message
                        for message in call_messages
                        if (message.get("ts", ""), message.get("type", "")) not in present_calls
                    )
                    file_messages.sort(key=lambda message: message.get("ts", ""))
                    file_view = {
                        "sessions": ActivityLogger._group_into_sessions(file_messages, 10),
                        "has_more": False,
                        "next_before": None,
                    }
                return _summarize_view_tool_calls(file_view) if tool_detail == "summary" else file_view
        return view

    @router.get("/animas/{name}/conversation/feed")
    async def get_conversation_feed(
        name: str,
        thread_id: str = "default",
        before: str | None = None,
        min_user_items: int = Query(default=10, ge=1, le=50),
        strict_thread: bool = False,
        request: Request = None,
    ):
        """Get user-centered chat history with lazy background activity groups."""
        if not name or name.startswith(".") or "/" in name:
            raise HTTPException(status_code=400, detail="Invalid Anima name")
        try:
            safe_thread_id = validate_thread_id(thread_id)
            if before is not None:
                datetime.fromisoformat(before.replace("Z", "+00:00"))
        except (ValueError, TypeError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from None

        anima_dir = request.app.state.animas_dir / name
        if not anima_dir.is_dir():
            raise HTTPException(status_code=404, detail=f"Anima not found: {name}")

        activity = ActivityLogger(anima_dir)
        return await asyncio.to_thread(
            activity.get_conversation_feed,
            thread_id=safe_thread_id,
            before=before,
            min_user_items=min_user_items,
            strict_thread=strict_thread,
        )

    @router.get("/animas/{name}/conversation/bg")
    async def get_conversation_background(
        name: str,
        thread_id: str = "default",
        from_ts: str = Query(...),
        to_ts: str = Query(...),
        strict_thread: bool = False,
        request: Request = None,
    ):
        """Load background activity for one collapsed feed group."""
        if not name or name.startswith(".") or "/" in name:
            raise HTTPException(status_code=400, detail="Invalid Anima name")
        try:
            safe_thread_id = validate_thread_id(thread_id)
            from_time = datetime.fromisoformat(from_ts.replace("Z", "+00:00"))
            to_time = datetime.fromisoformat(to_ts.replace("Z", "+00:00"))
            if from_time > to_time:
                raise ValueError("from_ts must be before or equal to to_ts")
        except (ValueError, TypeError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from None

        anima_dir = request.app.state.animas_dir / name
        if not anima_dir.is_dir():
            raise HTTPException(status_code=404, detail=f"Anima not found: {name}")

        activity = ActivityLogger(anima_dir)
        return await asyncio.to_thread(
            activity.get_conversation_background,
            thread_id=safe_thread_id,
            from_ts=from_ts,
            to_ts=to_ts,
            strict_thread=strict_thread,
        )

    @router.get("/animas/{name}/conversation/tool-calls")
    async def get_conversation_tool_calls(
        name: str,
        message_ts: str,
        thread_id: str = "default",
        request: Request = None,
    ):
        """Get the full, truncated tool-call details for one assistant message."""
        if not name or name.startswith(".") or "/" in name:
            raise HTTPException(status_code=400, detail="Invalid Anima name")
        try:
            safe_thread_id = validate_thread_id(thread_id)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from None

        anima_dir = request.app.state.animas_dir / name
        if not anima_dir.is_dir():
            raise HTTPException(status_code=404, detail=f"Anima not found: {name}")

        activity = ActivityLogger(anima_dir)
        tool_calls = await asyncio.to_thread(
            activity.get_conversation_tool_calls,
            message_ts,
            thread_id=safe_thread_id,
            strict_thread=safe_thread_id != "default",
        )
        if not tool_calls and safe_thread_id not in ("default", "inbox"):
            file_view = _thread_file_conversation_view(anima_dir, safe_thread_id)
            file_message = next(
                (
                    message
                    for session in (file_view or {}).get("sessions", [])
                    for message in session.get("messages", [])
                    if message.get("role") == "assistant" and message.get("ts") == message_ts
                ),
                None,
            )
            tool_calls = file_message.get("tool_calls") if file_message else None
        if not tool_calls:
            raise HTTPException(status_code=404, detail="Tool calls not found")
        return {"tool_calls": tool_calls}

    @router.get("/notifications/awaiting")
    async def list_awaiting_notifications(request: Request):
        """List call_human notifications from the last 24 hours that still wait for a reply, per Anima.

        Reads only notification, reply and incoming-message entries from the last two
        days of each activity log, so it stays cheap enough to poll.
        """
        animas_dir = request.app.state.animas_dir
        names = sorted(p.name for p in animas_dir.iterdir() if p.is_dir() and not p.name.startswith("."))

        def _waiting(anima_dir) -> list[dict[str, str]]:
            activity = ActivityLogger(anima_dir)
            entries = activity.recent(days=2, limit=5000, types=["human_notify", "human_reply", "message_received"])
            # Only chat-channel messages appear in the chat view; inbox traffic is not an answer.
            entries = [e for e in entries if e.type != "message_received" or e.channel == "chat"]
            messages = activity._entries_to_messages(entries)
            return [
                {
                    "ts": str(message.get("ts") or ""),
                    "subject": str(message.get("subject") or "")[:80],
                    "preview": " ".join(str(message.get("content") or "").split())[:80],
                }
                for message in messages
                if message.get("type") == "human_notify"
                and message.get("needs_reply", True)
                and not message.get("replied")
            ]

        def _collect() -> dict[str, list[dict[str, str]]]:
            now = now_local()
            since = (now - timedelta(hours=24)).isoformat()

            def _latest_log_signature(anima_dir) -> tuple[str, int, int] | None:
                log_dir = anima_dir / "activity_log"
                for day_offset in range(2):
                    day = now.date() - timedelta(days=day_offset)
                    latest = log_dir / f"{day.isoformat()}.jsonl"
                    try:
                        stat = latest.stat()
                    except OSError:
                        continue
                    if latest.is_file():
                        return latest.name, stat.st_size, stat.st_mtime_ns
                return None

            def _is_expired_cache(cache: dict[str, Any]) -> bool:
                expiry = cache.get("expires_at")
                return expiry is not None and now >= expiry

            def _next_expiry(found: dict[str, list[dict[str, str]]]) -> datetime | None:
                expiries: list[datetime] = []
                for rows in found.values():
                    for row in rows:
                        try:
                            timestamp = datetime.fromisoformat(row["ts"].replace("Z", "+00:00"))
                            if timestamp.tzinfo is None:
                                timestamp = timestamp.astimezone()
                            expiries.append(timestamp + timedelta(hours=24))
                        except (KeyError, ValueError):
                            continue
                return min(expiries) if expiries else None

            root_key = str(animas_dir.resolve())
            signatures = {name: _latest_log_signature(animas_dir / name) for name in names}
            with _AWAITING_CACHE_LOCK:
                cache = _AWAITING_CACHE_BY_ROOT.setdefault(
                    root_key,
                    {"names": (), "per_anima": {}, "response": None, "expires_at": None},
                )
                per_anima: dict[str, Any] = cache["per_anima"]
                for removed_name in set(per_anima) - set(names):
                    per_anima.pop(removed_name, None)

                unchanged = cache["names"] == tuple(names) and all(
                    name in per_anima and per_anima[name]["signature"] == signature
                    for name, signature in signatures.items()
                )
                if unchanged and cache["response"] is not None and not _is_expired_cache(cache):
                    return cache["response"]

                for name, signature in signatures.items():
                    cached = per_anima.get(name)
                    if cached and cached["signature"] == signature:
                        continue
                    try:
                        per_anima[name] = {"signature": signature, "items": _waiting(animas_dir / name)}
                    except Exception:
                        logger.debug("Failed to list awaiting notifications for %s", name, exc_info=True)

                found: dict[str, list[dict[str, str]]] = {}
                for name in names:
                    cached = per_anima.get(name)
                    if not cached:
                        continue
                    rows = [row for row in cached["items"] if row["ts"] >= since]
                    if rows:
                        found[name] = rows

                response = {"animas": {name: len(rows) for name, rows in found.items()}, "items": found}
                cache["names"] = tuple(names)
                cache["response"] = response
                cache["expires_at"] = _next_expiry(found)
                return response

        result = await asyncio.to_thread(_collect)
        return result

    @router.post("/animas/{name}/notifications/reply")
    async def reply_to_notification(
        name: str,
        body: NotificationReplyRequest,
        request: Request,
    ):
        """Route a reply to a human_notify entry back to its originating Anima."""
        if not name or name.startswith(".") or "/" in name:
            raise HTTPException(status_code=400, detail="Invalid Anima name")
        animas_dir = request.app.state.animas_dir
        anima_dir = animas_dir / name
        if not anima_dir.is_dir():
            raise HTTPException(status_code=404, detail=f"Anima not found: {name}")

        activity = ActivityLogger(anima_dir)
        notification = await asyncio.to_thread(_find_notification, activity, body.notification_id)
        if notification is None:
            raise HTTPException(status_code=404, detail="Notification not found")

        text = body.text.strip()
        if not text:
            raise HTTPException(status_code=400, detail="Reply text is required")
        actor = getattr(getattr(request.state, "user", None), "username", "") or "human"
        callback_id = str(notification.get("callback_id") or "")

        from core.notification.reply_routing import (
            _notification_context,
            _resolve_callback_text_reply,
            deliver_notification_reply,
        )

        if await _resolve_callback_text_reply(callback_id, text, actor):
            return {"status": "resolved", "notification_id": body.notification_id, "replied": True}

        notification_text = "\n".join(
            part for part in (str(notification.get("subject") or ""), str(notification.get("content") or "")) if part
        )
        from core.paths import get_shared_dir

        routed = await deliver_notification_reply(
            name,
            text,
            get_shared_dir(),
            actor=actor,
            source="web",
            source_message_id=body.notification_id,
            external_user_id=actor,
            external_channel_id="web",
            external_thread_ts=body.notification_id,
            thread_context=_notification_context(name, notification_text),
            notification_id=body.notification_id,
            callback_id=callback_id,
        )
        if not routed:
            raise HTTPException(status_code=400, detail="Reply could not be delivered")
        return {"status": "ok", "notification_id": body.notification_id, "replied": True}

    @router.post("/animas/{name}/conversation/calls")
    async def record_voice_call(name: str, request: Request, payload: dict[str, Any]):
        """Persist a completed voice call in the Anima's conversation history."""
        if not name or name.startswith(".") or "/" in name or ".." in name:
            raise HTTPException(status_code=400, detail="Invalid Anima name")
        anima_dir = request.app.state.animas_dir / name
        if not anima_dir.exists():
            raise HTTPException(status_code=404, detail=f"Anima not found: {name}")

        try:
            thread_id = validate_thread_id(payload.get("thread_id", "default"))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from None

        duration = payload.get("duration_seconds", 0)
        if isinstance(duration, bool) or not isinstance(duration, (int, float)):
            raise HTTPException(status_code=400, detail="duration_seconds must be a number")
        try:
            numeric_duration = float(duration)
        except (OverflowError, ValueError):
            raise HTTPException(status_code=400, detail="duration_seconds must be finite") from None
        if not math.isfinite(numeric_duration):
            raise HTTPException(status_code=400, detail="duration_seconds must be finite")
        duration_seconds = max(0, min(int(numeric_duration), 86_400))

        started_at = payload.get("started_at", "")
        if not isinstance(started_at, str) or len(started_at) > 64:
            raise HTTPException(status_code=400, detail="started_at must be an ISO timestamp")
        try:
            datetime.fromisoformat(started_at.replace("Z", "+00:00"))
        except ValueError:
            raise HTTPException(status_code=400, detail="started_at must be an ISO timestamp") from None

        summary = payload.get("summary", "")
        if not isinstance(summary, str):
            raise HTTPException(status_code=400, detail="summary must be a string")
        summary = summary[:60]

        raw_transcript = payload.get("transcript", [])
        if not isinstance(raw_transcript, list) or len(raw_transcript) > 500:
            raise HTTPException(status_code=400, detail="transcript must contain at most 500 lines")
        transcript: list[dict[str, str]] = []
        total_transcript_chars = 0
        for line in raw_transcript:
            if not isinstance(line, dict) or line.get("speaker") not in {"user", "assistant"}:
                raise HTTPException(status_code=400, detail="transcript lines need a valid speaker")
            text = line.get("text", "")
            if not isinstance(text, str):
                raise HTTPException(status_code=400, detail="transcript line text must be a string")
            total_transcript_chars += len(text)
            if total_transcript_chars > 100_000:
                raise HTTPException(status_code=400, detail="transcript is too long")
            transcript.append({"speaker": line["speaker"], "text": text})

        activity = ActivityLogger(anima_dir)
        entry = await asyncio.to_thread(
            activity.log,
            "voice_call",
            content=summary,
            summary=summary,
            channel="chat",
            origin="human",
            meta={
                "thread_id": thread_id,
                "started_at": started_at,
                "duration_seconds": duration_seconds,
                "transcript": transcript,
            },
        )
        return {"ok": True, "ts": entry.ts}

    @router.get("/animas/{name}/sessions/{session_id}")
    async def get_session_detail(
        name: str,
        session_id: str,
        request: Request,
    ):
        """Get archived session detail."""
        animas_dir = request.app.state.animas_dir
        anima_dir = animas_dir / name
        if not anima_dir.exists():
            raise HTTPException(status_code=404, detail=f"Anima not found: {name}")

        stm = ShortTermMemory(anima_dir, read_only=True)
        archive_dir = stm._archive_dir
        json_path = archive_dir / f"{session_id}.json"
        md_path = archive_dir / f"{session_id}.md"

        if not json_path.exists():
            raise HTTPException(status_code=404, detail="Session not found")

        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, TypeError):
            raise HTTPException(status_code=500, detail="Session data corrupted") from None

        markdown = ""
        if md_path.exists():
            markdown = md_path.read_text(encoding="utf-8")

        return {
            "anima": name,
            "session_id": session_id,
            "data": data,
            "markdown": markdown,
        }

    @router.get("/animas/{name}/transcripts/{date}")
    async def get_transcript(name: str, date: str, request: Request):
        """Get full conversation transcript for a specific date."""
        animas_dir = request.app.state.animas_dir
        anima_dir = animas_dir / name
        if not anima_dir.exists():
            raise HTTPException(status_code=404, detail=f"Anima not found: {name}")

        from core.config.models import load_model_config

        model_config = load_model_config(anima_dir)

        conv = ConversationMemory(anima_dir, model_config)
        messages = conv.load_transcript(date)
        return {
            "anima": name,
            "date": date,
            "has_summary": False,
            "compressed_summary": "",
            "compressed_turn_count": 0,
            "turns": messages,
        }

    return router
