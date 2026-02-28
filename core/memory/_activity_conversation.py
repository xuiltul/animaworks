from __future__ import annotations
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.

"""Conversation view mixin for ActivityLogger.

Internal module — import from :mod:`core.memory.activity` instead.
"""

import json
import logging
from datetime import timedelta
from typing import Any

from core.i18n import t
from core.memory._activity_models import (
    ActivityEntry,
    find_tool_result_fallback,
    time_diff,
)
from core.time_utils import now_jst

logger = logging.getLogger("animaworks.activity")


class ConversationMixin:
    """Mixin providing conversation view methods for ActivityLogger."""

    _CONVERSATION_TYPES = {
        "message_received", "response_sent",
        "tool_use", "tool_result",
        "heartbeat_start", "heartbeat_end",
        "cron_executed",
        "error",
    }

    def get_conversation_view(
        self,
        *,
        before: str | None = None,
        limit: int = 50,
        session_gap_minutes: int = 10,
        thread_id: str | None = None,
        strict_thread: bool = False,
    ) -> dict[str, Any]:
        """Build a conversation view from activity log entries.

        Reads activity log events, pairs tool_use/tool_result, groups into
        sessions (separated by gaps >= *session_gap_minutes*), and returns a
        structure ready for UI rendering.

        Args:
            before: If given, only include entries with ``ts < before``
                (cursor-based pagination, ISO 8601 timestamp).
            limit: Maximum number of *messages* to return (tool_calls are
                nested within assistant messages and do not count).
            session_gap_minutes: Minimum gap in minutes to start a new session.
            thread_id: If given, only include entries whose
                ``meta.thread_id`` matches. Entries without a ``thread_id``
                in meta are treated as belonging to ``"default"``.
            strict_thread: If True, require explicit ``meta.thread_id`` match.
                Entries without ``meta.thread_id`` are excluded.

        Returns:
            ``{"sessions": [...], "has_more": bool, "next_before": str | None}``
        """
        entries = self._load_conversation_entries(
            before=before,
            limit=limit,
            thread_id=thread_id,
            strict_thread=strict_thread,
        )
        messages = self._entries_to_messages(entries)

        has_more = len(messages) > limit
        if has_more:
            messages = messages[-limit:]

        next_before: str | None = None
        if has_more and messages:
            next_before = messages[0]["ts"]

        sessions = self._group_into_sessions(messages, session_gap_minutes)

        return {
            "sessions": sessions,
            "has_more": has_more,
            "next_before": next_before,
        }

    def _load_conversation_entries(
        self,
        *,
        before: str | None = None,
        limit: int = 50,
        thread_id: str | None = None,
        strict_thread: bool = False,
    ) -> list[ActivityEntry]:
        """Load conversation-relevant entries, scanning backwards.

        Returns entries in chronological order.  Scans enough days to
        collect at least ``limit * 3`` raw entries (to account for
        tool_use/tool_result pairs being folded into messages).

        When *thread_id* is given, only entries whose ``meta.thread_id``
        matches are returned.  Entries without the field are treated as
        belonging to ``"default"`` unless *strict_thread* is True.
        """
        target_raw = limit * 3 + 50
        entries: list[ActivityEntry] = []
        today = now_jst().date()
        max_scan_days = 365

        for day_offset in range(max_scan_days):
            target = today - timedelta(days=day_offset)
            path = self._log_dir / f"{target.isoformat()}.jsonl"  # type: ignore[attr-defined]
            if not path.exists():
                if day_offset > 30 and not entries:
                    break
                continue

            day_entries: list[ActivityEntry] = []
            try:
                for line_num, line in enumerate(
                    path.read_text(encoding="utf-8").splitlines(), start=1,
                ):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        raw = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if raw.get("type") not in self._CONVERSATION_TYPES:
                        continue
                    ts = raw.get("ts", "")
                    if before and ts >= before:
                        continue
                    if thread_id is not None:
                        meta = raw.get("meta", {})
                        has_thread_id = isinstance(meta, dict) and "thread_id" in meta
                        if strict_thread:
                            if not has_thread_id:
                                continue
                            if meta.get("thread_id") != thread_id:
                                continue
                        else:
                            entry_tid = meta.get("thread_id", "default") if isinstance(meta, dict) else "default"
                            if entry_tid != thread_id:
                                continue
                    if "from" in raw:
                        raw["from_person"] = raw.pop("from")
                    if "to" in raw:
                        raw["to_person"] = raw.pop("to")
                    entry = ActivityEntry(**{
                        k: v for k, v in raw.items()
                        if k in ActivityEntry.__dataclass_fields__
                    })
                    entry._line_number = line_num
                    day_entries.append(entry)
            except Exception:
                logger.exception("Failed to read activity log %s", path)

            entries = day_entries + entries
            if len(entries) >= target_raw:
                break

        return entries

    def _entries_to_messages(
        self,
        entries: list[ActivityEntry],
    ) -> list[dict[str, Any]]:
        """Convert raw activity entries to conversation messages.

        Pairs ``tool_use``/``tool_result`` by ``meta.tool_use_id`` and
        nests them into the preceding ``response_sent`` message.
        """
        tool_results: dict[str, ActivityEntry] = {}
        for e in entries:
            if e.type == "tool_result":
                tid = e.meta.get("tool_use_id", "")
                if tid:
                    tool_results[tid] = e

        messages: list[dict[str, Any]] = []
        pending_tool_calls: list[dict[str, Any]] = []

        for e in entries:
            if e.type == "message_received":
                self._flush_tool_calls(messages, pending_tool_calls)
                messages.append({
                    "ts": e.ts,
                    "role": "human",
                    "content": e.content,
                    "from_person": e.from_person,
                    "tool_calls": [],
                })

            elif e.type == "response_sent":
                msg = {
                    "ts": e.ts,
                    "role": "assistant",
                    "content": e.content,
                    "from_person": "",
                    "tool_calls": [],
                }
                if e.meta.get("thinking_text"):
                    msg["thinking_text"] = e.meta["thinking_text"]
                images = e.meta.get("images") or e.meta.get("artifacts") or []
                if isinstance(images, list) and images:
                    msg["images"] = images
                messages.append(msg)
                if pending_tool_calls:
                    msg["tool_calls"].extend(pending_tool_calls)
                    pending_tool_calls.clear()

            elif e.type == "tool_use":
                tid = e.meta.get("tool_use_id", "")
                result_entry = tool_results.get(tid) if tid else None
                if not result_entry:
                    result_entry = find_tool_result_fallback(entries, e)
                tc: dict[str, Any] = {
                    "tool_use_id": tid,
                    "tool_name": e.tool,
                    "input": e.meta.get("args", e.content),
                    "result": result_entry.content if result_entry else "",
                    "is_error": (
                        result_entry.meta.get("is_error", False)
                        if result_entry else False
                    ),
                }
                if e.meta.get("blocked"):
                    tc["is_error"] = True
                    tc["result"] = t("activity.blocked", reason=e.meta.get("reason", ""))
                pending_tool_calls.append(tc)

            elif e.type == "tool_result":
                pass

            elif e.type == "heartbeat_start":
                self._flush_tool_calls(messages, pending_tool_calls)
                messages.append({
                    "ts": e.ts,
                    "role": "system",
                    "content": e.summary or t("activity.heartbeat_start"),
                    "from_person": "",
                    "tool_calls": [],
                    "_trigger": "heartbeat",
                })

            elif e.type == "heartbeat_end":
                self._flush_tool_calls(messages, pending_tool_calls)
                messages.append({
                    "ts": e.ts,
                    "role": "system",
                    "content": e.summary or e.content or t("activity.heartbeat_end"),
                    "from_person": "",
                    "tool_calls": [],
                    "_trigger": "heartbeat",
                })

            elif e.type == "cron_executed":
                self._flush_tool_calls(messages, pending_tool_calls)
                task_name = e.meta.get("task_name", "")
                content = e.summary or e.content or task_name or t("activity.cron_task_exec")
                messages.append({
                    "ts": e.ts,
                    "role": "system",
                    "content": content,
                    "from_person": "",
                    "tool_calls": [],
                    "_trigger": "cron",
                })

            elif e.type == "error":
                self._flush_tool_calls(messages, pending_tool_calls)
                messages.append({
                    "ts": e.ts,
                    "role": "system",
                    "content": t("activity.error_prefix") + (e.content or e.summary or ""),
                    "from_person": "",
                    "tool_calls": [],
                })

        self._flush_tool_calls(messages, pending_tool_calls)

        return messages

    @staticmethod
    def _flush_tool_calls(
        messages: list[dict[str, Any]],
        pending: list[dict[str, Any]],
    ) -> None:
        """Attach pending tool calls to the last assistant message."""
        if not pending:
            return
        for msg in reversed(messages):
            if msg["role"] == "assistant":
                msg["tool_calls"].extend(pending)
                pending.clear()
                return
        pending.clear()

    @staticmethod
    def _group_into_sessions(
        messages: list[dict[str, Any]],
        gap_minutes: int,
    ) -> list[dict[str, Any]]:
        """Group messages into sessions based on time gaps."""
        if not messages:
            return []

        gap_seconds = gap_minutes * 60
        sessions: list[dict[str, Any]] = []
        current_msgs: list[dict[str, Any]] = []
        current_trigger = "chat"

        for msg in messages:
            msg_trigger = msg.pop("_trigger", None)
            if msg_trigger:
                current_trigger = msg_trigger

            if current_msgs:
                prev_ts = current_msgs[-1]["ts"]
                if time_diff(prev_ts, msg["ts"]) >= gap_seconds:
                    sessions.append({
                        "session_start": current_msgs[0]["ts"],
                        "session_end": current_msgs[-1]["ts"],
                        "trigger": current_trigger,
                        "messages": current_msgs,
                    })
                    current_msgs = []
                    current_trigger = msg_trigger or "chat"

            current_msgs.append(msg)

        if current_msgs:
            sessions.append({
                "session_start": current_msgs[0]["ts"],
                "session_end": current_msgs[-1]["ts"],
                "trigger": current_trigger,
                "messages": current_msgs,
            })

        return sessions
