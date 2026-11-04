from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.

"""Conversation view mixin for ActivityLogger.

Internal module — import from :mod:`core.activity.logger` instead.
"""

import json
import logging
import re
import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

from core.activity.format import EVENT_SETS
from core.activity.models import (
    ActivityEntry,
    time_diff,
)
from core.activity.slack_mirror import (
    aliased_slack_users,
    apply_slack_mirror,
    is_slack_send_command,
    is_user_addressed_slack_tool_use,
)
from core.i18n import t
from core.time_utils import now_local

logger = logging.getLogger("animaworks.activity")


_TOOL_RESULT_TRUNCATE = 1500
_TOOL_INPUT_TRUNCATE = 500
_FEED_CACHE_MAX = 256
_FEED_MAX_DAYS = 14
_FEED_MAX_FILES = 30
_FEED_MAX_RAW_ENTRIES = 20_000
_FEED_MAX_BACKGROUND_ITEMS = 200
_EXTERNAL_PLATFORM_SOURCES = frozenset({"slack", "chatwork", "googlechat", "discord", "zoom"})
_FEED_CACHE: OrderedDict[tuple[Any, ...], tuple[tuple[Any, ...] | None, dict[str, Any]]] = OrderedDict()
_FEED_CACHE_LOCK = threading.RLock()
_FEED_INFLIGHT: dict[tuple[Any, ...], threading.Event] = {}
_FEED_PREFETCH_LOCK = threading.Lock()
_FEED_PREFETCHING: set[str] = set()
_FEED_PREFETCH_EXECUTOR = ThreadPoolExecutor(max_workers=4, thread_name_prefix="conversation-feed")


def _iter_jsonl_lines_reverse(path: Path):
    """Yield non-empty UTF-8 lines from a JSONL file, newest first."""
    chunk_size = 64 * 1024
    with path.open("rb") as file:
        position = file.seek(0, 2)
        remainder = b""
        while position > 0:
            chunk_start = max(0, position - chunk_size)
            file.seek(chunk_start)
            chunk = file.read(position - chunk_start)
            position = chunk_start
            parts = (chunk + remainder).split(b"\n")
            remainder = parts.pop(0)
            for line in reversed(parts):
                if line.strip():
                    yield line.decode("utf-8", errors="replace")
        if remainder.strip():
            yield remainder.decode("utf-8", errors="replace")


def _source_key_for_entry(entry: ActivityEntry) -> str:
    """Return a compact source key for chat rendering."""
    if entry.type.startswith("heartbeat_"):
        return "heartbeat"
    if entry.type == "cron_executed":
        return "cron"
    if entry.type.startswith("task_exec_"):
        return "task"
    if entry.channel == "inbox":
        return "inbox"
    if entry.channel == "chat":
        return "chat"
    if entry.type == "message_sent":
        return "dm"
    return entry.channel or "chat"


def _is_inbox_raw_entry(raw: dict[str, Any]) -> bool:
    meta = raw.get("meta", {})
    trigger = meta.get("trigger", "") if isinstance(meta, dict) else ""
    session_type = meta.get("session_type", "") if isinstance(meta, dict) else ""
    return (
        raw.get("channel") == "inbox"
        or session_type == "inbox"
        or trigger == "inbox"
        or (isinstance(trigger, str) and trigger.startswith("inbox:"))
    )


def _is_slack_mirror_raw_candidate(raw: dict[str, Any], aliases: dict[str, str]) -> bool:
    """Return True for inbox rows needed to derive Slack mirror messages."""
    if not aliases:
        return False
    meta = raw.get("meta", {})
    if not isinstance(meta, dict):
        return False
    entry_type = raw.get("type", "")
    if entry_type == "message_received":
        trigger = str(meta.get("trigger", ""))
        if meta.get("from_type") != "slack" or not trigger.startswith("inbox:slack:"):
            return False
        return trigger[len("inbox:slack:") :].split(",", 1)[0].strip() in aliases
    if entry_type == "tool_use":
        if raw.get("tool") != "Bash":
            return False
        args = meta.get("args")
        command = args.get("command") if isinstance(args, dict) else ""
        return isinstance(command, str) and is_slack_send_command(command)
    return False


def _external_user_aliases() -> dict[str, str]:
    """Map external chat user IDs (Slack/Discord/Chatwork) to configured user aliases."""
    try:
        from core.config.io import load_config

        aliases = load_config().external_messaging.user_aliases
    except Exception:
        logger.debug("Failed to load external user aliases", exc_info=True)
        return {}
    mapping: dict[str, str] = {}
    for alias, cfg in aliases.items():
        for ext_id in (cfg.slack_user_id, cfg.discord_user_id, cfg.chatwork_room_id):
            if ext_id:
                mapping[str(ext_id)] = alias
    return mapping


def _split_interim_texts(content: str, interim: list[Any]) -> tuple[str, list[str]]:
    """Remove progress remarks spoken before tool calls from the start of a reply.

    Returns the remaining reply body and the remarks that were actually found.
    When nothing lines up, the content is returned unchanged.
    """
    rest = content.lstrip()
    found: list[str] = []
    for item in interim:
        text = str(item or "").strip()
        if not text:
            continue
        if not rest.startswith(text):
            break
        found.append(text)
        rest = rest[len(text) :].lstrip()
    if not found or not rest:
        return content, []
    return rest, found


def _truncate_tool_field(value: Any, limit: int) -> Any:
    """Truncate tool input/result for the conversation view API."""
    if isinstance(value, str):
        if len(value) <= limit:
            return value
        return value[:limit] + f"\n…({len(value) - limit} chars truncated)"
    if isinstance(value, (dict, list)):
        s = json.dumps(value, ensure_ascii=False)
        if len(s) <= limit:
            return value
        return s[:limit] + f"\n…({len(s) - limit} chars truncated)"
    return value


def _tool_target_from_input(value: Any) -> str:
    """Extract a compact, user-facing target from a tool's input."""
    obj = value
    if isinstance(value, str):
        try:
            obj = json.loads(value)
        except (TypeError, ValueError):
            obj = value

    target: Any = None
    if isinstance(obj, dict):
        for key in ("file_path", "path", "command", "query", "pattern", "url", "to", "name", "channel"):
            candidate = obj.get(key)
            if candidate:
                target = candidate
                break
    elif obj is not None:
        target = obj

    if target is None:
        return ""
    if isinstance(target, (dict, list)):
        target = json.dumps(target, ensure_ascii=False)
    compact = re.sub(r"\s+", " ", str(target)).strip()
    return compact[:80]


def _summarize_tool_call(tool_call: dict[str, Any]) -> dict[str, Any]:
    """Return only the fields needed to list a tool call in chat history."""
    return {
        "tool_name": tool_call.get("tool_name", ""),
        "tool_use_id": tool_call.get("tool_use_id", ""),
        "is_error": bool(tool_call.get("is_error", False)),
        "target": tool_call.get("target")
        or _tool_target_from_input(tool_call.get("input", tool_call.get("input_summary", ""))),
    }


class ConversationMixin:
    """Mixin providing conversation view methods for ActivityLogger."""

    _CONVERSATION_TYPES = EVENT_SETS["chat"]

    def get_conversation_view(
        self,
        *,
        before: str | None = None,
        limit: int = 50,
        session_gap_minutes: int = 10,
        thread_id: str | None = None,
        strict_thread: bool = False,
        tool_detail: str = "full",
    ) -> dict[str, Any]:
        """Build a conversation view from activity log entries.

        Pairs tool events, groups into sessions (by gap/trigger change),
        returns ``{"sessions": [...], "has_more": bool, "next_before": str}``.
        """
        entries = self._load_conversation_entries(
            before=before,
            limit=limit,
            thread_id=thread_id,
            strict_thread=strict_thread,
        )
        messages = self._entries_to_messages(entries, tool_detail=tool_detail)

        if len(messages) <= limit and entries:
            entries = self._load_conversation_entries(
                before=before,
                limit=limit,
                thread_id=thread_id,
                strict_thread=strict_thread,
                _target_multiplier=15,
            )
            messages = self._entries_to_messages(entries, tool_detail=tool_detail)

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

    def get_conversation_feed(
        self,
        *,
        thread_id: str = "default",
        before: str | None = None,
        min_user_items: int = 10,
        strict_thread: bool = False,
        _schedule_prefetch: bool = True,
    ) -> dict[str, Any]:
        """Return a compact feed with user messages and lazy background groups."""
        return _cached_conversation_feed(
            self,
            thread_id=thread_id,
            before=before,
            min_user_items=min_user_items,
            strict_thread=strict_thread,
            schedule_prefetch=_schedule_prefetch,
        )

    def get_conversation_background(
        self,
        *,
        thread_id: str = "default",
        from_ts: str,
        to_ts: str,
        strict_thread: bool = False,
    ) -> dict[str, Any]:
        """Return only background messages in a feed group, in history session shape."""
        scan_from = (_parse_timestamp(from_ts) - timedelta(days=1)).isoformat()
        entries, scan_truncated = self._load_feed_entries(
            thread_id=thread_id,
            strict_thread=strict_thread,
            from_ts=scan_from,
            to_ts=to_ts,
            scan_days=_FEED_MAX_DAYS + 1,
        )
        messages = self._entries_to_messages(entries, tool_detail="summary")
        sessions = self._group_into_sessions(messages, gap_minutes=10)
        classification = _classify_feed_sessions(self, sessions, entries)

        background: list[dict[str, Any]] = []
        for session in sessions:
            session_messages = [
                message
                for message in session["messages"]
                if not classification[id(message)][0]
                and _timestamp_at_or_after(str(message.get("ts") or ""), from_ts)
                and _timestamp_at_or_before(str(message.get("ts") or ""), to_ts)
            ]
            if not session_messages:
                continue
            background.append(
                {
                    "session_start": session_messages[0]["ts"],
                    "session_end": session_messages[-1]["ts"],
                    "trigger": session["trigger"],
                    "messages": session_messages,
                }
            )

        total = sum(len(session["messages"]) for session in background)
        truncated = scan_truncated or total > _FEED_MAX_BACKGROUND_ITEMS
        if truncated:
            remaining = _FEED_MAX_BACKGROUND_ITEMS
            limited: list[dict[str, Any]] = []
            for session in reversed(background):
                if remaining <= 0:
                    break
                messages_in_session = session["messages"][-remaining:]
                limited.append(
                    {
                        **session,
                        "session_start": messages_in_session[0]["ts"],
                        "messages": messages_in_session,
                    }
                )
                remaining -= len(messages_in_session)
            background = list(reversed(limited))

        return {"sessions": background, "truncated": truncated}

    def _load_feed_entries(
        self,
        *,
        thread_id: str,
        strict_thread: bool = False,
        before: str | None = None,
        min_user_items: int | None = None,
        from_ts: str | None = None,
        to_ts: str | None = None,
        scan_days: int = _FEED_MAX_DAYS,
    ) -> tuple[list[ActivityEntry], bool]:
        """Read recent conversation events newest-first within explicit scan bounds."""
        today = now_local().date()
        if from_ts and to_ts:
            range_start = _local_timestamp_date(from_ts)
            range_end = _local_timestamp_date(to_ts)
            first_day = min(today, range_end)
            last_day = max(range_start, first_day - timedelta(days=scan_days - 1))
            has_more = range_start < last_day
        else:
            cursor_day = _local_timestamp_date(before) if before else today
            first_day = min(today, cursor_day)
            last_day = first_day - timedelta(days=scan_days - 1)
            has_more = False

        sibling_names = self._sibling_anima_names()
        aliases = aliased_slack_users()
        mirror_enabled = thread_id != "inbox"
        descending_entries: list[ActivityEntry] = []
        context_entries: list[ActivityEntry] = []
        raw_count = 0
        user_count = 0
        target_response_ts = ""
        pending_mirror_stop = False
        mirror_lookahead = 0
        files_scanned = 0
        scan_dates = [first_day - timedelta(days=offset) for offset in range((first_day - last_day).days + 1)]
        # A named thread is a handful of rows in a busy log. Its rows must hold
        # the quoted id (validate_thread_id allows no characters JSON escapes),
        # so other lines are skipped unparsed and do not count towards the raw
        # cap, which would otherwise end the scan before reaching older rows.
        thread_marker = f'"{thread_id}"' if thread_id not in ("default", "inbox") else ""

        for day_index, target_day in enumerate(scan_dates):
            path = self._log_dir / f"{target_day.isoformat()}.jsonl"  # type: ignore[attr-defined]
            if not path.is_file():
                continue
            files_scanned += 1
            if files_scanned > _FEED_MAX_FILES:
                has_more = True
                break

            line_iter = iter(_iter_jsonl_lines_reverse(path))
            file_exhausted = False
            stop_scan = False
            while raw_count < _FEED_MAX_RAW_ENTRIES:
                try:
                    line = next(line_iter)
                except StopIteration:
                    file_exhausted = True
                    break
                if thread_marker and thread_marker not in line:
                    continue
                raw_count += 1
                try:
                    raw = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(raw, dict) or raw.get("type") not in self._CONVERSATION_TYPES:
                    continue
                raw_ts = str(raw.get("ts") or "")
                if before and raw_ts >= before:
                    continue
                if from_ts and not _timestamp_at_or_after(raw_ts, from_ts):
                    continue
                if to_ts and not _timestamp_at_or_before(raw_ts, to_ts):
                    continue

                is_inbox_entry = _is_inbox_raw_entry(raw)
                entry_type = raw.get("type", "")
                raw_meta = raw.get("meta", {})
                meta = raw_meta if isinstance(raw_meta, dict) else {}
                mirror_candidate = mirror_enabled and _is_slack_mirror_raw_candidate(raw, aliases)
                if (
                    thread_id != "inbox"
                    and is_inbox_entry
                    and entry_type not in ("human_notify", "human_reply", "external_reply_sent")
                    and not mirror_candidate
                ):
                    continue

                if thread_id is not None:
                    has_thread_id = "thread_id" in meta
                    if strict_thread:
                        if not has_thread_id or meta.get("thread_id") != thread_id:
                            continue
                    else:
                        default_tid = "inbox" if is_inbox_entry and thread_id == "inbox" else "default"
                        entry_tid = meta.get("thread_id", default_tid)
                        if entry_tid != thread_id and not (
                            thread_id == "default" and (mirror_candidate or entry_type == "external_reply_sent")
                        ):
                            continue

                if "from" in raw:
                    raw["from_person"] = raw.pop("from")
                if "to" in raw:
                    raw["to_person"] = raw.pop("to")
                raw["meta"] = meta
                try:
                    entry = ActivityEntry(
                        **{key: value for key, value in raw.items() if key in ActivityEntry.__dataclass_fields__}
                    )
                except (TypeError, ValueError, KeyError):
                    continue
                descending_entries.append(entry)
                context_entries.append(entry)

                if pending_mirror_stop:
                    if entry.type == "inbox_processing_start":
                        stop_scan = True
                        break
                    mirror_lookahead += 1
                    if mirror_lookahead > 100:
                        stop_scan = True
                        break

                if min_user_items is None or not _is_user_item_entry(entry, sibling_names, self._anima_name, aliases):
                    continue
                user_count += 1
                if user_count < min_user_items:
                    continue
                if entry.type == "response_sent":
                    target_response_ts = entry.ts
                    continue
                if target_response_ts and entry.type != "message_received":
                    continue
                if target_response_ts and entry.ts >= target_response_ts:
                    continue
                if mirror_candidate:
                    # Keep scanning until the inbox-processing-start row is
                    # included, so historical trigger resolution can see it.
                    pending_mirror_stop = True
                    continue
                stop_scan = True
                break

            if stop_scan:
                has_more = _feed_has_older_lines(line_iter, file_exhausted) or any(
                    (self._log_dir / f"{older_day.isoformat()}.jsonl").is_file()
                    for older_day in scan_dates[day_index + 1 :]
                )
                break

            if raw_count >= _FEED_MAX_RAW_ENTRIES:
                has_more = _feed_has_older_lines(line_iter, file_exhausted) or any(
                    (self._log_dir / f"{older_day.isoformat()}.jsonl").is_file()
                    for older_day in scan_dates[day_index + 1 :]
                )
                break

        if min_user_items is not None and not has_more and raw_count < _FEED_MAX_RAW_ENTRIES:
            has_more = _has_log_before_date(self._log_dir, last_day)

        descending_entries.reverse()
        if mirror_enabled and descending_entries:
            context_entries.reverse()
            descending_entries = apply_slack_mirror(
                descending_entries,
                context_entries=context_entries,
                anima_name=self._anima_name,
                aliases=aliases,
            )
        return descending_entries, has_more

    def get_conversation_tool_calls(
        self,
        message_ts: str,
        *,
        thread_id: str = "default",
        strict_thread: bool = False,
    ) -> list[dict[str, Any]] | None:
        """Load full tool calls attached to a specific assistant message."""
        try:
            message_time = datetime.fromisoformat(message_ts.replace("Z", "+00:00"))
        except (AttributeError, TypeError, ValueError):
            return None

        # Tool calls made after a message (until the next turn) are attached to it too,
        # so read the whole day rather than stopping at the message itself.
        entries = self._load_conversation_entries(
            before=None,
            limit=2000,
            thread_id=thread_id,
            strict_thread=strict_thread,
            _specific_day=message_time.date(),
        )

        message = next(
            (
                item
                for item in self._entries_to_messages(entries)
                if item.get("role") == "assistant" and item.get("ts") == message_ts
            ),
            None,
        )
        tool_calls = message.get("tool_calls") if message else None
        return tool_calls if tool_calls else None

    def _load_conversation_entries(
        self,
        *,
        before: str | None = None,
        limit: int = 50,
        thread_id: str | None = None,
        strict_thread: bool = False,
        _target_multiplier: int = 3,
        _specific_day: date | None = None,
    ) -> list[ActivityEntry]:
        """Load conversation-relevant entries, scanning enough days for *limit* messages."""
        target_raw = limit * _target_multiplier + 50
        entries: list[ActivityEntry] = []
        context_entries: list[ActivityEntry] = []
        aliases = aliased_slack_users()
        mirror_enabled = thread_id != "inbox"
        today = now_local().date()
        max_scan_days = 365
        # Thread filters may match few entries and never reach target_raw;
        # without a file cap the scan reads (and JSON-parses) every log file
        # ever written, blocking for seconds per request. Named threads have
        # a conversation-file fallback for older history.
        max_files_scanned = 30
        files_scanned = 0

        target_days = (
            [_specific_day]
            if _specific_day is not None
            else [today - timedelta(days=day_offset) for day_offset in range(max_scan_days)]
        )
        for target in target_days:
            day_offset = (today - target).days
            path = self._log_dir / f"{target.isoformat()}.jsonl"  # type: ignore[attr-defined]
            if not path.exists():
                if day_offset > 30 and not entries:
                    break
                continue

            day_entries: list[ActivityEntry] = []
            try:
                for line_num, line in enumerate(
                    path.read_text(encoding="utf-8").splitlines(),
                    start=1,
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
                    is_inbox_entry = _is_inbox_raw_entry(raw)
                    raw_meta = raw.get("meta", {})
                    meta = raw_meta if isinstance(raw_meta, dict) else {}
                    mirror_candidate = mirror_enabled and _is_slack_mirror_raw_candidate(raw, aliases)
                    # call_human notify/reply stay on the default thread even when
                    # the activity row was written during inbox/heartbeat handling.
                    if (
                        thread_id != "inbox"
                        and is_inbox_entry
                        and raw.get("type") not in ("human_notify", "human_reply", "external_reply_sent")
                        and not mirror_candidate
                    ):
                        continue
                    ts = raw.get("ts", "")
                    if before and ts >= before:
                        continue
                    if thread_id is not None:
                        has_thread_id = isinstance(meta, dict) and "thread_id" in meta
                        if strict_thread:
                            if not has_thread_id:
                                continue
                            if meta.get("thread_id") != thread_id:
                                continue
                        else:
                            default_tid = "inbox" if is_inbox_entry and thread_id == "inbox" else "default"
                            entry_tid = meta.get("thread_id", default_tid) if isinstance(meta, dict) else default_tid
                            if entry_tid != thread_id and not (
                                thread_id == "default"
                                and (mirror_candidate or raw.get("type") == "external_reply_sent")
                            ):
                                continue
                    if "from" in raw:
                        raw["from_person"] = raw.pop("from")
                    if "to" in raw:
                        raw["to_person"] = raw.pop("to")
                    raw["meta"] = meta
                    try:
                        entry = ActivityEntry(
                            **{k: v for k, v in raw.items() if k in ActivityEntry.__dataclass_fields__}
                        )
                    except (TypeError, ValueError, KeyError):
                        logger.debug("Skipping malformed entry at line %d in %s", line_num, path)
                        continue
                    entry._line_number = line_num
                    day_entries.append(entry)
            except Exception:
                logger.exception("Failed to read activity log %s", path)

            entries = day_entries + entries
            context_entries = day_entries + context_entries
            files_scanned += 1
            if len(entries) >= target_raw or files_scanned >= max_files_scanned:
                break

        if mirror_enabled and entries:
            entries = apply_slack_mirror(
                entries,
                context_entries=context_entries,
                anima_name=self._anima_name,
                aliases=aliases,
            )
        return entries

    def _entries_to_messages(
        self,
        entries: list[ActivityEntry],
        *,
        tool_detail: str = "full",
    ) -> list[dict[str, Any]]:
        """Convert raw activity entries to conversation messages (pair + nest tools)."""
        from core.activity.format import pair_tool_events
        from core.anima.emotion_tag import extract_emotion

        result_by_use_id: dict[int, ActivityEntry] = {
            id(ex.tool_use): ex.result for ex in pair_tool_events(entries) if ex.result is not None
        }

        messages: list[dict[str, Any]] = []
        pending_tool_calls: list[dict[str, Any]] = []
        user_aliases = _external_user_aliases()

        for e in entries:
            if e.type == "message_received":
                self._flush_tool_calls(messages, pending_tool_calls)
                msg = {
                    "ts": e.ts,
                    "role": "human",
                    "content": e.content,
                    "from_person": user_aliases.get(e.from_person, e.from_person),
                    "to_person": e.to_person,
                    "source_key": _source_key_for_entry(e),
                    "tool_calls": [],
                }
                files = e.meta.get("files") or []
                if isinstance(files, list) and files:
                    msg["files"] = files
                messages.append(msg)

            elif e.type == "message_sent":
                self._flush_tool_calls(messages, pending_tool_calls)
                messages.append(
                    {
                        "ts": e.ts,
                        "role": "assistant",
                        "content": e.content,
                        "from_person": "",
                        "to_person": e.to_person,
                        "source_key": _source_key_for_entry(e),
                        "tool_calls": [],
                    }
                )

            elif e.type == "response_sent":
                clean_content, emotion = extract_emotion(e.content)
                msg = {
                    "ts": e.ts,
                    "role": "assistant",
                    "content": clean_content,
                    "emotion": emotion if clean_content != e.content else None,
                    "from_person": "",
                    "to_person": e.to_person,
                    "source_key": _source_key_for_entry(e),
                    "tool_calls": [],
                }
                interim = e.meta.get("interim_texts")
                if isinstance(interim, list) and interim:
                    body, found = _split_interim_texts(clean_content, interim)
                    if found:
                        msg["content"] = body
                        msg["interim_texts"] = found
                if e.meta.get("thinking_text"):
                    msg["thinking_text"] = e.meta["thinking_text"]
                images = e.meta.get("images") or e.meta.get("artifacts") or []
                if isinstance(images, list) and images:
                    msg["images"] = images
                files = e.meta.get("files") or []
                if isinstance(files, list) and files:
                    msg["files"] = files
                messages.append(msg)
                if pending_tool_calls:
                    msg["tool_calls"].extend(pending_tool_calls)
                    pending_tool_calls.clear()

            elif e.type == "tool_use":
                # human_notify card is the canonical display for call_human.
                if e.tool == "call_human":
                    continue
                tid = e.meta.get("tool_use_id", "")
                result_entry = result_by_use_id.get(id(e))
                raw_input = e.meta.get("args", e.content)
                raw_result = result_entry.content if result_entry else ""
                tc: dict[str, Any] = {
                    "tool_use_id": tid,
                    "tool_name": e.tool,
                    "input": _truncate_tool_field(raw_input, _TOOL_INPUT_TRUNCATE),
                    "result": _truncate_tool_field(raw_result, _TOOL_RESULT_TRUNCATE),
                    "is_error": (result_entry.meta.get("is_error", False) if result_entry else False),
                }
                if e.meta.get("blocked"):
                    tc["is_error"] = True
                    tc["result"] = t("activity.blocked", reason=e.meta.get("reason", ""))
                if tool_detail == "summary":
                    tc = _summarize_tool_call({**tc, "input": raw_input})
                pending_tool_calls.append(tc)

            elif e.type == "tool_result":
                pass

            elif e.type == "human_notify":
                self._flush_tool_calls(messages, pending_tool_calls)
                messages.append(
                    {
                        "ts": e.ts,
                        "role": "system",
                        "type": "human_notify",
                        "content": e.content,
                        "subject": e.meta.get("subject", ""),
                        "priority": e.meta.get("priority", "normal"),
                        "callback_id": e.meta.get("callback_id", ""),
                        "notification_id": e.meta.get("notification_id") or e.ts,
                        # Low-priority notices (FYI, news digests) are for reading, not answering.
                        "needs_reply": e.meta.get("priority", "normal") != "low" or bool(e.meta.get("callback_id")),
                        "replied": False,
                        "emotion": None,
                        "from_person": "",
                        "source_key": "call_human",
                        "tool_calls": [],
                    }
                )

            elif e.type == "external_reply_sent":
                self._flush_tool_calls(messages, pending_tool_calls)
                msg = {
                    "ts": e.ts,
                    "role": "assistant",
                    "type": "external_reply",
                    "content": e.content,
                    "via": e.via,
                    "to_person": user_aliases.get(e.to_person, e.to_person),
                    "from_person": "",
                    "source_key": "external_reply",
                    "tool_calls": [],
                }
                if e.meta.get("derived"):
                    msg["derived"] = True
                messages.append(msg)

            elif e.type == "human_reply":
                self._flush_tool_calls(messages, pending_tool_calls)
                msg = {
                    "ts": e.ts,
                    "role": "human",
                    "type": "human_reply",
                    "content": e.content,
                    "from_person": user_aliases.get(e.from_person, e.from_person),
                    "via": e.via,
                    "notification_id": e.meta.get("notification_id", ""),
                    "callback_id": e.meta.get("callback_id", ""),
                    "source_key": "call_human_reply",
                    "tool_calls": [],
                }
                if e.meta.get("derived"):
                    msg["derived"] = True
                messages.append(msg)

            elif e.type == "voice_call":
                self._flush_tool_calls(messages, pending_tool_calls)
                messages.append(
                    {
                        "ts": e.ts,
                        "role": "assistant",
                        "type": "voice_call",
                        "content": e.summary or e.content,
                        "started_at": e.meta.get("started_at", ""),
                        "duration_seconds": e.meta.get("duration_seconds", 0),
                        "transcript": e.meta.get("transcript", []),
                        "source_key": "voice_call",
                        "tool_calls": [],
                    }
                )

            elif e.type == "heartbeat_start":
                self._flush_tool_calls(messages, pending_tool_calls)
                messages.append(
                    {
                        "ts": e.ts,
                        "role": "system",
                        "content": e.summary or t("activity.heartbeat_start"),
                        "from_person": "",
                        "source_key": _source_key_for_entry(e),
                        "tool_calls": [],
                        "_trigger": "heartbeat",
                    }
                )

            elif e.type == "heartbeat_end":
                self._flush_tool_calls(messages, pending_tool_calls)
                messages.append(
                    {
                        "ts": e.ts,
                        "role": "system",
                        "content": e.summary or e.content or t("activity.heartbeat_end"),
                        "from_person": "",
                        "source_key": _source_key_for_entry(e),
                        "tool_calls": [],
                        "_trigger": "heartbeat",
                    }
                )

            elif e.type == "cron_executed":
                self._flush_tool_calls(messages, pending_tool_calls)
                task_name = e.meta.get("task_name", "")
                content = e.summary or e.content or task_name or t("activity.cron_task_exec")
                messages.append(
                    {
                        "ts": e.ts,
                        "role": "system",
                        "content": content,
                        "from_person": "",
                        "source_key": _source_key_for_entry(e),
                        "tool_calls": [],
                        "_trigger": "cron",
                    }
                )

            elif e.type == "task_exec_start":
                self._flush_tool_calls(messages, pending_tool_calls)
                messages.append(
                    {
                        "ts": e.ts,
                        "role": "system",
                        "content": e.summary or t("activity.task_exec_start_label"),
                        "from_person": "",
                        "source_key": _source_key_for_entry(e),
                        "tool_calls": [],
                        "_trigger": "task",
                    }
                )

            elif e.type == "task_exec_end":
                self._flush_tool_calls(messages, pending_tool_calls)
                messages.append(
                    {
                        "ts": e.ts,
                        "role": "system",
                        "content": e.summary or e.content or t("activity.task_exec_end_label"),
                        "from_person": "",
                        "source_key": _source_key_for_entry(e),
                        "tool_calls": [],
                        "_trigger": "task",
                    }
                )

            elif e.type == "error":
                self._flush_tool_calls(messages, pending_tool_calls)
                messages.append(
                    {
                        "ts": e.ts,
                        "role": "system",
                        "content": t("activity.error_prefix") + (e.content or e.summary or ""),
                        "from_person": "",
                        "source_key": _source_key_for_entry(e),
                        "tool_calls": [],
                    }
                )

        self._flush_tool_calls(messages, pending_tool_calls)

        notifications = [msg for msg in messages if msg.get("type") == "human_notify"]
        for reply in (msg for msg in messages if msg.get("type") == "human_reply"):
            reply_ts = str(reply.get("ts") or "")
            candidates = [notify for notify in notifications if str(notify.get("ts") or "") <= reply_ts]
            if not candidates:
                continue

            notification_id = reply.get("notification_id")
            callback_id = reply.get("callback_id")
            matched = next(
                (
                    notify
                    for notify in reversed(candidates)
                    if (notification_id and notify.get("notification_id") == notification_id)
                    or (callback_id and notify.get("callback_id") == callback_id)
                ),
                None,
            )
            if matched is None:
                matched = next((notify for notify in reversed(candidates) if not notify["replied"]), candidates[-1])
            matched["replied"] = True
            if notification_id or callback_id:
                quote = str(matched.get("subject") or "") or " ".join(str(matched.get("content") or "").split())
                reply["reply_quote"] = quote[:80]

        # Answering in the normal chat (not via the reply button) also answers earlier notifications.
        if notifications:
            anima_names = self._sibling_anima_names()
            human_ts = [
                str(msg.get("ts") or "")
                for msg in messages
                if msg.get("role") == "human"
                and msg.get("type") != "human_reply"
                and msg.get("from_person")
                and msg.get("from_person") not in anima_names
            ]
            for notify in notifications:
                if not notify["replied"] and any(ts > str(notify.get("ts") or "") for ts in human_ts):
                    notify["replied"] = True

        if tool_detail == "summary":
            for message in messages:
                calls = message.get("tool_calls")
                if message.get("role") == "assistant" and calls:
                    message["tool_calls"] = [_summarize_tool_call(call) for call in calls]
                    message["tool_calls_detail"] = "lazy"

        return messages

    def _sibling_anima_names(self) -> set[str]:
        anima_dir = getattr(self, "anima_dir", None)
        try:
            return {p.name for p in anima_dir.parent.iterdir() if p.is_dir()} if anima_dir else set()
        except OSError:
            return set()

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
        """Group messages into sessions by time gaps and trigger changes."""
        if not messages:
            return []

        gap_seconds = gap_minutes * 60
        sessions: list[dict[str, Any]] = []
        current_msgs: list[dict[str, Any]] = []
        current_trigger = "chat"

        for msg in messages:
            msg_trigger = msg.pop("_trigger", None)
            effective_trigger = msg_trigger or "chat"

            if current_msgs:
                prev_ts = current_msgs[-1]["ts"]
                trigger_changed = effective_trigger != current_trigger
                time_gap = time_diff(prev_ts, msg["ts"]) >= gap_seconds

                if trigger_changed or time_gap:
                    sessions.append(
                        {
                            "session_start": current_msgs[0]["ts"],
                            "session_end": current_msgs[-1]["ts"],
                            "trigger": current_trigger,
                            "messages": current_msgs,
                        }
                    )
                    current_msgs = []

            current_trigger = effective_trigger
            current_msgs.append(msg)

        if current_msgs:
            sessions.append(
                {
                    "session_start": current_msgs[0]["ts"],
                    "session_end": current_msgs[-1]["ts"],
                    "trigger": current_trigger,
                    "messages": current_msgs,
                }
            )

        return sessions


def _parse_timestamp(value: str | None) -> datetime:
    if not value:
        return now_local()
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return parsed.replace(tzinfo=now_local().tzinfo) if parsed.tzinfo is None else parsed


def _local_timestamp_date(value: str) -> date:
    local_timezone = now_local().tzinfo
    return _parse_timestamp(value).astimezone(local_timezone).date()


def _timestamp_at_or_after(value: str, bound: str) -> bool:
    try:
        return _parse_timestamp(value).astimezone(UTC) >= _parse_timestamp(bound).astimezone(UTC)
    except (AttributeError, TypeError, ValueError):
        return False


def _timestamp_at_or_before(value: str, bound: str) -> bool:
    try:
        return _parse_timestamp(value).astimezone(UTC) <= _parse_timestamp(bound).astimezone(UTC)
    except (AttributeError, TypeError, ValueError):
        return False


def _feed_has_older_lines(line_iter: Any, file_exhausted: bool) -> bool:
    if file_exhausted:
        return False
    try:
        next(line_iter)
    except StopIteration:
        return False
    return True


def _has_log_before_date(log_dir: Path, cutoff: date) -> bool:
    if not log_dir.is_dir():
        return False
    cutoff_name = cutoff.isoformat()
    return any(path.stem < cutoff_name for path in log_dir.glob("*.jsonl"))


def _is_user_item_entry(
    entry: ActivityEntry,
    anima_names: set[str],
    self_name: str,
    aliases: dict[str, str] | None = None,
) -> bool:
    """Mirror the chat UI's user/background boundary using source events."""
    if entry.type == "external_reply_sent":
        return True
    if entry.type == "tool_use" and is_user_addressed_slack_tool_use(
        entry,
        anima_name=self_name,
        aliases=aliases,
    ):
        return True
    if entry.type == "message_received" and aliases:
        meta = entry.meta or {}
        trigger = str(meta.get("trigger", ""))
        if meta.get("from_type") == "slack" and trigger.startswith("inbox:slack:"):
            uid = trigger[len("inbox:slack:") :].split(",", 1)[0].strip()
            if uid in aliases:
                return True
    if entry.type == "message_received":
        from_type = entry.meta.get("from_type", "")
        is_human_channel = entry.channel in {"chat", "inbox"} or entry.channel in _EXTERNAL_PLATFORM_SOURCES
        is_external_human = from_type in {"human", "external"} or from_type in _EXTERNAL_PLATFORM_SOURCES
        if not is_human_channel and not is_external_human:
            return False
        return entry.from_person not in anima_names and from_type != "anima"
    if entry.type == "response_sent":
        return not (entry.to_person and entry.to_person in anima_names)
    if entry.type in {"human_reply", "voice_call"}:
        return True
    if entry.type == "human_notify":
        return not (entry.from_person == self_name and entry.to_person == self_name)
    return False


def _entry_for_message(
    message: dict[str, Any],
    entries_by_ts: dict[str, list[ActivityEntry]],
) -> ActivityEntry | None:
    candidates = entries_by_ts.get(str(message.get("ts") or ""), [])
    preferred_types = {
        "human_notify": ("human_notify",),
        "human_reply": ("human_reply",),
        "external_reply": ("external_reply_sent",),
        "voice_call": ("voice_call",),
    }.get(str(message.get("type") or ""))
    if not preferred_types:
        if message.get("role") == "human":
            preferred_types = ("message_received", "human_reply")
        elif message.get("role") == "assistant":
            preferred_types = ("response_sent", "message_sent", "external_reply_sent", "voice_call")
        elif message.get("_trigger"):
            preferred_types = ("heartbeat_start", "heartbeat_end", "cron_executed", "task_exec_start", "task_exec_end")
        else:
            preferred_types = (
                "error",
                "heartbeat_start",
                "heartbeat_end",
                "cron_executed",
                "task_exec_start",
                "task_exec_end",
            )
    return next((entry for entry in candidates if entry.type in preferred_types), candidates[0] if candidates else None)


def _classify_feed_sessions(
    activity: ConversationMixin,
    sessions: list[dict[str, Any]],
    entries: list[ActivityEntry],
) -> dict[int, tuple[bool, str, str]]:
    """Return message identity to (is_user_item, background kind, DM peer)."""
    anima_names = activity._sibling_anima_names()
    self_name = str(getattr(activity, "_anima_name", ""))
    aliases = aliased_slack_users()
    entries_by_ts: dict[str, list[ActivityEntry]] = {}
    for entry in entries:
        entries_by_ts.setdefault(entry.ts, []).append(entry)

    classification: dict[int, tuple[bool, str, str]] = {}
    for session in sessions:
        trigger = str(session.get("trigger") or "chat")
        messages = session.get("messages") or []
        background_session = (
            trigger in {"heartbeat", "cron", "task"}
            and bool(messages)
            and all(message.get("role") == "system" for message in messages)
        )
        for message in messages:
            entry = _entry_for_message(message, entries_by_ts)
            if background_session:
                classification[id(message)] = (False, trigger, "")
                continue

            is_user_item = bool(entry and _is_user_item_entry(entry, anima_names, self_name, aliases))
            from_person = str((entry.from_person if entry else message.get("from_person")) or "")
            to_person = str((entry.to_person if entry else message.get("to_person")) or "")
            if is_user_item:
                classification[id(message)] = (True, "", "")
                continue

            if from_person == self_name and to_person == self_name and self_name:
                classification[id(message)] = (False, "notice", "")
                continue

            peer = ""
            if from_person in anima_names and from_person != self_name:
                peer = from_person
            elif to_person in anima_names and to_person != self_name:
                peer = to_person
            if peer:
                classification[id(message)] = (False, "dm", peer)
                continue

            if trigger in {"heartbeat", "cron", "task"}:
                kind = trigger
            elif entry and entry.type.startswith("heartbeat_"):
                kind = "heartbeat"
            elif entry and entry.type == "cron_executed":
                kind = "cron"
            elif entry and entry.type.startswith("task_exec_"):
                kind = "task"
            else:
                kind = "task"
            classification[id(message)] = (False, kind, "")
    return classification


def _build_feed_items(
    activity: ConversationMixin,
    entries: list[ActivityEntry],
    *,
    tool_detail: str = "summary",
) -> list[dict[str, Any]]:
    messages = activity._entries_to_messages(entries, tool_detail=tool_detail)
    sessions = activity._group_into_sessions(messages, gap_minutes=10)
    classification = _classify_feed_sessions(activity, sessions, entries)
    items: list[dict[str, Any]] = []
    current_group: dict[str, Any] | None = None

    def flush_group() -> None:
        nonlocal current_group
        if current_group is None:
            return
        kinds = current_group["kinds"]
        priority = ("dm", "cron", "heartbeat", "task", "notice")
        label_hint = max(priority, key=lambda kind: (kinds.get(kind, 0), -priority.index(kind)))
        items.append(
            {
                "kind": "bg_group",
                "from_ts": current_group["from_ts"],
                "to_ts": current_group["to_ts"],
                "count": current_group["count"],
                "kinds": kinds,
                "participants": sorted(current_group["participants"])[:5],
                "label_hint": label_hint,
            }
        )
        current_group = None

    for session in sessions:
        for message in session["messages"]:
            is_user_item, kind, peer = classification[id(message)]
            if is_user_item:
                flush_group()
                items.append({"kind": "message", "message": message})
                continue

            ts = str(message.get("ts") or "")
            if current_group is None:
                current_group = {
                    "from_ts": ts,
                    "to_ts": ts,
                    "count": 0,
                    "kinds": {},
                    "participants": set(),
                }
            current_group["to_ts"] = ts
            current_group["count"] += 1
            current_group["kinds"][kind] = current_group["kinds"].get(kind, 0) + 1
            if peer and len(current_group["participants"]) < 5:
                current_group["participants"].add(peer)

    flush_group()
    return items


def _latest_activity_file_stamp(activity: ConversationMixin) -> tuple[Any, ...] | None:
    log_dir = getattr(activity, "_log_dir", None)
    if not log_dir or not log_dir.is_dir():
        return None
    latest_path = max(log_dir.glob("*.jsonl"), key=lambda path: path.name, default=None)
    if latest_path is None:
        return None
    try:
        stat = latest_path.stat()
    except OSError:
        return None
    return latest_path.name, stat.st_size, stat.st_mtime_ns


def _cached_conversation_feed(
    activity: ConversationMixin,
    *,
    thread_id: str,
    before: str | None,
    min_user_items: int,
    strict_thread: bool,
    schedule_prefetch: bool,
) -> dict[str, Any]:
    cache_key = (activity._anima_name, thread_id, before, min_user_items, strict_thread)  # type: ignore[attr-defined]
    result = None
    while result is None:
        stamp = _latest_activity_file_stamp(activity) if before is None else None
        with _FEED_CACHE_LOCK:
            cached = _FEED_CACHE.get(cache_key)
            if cached and (before is not None or cached[0] == stamp):
                _FEED_CACHE.move_to_end(cache_key)
                result = json.loads(json.dumps(cached[1], ensure_ascii=False))
                owner = False
                pending = None
            else:
                _FEED_CACHE.pop(cache_key, None)
                pending = _FEED_INFLIGHT.get(cache_key)
                owner = pending is None
                if owner:
                    pending = threading.Event()
                    _FEED_INFLIGHT[cache_key] = pending

        if result is not None:
            break
        if not owner:
            pending.wait()  # type: ignore[union-attr]
            continue

        try:
            entries, has_more = activity._load_feed_entries(
                thread_id=thread_id,
                before=before,
                min_user_items=min_user_items,
                strict_thread=strict_thread,
            )
            items = _build_feed_items(activity, entries)
            next_before = None
            if items:
                oldest = items[0]
                next_before = oldest.get("from_ts") if oldest.get("kind") == "bg_group" else oldest["message"].get("ts")
            elif entries:
                next_before = entries[0].ts
            elif has_more:
                cursor_day = _local_timestamp_date(before) if before else now_local().date()
                next_before = (cursor_day - timedelta(days=_FEED_MAX_DAYS - 1)).isoformat()
            built_result = {"items": items, "has_more": has_more, "next_before": next_before if has_more else None}
            with _FEED_CACHE_LOCK:
                _FEED_CACHE[cache_key] = (stamp, built_result)
                _FEED_CACHE.move_to_end(cache_key)
                while len(_FEED_CACHE) > _FEED_CACHE_MAX:
                    _FEED_CACHE.popitem(last=False)
            result = json.loads(json.dumps(built_result, ensure_ascii=False))
        finally:
            with _FEED_CACHE_LOCK:
                pending = _FEED_INFLIGHT.pop(cache_key, None)
                if pending is not None:
                    pending.set()

    if schedule_prefetch and result.get("has_more") and result.get("next_before"):
        _schedule_feed_prefetch(
            activity,
            thread_id=thread_id,
            before=result["next_before"],
            min_user_items=min_user_items,
            strict_thread=strict_thread,
        )
    return result


def _schedule_feed_prefetch(
    activity: ConversationMixin,
    *,
    thread_id: str,
    before: str,
    min_user_items: int,
    strict_thread: bool,
) -> None:
    anima_name = str(getattr(activity, "_anima_name", ""))
    with _FEED_PREFETCH_LOCK:
        if anima_name in _FEED_PREFETCHING:
            return
        _FEED_PREFETCHING.add(anima_name)

    def prefetch() -> None:
        try:
            activity.get_conversation_feed(
                thread_id=thread_id,
                before=before,
                min_user_items=min_user_items,
                strict_thread=strict_thread,
                _schedule_prefetch=False,
            )
        except Exception:
            logger.debug("Conversation feed prefetch failed for %s", anima_name, exc_info=True)
        finally:
            with _FEED_PREFETCH_LOCK:
                _FEED_PREFETCHING.discard(anima_name)

    _FEED_PREFETCH_EXECUTOR.submit(prefetch)


def clear_conversation_feed_cache() -> None:
    """Clear the process-local conversation feed cache (primarily for tests)."""
    with _FEED_CACHE_LOCK:
        _FEED_CACHE.clear()
