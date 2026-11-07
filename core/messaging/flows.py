from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Build the Board and DM "call" flows from recorded facts.

The builder only joins recorded timestamps and identifiers.  Two inferences are
explicit and deliberately narrow: an old Board row without a delivery record is
treated as delivered to the Animas it mentions, and a reply without a recorded
``reply_to``/``trigger_id`` is inferred from the next message by the same person
within 24 hours.  Both are marked with ``inferred``.
"""

import json
import logging
import statistics
import threading
import time
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Literal

from core.activity.models import ActivityEntry
from core.messaging.board import anima_principal, board_access, human_principal
from core.messaging.board_deliveries import load_deliveries
from core.messaging.board_reads import load_reads, read_board_entries, unread_summary
from core.messaging.mentions import extract_mentions
from core.time_utils import ensure_aware, now_local

logger = logging.getLogger(__name__)

Via = Literal["all", "board", "dm"]

_MAX_CALLS = 5000
_REPLY_WINDOW = timedelta(hours=24)
_WAITING_WINDOW = timedelta(days=2)
_CACHE_TTL_SECONDS = 60.0
_TEXT_LIMIT = 80

# shared_dir -> (days, via, username) -> (built_at, payload)
_CACHE_LOCK = threading.Lock()
_CACHE: dict[tuple[str, int, str, str], tuple[float, dict[str, Any]]] = {}


@dataclass
class _Logged:
    owner: str
    entry: Any


def clear_flows_cache() -> None:
    """Drop the assembled-flow cache; mainly for tests."""
    with _CACHE_LOCK:
        _CACHE.clear()
    with _FILE_CACHE_LOCK:
        _file_cache.clear()


def _parse_ts(value: object) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return ensure_aware(parsed).astimezone(UTC)


def _head(text: object) -> str:
    return str(text or "")[:_TEXT_LIMIT]


def _meta(entry: Any) -> dict[str, Any]:
    meta = getattr(entry, "meta", None)
    return meta if isinstance(meta, dict) else {}


def _str_value(value: object) -> str:
    return value if isinstance(value, str) else ""


def _anima_names(shared_dir: Path) -> list[str]:
    animas_dir = shared_dir.parent / "animas"
    if not animas_dir.is_dir():
        return []
    return sorted(path.name for path in animas_dir.iterdir() if path.is_dir())


def _record_kind(record: dict[str, Any]) -> str:
    kind = record.get("kind")
    if isinstance(kind, str) and kind:
        return kind
    source = str(record.get("source") or "").lower()
    if source == "anima":
        return "anima"
    if source == "human":
        return "human"
    if source in {"slack", "discord"}:
        return "external_user"
    return "operator"


_MESSAGE_TYPES = ("message_sent", "message_received", "dm_sent", "dm_received")
_MESSAGE_MARKERS = tuple(t.encode() for t in _MESSAGE_TYPES)
_FILE_CACHE_LOCK = threading.Lock()
# path -> (mtime, size, message entries in that file)
_file_cache: dict[str, tuple[float, int, list[ActivityEntry]]] = {}


def _message_entries_in(path: Path) -> list[ActivityEntry]:
    """Return message entries of one activity log file, cached by mtime and size.

    Activity logs also hold tool calls and long outputs (hundreds of MB per
    month), so lines are filtered by a byte search before JSON parsing and
    unchanged day files are not re-read.
    """
    try:
        stat = path.stat()
    except OSError:
        return []
    key = str(path)
    with _FILE_CACHE_LOCK:
        cached = _file_cache.get(key)
        if cached is not None and cached[0] == stat.st_mtime and cached[1] == stat.st_size:
            return cached[2]

    entries: list[ActivityEntry] = []
    try:
        with path.open("rb") as fh:
            for raw_line in fh:
                if not any(marker in raw_line for marker in _MESSAGE_MARKERS):
                    continue
                try:
                    raw = json.loads(raw_line)
                except (json.JSONDecodeError, UnicodeDecodeError):
                    continue
                if not isinstance(raw, dict):
                    continue
                if "event" in raw and "type" not in raw:
                    raw["type"] = raw.pop("event")
                if raw.get("type") not in _MESSAGE_TYPES:
                    continue
                if "timestamp" in raw and "ts" not in raw:
                    raw["ts"] = raw.pop("timestamp")
                if "from" in raw:
                    raw["from_person"] = raw.pop("from")
                if "to" in raw:
                    raw["to_person"] = raw.pop("to")
                try:
                    entries.append(
                        ActivityEntry(**{k: v for k, v in raw.items() if k in ActivityEntry.__dataclass_fields__})
                    )
                except (TypeError, ValueError, KeyError):
                    continue
    except OSError:
        logger.warning("Failed to read activity log %s", path, exc_info=True)
        return []

    with _FILE_CACHE_LOCK:
        _file_cache[key] = (stat.st_mtime, stat.st_size, entries)
    return entries


def _load_activity(shared_dir: Path, anima_names: list[str], days: int) -> tuple[list[_Logged], list[_Logged]]:
    animas_dir = shared_dir.parent / "animas"
    sent: list[_Logged] = []
    received: list[_Logged] = []
    today = now_local().date()
    for name in anima_names:
        log_dir = animas_dir / name / "activity_log"
        entries: list[ActivityEntry] = []
        for offset in range(days + 1):
            entries.extend(_message_entries_in(log_dir / f"{(today - timedelta(days=offset)).isoformat()}.jsonl"))
        entries.sort(key=lambda e: e.ts)
        for entry in entries:
            event_type = entry.type
            if event_type in {"message_sent", "dm_sent"}:
                sent.append(_Logged(owner=name, entry=entry))
            elif event_type in {"message_received", "dm_received"}:
                received.append(_Logged(owner=name, entry=entry))
    return sent, received


def _readable_boards(shared_dir: Path, username: str) -> list[str]:
    channels_dir = shared_dir / "channels"
    if not channels_dir.is_dir():
        return []
    principal = human_principal(username)
    names: list[str] = []
    for path in sorted(channels_dir.glob("*.jsonl")):
        channel = path.stem
        if board_access(shared_dir, channel, principal, "read").allowed:
            names.append(channel)
    return names


def _targets_for_row(
    row: dict[str, Any],
    *,
    channel: str,
    post_id: str,
    deliveries: dict[str, dict[str, Any]],
    anima_names: list[str],
) -> tuple[list[tuple[str, bool, str]], bool]:
    """Return (to, delivered, reason) targets and whether they were inferred."""
    record = deliveries.get(post_id) if post_id else None
    if record is not None:
        targets: list[tuple[str, bool, str]] = []
        seen: set[str] = set()
        notified = record.get("notified")
        if isinstance(notified, list):
            for name in notified:
                if isinstance(name, str) and name and name not in seen:
                    seen.add(name)
                    targets.append((name, True, ""))
        not_notified = record.get("not_notified")
        if isinstance(not_notified, dict):
            for name, reason in not_notified.items():
                if isinstance(name, str) and name and name not in seen:
                    seen.add(name)
                    targets.append((name, False, str(reason or "")))
        return targets, False

    text = str(row.get("text") or "")
    mentions = extract_mentions(text, anima_names)
    if "all" in mentions:
        targets_set = set(anima_names)
    else:
        targets_set = mentions - {"all"}
    sender = str(row.get("from") or "")
    if sender in targets_set:
        targets_set.discard(sender)
    return [(name, True, "") for name in sorted(targets_set)], True


def _closest(entries: list[_Logged], after: datetime, *, within: timedelta | None = None) -> _Logged | None:
    candidates: list[tuple[datetime, _Logged]] = []
    for logged in entries:
        ts = _parse_ts(logged.entry.ts)
        if ts is None or ts <= after:
            continue
        if within is not None and ts - after > within:
            continue
        candidates.append((ts, logged))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    return candidates[0][1]


def _inferred_read(
    received_by_owner_from: dict[tuple[str, str], list[_Logged]],
    recipient: str,
    sender: str,
    sent_at: datetime | None,
) -> str | None:
    """Legacy fallback: first message the recipient took in from the sender after sending.

    Older activity entries carry no message id, so the read time is the first
    ``message_received`` from the same sender within 24 hours (marked inferred).
    """
    if sent_at is None:
        return None
    match = _closest(
        received_by_owner_from.get((recipient, sender), []),
        sent_at - timedelta(seconds=1),
        within=timedelta(hours=24),
    )
    return str(match.entry.ts) if match is not None else None


def _min_ts(entries: list[_Logged]) -> str | None:
    values = [(ts, logged.entry.ts) for logged in entries if (ts := _parse_ts(logged.entry.ts)) is not None]
    if not values:
        return None
    values.sort(key=lambda item: item[0])
    return str(values[0][1])


def _board_reply(
    *,
    call_ts: datetime | None,
    channel: str,
    post_id: str,
    sender: str,
    recipient: str,
    rows_by_channel: dict[str, list[dict[str, Any]]],
    sent_by_owner: dict[str, list[_Logged]],
    trigger_message_ids: set[str],
) -> tuple[str | None, str | None, str, bool]:
    if call_ts is not None and post_id:
        for row in rows_by_channel.get(channel, []):
            if _str_value(row.get("reply_to")) != post_id:
                continue
            if str(row.get("from") or "") != recipient:
                continue
            row_ts = _parse_ts(row.get("ts"))
            if row_ts is None or row_ts <= call_ts:
                continue
            return str(row.get("ts") or ""), _str_value(row.get("id")) or None, _head(row.get("text")), False

    if call_ts is not None and trigger_message_ids:
        for logged in sent_by_owner.get(recipient, []):
            entry = logged.entry
            if str(getattr(entry, "to_person", "") or "") != sender:
                continue
            if _str_value(_meta(entry).get("trigger_id")) not in trigger_message_ids:
                continue
            entry_ts = _parse_ts(entry.ts)
            if entry_ts is None or entry_ts <= call_ts:
                continue
            return (
                str(entry.ts),
                _str_value(_meta(entry).get("message_id")) or None,
                _head(entry.content),
                False,
            )

    if call_ts is not None:
        candidates = [
            row
            for row in rows_by_channel.get(channel, [])
            if str(row.get("from") or "") == recipient
            and (row_ts := _parse_ts(row.get("ts"))) is not None
            and call_ts < row_ts <= call_ts + _REPLY_WINDOW
        ]
        if candidates:
            candidates.sort(key=lambda row: _parse_ts(row.get("ts")) or datetime.min.replace(tzinfo=UTC))
            row = candidates[0]
            return str(row.get("ts") or ""), _str_value(row.get("id")) or None, _head(row.get("text")), True
    return None, None, "", False


def _dm_reply(
    *,
    call_ts: datetime | None,
    message_id: str,
    sender: str,
    recipient: str,
    sent_by_owner: dict[str, list[_Logged]],
) -> tuple[str | None, str | None, str, bool]:
    replies = sent_by_owner.get(recipient, [])
    if call_ts is not None and message_id:
        direct = [
            logged
            for logged in replies
            if str(getattr(logged.entry, "to_person", "") or "") == sender
            and _str_value(_meta(logged.entry).get("reply_to")) == message_id
            and (_parse_ts(logged.entry.ts) or datetime.min.replace(tzinfo=UTC)) > call_ts
        ]
        if direct:
            direct.sort(key=lambda logged: _parse_ts(logged.entry.ts) or datetime.min.replace(tzinfo=UTC))
            entry = direct[0].entry
            return (
                str(entry.ts),
                _str_value(_meta(entry).get("message_id")) or None,
                _head(entry.content),
                False,
            )

    if call_ts is not None:
        inferred = _closest(
            [logged for logged in replies if str(getattr(logged.entry, "to_person", "") or "") == sender],
            call_ts,
            within=_REPLY_WINDOW,
        )
        if inferred is not None:
            entry = inferred.entry
            return (
                str(entry.ts),
                _str_value(_meta(entry).get("message_id")) or None,
                _head(entry.content),
                True,
            )
    return None, None, "", False


def _summarize(calls: list[dict[str, Any]], now: datetime) -> dict[str, Any]:
    delivered = sum(1 for call in calls if call["delivered"])
    read = sum(1 for call in calls if call["read"])
    replied_calls = [call for call in calls if call["reply"]]
    reply_seconds: list[float] = []
    for call in replied_calls:
        start = _parse_ts(call["t"])
        end = _parse_ts(call["reply"])
        if start is not None and end is not None:
            reply_seconds.append((end - start).total_seconds())
    waiting = 0
    for call in calls:
        if not call["delivered"] or call["reply"]:
            continue
        start = _parse_ts(call["t"])
        if start is not None and timedelta(0) <= now - start <= _WAITING_WINDOW:
            waiting += 1
    median_reply_seconds = round(statistics.median(reply_seconds)) if reply_seconds else None
    return {
        "calls": len(calls),
        "delivered": delivered,
        "read": read,
        "replied": len(replied_calls),
        "median_reply_seconds": median_reply_seconds,
        "waiting": waiting,
    }


def _build(shared_dir: Path, days: int, via: Via, username: str) -> dict[str, Any]:
    now = now_local().astimezone(UTC)
    cutoff = now - timedelta(days=days)
    anima_names = _anima_names(shared_dir)
    anima_set = set(anima_names)

    sent_logs, received_logs = _load_activity(shared_dir, anima_names, days)
    sent_by_owner: dict[str, list[_Logged]] = {}
    received_by_owner_message: dict[tuple[str, str], list[_Logged]] = {}
    received_by_owner_board: dict[tuple[str, str], list[_Logged]] = {}
    board_mention_by_inbox: dict[str, tuple[str, str]] = {}
    received_by_owner_from: dict[tuple[str, str], list[_Logged]] = {}
    for logged in sent_logs:
        sent_by_owner.setdefault(logged.owner, []).append(logged)
    for logged in received_logs:
        entry = logged.entry
        meta = _meta(entry)
        received_from = str(getattr(entry, "from_person", "") or "")
        if received_from:
            received_by_owner_from.setdefault((logged.owner, received_from), []).append(logged)
        message_id = _str_value(meta.get("message_id"))
        if message_id:
            received_by_owner_message.setdefault((logged.owner, message_id), []).append(logged)
        post_id = _str_value(meta.get("board_post_id"))
        if post_id:
            received_by_owner_board.setdefault((logged.owner, post_id), []).append(logged)
            if message_id:
                board_mention_by_inbox[message_id] = (post_id, logged.owner)

    rows_by_channel: dict[str, list[dict[str, Any]]] = {}
    for channel in _readable_boards(shared_dir, username):
        rows_by_channel[channel] = read_board_entries(shared_dir, channel)

    calls: list[dict[str, Any]] = []
    triggers: dict[str, tuple[str, str]] = {}
    board_index: dict[tuple[str, str], dict[str, Any]] = {}
    dm_index: dict[str, dict[str, Any]] = {}

    for channel, rows in rows_by_channel.items():
        deliveries = load_deliveries(shared_dir, channel)
        for row_index, row in enumerate(rows):
            row_ts = _parse_ts(row.get("ts"))
            if row_ts is None or row_ts < cutoff:
                continue
            sender = str(row.get("from") or "")
            post_id = _str_value(row.get("id"))
            targets, inferred_call = _targets_for_row(
                row,
                channel=channel,
                post_id=post_id,
                deliveries=deliveries,
                anima_names=anima_names,
            )
            text = _head(row.get("text"))
            trigger = row.get("trigger")
            trigger_type = ""
            trigger_id = ""
            if isinstance(trigger, dict):
                trigger_type = _str_value(trigger.get("type"))
                trigger_id = _str_value(trigger.get("id"))

            for recipient, delivered, reason in targets:
                if post_id:
                    call_id = f"board:{post_id}:{recipient}"
                else:
                    call_id = f"board:{channel}:{row_index}:{recipient}"
                received = received_by_owner_board.get((recipient, post_id), []) if post_id else []
                read_ts = _min_ts(received)
                inferred_read = False
                if read_ts is None and delivered:
                    read_ts = _inferred_read(received_by_owner_from, recipient, sender, row_ts)
                    inferred_read = read_ts is not None
                trigger_message_ids = {
                    message_id
                    for logged in received
                    if (message_id := _str_value(_meta(logged.entry).get("message_id")))
                }
                reply_ts, reply_id, reply_text, inferred_reply = _board_reply(
                    call_ts=row_ts,
                    channel=channel,
                    post_id=post_id,
                    sender=sender,
                    recipient=recipient,
                    rows_by_channel=rows_by_channel,
                    sent_by_owner=sent_by_owner,
                    trigger_message_ids=trigger_message_ids,
                )
                call = {
                    "id": call_id,
                    "via": "board",
                    "place": channel,
                    "from": sender,
                    "to": recipient,
                    "t": str(row.get("ts") or ""),
                    "delivered": delivered,
                    "reason": reason,
                    "read": read_ts,
                    "reply": reply_ts,
                    "reply_id": reply_id,
                    "reply_text": reply_text,
                    "inferred": inferred_call or inferred_reply or inferred_read,
                    "parent": None,
                    "text": text,
                }
                calls.append(call)
                if post_id:
                    board_index[(post_id, recipient)] = call
                if trigger_id:
                    triggers[call_id] = (trigger_type, trigger_id)

    for sender in anima_names:
        for logged in sent_by_owner.get(sender, []):
            entry = logged.entry
            recipient = str(getattr(entry, "to_person", "") or "")
            if recipient not in anima_set or recipient == sender:
                continue
            entry_ts = _parse_ts(entry.ts)
            if entry_ts is None or entry_ts < cutoff:
                continue
            meta = _meta(entry)
            message_id = _str_value(meta.get("message_id"))
            call_id = f"dm:{message_id}" if message_id else f"dm:{sender}:{entry.ts}"
            received = received_by_owner_message.get((recipient, message_id), []) if message_id else []
            read_ts = _min_ts(received)
            inferred_read = False
            if read_ts is None:
                read_ts = _inferred_read(received_by_owner_from, recipient, sender, entry_ts)
                inferred_read = read_ts is not None
            reply_ts, reply_id, reply_text, inferred_reply = _dm_reply(
                call_ts=entry_ts,
                message_id=message_id,
                sender=sender,
                recipient=recipient,
                sent_by_owner=sent_by_owner,
            )
            call = {
                "id": call_id,
                "via": "dm",
                "place": None,
                "from": sender,
                "to": recipient,
                "t": str(entry.ts),
                "delivered": True,
                "reason": "",
                "read": read_ts,
                "reply": reply_ts,
                "reply_id": reply_id,
                "reply_text": reply_text,
                "inferred": inferred_reply or inferred_read,
                "parent": None,
                "text": _head(entry.content),
            }
            calls.append(call)
            if message_id:
                dm_index[message_id] = call
            trigger_id = _str_value(meta.get("trigger_id"))
            if trigger_id:
                triggers[call_id] = ("", trigger_id)

    for call in calls:
        trigger_type, trigger_id = triggers.get(call["id"], ("", ""))
        parent: dict[str, Any] | None = None
        if trigger_type == "board_mention":
            parent = board_index.get((trigger_id, call["from"]))
        elif trigger_id:
            parent = dm_index.get(trigger_id)
            if parent is None and trigger_id in board_mention_by_inbox:
                post_id, owner = board_mention_by_inbox[trigger_id]
                if owner == call["from"]:
                    parent = board_index.get((post_id, call["from"]))
        if parent is not None:
            call["parent"] = parent["id"]

    if via != "all":
        calls = [call for call in calls if call["via"] == via]

    calls.sort(
        key=lambda call: (_parse_ts(call["t"]) or datetime.min.replace(tzinfo=UTC), call["id"]),
        reverse=True,
    )
    calls = calls[:_MAX_CALLS]
    return {"calls": calls, "summary": _summarize(calls, now)}


def build_flows(shared_dir: Path, *, days: int = 30, via: Via = "all", username: str = "human") -> dict[str, Any]:
    """Return the assembled calls and summary, cached for 60 seconds."""
    key = (str(shared_dir), days, via, username)
    now = time.monotonic()
    with _CACHE_LOCK:
        cached = _CACHE.get(key)
        if cached is not None and now - cached[0] < _CACHE_TTL_SECONDS:
            return cached[1]
    payload = _build(shared_dir, days, via, username)
    with _CACHE_LOCK:
        _CACHE[key] = (time.monotonic(), payload)
    return payload


def channel_info(shared_dir: Path, channel: str) -> dict[str, Any]:
    """Return the informational panel for one Board channel."""
    from core.messaging.messenger import load_channel_meta

    anima_names = _anima_names(shared_dir)
    animas_dir = shared_dir.parent / "animas"
    meta = load_channel_meta(shared_dir, channel)
    members = list(meta.members or []) if meta is not None else []

    mention_targets = [
        name
        for name in anima_names
        if board_access(
            shared_dir,
            channel,
            anima_principal(name, animas_dir=animas_dir),
            "mention_target",
        ).allowed
    ]

    reads = load_reads(shared_dir, channel)
    readers: list[dict[str, Any]] = []
    for reader, record in sorted(reads["readers"].items()):
        self_name = reader.split(":", 1)[1] if ":" in reader else reader
        try:
            unread = int(unread_summary(shared_dir, channel, reader, self_name=self_name)["unread"])
        except Exception:
            logger.warning("Failed to count unread posts for %s on #%s", reader, channel, exc_info=True)
            unread = 0
        readers.append(
            {
                "reader": reader,
                "unread": unread,
                "updated_at": str(record.get("updated_at") or ""),
            }
        )

    now = now_local().astimezone(UTC)
    cutoff = now - timedelta(days=30)
    mentions: Counter[str] = Counter()
    kinds: Counter[str] = Counter({"anima": 0, "human": 0, "external": 0, "operator": 0})
    for entry in read_board_entries(shared_dir, channel):
        kind = _record_kind(entry)
        category = {
            "anima": "anima",
            "human": "human",
            "external": "external",
            "external_user": "external",
            "external_bot": "external",
        }.get(kind, "operator")
        kinds[category] += 1

        entry_ts = _parse_ts(entry.get("ts"))
        if entry_ts is None or entry_ts < cutoff:
            continue
        text = str(entry.get("text") or "")
        for name in extract_mentions(text, anima_names):
            if name != "all":
                mentions[name] += 1

    mentions_30d = [[name, count] for name, count in sorted(mentions.items(), key=lambda item: (-item[1], item[0]))[:8]]

    return {
        "members": members,
        "mention_targets": mention_targets,
        "readers": readers,
        "mentions_30d": mentions_30d,
        "kinds": dict(kinds),
    }
