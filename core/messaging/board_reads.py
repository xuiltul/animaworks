from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Per-reader read positions for Board channels.

Read positions live outside ``shared/channels`` so they never appear as
channels themselves.  Each channel gets one JSON document containing every
reader's contiguous position (``upto``) and individually read posts
(``picked``).  Writes are serialized with the adjacent persistent lock file
and atomically replace the document.
"""

import json
import logging
import threading
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from core.messaging.mentions import extract_mentions
from core.platform.atomic_io import atomic_write_json
from core.platform.locks import locked_path
from core.time_utils import now_iso

logger = logging.getLogger(__name__)

_MAX_PICKED = 500
_READ_HOOKS: dict[str, Callable[[dict[str, Any]], Any]] = {}
_READ_HOOKS_GUARD = threading.Lock()


def register_read_hook(name: str, fn: Callable[[dict[str, Any]], Any]) -> None:
    """Register a notification callback for read-position updates."""
    with _READ_HOOKS_GUARD:
        _READ_HOOKS[name] = fn


def unregister_read_hook(name: str) -> None:
    """Remove a previously registered read-position callback."""
    with _READ_HOOKS_GUARD:
        _READ_HOOKS.pop(name, None)


def clear_read_hooks() -> None:
    """Remove every read-position callback (mainly for tests)."""
    with _READ_HOOKS_GUARD:
        _READ_HOOKS.clear()


def _run_read_hooks(payload: dict[str, Any]) -> None:
    with _READ_HOOKS_GUARD:
        hooks = list(_READ_HOOKS.items())
    for name, fn in hooks:
        try:
            fn(dict(payload))
        except Exception:
            logger.warning("Board read hook %s failed for #%s", name, payload.get("channel"), exc_info=True)


def reads_path(shared_dir: Path, channel: str) -> Path:
    """Return the read-position document path for one Board channel."""
    return Path(shared_dir) / "reads" / "board" / f"{channel}.json"


def _lock_path(path: Path) -> Path:
    return path.with_name(f"{path.name}.lock")


def _empty_reads(channel: str) -> dict[str, Any]:
    return {"stream": f"board/{channel}", "readers": {}, "picked": {}}


def _load_unlocked(path: Path, channel: str) -> dict[str, Any]:
    empty = _empty_reads(channel)
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return empty
    except (OSError, UnicodeDecodeError) as exc:
        logger.warning("Failed to read Board read positions at %s: %s", path, exc)
        return empty
    try:
        payload = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        logger.warning("Corrupt Board read positions at %s; treating as empty: %s", path, exc)
        return empty
    if not isinstance(payload, dict):
        logger.warning("Invalid Board read positions at %s; treating as empty", path)
        return empty

    readers: dict[str, dict[str, str]] = {}
    raw_readers = payload.get("readers")
    if isinstance(raw_readers, dict):
        for reader, value in raw_readers.items():
            if isinstance(reader, str) and isinstance(value, dict):
                readers[reader] = {
                    "upto_id": str(value.get("upto_id") or ""),
                    "upto_ts": str(value.get("upto_ts") or ""),
                    "updated_at": str(value.get("updated_at") or ""),
                }

    picked: dict[str, list[str]] = {}
    raw_picked = payload.get("picked")
    if isinstance(raw_picked, dict):
        for reader, values in raw_picked.items():
            if isinstance(reader, str) and isinstance(values, list):
                picked[reader] = [str(value) for value in values if isinstance(value, str) and value]

    return {"stream": f"board/{channel}", "readers": readers, "picked": picked}


def load_reads(shared_dir: Path, channel: str) -> dict[str, Any]:
    """Load the read-position document, returning an empty shape when absent."""
    return _load_unlocked(reads_path(shared_dir, channel), channel)


def _parse_ts(value: object) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def is_after_ts(candidate: object, reference: object) -> bool:
    """Return whether *candidate* is strictly newer than *reference*."""
    candidate_ts = str(candidate or "")
    reference_ts = str(reference or "")
    if not reference_ts:
        return bool(candidate_ts)
    if not candidate_ts:
        return False
    candidate_dt = _parse_ts(candidate_ts)
    reference_dt = _parse_ts(reference_ts)
    if candidate_dt is not None and reference_dt is not None:
        return candidate_dt > reference_dt
    return candidate_ts > reference_ts


def read_board_entries(shared_dir: Path, channel: str) -> list[dict[str, Any]]:
    """Return all readable JSONL records for one Board channel."""
    path = Path(shared_dir) / "channels" / f"{channel}.jsonl"
    try:
        lines = path.read_text(encoding="utf-8").strip().splitlines()
    except OSError:
        return []
    entries: list[dict[str, Any]] = []
    for line in lines:
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(entry, dict):
            entries.append(entry)
    return entries


def entries_after_ts(entries: list[dict[str, Any]], upto_ts: str) -> list[dict[str, Any]]:
    """Return entries strictly newer than *upto_ts* in file order."""
    return [entry for entry in entries if is_after_ts(entry.get("ts", ""), upto_ts)]


def find_post_ts(shared_dir: Path, channel: str, post_id: str) -> str | None:
    """Return the timestamp of a post id, or ``None`` when it does not exist."""
    if not post_id:
        return None
    for entry in read_board_entries(shared_dir, channel):
        if entry.get("id") == post_id:
            return str(entry.get("ts") or "")
    return None


def advance_upto(
    shared_dir: Path,
    channel: str,
    reader: str,
    *,
    post_id: str | None,
    post_ts: str,
) -> bool:
    """Advance *reader*'s contiguous position when *post_ts* is newer."""
    path = reads_path(shared_dir, channel)
    new_ts = str(post_ts or "")
    new_id = str(post_id or "")
    path.parent.mkdir(parents=True, exist_ok=True)
    changed = False
    with locked_path(_lock_path(path), exclusive=True, thread_lock=True):
        payload = _load_unlocked(path, channel)
        readers = payload["readers"]
        current = readers.get(reader)
        current_ts = str(current.get("upto_ts") or "") if isinstance(current, dict) else ""
        if current is not None and not is_after_ts(new_ts, current_ts):
            return False
        readers[reader] = {
            "upto_id": new_id,
            "upto_ts": new_ts,
            "updated_at": now_iso(),
        }
        atomic_write_json(path, payload, indent=2, ensure_ascii=False)
        changed = True

    if changed:
        _run_read_hooks(
            {
                "channel": channel,
                "reader": reader,
                "upto_id": new_id,
                "upto_ts": new_ts,
                "picked_id": "",
            }
        )
    return changed


def add_picked(shared_dir: Path, channel: str, reader: str, post_id: str) -> bool:
    """Mark one post as individually read for *reader*."""
    if not post_id:
        return False
    path = reads_path(shared_dir, channel)
    path.parent.mkdir(parents=True, exist_ok=True)
    changed = False
    updated: dict[str, str] = {}
    with locked_path(_lock_path(path), exclusive=True, thread_lock=True):
        payload = _load_unlocked(path, channel)
        picked = payload["picked"].setdefault(reader, [])
        if post_id in picked:
            return False
        picked.append(post_id)
        if len(picked) > _MAX_PICKED:
            payload["picked"][reader] = picked[-_MAX_PICKED:]
        atomic_write_json(path, payload, indent=2, ensure_ascii=False)
        current = payload["readers"].get(reader)
        if isinstance(current, dict):
            updated = dict(current)
        changed = True

    if changed:
        _run_read_hooks(
            {
                "channel": channel,
                "reader": reader,
                "upto_id": updated.get("upto_id", ""),
                "upto_ts": updated.get("upto_ts", ""),
                "picked_id": post_id,
            }
        )
    return changed


def unread_summary(
    shared_dir: Path,
    channel: str,
    reader: str,
    *,
    self_name: str | None = None,
) -> dict[str, int]:
    """Count unread posts and mentions addressed to *self_name*."""
    payload = load_reads(shared_dir, channel)
    record = payload["readers"].get(reader)
    upto_ts = str(record.get("upto_ts") or "") if isinstance(record, dict) else ""
    picked = set(payload["picked"].get(reader, []))
    has_record = isinstance(record, dict)

    unread = 0
    to_me = 0
    for entry in read_board_entries(shared_dir, channel):
        if has_record and not is_after_ts(entry.get("ts", ""), upto_ts):
            continue
        if self_name and entry.get("from") == self_name:
            continue
        post_id = entry.get("id")
        if isinstance(post_id, str) and post_id in picked:
            continue
        unread += 1
        if self_name:
            text = entry.get("text")
            if isinstance(text, str) and extract_mentions(text, {self_name}):
                to_me += 1

    return {"unread": unread, "to_me": to_me}
