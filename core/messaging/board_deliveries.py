from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Record of who received a Board mention and who did not.

Deliveries live in ``shared/board_deliveries/<channel>.jsonl``; everything in
``shared/channels/`` is treated as a Board channel, so these files must stay
outside that directory.
"""

import json
import logging
import threading
from pathlib import Path
from typing import Any

from core.platform.atomic_io import append_jsonl_locked

logger = logging.getLogger("animaworks.messaging.board_deliveries")

_CACHE_LOCK = threading.Lock()
# path -> (mtime, size, deliveries)
_cache: dict[str, tuple[float, int, dict[str, dict[str, Any]]]] = {}


def deliveries_path(shared_dir: Path, channel: str) -> Path:
    """Return the delivery log path for *channel*."""
    return shared_dir / "board_deliveries" / f"{channel}.jsonl"


def append_delivery(
    shared_dir: Path,
    channel: str,
    *,
    post_id: str,
    ts: str,
    notified: list[str],
    not_notified: dict[str, str],
) -> None:
    """Append one delivery record; the caller decides whether failures matter."""
    record = {
        "post_id": post_id,
        "ts": ts,
        "notified": list(notified),
        "not_notified": dict(not_notified),
    }
    append_jsonl_locked(deliveries_path(shared_dir, channel), record)


def load_deliveries(shared_dir: Path, channel: str) -> dict[str, dict[str, Any]]:
    """Return ``post_id -> delivery record``, cached by mtime and size."""
    path = deliveries_path(shared_dir, channel)
    key = str(path)
    try:
        stat = path.stat()
    except OSError:
        with _CACHE_LOCK:
            _cache.pop(key, None)
        return {}

    with _CACHE_LOCK:
        cached = _cache.get(key)
        if cached is not None and cached[0] == stat.st_mtime and cached[1] == stat.st_size:
            return cached[2]

    deliveries: dict[str, dict[str, Any]] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        lines = []
    for line in lines:
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(record, dict):
            continue
        post_id = record.get("post_id")
        if not isinstance(post_id, str) or not post_id:
            continue
        deliveries[post_id] = record

    with _CACHE_LOCK:
        _cache[key] = (stat.st_mtime, stat.st_size, deliveries)
    return deliveries


def attach_deliveries(
    shared_dir: Path,
    channel: str,
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return copies of *messages* with their delivery record attached."""
    if not messages:
        return messages
    deliveries = load_deliveries(shared_dir, channel)
    if not deliveries:
        return messages

    attached: list[dict[str, Any]] = []
    for message in messages:
        copy = dict(message)
        post_id = copy.get("id")
        record = deliveries.get(post_id) if isinstance(post_id, str) else None
        if record is not None:
            copy["delivery"] = {
                "notified": list(record.get("notified") or []),
                "not_notified": dict(record.get("not_notified") or {}),
            }
        attached.append(copy)
    return attached
