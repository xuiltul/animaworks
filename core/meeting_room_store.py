from __future__ import annotations

import json
import logging
import os
import re
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from core.time_utils import now_iso

logger = logging.getLogger("animaworks.meeting_room_store")

_ROOM_ID_RE = re.compile(r"^[a-f0-9]{12}$")
_ROOM_LOCKS: dict[Path, threading.RLock] = {}
_ROOM_LOCKS_GUARD = threading.Lock()


def _validate_room_id(room_id: str) -> None:
    if not _ROOM_ID_RE.match(room_id):
        raise ValueError(f"Invalid room_id: {room_id!r}")


def _process_lock(path: Path) -> threading.RLock:
    resolved = path.resolve()
    with _ROOM_LOCKS_GUARD:
        if resolved not in _ROOM_LOCKS:
            _ROOM_LOCKS[resolved] = threading.RLock()
        return _ROOM_LOCKS[resolved]


@contextmanager
def _locked_room(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_suffix(path.suffix + ".lock")
    thread_lock = _process_lock(lock_path)
    with thread_lock, lock_path.open("a+", encoding="utf-8") as lock_file:
        locked = False
        try:
            from core.platform.locks import acquire_file_lock

            acquire_file_lock(lock_file, exclusive=True)
            locked = True
        except OSError:
            logger.debug("OS file lock unavailable for %s", lock_path, exc_info=True)
        try:
            yield
        finally:
            if locked:
                try:
                    from core.platform.locks import release_file_lock

                    release_file_lock(lock_file)
                except OSError:
                    logger.debug("Failed to release meeting room lock %s", lock_path, exc_info=True)


def _room_path(meetings_dir: Path, room_id: str) -> Path:
    _validate_room_id(room_id)
    return Path(meetings_dir) / f"{room_id}.json"


def _load_room_data(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ValueError(f"Meeting room not found: {path.stem}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Invalid meeting room file: {path}")
    conversation = data.setdefault("conversation", [])
    if not isinstance(conversation, list):
        data["conversation"] = []
    return data


def _write_room_data(path: Path, data: dict[str, Any]) -> None:
    tmp_path = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    tmp_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp_path.replace(path)


def append_room_message(
    meetings_dir: Path,
    room_id: str,
    speaker: str,
    role: str,
    text: str,
    *,
    meta: dict[str, Any] | None = None,
    dedup_key: str = "",
) -> dict[str, Any]:
    """Append a room conversation entry with an interprocess file lock."""
    path = _room_path(meetings_dir, room_id)
    with _locked_room(path):
        data = _load_room_data(path)
        conversation = data.setdefault("conversation", [])
        if dedup_key:
            for existing in conversation:
                existing_meta = existing.get("meta", {}) if isinstance(existing, dict) else {}
                if isinstance(existing_meta, dict) and existing_meta.get("dedup_key") == dedup_key:
                    return existing
        entry: dict[str, Any] = {
            "speaker": speaker,
            "role": role,
            "text": text,
            "ts": now_iso(),
        }
        if meta or dedup_key:
            entry_meta = dict(meta or {})
            if dedup_key:
                entry_meta["dedup_key"] = dedup_key
            entry["meta"] = entry_meta
        conversation.append(entry)
        _write_room_data(path, data)
        return entry


def append_meeting_redirect(
    meetings_dir: Path,
    room_id: str,
    *,
    from_name: str,
    to_name: str,
    content: str,
    intent: str = "",
    redirect_id: str = "",
) -> dict[str, Any]:
    """Durably append a meeting-local redirect entry to the room store."""
    path = _room_path(meetings_dir, room_id)
    dedup_key = f"meeting_redirect:{redirect_id}" if redirect_id else ""
    with _locked_room(path):
        data = _load_room_data(path)
        conversation = data.setdefault("conversation", [])
        if dedup_key:
            for existing in conversation:
                existing_meta = existing.get("meta", {}) if isinstance(existing, dict) else {}
                if isinstance(existing_meta, dict) and existing_meta.get("dedup_key") == dedup_key:
                    return existing
        participants = {str(name) for name in data.get("participants", [])}
        if to_name not in participants:
            raise ValueError(f"Meeting redirect target is not a participant: {to_name}")
        chair = str(data.get("chair") or "")
        role = "chair" if from_name == chair else "participant"
        entry: dict[str, Any] = {
            "speaker": from_name,
            "role": role,
            "text": f"@{to_name} {content}",
            "ts": now_iso(),
            "meta": {
                "type": "meeting_redirect",
                "to": to_name,
                "intent": intent,
                "redirect_id": redirect_id,
                "dedup_key": dedup_key,
            },
        }
        conversation.append(entry)
        _write_room_data(path, data)
        return entry
