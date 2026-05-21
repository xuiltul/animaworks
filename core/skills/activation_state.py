from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""State helpers for thread-scoped explicit skill activation."""

import json
import logging
import re
from pathlib import Path
from typing import Any

from core.time_utils import now_iso

logger = logging.getLogger(__name__)

ACTIVE_SKILLS_STATE_FILE = "active_skills.json"
_THREAD_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_-]{1,36}$")


def validate_thread_id(thread_id: str) -> str:
    value = str(thread_id or "default").strip() or "default"
    if not _THREAD_ID_PATTERN.match(value):
        raise ValueError(f"Invalid thread_id: {value!r}. Must be 1-36 alphanumeric, underscore, or hyphen characters.")
    return value


def get_active_skill_refs(anima_dir: Path, thread_id: str = "default") -> list[str]:
    """Return stored active skill refs for an anima thread."""
    thread_id = validate_thread_id(thread_id)
    state = _read_state(anima_dir)
    thread_entry = state.get("threads", {}).get(thread_id, {})
    if isinstance(thread_entry, list):
        refs = thread_entry
    elif isinstance(thread_entry, dict):
        refs = thread_entry.get("refs", [])
    else:
        refs = []
    return dedupe_refs([str(ref) for ref in refs])


def write_thread_refs(anima_dir: Path, thread_id: str, refs: list[str]) -> None:
    state = _read_state(anima_dir)
    threads = state.setdefault("threads", {})
    if refs:
        threads[thread_id] = {
            "refs": dedupe_refs(refs),
            "updated_at": now_iso(),
        }
    else:
        threads.pop(thread_id, None)
    path = _state_path(anima_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def dedupe_refs(skill_refs: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for ref in skill_refs:
        value = str(ref).strip()
        if value and value not in seen:
            result.append(value)
            seen.add(value)
    return result


def _state_path(anima_dir: Path) -> Path:
    return anima_dir / "state" / ACTIVE_SKILLS_STATE_FILE


def _read_state(anima_dir: Path) -> dict[str, Any]:
    path = _state_path(anima_dir)
    if not path.is_file():
        return {"version": 1, "threads": {}}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        logger.warning("Failed to read active skill state from %s; using empty state", path)
        return {"version": 1, "threads": {}}
    if not isinstance(data, dict):
        return {"version": 1, "threads": {}}
    threads = data.get("threads")
    if not isinstance(threads, dict):
        data["threads"] = {}
    data["version"] = 1
    return data
