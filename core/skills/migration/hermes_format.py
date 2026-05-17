from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Format adapters for Hermes migration sources."""

from pathlib import Path
from typing import Any

from core.skills.models import SkillUsageEventType

from ._common import read_json


def parse_hermes_usage(path: Path, *, import_time: str) -> tuple[list[dict[str, Any]], set[str]]:
    raw = read_json(path)
    events: list[dict[str, Any]] = []
    skill_names: set[str] = set()
    if isinstance(raw, list):
        for item in raw:
            if isinstance(item, dict):
                event = _usage_event_from_dict(item, import_time=import_time)
                if event:
                    events.append(event)
                    skill_names.add(event["skill_name"])
        return events, skill_names
    if isinstance(raw, dict) and isinstance(raw.get("events"), list):
        for item in raw["events"]:
            if isinstance(item, dict):
                event = _usage_event_from_dict(item, import_time=import_time)
                if event:
                    events.append(event)
                    skill_names.add(event["skill_name"])
        return events, skill_names
    if isinstance(raw, dict):
        for skill_name, record in raw.items():
            if str(skill_name).startswith(".") or not isinstance(record, dict):
                continue
            skill_names.add(str(skill_name))
            before = len(events)
            for event_type, aliases in _USAGE_ALIASES.items():
                count = max(_int(record.get(alias)) for alias in aliases)
                for index in range(count):
                    events.append(_counter_usage_event(str(skill_name), event_type, record, import_time, index))
            if len(events) == before and record.get("created_at"):
                events.append(_counter_usage_event(str(skill_name), SkillUsageEventType.create, record, import_time, 0))
    return events, skill_names


def coerce_lock_entries(raw: Any) -> list[dict[str, Any]]:
    if isinstance(raw, list):
        return [item for item in raw if isinstance(item, dict)]
    if isinstance(raw, dict):
        for key in ("entries", "skills", "locks"):
            if isinstance(raw.get(key), list):
                return [item for item in raw[key] if isinstance(item, dict)]
        return [raw]
    return []


def coerce_task_entries(raw: Any) -> list[dict[str, Any]]:
    if isinstance(raw, list):
        return [item for item in raw if isinstance(item, dict)]
    if isinstance(raw, dict):
        for key in ("tasks", "cards", "items"):
            if isinstance(raw.get(key), list):
                return [item for item in raw[key] if isinstance(item, dict)]
    return []


def task_status(value: str) -> str:
    normalized = value.strip().lower()
    if normalized in {"done", "completed", "closed"}:
        return "done"
    if normalized == "blocked":
        return "blocked"
    if normalized in {"in_progress", "running", "doing"}:
        return "in_progress"
    return "pending"


def convert_cron_skills_field(text: str) -> str:
    lines: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("skill:") and not stripped.startswith("skills:"):
            value = stripped.split(":", 1)[1].strip()
            indent = line[: len(line) - len(line.lstrip())]
            lines.append(f"{indent}skills:")
            if value:
                lines.append(f"{indent}  - {value}")
        else:
            lines.append(line)
    return "\n".join(lines).rstrip() + "\n"


_USAGE_ALIASES: dict[SkillUsageEventType, tuple[str, ...]] = {
    SkillUsageEventType.view: ("view", "views", "view_count", "bump_view"),
    SkillUsageEventType.use: ("use", "uses", "use_count", "bump_use"),
    SkillUsageEventType.success: ("success", "successes", "success_count"),
    SkillUsageEventType.failure: ("failure", "failures", "failure_count"),
    SkillUsageEventType.patch: ("patch", "patches", "patch_count", "bump_patch"),
}


def _usage_event_from_dict(raw: dict[str, Any], *, import_time: str) -> dict[str, Any] | None:
    skill_name = str(raw.get("skill_name") or raw.get("name") or raw.get("skill") or "").strip()
    event_raw = str(raw.get("event_type") or raw.get("event") or raw.get("type") or "").strip()
    event_type = _usage_event_type(event_raw)
    if not skill_name or event_type is None:
        return None
    ts = str(raw.get("ts") or raw.get("timestamp") or raw.get("created_at") or import_time)
    notes = str(raw.get("notes") or "")
    if ts == import_time and not raw.get("ts") and not raw.get("timestamp") and not raw.get("created_at"):
        notes = _join_notes(notes, "imported_without_original_timestamp")
    return {
        "ts": ts,
        "skill_name": skill_name,
        "event_type": event_type.value,
        "is_common": bool(raw.get("is_common", False)),
        "notes": _join_notes(notes, "source_usage_completeness:sparse"),
    }


def _counter_usage_event(
    skill_name: str,
    event_type: SkillUsageEventType,
    record: dict[str, Any],
    import_time: str,
    index: int,
) -> dict[str, Any]:
    ts = str(
        record.get(f"{event_type.value}_at") or record.get("last_used_at") or record.get("created_at") or import_time
    )
    notes = "source_usage_completeness:sparse"
    if ts == import_time:
        notes = _join_notes(notes, "imported_without_original_timestamp")
    if index:
        notes = _join_notes(notes, f"imported_counter_index:{index}")
    return {
        "ts": ts,
        "skill_name": skill_name,
        "event_type": event_type.value,
        "is_common": bool(record.get("is_common", False)),
        "notes": notes,
    }


def _usage_event_type(value: str) -> SkillUsageEventType | None:
    normalized = value.strip().lower()
    normalized = {"bump_view": "view", "bump_use": "use", "bump_patch": "patch"}.get(normalized, normalized)
    try:
        return SkillUsageEventType(normalized)
    except ValueError:
        return None


def _int(value: Any) -> int:
    if isinstance(value, bool):
        return int(value)
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


def _join_notes(*parts: str) -> str:
    return "; ".join(part for part in parts if part)
