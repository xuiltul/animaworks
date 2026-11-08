from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Append-only company timeline shared by humans and Anima.

Items live in ``shared/timeline/YYYY-MM.jsonl`` (one month per file) and are
never rewritten.  Human read positions live in ``shared/reads/timeline.json``
and reuse the Board read-document format.
"""

import json
import logging
import re
import secrets
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from core.messaging.board_reads import advance_upto_at, is_after_ts, load_reads_at
from core.platform.atomic_io import append_jsonl_locked
from core.time_utils import now_iso, now_local

logger = logging.getLogger(__name__)

KINDS: frozenset[str] = frozenset({"notified", "done", "decided"})
SOURCES: frozenset[str] = frozenset({"anima", "system"})
PRIORITIES: frozenset[str] = frozenset({"low", "normal", "high", "urgent"})
MAX_TEXT_CHARS = 400
_ABOUT_RE = re.compile(r"^tl_[0-9a-f]{16}$")


def timeline_dir(shared_dir: Path) -> Path:
    """Return the timeline storage directory."""
    return Path(shared_dir) / "timeline"


def reads_path(shared_dir: Path) -> Path:
    """Return the timeline read-position document path."""
    return Path(shared_dir) / "reads" / "timeline.json"


def _month_path(shared_dir: Path, month: str) -> Path:
    return timeline_dir(shared_dir) / f"{month}.jsonl"


def _current_and_previous_months(now: datetime | None = None) -> tuple[str, str]:
    reference = now or now_local()
    current = reference.strftime("%Y-%m")
    previous = (reference.replace(day=1) - timedelta(days=1)).strftime("%Y-%m")
    return current, previous


def month_paths(shared_dir: Path) -> list[Path]:
    """Return the current and previous month files, newest first."""
    current, previous = _current_and_previous_months()
    return [_month_path(shared_dir, current), _month_path(shared_dir, previous)]


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


def _timestamp_rank(item: dict[str, Any]) -> float:
    parsed = _parse_ts(item.get("ts"))
    return parsed.timestamp() if parsed is not None else 0.0


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return []
    except (OSError, UnicodeDecodeError) as exc:
        logger.warning("Failed to read timeline file at %s: %s", path, exc)
        return []

    items: list[dict[str, Any]] = []
    for line in lines:
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            logger.warning("Corrupt timeline line in %s: %s", path, exc)
            continue
        if isinstance(item, dict) and isinstance(item.get("id"), str) and isinstance(item.get("ts"), str):
            items.append(item)
    return items


def _find_in_months(shared_dir: Path, item_id: str) -> dict[str, Any] | None:
    for path in month_paths(shared_dir):
        for item in _read_jsonl(path):
            if item.get("id") == item_id:
                return item
    return None


def find_item(shared_dir: Path, item_id: str) -> dict[str, Any] | None:
    """Return a timeline item by id, searching every month file."""
    if not item_id:
        return None
    paths = sorted(timeline_dir(shared_dir).glob("*.jsonl"), reverse=True)
    for path in paths:
        for item in _read_jsonl(path):
            if item.get("id") == item_id:
                return item
    return None


def append_item(
    shared_dir: Path,
    *,
    from_name: str,
    company: str,
    kind: str,
    text: str,
    source: str,
    about: str | None = None,
    urgent: bool = False,
    ref: str | None = None,
    priority: str | None = None,
    delivered: bool | None = None,
) -> dict[str, Any]:
    """Append one timeline item and return the stored record."""
    kind_value = str(kind or "")
    if kind_value not in KINDS:
        raise ValueError(f"Invalid timeline kind: {kind_value!r}")
    source_value = str(source or "")
    if source_value not in SOURCES:
        raise ValueError(f"Invalid timeline source: {source_value!r}")
    if priority is not None:
        priority = str(priority)
        if priority not in PRIORITIES:
            raise ValueError(f"Invalid timeline priority: {priority!r}")

    text_value = str(text or "").strip()
    if not text_value:
        raise ValueError("Timeline text must not be empty")
    text_value = text_value[:MAX_TEXT_CHARS]

    company_value = str(company or "").strip()
    if not company_value:
        raise ValueError("Timeline company must not be empty")

    about_value: str | None = None
    if about is not None:
        about_value = str(about).strip()
        if not _ABOUT_RE.fullmatch(about_value):
            raise ValueError(f"Invalid timeline item id: {about_value!r}")
        if _find_in_months(shared_dir, about_value) is None:
            raise ValueError(f"Timeline item not found: {about_value!r}")

    ts = now_iso()
    item: dict[str, Any] = {
        "id": f"tl_{secrets.token_hex(8)}",
        "ts": ts,
        "from": str(from_name or ""),
        "company": company_value,
        "kind": kind_value,
        "text": text_value,
        "about": about_value,
        "urgent": bool(urgent),
        "ref": None if ref is None else str(ref),
        "source": source_value,
        "priority": priority,
        "delivered": None if delivered is None else bool(delivered),
    }
    append_jsonl_locked(_month_path(shared_dir, ts[:7]), item)
    return item


def list_items(
    shared_dir: Path,
    *,
    company: str,
    since: str | None = None,
    limit: int = 200,
) -> list[dict[str, Any]]:
    """Return company items from the current and previous month, newest first."""
    items: list[dict[str, Any]] = []
    for path in month_paths(shared_dir):
        for item in _read_jsonl(path):
            if item.get("company") != company:
                continue
            if since and not is_after_ts(item.get("ts", ""), since):
                continue
            items.append(item)
    items.sort(key=_timestamp_rank, reverse=True)
    if limit is not None and limit >= 0:
        return items[:limit]
    return items


def load_timeline_reads(shared_dir: Path) -> dict[str, Any]:
    """Load the timeline read-position document."""
    return load_reads_at(reads_path(shared_dir), "timeline")


def advance_timeline_read(shared_dir: Path, reader: str, upto_id: str) -> bool:
    """Advance *reader*'s position in the timeline to *upto_id*."""
    item = find_item(shared_dir, upto_id)
    if item is None:
        return False
    return advance_upto_at(
        reads_path(shared_dir),
        "timeline",
        reader,
        post_id=upto_id,
        post_ts=str(item.get("ts") or ""),
    )


def _ts_at_least(value: object, reference: object) -> bool:
    candidate = str(value or "")
    threshold = str(reference or "")
    if not candidate or not threshold:
        return False
    candidate_dt = _parse_ts(candidate)
    threshold_dt = _parse_ts(threshold)
    if candidate_dt is not None and threshold_dt is not None:
        return candidate_dt <= threshold_dt
    return candidate <= threshold


def is_read_for_reader(item: dict[str, Any], reads: dict[str, Any], reader: str) -> bool:
    """Return whether *reader* has read *item*."""
    item_id = str(item.get("id") or "")
    if item_id and item_id in reads.get("picked", {}).get(reader, []):
        return True
    record = reads.get("readers", {}).get(reader)
    if not isinstance(record, dict):
        return False
    return _ts_at_least(item.get("ts"), record.get("upto_ts"))


def is_read_by_human(item: dict[str, Any], reads: dict[str, Any]) -> bool:
    """Return whether any human reader has read *item* (position or picked)."""
    readers = reads.get("readers", {})
    picked = reads.get("picked", {})
    if not isinstance(readers, dict):
        readers = {}
    if not isinstance(picked, dict):
        picked = {}
    human_readers = {
        reader
        for reader in [*readers.keys(), *picked.keys()]
        if isinstance(reader, str) and reader.startswith("human:")
    }
    return any(is_read_for_reader(item, reads, reader) for reader in human_readers)


def decorate_items(
    items: list[dict[str, Any]],
    *,
    reads: dict[str, Any],
    reader: str,
    reference_items: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Add read/update flags used by the API responses."""
    reference = reference_items if reference_items is not None else items
    by_id = {str(item["id"]): item for item in reference if isinstance(item.get("id"), str)}

    updates: dict[str, list[str]] = {}
    for item in reference:
        item_id = item.get("id")
        about = item.get("about")
        if isinstance(item_id, str) and isinstance(about, str) and about in by_id:
            updates.setdefault(about, []).append(item_id)

    decorated: list[dict[str, Any]] = []
    for item in items:
        result = dict(item)
        read = is_read_for_reader(result, reads, reader) if reader else False
        result["read"] = read
        result["read_by_human"] = is_read_by_human(result, reads)
        item_id = str(result.get("id") or "")
        update_ids = updates.get(item_id, [])
        result["updates"] = update_ids
        about = result.get("about")
        changed = False
        if reader and isinstance(about, str) and about in by_id:
            original = by_id[about]
            changed = not read and is_read_for_reader(original, reads, reader)
        result["changed"] = changed
        decorated.append(result)
    return decorated
