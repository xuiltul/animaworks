from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Secret-safe verification helpers for the final Anima merge phase."""

import json
import re
import sqlite3
from pathlib import Path
from typing import Any

from core.memory.entity_index import normalize_entity_key
from core.memory.facts import fact_entity_names, iter_fact_records

_ATTRIBUTION_FIELDS = frozenset(
    {
        "created_by",
        "from",
        "from_person",
        "origin_chain",
        "sender",
        "speaker",
    }
)
_INTRINSIC_SOURCE_FIELDS = frozenset({"anima", "anima_name", "name"})
_CONTENT_FIELDS = frozenset(
    {"content", "description", "message", "notes", "original_instruction", "summary", "text"}
)
_TOKEN_BOUNDARY = r"(?<![0-9A-Za-z_.-]){}(?![0-9A-Za-z_.-])"


def estimate_probe_counts(source_dir: Path) -> dict[str, int]:
    """Estimate VERIFY probes without initializing a search backend."""

    counts = {
        "knowledge": _count_files(source_dir / "knowledge", "*.md"),
        "episodes": _count_files(source_dir / "episodes", "*.md"),
        "procedures": _count_files(source_dir / "procedures", "*.md"),
        "skills": _count_files(source_dir / "skills", "SKILL.md"),
        "facts": min(3, sum(1 for _record in iter_fact_records(source_dir, include_expired=True))),
        "conversation_summary": 0,
        "entities": 0,
    }
    conversation = source_dir / "state" / "conversation.json"
    if conversation.is_file():
        try:
            value = json.loads(conversation.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            value = {}
        counts["conversation_summary"] = int(
            isinstance(value, dict) and bool(str(value.get("compressed_summary", "")).strip())
        )
    entities: set[str] = set()
    for record in iter_fact_records(source_dir, include_expired=True):
        entities.update(normalize_entity_key(name) for name in fact_entity_names(record))
    counts["entities"] = min(3, len(entities - {""}))
    return counts


def _count_files(root: Path, pattern: str) -> int:
    if not root.is_dir():
        return 0
    return min(3, sum(1 for path in root.rglob(pattern) if path.is_file()))


def probe_query(content: str, *, limit: int = 240) -> str:
    """Choose a compact content-derived query without persisting the content."""

    in_frontmatter = False
    candidates: list[str] = []
    for raw in content[:16_384].splitlines():
        line = raw.strip()
        if line == "---":
            in_frontmatter = not in_frontmatter
            continue
        if in_frontmatter or not line:
            continue
        line = re.sub(r"^#{1,6}\s*", "", line).strip()
        if line:
            candidates.append(line)
        if sum(len(item) for item in candidates) >= limit:
            break
    return " ".join(candidates)[:limit].strip()


def source_reference_report(data_dir: Path, source: str) -> dict[str, Any]:
    """Scan active reference surfaces and return locations, never values."""

    data_dir = Path(data_dir)
    residual: list[str] = []
    allowed: list[str] = []
    checked: list[str] = []

    config_path = data_dir / "config.json"
    if config_path.is_file():
        checked.append("config.json")
        config = _read_json(config_path)
        animas = config.get("animas")
        if isinstance(animas, dict) and source in animas:
            allowed.append(f"config.json.animas.{source}")
        _scan_json(
            config,
            source,
            "config.json",
            residual,
            allowed,
            allow=lambda path, _key: path == f"config.json.animas.{source}"
            or path.startswith(f"config.json.animas.{source}."),
        )

    animas_dir = data_dir / "animas"
    if animas_dir.is_dir():
        for path in sorted(animas_dir.glob("*/status.json")):
            relative = path.relative_to(data_dir).as_posix()
            checked.append(relative)
            _scan_json(
                _read_json(path),
                source,
                relative,
                residual,
                allowed,
                allow=lambda location, key, relative=relative: (
                    relative == f"animas/{source}/status.json" and key in _INTRINSIC_SOURCE_FIELDS
                ),
            )

    inbox_dir = data_dir / "shared" / "inbox"
    if inbox_dir.is_dir():
        # Only direct per-Anima inbox entries are active routing surfaces.
        # processed/expired/quarantine descendants are immutable history and
        # intentionally remain source-attributed after REWRITE_REFS.
        for path in sorted(inbox_dir.glob("*/*.json")):
            relative = path.relative_to(data_dir).as_posix()
            checked.append(relative)
            _scan_json(
                _read_json(path),
                source,
                relative,
                residual,
                allowed,
                allow=lambda _location, key: key in _ATTRIBUTION_FIELDS,
            )

    for path in (
        data_dir / "run" / "notification_map.json",
        data_dir / "run" / "discord_thread_map.json",
    ):
        if not path.is_file():
            continue
        relative = path.relative_to(data_dir).as_posix()
        checked.append(relative)
        # notification_textは配信済み通知の文面(不変履歴)。routing上意味を持つのは
        # anima/channel等のフィールドのみなので、文面中の名前言及は残存参照として扱わない。
        _scan_json(
            _read_json(path),
            source,
            relative,
            residual,
            allowed,
            allow=lambda _location, key: key == "notification_text",
        )

    taskboard = data_dir / "shared" / "taskboard.sqlite3"
    if taskboard.is_file():
        checked.append("shared/taskboard.sqlite3")
        _scan_taskboard(taskboard, source, residual, allowed)

    return {
        "surfaces_checked": checked,
        "references_allowed": sorted(set(allowed)),
        "residual_references": sorted(set(residual)),
    }


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid JSON file {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _scan_json(
    value: Any,
    source: str,
    path: str,
    residual: list[str],
    allowed: list[str],
    *,
    key: str = "",
    allow: Any = None,
) -> None:
    if isinstance(value, dict):
        for child_key, child in value.items():
            child_path = f"{path}.{child_key}"
            _scan_json(child, source, child_path, residual, allowed, key=str(child_key), allow=allow)
        return
    if isinstance(value, list):
        for index, child in enumerate(value):
            _scan_json(child, source, f"{path}[{index}]", residual, allowed, key=key, allow=allow)
        return
    if key in _CONTENT_FIELDS:
        return
    if not isinstance(value, str) or not _references_source(value, source):
        return
    if allow is not None and allow(path, key):
        allowed.append(path)
    else:
        residual.append(path)


def _references_source(value: str, source: str) -> bool:
    if value == source:
        return True
    if value.startswith(f"task_queue:{source}:"):
        return True
    if f"/animas/{source}/" in value or value.endswith(f"/animas/{source}"):
        return True
    return bool(re.search(_TOKEN_BOUNDARY.format(re.escape(source)), value))


def _scan_taskboard(path: Path, source: str, residual: list[str], allowed: list[str]) -> None:
    uri = f"file:{path.resolve().as_posix()}?mode=ro"
    try:
        with sqlite3.connect(uri, uri=True) as conn:
            conn.row_factory = sqlite3.Row
            tables = {
                str(row[0])
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name IN "
                    "('taskboard_metadata', 'taskboard_events')"
                )
            }
            for table in sorted(tables):
                rows = conn.execute(f"SELECT rowid AS _scan_rowid, * FROM {table}")
                for row in rows:
                    row_id = row["_scan_rowid"]
                    columns = row.keys()
                    for column in columns:
                        if column == "_scan_rowid":
                            continue
                        value = row[column]
                        location = f"shared/taskboard.sqlite3:{table}[{row_id}].{column}"
                        if isinstance(value, str) and column.endswith("_json"):
                            try:
                                parsed = json.loads(value)
                            except json.JSONDecodeError:
                                parsed = value
                            _scan_json(
                                parsed,
                                source,
                                location,
                                residual,
                                allowed,
                                key=column,
                                allow=lambda _location, key: key in _ATTRIBUTION_FIELDS,
                            )
                        elif isinstance(value, str) and _references_source(value, source):
                            residual.append(location)
    except sqlite3.Error as exc:
        raise ValueError(f"Invalid TaskBoard database {path}: {exc}") from exc


__all__ = ["estimate_probe_counts", "probe_query", "source_reference_report"]
