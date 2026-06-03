from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Append-only JSONL store for legacy atomic facts."""

import contextlib
import hashlib
import json
import logging
import os
import threading
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from core.memory._io import atomic_write_text
from core.time_utils import now_iso, today_local

logger = logging.getLogger("animaworks.memory.facts")

_LOCKS: dict[Path, threading.Lock] = {}
_LOCKS_GUARD = threading.Lock()


def _normalize(value: object) -> str:
    return str(value or "").strip().lower()


def _unique_strings(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        clean = str(value or "").strip()
        if not clean:
            continue
        key = clean.casefold()
        if key in seen:
            continue
        seen.add(key)
        out.append(clean)
    return out


def _stable_fact_id(
    text: str,
    source_entity: str,
    target_entity: str,
    edge_type: str,
    valid_at: str,
) -> str:
    raw = "\u241f".join(
        (
            _normalize(source_entity),
            _normalize(edge_type),
            _normalize(target_entity),
            _normalize(text),
            _normalize(valid_at),
        )
    )
    return "fact_" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


@dataclass
class FactRecord:
    """A durable legacy atomic fact extracted from conversation episodes."""

    text: str
    source_entity: str = ""
    target_entity: str = ""
    edge_type: str = "RELATES_TO"
    raw_edge_type: str = ""
    valid_at: str = ""
    recorded_at: str = ""
    valid_until: str = ""
    entities: list[str] = field(default_factory=list)
    source_episode: str = ""
    source_session_id: str = ""
    confidence: float = 0.0
    fact_id: str = ""

    def __post_init__(self) -> None:
        self.text = str(self.text or "").strip()
        self.source_entity = str(self.source_entity or "").strip()
        self.target_entity = str(self.target_entity or "").strip()
        self.edge_type = str(self.edge_type or "RELATES_TO").strip() or "RELATES_TO"
        self.raw_edge_type = str(self.raw_edge_type or "").strip()
        self.valid_at = str(self.valid_at or "").strip()
        self.recorded_at = str(self.recorded_at or "").strip() or now_iso()
        self.valid_until = str(self.valid_until or "").strip()
        self.source_episode = str(self.source_episode or "").strip()
        self.source_session_id = str(self.source_session_id or "").strip()
        self.entities = _unique_strings([*self.entities, self.source_entity, self.target_entity])
        try:
            self.confidence = float(self.confidence)
        except (TypeError, ValueError):
            self.confidence = 0.0
        self.fact_id = str(self.fact_id or "").strip() or _stable_fact_id(
            self.text,
            self.source_entity,
            self.target_entity,
            self.edge_type,
            self.valid_at,
        )

    @property
    def dedup_key(self) -> tuple[str, str, str, str, str]:
        return (
            _normalize(self.source_entity),
            _normalize(self.edge_type),
            _normalize(self.target_entity),
            _normalize(self.text),
            _normalize(self.valid_at),
        )

    @property
    def storage_date(self) -> str:
        ts = self.recorded_at or self.valid_at
        if ts:
            try:
                return datetime.fromisoformat(ts.replace("Z", "+00:00")).date().isoformat()
            except ValueError:
                pass
        return today_local().isoformat()

    def is_active(self) -> bool:
        return self.valid_until == ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "fact_id": self.fact_id,
            "text": self.text,
            "source_entity": self.source_entity,
            "target_entity": self.target_entity,
            "edge_type": self.edge_type,
            "raw_edge_type": self.raw_edge_type,
            "valid_at": self.valid_at,
            "recorded_at": self.recorded_at,
            "valid_until": self.valid_until,
            "entities": list(self.entities),
            "source_episode": self.source_episode,
            "source_session_id": self.source_session_id,
            "confidence": self.confidence,
        }

    def to_json_line(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FactRecord:
        return cls(
            fact_id=str(data.get("fact_id", "") or ""),
            text=str(data.get("text", data.get("fact", "")) or ""),
            source_entity=str(data.get("source_entity", "") or ""),
            target_entity=str(data.get("target_entity", "") or ""),
            edge_type=str(data.get("edge_type", "RELATES_TO") or "RELATES_TO"),
            raw_edge_type=str(data.get("raw_edge_type", "") or ""),
            valid_at=str(data.get("valid_at", "") or ""),
            recorded_at=str(data.get("recorded_at", "") or ""),
            valid_until=str(data.get("valid_until", "") or ""),
            entities=[str(v) for v in data.get("entities", []) if str(v).strip()]
            if isinstance(data.get("entities", []), list)
            else [],
            source_episode=str(data.get("source_episode", "") or ""),
            source_session_id=str(data.get("source_session_id", "") or ""),
            confidence=float(data.get("confidence", 0.0) or 0.0),
        )

    @classmethod
    def from_json_line(cls, line: str) -> FactRecord:
        data = json.loads(line)
        if not isinstance(data, dict):
            raise ValueError("fact line must be a JSON object")
        return cls.from_dict(data)


def facts_dir(anima_dir: Path) -> Path:
    return Path(anima_dir) / "facts"


def fact_file_for_record(anima_dir: Path, record: FactRecord) -> Path:
    return facts_dir(anima_dir) / f"{record.storage_date}.jsonl"


def is_fact_active(record: FactRecord) -> bool:
    return record.is_active()


def _process_lock(path: Path) -> threading.Lock:
    resolved = path.resolve()
    with _LOCKS_GUARD:
        if resolved not in _LOCKS:
            _LOCKS[resolved] = threading.Lock()
        return _LOCKS[resolved]


@contextlib.contextmanager
def _locked_file(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_suffix(path.suffix + ".lock")
    thread_lock = _process_lock(lock_path)
    with thread_lock, open(lock_path, "a+", encoding="utf-8") as lock_file:
        try:
            import fcntl

            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        except (ImportError, OSError):
            logger.debug("OS file lock unavailable for %s", lock_path, exc_info=True)
        try:
            yield
        finally:
            try:
                import fcntl

                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
            except (ImportError, OSError):
                pass


def read_fact_records(path: Path, *, include_expired: bool = False) -> list[FactRecord]:
    if not path.is_file():
        return []

    records: list[FactRecord] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        logger.warning("Failed to read facts file %s", path, exc_info=True)
        return []

    for line_no, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            record = FactRecord.from_json_line(line)
        except (json.JSONDecodeError, TypeError, ValueError):
            logger.warning("Skipping invalid fact line %s:%d", path, line_no)
            continue
        if include_expired or record.is_active():
            records.append(record)
    return records


def iter_fact_records(anima_dir: Path, *, include_expired: bool = False) -> Iterator[FactRecord]:
    directory = facts_dir(anima_dir)
    if not directory.is_dir():
        return
    for path in sorted(directory.glob("*.jsonl")):
        yield from read_fact_records(path, include_expired=include_expired)


def rewrite_fact_records(path: Path, records: list[FactRecord]) -> None:
    with _locked_file(path):
        unique = _dedup_records(records)
        content = "\n".join(record.to_json_line() for record in unique)
        atomic_write_text(path, content + ("\n" if content else ""))


def append_fact_records(anima_dir: Path, records: list[FactRecord]) -> list[FactRecord]:
    """Append records by JSONL day file, skipping stable duplicates."""
    by_path: dict[Path, list[FactRecord]] = {}
    for record in records:
        if record.text:
            by_path.setdefault(fact_file_for_record(anima_dir, record), []).append(record)

    stored: list[FactRecord] = []
    for path, new_records in by_path.items():
        with _locked_file(path):
            existing = read_fact_records(path, include_expired=True)
            seen_ids = {record.fact_id for record in existing}
            seen_keys = {record.dedup_key for record in existing}
            additions: list[FactRecord] = []
            for record in new_records:
                if record.fact_id in seen_ids or record.dedup_key in seen_keys:
                    continue
                seen_ids.add(record.fact_id)
                seen_keys.add(record.dedup_key)
                additions.append(record)
            if not additions:
                continue
            with open(path, "a", encoding="utf-8") as f:
                for record in additions:
                    f.write(record.to_json_line() + "\n")
                f.flush()
                os.fsync(f.fileno())
            stored.extend(additions)
    return stored


def _dedup_records(records: list[FactRecord]) -> list[FactRecord]:
    seen_ids: set[str] = set()
    seen_keys: set[tuple[str, str, str, str, str]] = set()
    out: list[FactRecord] = []
    for record in records:
        if not record.text:
            continue
        if record.fact_id in seen_ids or record.dedup_key in seen_keys:
            continue
        seen_ids.add(record.fact_id)
        seen_keys.add(record.dedup_key)
        out.append(record)
    return out
