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
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from core.memory._io import atomic_write_text
from core.time_utils import ensure_aware, now_iso, now_local, today_local

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

    def is_active(self, as_of_time: str | datetime | None = None) -> bool:
        return is_valid_until_active(self.valid_until, as_of_time=as_of_time)

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


@dataclass(frozen=True)
class FactRecordUpdate:
    """A fact row updated in a JSONL facts file."""

    path: Path
    line_no: int
    record: FactRecord


def _parse_timestamp(value: str | datetime | None) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return ensure_aware(value)
    raw = str(value or "").strip()
    if not raw:
        return None
    try:
        return ensure_aware(datetime.fromisoformat(raw.replace("Z", "+00:00")))
    except (TypeError, ValueError):
        return None


def is_valid_until_active(valid_until: str, *, as_of_time: str | datetime | None = None) -> bool:
    """Return whether a ``valid_until`` value is still active at ``as_of_time``."""
    if not str(valid_until or "").strip():
        return True
    until = _parse_timestamp(valid_until)
    if until is None:
        return False
    as_of = _parse_timestamp(as_of_time) or now_local()
    return until > as_of


def is_fact_active(record: FactRecord, *, as_of_time: str | datetime | None = None) -> bool:
    return record.is_active(as_of_time=as_of_time)


def fact_entity_names(record: FactRecord) -> list[str]:
    """Return stable unique entity names carried by a fact record."""
    return _unique_strings([*record.entities, record.source_entity, record.target_entity])


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


def read_fact_records(
    path: Path,
    *,
    include_expired: bool = False,
    as_of_time: str | datetime | None = None,
) -> list[FactRecord]:
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
        if include_expired or record.is_active(as_of_time=as_of_time):
            records.append(record)
    return records


def iter_fact_records(
    anima_dir: Path,
    *,
    include_expired: bool = False,
    as_of_time: str | datetime | None = None,
) -> Iterator[FactRecord]:
    directory = facts_dir(anima_dir)
    if not directory.is_dir():
        return
    for path in sorted(directory.glob("*.jsonl")):
        yield from read_fact_records(path, include_expired=include_expired, as_of_time=as_of_time)


def iter_active_fact_records(
    anima_dir: Path,
    *,
    as_of_time: str | datetime | None = None,
) -> Iterator[FactRecord]:
    """Iterate active facts for retrieval/graph consumers."""
    yield from iter_fact_records(anima_dir, include_expired=False, as_of_time=as_of_time)


def rewrite_fact_records(path: Path, records: list[FactRecord]) -> None:
    with _locked_file(path):
        unique = _dedup_records(records)
        _write_fact_records_unlocked(path, unique)


def find_fact_record(anima_dir: Path, fact_id: str) -> FactRecordUpdate | None:
    """Find a fact by id, including expired rows."""
    wanted = str(fact_id or "").strip()
    if not wanted:
        return None
    directory = facts_dir(anima_dir)
    if not directory.is_dir():
        return None
    for path in sorted(directory.glob("*.jsonl")):
        records = read_fact_records(path, include_expired=True)
        for idx, record in enumerate(records, 1):
            if record.fact_id == wanted:
                return FactRecordUpdate(path=path, line_no=idx, record=record)
    return None


def update_fact_record_by_id(
    anima_dir: Path,
    fact_id: str,
    updater: Callable[[FactRecord], FactRecord | None],
    *,
    path: Path | None = None,
) -> FactRecordUpdate | None:
    _stored, updates = update_fact_records_and_append(
        anima_dir,
        {fact_id: updater},
        [],
        update_paths=[path] if path is not None else None,
    )
    return updates[0] if updates else None


def update_fact_records_and_append(
    anima_dir: Path,
    updaters: dict[str, Callable[[FactRecord], FactRecord | None]],
    records_to_append: list[FactRecord],
    *,
    update_paths: list[Path] | None = None,
) -> tuple[list[FactRecord], list[FactRecordUpdate]]:
    """Apply fact_id updates and append new records under the same file locks."""
    pending = {
        str(fact_id or "").strip(): updater for fact_id, updater in updaters.items() if str(fact_id or "").strip()
    }
    append_by_path: dict[Path, list[FactRecord]] = {}
    for record in records_to_append:
        if record.text:
            append_by_path.setdefault(fact_file_for_record(anima_dir, record), []).append(record)

    directory = facts_dir(anima_dir)
    candidate_paths = set(append_by_path)
    if update_paths is not None:
        candidate_paths.update(Path(path) for path in update_paths)
    elif pending and directory.is_dir():
        candidate_paths.update(directory.glob("*.jsonl"))
    if not candidate_paths:
        return [], []

    stored: list[FactRecord] = []
    updated: list[FactRecordUpdate] = []
    original_by_path: dict[Path, list[FactRecord]] = {}
    existed_by_path: dict[Path, bool] = {}
    changed_by_path: dict[Path, list[FactRecord]] = {}

    with contextlib.ExitStack() as stack:
        for path in sorted(candidate_paths):
            stack.enter_context(_locked_file(path))

        for path in sorted(candidate_paths):
            existed_by_path[path] = path.exists()
            records = read_fact_records(path, include_expired=True)
            original_by_path[path] = list(records)
            changed = False
            for idx, record in enumerate(records):
                updater = pending.get(record.fact_id)
                if updater is None:
                    continue
                replacement = updater(record)
                pending.pop(record.fact_id, None)
                if replacement is None:
                    continue
                records[idx] = replacement
                updated.append(FactRecordUpdate(path=path, line_no=idx + 1, record=replacement))
                changed = True

            additions = _appendable_records(records, append_by_path.get(path, []))
            if additions:
                records.extend(additions)
                stored.extend(additions)
                changed = True
            if changed:
                changed_by_path[path] = records

        if pending:
            missing = ", ".join(sorted(pending))
            raise KeyError(f"missing fact ids: {missing}")

        try:
            for path, records in changed_by_path.items():
                _write_fact_records_unlocked(path, _dedup_records(records))
        except Exception:
            for path, records in original_by_path.items():
                if path in changed_by_path:
                    try:
                        if existed_by_path.get(path, False):
                            _write_fact_records_unlocked(path, records)
                        elif path.exists():
                            path.unlink()
                    except Exception:
                        logger.warning("Failed to roll back facts file %s", path, exc_info=True)
            raise

    return stored, updated


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
            additions = _appendable_records(existing, new_records)
            if not additions:
                continue
            with open(path, "a", encoding="utf-8") as f:
                for record in additions:
                    f.write(record.to_json_line() + "\n")
                f.flush()
                os.fsync(f.fileno())
            stored.extend(additions)
    return stored


def _appendable_records(existing: list[FactRecord], new_records: list[FactRecord]) -> list[FactRecord]:
    seen_ids = {record.fact_id for record in existing}
    seen_keys = {record.dedup_key for record in existing}
    additions: list[FactRecord] = []
    for record in new_records:
        if record.fact_id in seen_ids or record.dedup_key in seen_keys:
            continue
        seen_ids.add(record.fact_id)
        seen_keys.add(record.dedup_key)
        additions.append(record)
    return additions


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


def _write_fact_records_unlocked(path: Path, records: list[FactRecord]) -> None:
    content = "\n".join(record.to_json_line() for record in records if record.text)
    atomic_write_text(path, content + ("\n" if content else ""))
