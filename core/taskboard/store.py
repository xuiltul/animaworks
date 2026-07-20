"""SQLite-backed TaskBoard metadata store."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from core.paths import get_taskboard_db_path
from core.taskboard.models import AttentionVisibility, BoardColumn, TaskBoardMetadata
from core.time_utils import now_iso

TASKBOARD_EVENT_TYPES = {
    "metadata_upserted",
    "visibility_changed",
    "column_changed",
    "snoozed",
    "expired",
    "archived",
    "tombstoned",
    "notification_acknowledged",
    "surface_recorded",
    "stale_processing_recovered",
}

_METADATA_COLUMNS = [
    "anima_name",
    "task_id",
    "visibility",
    "column",
    "position",
    "expires_at",
    "snoozed_until",
    "last_notified_at",
    "notification_key",
    "surface_count",
    "source_ref",
    "replaced_by",
    "tombstone_reason",
    "updated_at",
    "updated_by",
]

_MUTABLE_METADATA_FIELDS = set(_METADATA_COLUMNS) - {"anima_name", "task_id"}

_VISIBILITY_VALUES = "', '".join(visibility.value for visibility in AttentionVisibility)
_COLUMN_VALUES = "', '".join(column.value for column in BoardColumn)

_SCHEMA_SQL = f"""
CREATE TABLE IF NOT EXISTS taskboard_metadata (
    anima_name TEXT NOT NULL,
    task_id TEXT NOT NULL,
    visibility TEXT NOT NULL DEFAULT 'active'
        CHECK(visibility IN ('{_VISIBILITY_VALUES}')),
    column TEXT CHECK(column IS NULL OR column IN ('{_COLUMN_VALUES}')),
    position REAL,
    expires_at TEXT,
    snoozed_until TEXT,
    last_notified_at TEXT,
    notification_key TEXT,
    surface_count INTEGER NOT NULL DEFAULT 0 CHECK(surface_count >= 0),
    source_ref TEXT,
    replaced_by TEXT,
    tombstone_reason TEXT,
    updated_at TEXT NOT NULL,
    updated_by TEXT NOT NULL DEFAULT 'system',
    PRIMARY KEY (anima_name, task_id)
);

CREATE INDEX IF NOT EXISTS idx_taskboard_metadata_visibility
    ON taskboard_metadata(visibility);
CREATE INDEX IF NOT EXISTS idx_taskboard_metadata_column_position
    ON taskboard_metadata(column, position);

CREATE TABLE IF NOT EXISTS taskboard_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    actor TEXT NOT NULL,
    event_type TEXT NOT NULL
        CHECK(event_type IN ({", ".join(repr(event_type) for event_type in sorted(TASKBOARD_EVENT_TYPES))})),
    anima_name TEXT NOT NULL,
    task_id TEXT NOT NULL,
    payload_json TEXT NOT NULL DEFAULT '{{}}'
);

CREATE INDEX IF NOT EXISTS idx_taskboard_events_task
    ON taskboard_events(anima_name, task_id, id);
"""


class TaskBoardStore:
    """SQLite store for TaskBoard metadata and audit events."""

    def __init__(self, db_path: Path | str | None = None) -> None:
        self.db_path = Path(db_path) if db_path is not None else get_taskboard_db_path()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        """Open a short-lived SQLite connection and always close it.

        ``sqlite3.Connection`` implements a context manager, but its ``__exit__``
        only commits or rolls back the active transaction; it does not close the
        connection. TaskBoard opens many WAL-mode connections from long-running
        Anima processes, so every call site must release the main DB/WAL/SHM file
        descriptors deterministically.
        """
        conn = self._connect()
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def _ensure_schema(self) -> None:
        with self._connection() as conn:
            conn.executescript(_SCHEMA_SQL)
            _migrate_taskboard_events_constraint(conn)

    def get_metadata(self, anima_name: str, task_id: str) -> TaskBoardMetadata | None:
        """Return metadata for a task, if present."""
        with self._connection() as conn:
            row = conn.execute(
                """
                SELECT *
                FROM taskboard_metadata
                WHERE anima_name = ? AND task_id = ?
                """,
                (anima_name, task_id),
            ).fetchone()
        return _metadata_from_row(row) if row else None

    def list_metadata(self, anima_name: str | None = None) -> list[TaskBoardMetadata]:
        """Return metadata rows, optionally restricted to one Anima."""
        if anima_name is None:
            query = """
                SELECT *
                FROM taskboard_metadata
                ORDER BY anima_name, task_id
            """
            params: tuple[str, ...] = ()
        else:
            query = """
                SELECT *
                FROM taskboard_metadata
                WHERE anima_name = ?
                ORDER BY task_id
            """
            params = (anima_name,)

        with self._connection() as conn:
            rows = conn.execute(query, params).fetchall()
        return [_metadata_from_row(row) for row in rows]

    def upsert_metadata(
        self,
        *,
        anima_name: str,
        task_id: str,
        actor: str = "system",
        event_type: str | None = None,
        **updates: Any,
    ) -> TaskBoardMetadata:
        """Create or update a metadata row and append one audit event."""
        unknown = set(updates) - _MUTABLE_METADATA_FIELDS
        if unknown:
            fields = ", ".join(sorted(unknown))
            raise TypeError(f"unknown TaskBoard metadata fields: {fields}")

        with self._connection() as conn:
            existing_row = conn.execute(
                """
                SELECT *
                FROM taskboard_metadata
                WHERE anima_name = ? AND task_id = ?
                """,
                (anima_name, task_id),
            ).fetchone()
            existing = _metadata_from_row(existing_row) if existing_row else None

            values = (
                existing.model_dump(mode="json")
                if existing is not None
                else {
                    "anima_name": anima_name,
                    "task_id": task_id,
                    "visibility": AttentionVisibility.ACTIVE.value,
                    "column": None,
                    "position": None,
                    "expires_at": None,
                    "snoozed_until": None,
                    "last_notified_at": None,
                    "notification_key": None,
                    "surface_count": 0,
                    "source_ref": None,
                    "replaced_by": None,
                    "tombstone_reason": None,
                    "updated_at": now_iso(),
                    "updated_by": actor,
                }
            )
            values.update(updates)
            if "updated_at" not in updates:
                values["updated_at"] = now_iso()
            if "updated_by" not in updates:
                values["updated_by"] = actor
            values["updated_at"] = str(values["updated_at"])
            values["updated_by"] = str(values["updated_by"])

            try:
                metadata = TaskBoardMetadata(**values)
            except ValidationError as exc:
                raise ValueError(str(exc)) from exc

            row_values = _upsert_metadata_row(conn, metadata)
            resolved_event_type = event_type or _event_type_for_update(existing, metadata, updates)
            payload = {
                "metadata": row_values,
                "updates": _json_safe_updates(updates),
            }
            _append_event(
                conn,
                event_type=resolved_event_type,
                anima_name=anima_name,
                task_id=task_id,
                actor=actor,
                payload=payload,
            )
            return metadata

    def append_event(
        self,
        *,
        event_type: str,
        anima_name: str,
        task_id: str,
        actor: str = "system",
        payload: Mapping[str, Any] | None = None,
        ts: str | None = None,
    ) -> int:
        """Append an audit event and return its row id."""
        with self._connection() as conn:
            return _append_event(
                conn,
                event_type=event_type,
                anima_name=anima_name,
                task_id=task_id,
                actor=actor,
                payload=payload or {},
                ts=ts,
            )

    def record_surface(
        self,
        *,
        anima_name: str,
        task_id: str,
        actor: str = "system",
        notification_key: str | None = None,
    ) -> TaskBoardMetadata:
        """Increment surface_count and record a surface event."""
        with self._connection() as conn:
            existing_row = conn.execute(
                """
                SELECT *
                FROM taskboard_metadata
                WHERE anima_name = ? AND task_id = ?
                """,
                (anima_name, task_id),
            ).fetchone()
            existing = _metadata_from_row(existing_row) if existing_row else None
            values = (
                existing.model_dump(mode="json")
                if existing is not None
                else {
                    "anima_name": anima_name,
                    "task_id": task_id,
                    "visibility": AttentionVisibility.ACTIVE.value,
                    "column": None,
                    "position": None,
                    "expires_at": None,
                    "snoozed_until": None,
                    "last_notified_at": None,
                    "notification_key": None,
                    "surface_count": 0,
                    "source_ref": None,
                    "replaced_by": None,
                    "tombstone_reason": None,
                    "updated_at": now_iso(),
                    "updated_by": actor,
                }
            )
            values["surface_count"] = int(values["surface_count"]) + 1
            values["last_notified_at"] = now_iso()
            values["updated_at"] = now_iso()
            values["updated_by"] = actor
            if notification_key is not None:
                values["notification_key"] = notification_key

            metadata = TaskBoardMetadata(**values)
            row_values = _upsert_metadata_row(conn, metadata)
            updates = {
                "surface_count": metadata.surface_count,
                "last_notified_at": metadata.last_notified_at,
            }
            if notification_key is not None:
                updates["notification_key"] = notification_key
            _append_event(
                conn,
                event_type="surface_recorded",
                anima_name=anima_name,
                task_id=task_id,
                actor=actor,
                payload={
                    "metadata": row_values,
                    "updates": updates,
                },
            )
            return metadata

    def delete_metadata(self, anima_name: str, task_id: str) -> bool:
        """Physically delete a metadata row. Does not append an audit event.

        Returns True if a row was deleted, False if no matching row existed.
        """
        with self._connection() as conn:
            cursor = conn.execute(
                """
                DELETE FROM taskboard_metadata
                WHERE anima_name = ? AND task_id = ?
                """,
                (anima_name, task_id),
            )
            return cursor.rowcount > 0

    def list_events(
        self,
        *,
        anima_name: str | None = None,
        task_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Return audit events as dictionaries with decoded payloads."""
        filters = []
        params: list[str] = []
        if anima_name is not None:
            filters.append("anima_name = ?")
            params.append(anima_name)
        if task_id is not None:
            filters.append("task_id = ?")
            params.append(task_id)

        where_clause = f"WHERE {' AND '.join(filters)}" if filters else ""
        with self._connection() as conn:
            rows = conn.execute(
                f"""
                SELECT *
                FROM taskboard_events
                {where_clause}
                ORDER BY id
                """,
                tuple(params),
            ).fetchall()

        events: list[dict[str, Any]] = []
        for row in rows:
            event = dict(row)
            event["payload"] = json.loads(event["payload_json"])
            events.append(event)
        return events


def _metadata_from_row(row: sqlite3.Row) -> TaskBoardMetadata:
    return TaskBoardMetadata(**{column: row[column] for column in _METADATA_COLUMNS})


def _upsert_metadata_row(conn: sqlite3.Connection, metadata: TaskBoardMetadata) -> dict[str, Any]:
    row_values = metadata.model_dump(mode="json")
    placeholders = ", ".join("?" for _ in _METADATA_COLUMNS)
    update_assignments = ", ".join(
        f"{column} = excluded.{column}" for column in _METADATA_COLUMNS if column not in {"anima_name", "task_id"}
    )
    conn.execute(
        f"""
        INSERT INTO taskboard_metadata ({", ".join(_METADATA_COLUMNS)})
        VALUES ({placeholders})
        ON CONFLICT(anima_name, task_id) DO UPDATE SET
            {update_assignments}
        """,
        tuple(row_values[column] for column in _METADATA_COLUMNS),
    )
    return row_values


def _append_event(
    conn: sqlite3.Connection,
    *,
    event_type: str,
    anima_name: str,
    task_id: str,
    actor: str,
    payload: Mapping[str, Any],
    ts: str | None = None,
) -> int:
    if event_type not in TASKBOARD_EVENT_TYPES:
        raise ValueError(f"unknown TaskBoard event type: {event_type}")
    cursor = conn.execute(
        """
        INSERT INTO taskboard_events (ts, actor, event_type, anima_name, task_id, payload_json)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            ts or now_iso(),
            actor,
            event_type,
            anima_name,
            task_id,
            json.dumps(payload, ensure_ascii=False, sort_keys=True),
        ),
    )
    return int(cursor.lastrowid)


def _migrate_taskboard_events_constraint(conn: sqlite3.Connection) -> None:
    row = conn.execute(
        """
        SELECT sql
        FROM sqlite_master
        WHERE type = 'table' AND name = 'taskboard_events'
        """
    ).fetchone()
    if row is None:
        return
    create_sql = str(row["sql"] or "")
    if "stale_processing_recovered" in create_sql:
        return

    conn.execute("ALTER TABLE taskboard_events RENAME TO taskboard_events_legacy")
    conn.execute(
        f"""
        CREATE TABLE taskboard_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL,
            actor TEXT NOT NULL,
            event_type TEXT NOT NULL
                CHECK(event_type IN ({", ".join(repr(event_type) for event_type in sorted(TASKBOARD_EVENT_TYPES))})),
            anima_name TEXT NOT NULL,
            task_id TEXT NOT NULL,
            payload_json TEXT NOT NULL DEFAULT '{{}}'
        )
        """
    )
    conn.execute(
        """
        INSERT INTO taskboard_events (id, ts, actor, event_type, anima_name, task_id, payload_json)
        SELECT id, ts, actor, event_type, anima_name, task_id, payload_json
        FROM taskboard_events_legacy
        """
    )
    conn.execute("DROP TABLE taskboard_events_legacy")
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_taskboard_events_task
            ON taskboard_events(anima_name, task_id, id)
        """
    )


def _event_type_for_update(
    existing: TaskBoardMetadata | None,
    metadata: TaskBoardMetadata,
    updates: Mapping[str, Any],
) -> str:
    if existing is None:
        return "metadata_upserted"
    if "visibility" in updates and metadata.visibility != existing.visibility:
        if metadata.visibility in {
            AttentionVisibility.SNOOZED,
            AttentionVisibility.EXPIRED,
            AttentionVisibility.ARCHIVED,
            AttentionVisibility.TOMBSTONED,
        }:
            return metadata.visibility.value
        return "visibility_changed"
    if "column" in updates and metadata.column != existing.column:
        return "column_changed"
    return "metadata_upserted"


def _json_safe_updates(updates: Mapping[str, Any]) -> dict[str, Any]:
    safe: dict[str, Any] = {}
    for key, value in updates.items():
        if isinstance(value, AttentionVisibility | BoardColumn):
            safe[key] = value.value
        else:
            safe[key] = value
    return safe
