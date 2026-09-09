"""Durable execution records, separate from TaskBoard presentation metadata.

One task owns its complete input. Attempts carry fenced execution identities;
JSONL and pending files are import/export formats, never a second authority.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Any

from core.schemas import TaskEntry
from core.time_utils import now_iso

logger = logging.getLogger(__name__)

_attempt_identity: ContextVar[dict[str, str] | None] = ContextVar("task_attempt_identity", default=None)


def current_attempt_identity() -> dict[str, str] | None:
    value = _attempt_identity.get()
    if value is not None:
        return value
    raw = os.environ.get("ANIMAWORKS_TASK_IDENTITY", "")
    if not raw:
        return None
    value = json.loads(raw)
    if not isinstance(value, dict) or any(
        not isinstance(value.get(key), str) or not value[key] for key in ("anima", "task_id", "token")
    ):
        raise ValueError("Invalid task execution identity")
    return value


@contextmanager
def attempt_scope(identity: dict[str, str] | None) -> Iterator[None]:
    token = _attempt_identity.set(identity)
    try:
        yield
    finally:
        _attempt_identity.reset(token)


_SCHEMA = """
CREATE TABLE IF NOT EXISTS tasks (
    anima TEXT NOT NULL,
    task_id TEXT NOT NULL,
    entry_json TEXT NOT NULL,
    input_json TEXT,
    ready INTEGER NOT NULL DEFAULT 0 CHECK(ready IN (0,1)),
    current_attempt TEXT,
    archived INTEGER NOT NULL DEFAULT 0 CHECK(archived IN (0,1)),
    PRIMARY KEY(anima, task_id)
);
CREATE TABLE IF NOT EXISTS task_attempts (
    token TEXT PRIMARY KEY,
    anima TEXT NOT NULL,
    task_id TEXT NOT NULL,
    number INTEGER NOT NULL,
    identity_json TEXT NOT NULL,
    started_at TEXT NOT NULL,
    ended_at TEXT,
    stop_kind TEXT,
    result_ref TEXT,
    UNIQUE(anima, task_id, number)
);
CREATE INDEX IF NOT EXISTS task_attempts_active ON task_attempts(anima, ended_at);
CREATE TABLE IF NOT EXISTS task_aliases (
    viewer TEXT NOT NULL,
    alias TEXT NOT NULL,
    anima TEXT NOT NULL,
    task_id TEXT NOT NULL,
    PRIMARY KEY(viewer, alias)
);
CREATE TABLE IF NOT EXISTS task_imports (
    anima TEXT PRIMARY KEY,
    imported_at TEXT NOT NULL,
    source_path TEXT NOT NULL,
    report_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS task_wakeups (
    anima TEXT NOT NULL,
    task_id TEXT NOT NULL,
    attempt_token TEXT NOT NULL,
    reason TEXT NOT NULL,
    created_at TEXT NOT NULL,
    acknowledged_at TEXT,
    PRIMARY KEY(anima, task_id, attempt_token)
);
CREATE TABLE IF NOT EXISTS task_claim_control (
    anima TEXT PRIMARY KEY,
    paused INTEGER NOT NULL CHECK(paused IN (0,1))
);
"""


def task_database_path(anima_dir: Path) -> Path:
    """Use the runtime's existing DB, including isolated fixture runtimes."""
    parent = anima_dir.resolve().parent
    root = parent.parent if parent.name == "animas" else parent
    return root / "shared" / "taskboard.sqlite3"


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def process_identity() -> dict[str, Any]:
    """PID plus creation time prevents a reused PID from retaining a claim."""
    import psutil

    process = psutil.Process()
    return {"pid": process.pid, "process_start_time": process.create_time()}


def identity_liveness(identity: dict[str, Any]) -> str:
    import psutil

    try:
        pid = int(identity.get("task_pid") or identity["pid"])
        process = psutil.Process(pid)
        if abs(process.create_time() - float(identity["process_start_time"])) > 0.01:
            return "dead"
        return "dead" if process.status() == psutil.STATUS_ZOMBIE else "live"
    except (psutil.NoSuchProcess, ValueError):
        return "dead"
    except (psutil.AccessDenied, KeyError, TypeError, OSError):
        return "unknown"


class TaskStore:
    """Short SQLite transactions; no connection is held during model work."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self._local = threading.local()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as db:
            if not db.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='task_claim_control'"
            ).fetchone():
                db.executescript(_SCHEMA)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        db = sqlite3.connect(self.db_path, timeout=30, isolation_level=None)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA busy_timeout=30000")
        db.execute("PRAGMA synchronous=FULL")
        try:
            yield db
        finally:
            db.close()

    @contextmanager
    def reader(self) -> Iterator[sqlite3.Connection]:
        current = getattr(self._local, "connection", None)
        if current is not None:
            yield current
            return
        with self._connect() as db:
            db.execute("BEGIN")
            self._local.connection = db
            self._local.read_only = True
            try:
                yield db
            finally:
                db.rollback()
                self._local.connection = None
                self._local.read_only = False

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        current = getattr(self._local, "connection", None)
        if current is not None:
            if getattr(self._local, "read_only", False):
                raise RuntimeError("Cannot mutate tasks inside a read-only snapshot")
            yield current
            return
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            self._local.connection = db
            self._local.after_commit = []
            committed = False
            try:
                yield db
                db.commit()
                committed = True
            except BaseException:
                db.rollback()
                raise
            finally:
                self._local.connection = None
                callbacks = self._local.after_commit
                self._local.after_commit = []
        if committed:
            for callback in callbacks:
                try:
                    callback()
                except Exception:
                    logger.warning("Task post-commit projection failed", exc_info=True)

    def after_commit(self, callback: Callable[[], Any]) -> None:
        """Presentation writes must not contend with the task's own transaction."""
        if getattr(self._local, "connection", None) is not None:
            self._local.after_commit.append(callback)
        else:
            callback()

    def _resolve(self, db: sqlite3.Connection, anima: str, task_id: str) -> tuple[str, str]:
        alias = db.execute(
            "SELECT anima, task_id FROM task_aliases WHERE viewer=? AND alias=?", (anima, task_id)
        ).fetchone()
        return (alias["anima"], alias["task_id"]) if alias else (anima, task_id)

    def read(self, anima: str, *, archived: bool = False) -> dict[str, TaskEntry]:
        with self.reader() as db:
            rows = db.execute(
                "SELECT entry_json FROM tasks WHERE anima=?" + ("" if archived else " AND archived=0"), (anima,)
            ).fetchall()
            result = {entry.task_id: entry for row in rows if (entry := TaskEntry(**json.loads(row[0])))}
            aliases = db.execute(
                "SELECT a.alias, a.anima, a.task_id, t.entry_json FROM task_aliases a "
                "JOIN tasks t ON t.anima=a.anima AND t.task_id=a.task_id WHERE a.viewer=?"
                + ("" if archived else " AND t.archived=0"),
                (anima,),
            ).fetchall()
            for row in aliases:
                entry = TaskEntry(**json.loads(row["entry_json"]))
                delegated_status = entry.status
                entry.task_id = row["alias"]
                if entry.status in {"pending", "in_progress"}:
                    entry.status = "delegated"
                entry.meta = {
                    **entry.meta,
                    "delegated_to": row["anima"],
                    "delegated_task_id": row["task_id"],
                    "delegated_status": delegated_status,
                }
                result[entry.task_id] = entry
            return result

    def apply(self, anima: str, event: dict[str, Any]) -> None:
        """Compatibility entry/update API; merges within the write transaction."""
        with self.transaction() as db:
            owner, task_id = self._resolve(db, anima, str(event["task_id"]))
            existing = db.execute("SELECT * FROM tasks WHERE anima=? AND task_id=?", (owner, task_id)).fetchone()
            identity = current_attempt_identity()
            if identity and (identity["anima"], identity["task_id"]) == (owner, task_id):
                if not existing or existing["current_attempt"] != identity["token"]:
                    raise ValueError("Stale task attempt cannot change the current task")
            if event.get("_event") == "update":
                if not existing:
                    return
                entry = json.loads(existing["entry_json"])
                new_status = event.get("status", entry["status"])
                if entry["status"] == "cancelled" and new_status == "pending" and existing["current_attempt"]:
                    raise ValueError("Cannot reopen a cancelled task until its active attempt stops")
                if entry["status"] == "cancelled" and new_status not in {"cancelled", "pending"}:
                    return
                for key in ("status", "summary", "updated_at"):
                    if key in event:
                        entry[key] = event[key]
                if "meta" in event:
                    entry["meta"] = {**entry.get("meta", {}), **event["meta"]}
                db.execute(
                    "UPDATE tasks SET entry_json=?, archived=0, ready=CASE WHEN ? IN ('done','cancelled') "
                    "THEN 0 ELSE ready END WHERE anima=? AND task_id=?",
                    (_json(entry), new_status, owner, task_id),
                )
                return
            entry = TaskEntry(**{key: value for key, value in event.items() if key != "_event"})
            target = entry.meta.get("delegated_to")
            child_id = entry.meta.get("delegated_task_id")
            if entry.status == "delegated" and target and child_id:
                self.alias(anima, entry.task_id, str(target), str(child_id))
                return
            if existing and existing["current_attempt"]:
                raise ValueError(f"Task {task_id} is running; cannot replace its input")
            if existing and existing["input_json"] is not None:
                raise ValueError(f"Task {task_id} already has execution input; use update or explicit resume")
            # Backlog entries are durable but intentionally not scheduled until submit.
            db.execute(
                "INSERT INTO tasks(anima,task_id,entry_json) VALUES(?,?,?) "
                "ON CONFLICT(anima,task_id) DO UPDATE SET entry_json=excluded.entry_json,archived=0",
                (anima, entry.task_id, _json(entry.model_dump())),
            )

    def get(self, anima: str, task_id: str) -> TaskEntry | None:
        """Primary-key lookup, including archived work and requester aliases."""
        with self.reader() as db:
            owner, resolved = self._resolve(db, anima, task_id)
            row = db.execute("SELECT entry_json FROM tasks WHERE anima=? AND task_id=?", (owner, resolved)).fetchone()
            if row is None:
                return None
            entry = TaskEntry(**json.loads(row[0]))
            if (owner, resolved) != (anima, task_id):
                entry.meta = {
                    **entry.meta,
                    "delegated_to": owner,
                    "delegated_task_id": resolved,
                    "delegated_status": entry.status,
                }
                entry.task_id = task_id
                if entry.status in {"pending", "in_progress"}:
                    entry.status = "delegated"
            return entry

    def alias(self, viewer: str, alias: str, anima: str, task_id: str) -> None:
        with self.transaction() as db:
            existing = db.execute(
                "SELECT anima,task_id FROM task_aliases WHERE viewer=? AND alias=?", (viewer, alias)
            ).fetchone()
            if existing and tuple(existing) != (anima, task_id):
                raise ValueError("Task alias already names another task")
            db.execute("INSERT OR IGNORE INTO task_aliases VALUES(?,?,?,?)", (viewer, alias, anima, task_id))

    def submit(self, anima: str, entry: TaskEntry, payload: dict[str, Any], *, resume: bool = False) -> bool:
        """Atomically publish both task and complete execution input.

        Re-delivery is idempotent even after completion. Explicit resume may
        reopen an ended attempt; an active attempt must finish/cancel first.
        """
        if payload.get("task_id") != entry.task_id:
            raise ValueError("Task input ID differs from task entry ID")
        with self.transaction() as db:
            previous = db.execute("SELECT * FROM tasks WHERE anima=? AND task_id=?", (anima, entry.task_id)).fetchone()
            identity = current_attempt_identity()
            if identity and (identity["anima"], identity["task_id"]) == (anima, entry.task_id):
                if not previous or previous["current_attempt"] != identity["token"]:
                    raise ValueError("Stale task attempt cannot publish or resume itself")
            if previous:
                if previous["current_attempt"]:
                    if resume:
                        raise ValueError("Cannot resume an active attempt")
                    return False
                if previous["input_json"] is not None and not resume:
                    return False
                if resume and json.loads(previous["entry_json"])["status"] in {"done", "cancelled"}:
                    raise ValueError("A terminal task cannot be resumed; create a new task")
            entry = entry.model_copy(update={"status": "pending", "updated_at": now_iso()})
            db.execute(
                "INSERT INTO tasks(anima,task_id,entry_json,input_json,ready) VALUES(?,?,?,?,1) "
                "ON CONFLICT(anima,task_id) DO UPDATE SET entry_json=excluded.entry_json, "
                "input_json=excluded.input_json,ready=1,archived=0",
                (anima, entry.task_id, _json(entry.model_dump()), _json(payload)),
            )
            db.execute(
                "UPDATE task_wakeups SET acknowledged_at=? WHERE anima=? AND task_id=? AND acknowledged_at IS NULL",
                (now_iso(), anima, entry.task_id),
            )
            return True

    def pending(self, anima: str) -> list[dict[str, Any]]:
        with self.reader() as db:
            rows = db.execute(
                "SELECT input_json FROM tasks WHERE anima=? AND ready=1 AND current_attempt IS NULL "
                "AND archived=0 ORDER BY rowid",
                (anima,),
            ).fetchall()
            return [json.loads(row[0]) for row in rows]

    def executable_ids(self, anima: str) -> set[str]:
        with self.reader() as db:
            return {
                row[0]
                for row in db.execute(
                    "SELECT task_id FROM tasks WHERE anima=? AND input_json IS NOT NULL AND archived=0", (anima,)
                )
            }

    def claim(
        self, anima: str, task_id: str, identity: dict[str, Any], *, max_active: int = 1
    ) -> dict[str, Any] | None:
        """Claim one ready task. Dependency status is checked in the same transaction."""
        with self.transaction() as db:
            paused = db.execute("SELECT paused FROM task_claim_control WHERE anima=?", (anima,)).fetchone()
            if paused and paused[0]:
                return None
            row = db.execute("SELECT * FROM tasks WHERE anima=? AND task_id=?", (anima, task_id)).fetchone()
            if not row or not row["ready"] or row["current_attempt"] or not row["input_json"]:
                return None
            entry = TaskEntry(**json.loads(row["entry_json"]))
            if entry.status != "pending":
                return None
            payload = json.loads(row["input_json"])
            active = db.execute(
                "SELECT input_json FROM tasks WHERE anima=? AND current_attempt IS NOT NULL", (anima,)
            ).fetchall()
            if len(active) >= max_active:
                return None
            is_serial = bool(payload.get("batch_id")) and not payload.get("parallel", False)
            same_batch = [
                p
                for item in active
                if (p := json.loads(item[0])).get("batch_id") and p.get("batch_id") == payload.get("batch_id")
            ]
            if same_batch and (is_serial or any(not item.get("parallel", False) for item in same_batch)):
                return None
            for dependency in payload.get("depends_on", []):
                dependency_owner, dependency_id = self._resolve(db, anima, dependency)
                dep = db.execute(
                    "SELECT entry_json,current_attempt FROM tasks WHERE anima=? AND task_id=?",
                    (dependency_owner, dependency_id),
                ).fetchone()
                if dep and json.loads(dep["entry_json"])["status"] == "cancelled":
                    db.execute("UPDATE tasks SET ready=0 WHERE anima=? AND task_id=?", (anima, task_id))
                    db.execute(
                        "INSERT OR IGNORE INTO task_wakeups VALUES(?,?,?,?,?,NULL)",
                        (anima, task_id, f"dependency-{task_id}-{dependency}", "dependency_cancelled", now_iso()),
                    )
                    return None
                if not dep or dep["current_attempt"] or json.loads(dep["entry_json"])["status"] != "done":
                    return None
            number = db.execute(
                "SELECT COALESCE(MAX(number),0)+1 FROM task_attempts WHERE anima=? AND task_id=?", (anima, task_id)
            ).fetchone()[0]
            token = uuid.uuid4().hex
            started = now_iso()
            entry.status = "in_progress"
            entry.updated_at = started
            # These fields describe this execution, not the immutable task.
            # Leaving them on a resumed claim can make the runner classify a
            # successful attempt using the previous attempt's crash. Historical
            # outcomes and result references remain in task_attempts.
            for key in ("last_run_stop_kind", "last_run_ended_at", "last_run_note"):
                entry.meta.pop(key, None)
            db.execute(
                "INSERT INTO task_attempts(token,anima,task_id,number,identity_json,started_at) VALUES(?,?,?,?,?,?)",
                (token, anima, task_id, number, _json(identity), started),
            )
            db.execute(
                "UPDATE tasks SET current_attempt=?,ready=0,entry_json=? WHERE anima=? AND task_id=?",
                (token, _json(entry.model_dump()), anima, task_id),
            )
            return {**payload, "_attempt_token": token, "_attempt_number": number}

    def pause_claims(self, anima: str, *, paused: bool = True) -> None:
        """Durable offline-migration gate, checked in each claim transaction."""
        with self.transaction() as db:
            db.execute(
                "INSERT INTO task_claim_control VALUES(?,?) ON CONFLICT(anima) DO UPDATE SET paused=excluded.paused",
                (anima, int(paused)),
            )

    def get_input(self, anima: str, task_id: str) -> dict[str, Any] | None:
        with self.reader() as db:
            owner, resolved = self._resolve(db, anima, task_id)
            row = db.execute("SELECT input_json FROM tasks WHERE anima=? AND task_id=?", (owner, resolved)).fetchone()
            return json.loads(row[0]) if row and row[0] else None

    def set_identity(self, token: str, identity: dict[str, Any]) -> bool:
        with self.transaction() as db:
            row = db.execute(
                "SELECT identity_json FROM task_attempts WHERE token=? AND ended_at IS NULL", (token,)
            ).fetchone()
            if not row:
                return False
            merged = {**json.loads(row[0]), **identity}
            db.execute("UPDATE task_attempts SET identity_json=? WHERE token=?", (_json(merged), token))
            return True

    def finish(
        self, token: str, *, status: str, stop_kind: str, summary: str | None = None, result_ref: str = ""
    ) -> bool:
        """End only the matching attempt. Incomplete work gets one durable wakeup."""
        if status not in {"pending", "done", "cancelled"}:
            raise ValueError(f"Invalid attempt outcome: {status}")
        with self.transaction() as db:
            row = db.execute(
                "SELECT t.*,a.token FROM tasks t JOIN task_attempts a ON a.token=t.current_attempt "
                "WHERE a.token=? AND a.ended_at IS NULL",
                (token,),
            ).fetchone()
            if not row:
                return False
            entry = TaskEntry(**json.loads(row["entry_json"]))
            if entry.status == "cancelled":
                status = "cancelled"
            entry.status = status
            entry.updated_at = now_iso()
            if summary is not None:
                entry.summary = summary
            entry.meta = {**entry.meta, "last_run_stop_kind": stop_kind, "last_attempt_token": token}
            db.execute(
                "UPDATE task_attempts SET ended_at=?,stop_kind=?,result_ref=? WHERE token=?",
                (entry.updated_at, stop_kind, result_ref, token),
            )
            db.execute(
                "UPDATE tasks SET entry_json=?,current_attempt=NULL,ready=0 WHERE anima=? AND task_id=?",
                (_json(entry.model_dump()), row["anima"], row["task_id"]),
            )
            if status == "pending":
                db.execute(
                    "INSERT OR IGNORE INTO task_wakeups VALUES(?,?,?,?,?,NULL)",
                    (row["anima"], row["task_id"], token, stop_kind, entry.updated_at),
                )
            elif status == "done" and row["input_json"] and json.loads(row["input_json"]).get("reply_to"):
                db.execute(
                    "INSERT OR IGNORE INTO task_wakeups VALUES(?,?,?,?,?,NULL)",
                    (row["anima"], row["task_id"], token, "completion", entry.updated_at),
                )
            return True

    def active_attempts(self, anima: str) -> list[dict[str, Any]]:
        with self.reader() as db:
            return [
                dict(row)
                for row in db.execute("SELECT * FROM task_attempts WHERE anima=? AND ended_at IS NULL", (anima,))
            ]

    def wakeups(self, anima: str) -> list[dict[str, Any]]:
        with self.reader() as db:
            return [
                dict(row)
                for row in db.execute(
                    "SELECT * FROM task_wakeups WHERE anima=? AND acknowledged_at IS NULL ORDER BY created_at", (anima,)
                )
            ]

    def acknowledge_wakeup(self, anima: str, token: str) -> None:
        with self.transaction() as db:
            db.execute(
                "UPDATE task_wakeups SET acknowledged_at=? WHERE anima=? AND attempt_token=?",
                (now_iso(), anima, token),
            )

    def compact(self, anima: str) -> int:
        with self.transaction() as db:
            cursor = db.execute(
                "UPDATE tasks SET archived=1 WHERE anima=? AND archived=0 AND current_attempt IS NULL "
                "AND json_extract(entry_json,'$.status') IN ('done','cancelled')",
                (anima,),
            )
            return cursor.rowcount

    def reference_records(self, anima: str) -> dict[str, dict[str, Any]]:
        """Current unfinished task references, including their execution inputs."""
        with self.reader() as db:
            rows = db.execute(
                "SELECT task_id,entry_json,input_json FROM tasks WHERE anima=? AND archived=0 "
                "AND json_extract(entry_json,'$.status') NOT IN ('done','cancelled')",
                (anima,),
            )
            return {
                row["task_id"]: {
                    "entry": json.loads(row["entry_json"]),
                    "input": json.loads(row["input_json"]) if row["input_json"] else None,
                }
                for row in rows
            }

    def rewrite_references(
        self,
        anima: str,
        transform: Callable[[dict[str, Any]], dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Update idle task references atomically without republishing work.

        The transform may change reference fields, but not ownership, ID,
        status, or immutable user instructions. Changed active attempts abort
        the entire operation. Readiness and attempt history remain untouched.
        """
        changes = []
        with self.transaction() as db:
            for task_id, before in self.reference_records(anima).items():
                after = transform(json.loads(_json(before)))
                if before == after:
                    continue
                for field in ("task_id", "assignee", "status", "original_instruction"):
                    if after["entry"].get(field) != before["entry"].get(field):
                        raise ValueError(f"Reference rewrite cannot change {field}")
                if (after["input"] or {}).get("task_id") != (before["input"] or {}).get("task_id"):
                    raise ValueError("Reference rewrite cannot change input task ID")
                if db.execute(
                    "SELECT current_attempt FROM tasks WHERE anima=? AND task_id=?", (anima, task_id)
                ).fetchone()[0]:
                    raise RuntimeError("Cannot rewrite references for an active task attempt")
                TaskEntry(**after["entry"])
                db.execute(
                    "UPDATE tasks SET entry_json=?,input_json=? WHERE anima=? AND task_id=?",
                    (
                        _json(after["entry"]),
                        _json(after["input"]) if after["input"] is not None else None,
                        anima,
                        task_id,
                    ),
                )
                changes.append({"task_id": task_id, "before": before, "after": after})
        return changes

    def maintenance_status(self, anima: str) -> dict[str, Any]:
        """Expose counts for an operator without dumping task instructions."""
        with self.transaction() as db:
            gate = db.execute("SELECT paused FROM task_claim_control WHERE anima=?", (anima,)).fetchone()
            imported = db.execute("SELECT report_json FROM task_imports WHERE anima=?", (anima,)).fetchone()
            return {
                "anima": anima,
                "quiesced": bool(gate and gate[0]),
                "invalid_import_rows": json.loads(imported[0]).get("invalid_rows", 0) if imported else 0,
                "active_attempts": db.execute(
                    "SELECT COUNT(*) FROM task_attempts WHERE anima=? AND ended_at IS NULL", (anima,)
                ).fetchone()[0],
                "owned_tasks": db.execute("SELECT COUNT(*) FROM tasks WHERE anima=?", (anima,)).fetchone()[0],
                "ready_tasks": db.execute("SELECT COUNT(*) FROM tasks WHERE anima=? AND ready=1", (anima,)).fetchone()[
                    0
                ],
            }

    def export(self, anima_dir: Path, *, destination: Path | None = None, descriptors: bool = False) -> int:
        """Produce a current rollback/export view; never replay old completed work."""
        from core.memory._io import atomic_write_text

        target_dir = destination or anima_dir
        with self.transaction() as db:
            if descriptors and self.active_attempts(anima_dir.name):
                raise RuntimeError("Cannot export runnable descriptors while attempts are active")
            entries = self.read(anima_dir.name, archived=True)
            payloads = (
                db.execute(
                    "SELECT task_id,input_json FROM tasks WHERE anima=? AND ready=1 AND archived=0 "
                    "AND current_attempt IS NULL AND json_extract(entry_json,'$.status')='pending'",
                    (anima_dir.name,),
                ).fetchall()
                if descriptors
                else []
            )
            pending = target_dir / "state" / "pending"
            if descriptors and pending.exists() and any(pending.iterdir()):
                raise FileExistsError("Export requires an empty pending directory")
            for row in payloads:
                task_id = row["task_id"]
                if Path(task_id).name != task_id or task_id in {".", ".."} or "\\" in task_id:
                    raise ValueError("Task ID cannot be used as a legacy descriptor filename")
            target = target_dir / "state" / "task_queue.jsonl"
            target.parent.mkdir(parents=True, exist_ok=True)
            atomic_write_text(target, "".join(_json(entry.model_dump()) + "\n" for entry in entries.values()))
            if descriptors:
                pending.mkdir(parents=True, exist_ok=True)
                for row in payloads:
                    atomic_write_text(pending / f"{row['task_id']}.json", row["input_json"] + "\n")
                # Retain inputs for non-runnable unfinished work without publishing it.
                inputs = [
                    dict(row)
                    for row in db.execute(
                        "SELECT task_id,input_json,ready FROM tasks WHERE anima=? AND input_json IS NOT NULL",
                        (anima_dir.name,),
                    )
                ]
                atomic_write_text(target_dir / "state" / "task_inputs_export.json", _json(inputs) + "\n")
            return len(entries)

    def transfer_anima_tasks(self, source: str, target: str, mapping: dict[str, str], transform) -> dict[str, Any]:
        """Move task ownership and references atomically after workers are quiesced.

        Inputs, archived records, attempt tokens/history, aliases and durable
        wakeups move together. A live claim on any changed task rejects the
        entire transaction; legacy interchange files remain untouched.
        """
        if source == target or len(set(mapping.values())) != len(mapping):
            raise ValueError("Invalid task ownership transfer")
        moved: list[str] = []
        updated: set[str] = set()
        with self.transaction() as db:
            if db.execute(
                "SELECT 1 FROM task_attempts WHERE anima IN (?,?) AND ended_at IS NULL", (source, target)
            ).fetchone():
                raise ValueError("Cannot merge an owner with an active attempt")
            rows = db.execute("SELECT * FROM tasks").fetchall()
            changes = []
            for row in rows:
                owner, old_id = row["anima"], row["task_id"]
                entry = json.loads(row["entry_json"])
                payload = json.loads(row["input_json"]) if row["input_json"] is not None else None
                rewritten = transform(owner, entry)
                rewritten_input = transform(owner, payload) if payload is not None else None
                if owner == source:
                    if old_id not in mapping:
                        raise ValueError(f"Missing task ID mapping: {old_id}")
                    new_id = mapping[old_id]
                    if db.execute("SELECT 1 FROM tasks WHERE anima=? AND task_id=?", (target, new_id)).fetchone():
                        raise ValueError(f"Conflicting canonical target task: {new_id}")
                else:
                    new_id = old_id
                if owner != source and rewritten == entry and rewritten_input == payload:
                    continue
                if (
                    row["current_attempt"]
                    or db.execute(
                        "SELECT 1 FROM task_attempts WHERE anima=? AND task_id=? AND ended_at IS NULL", (owner, old_id)
                    ).fetchone()
                ):
                    raise ValueError(f"Cannot merge a task with an active attempt: {owner}/{old_id}")
                rewritten["task_id"] = new_id
                TaskEntry(**rewritten)
                changes.append((row, new_id, rewritten, rewritten_input))
            for row, new_id, entry, payload in changes:
                owner, old_id = row["anima"], row["task_id"]
                new_owner = target if owner == source else owner
                db.execute(
                    "UPDATE tasks SET anima=?,task_id=?,entry_json=?,input_json=? WHERE anima=? AND task_id=?",
                    (new_owner, new_id, _json(entry), _json(payload) if payload is not None else None, owner, old_id),
                )
                updated.add(new_owner)
                if owner == source:
                    moved.append(new_id)
                    for table in ("task_attempts", "task_wakeups", "task_aliases"):
                        db.execute(
                            f"UPDATE {table} SET anima=?,task_id=? WHERE anima=? AND task_id=?",
                            (target, new_id, source, old_id),
                        )
                    for attempt in db.execute(
                        "SELECT token,result_ref FROM task_attempts WHERE anima=? AND task_id=?", (target, new_id)
                    ).fetchall():
                        ref = attempt["result_ref"] or ""
                        old_prefix = f"state/task_results/{old_id}/"
                        if ref.startswith(old_prefix):
                            db.execute(
                                "UPDATE task_attempts SET result_ref=? WHERE token=?",
                                (f"state/task_results/{new_id}/" + ref[len(old_prefix) :], attempt["token"]),
                            )
            # Source-owned aliases are references, not independent execution rows.
            for row in db.execute("SELECT * FROM task_aliases WHERE viewer=?", (source,)).fetchall():
                alias = mapping.get(row["alias"], row["alias"])
                existing = db.execute(
                    "SELECT anima,task_id FROM task_aliases WHERE viewer=? AND alias=?", (target, alias)
                ).fetchone()
                if existing and tuple(existing) != (row["anima"], row["task_id"]):
                    raise ValueError(f"Conflicting target task alias: {alias}")
                if db.execute("SELECT 1 FROM tasks WHERE anima=? AND task_id=?", (target, alias)).fetchone():
                    raise ValueError(f"Task alias conflicts with target execution: {alias}")
                db.execute(
                    "INSERT OR IGNORE INTO task_aliases(viewer,alias,anima,task_id) VALUES(?,?,?,?)",
                    (target, alias, row["anima"], row["task_id"]),
                )
                db.execute("DELETE FROM task_aliases WHERE viewer=? AND alias=?", (source, row["alias"]))
        return {
            "source_tasks_recreated": sorted(moved),
            "task_store_owners_updated": sorted(updated),
            "task_queue_files_updated": [],
        }

    def backup(self, destination: Path) -> None:
        """SQLite backup includes committed WAL data and refuses overwrites."""
        destination.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(destination, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        os.close(fd)
        with self._connect() as source:
            target = sqlite3.connect(destination)
            try:
                source.backup(target)
            finally:
                target.close()

    def import_legacy(self, anima_dir: Path) -> dict[str, Any]:
        """One-time transactional import; preserve originals for verification.

        Active/ambiguous legacy leases must be drained by the migration caller.
        Ledger-only pending work stays unscheduled, rather than being resurrected.
        """
        anima = anima_dir.name
        with self.reader() as reader:
            prior = reader.execute("SELECT report_json FROM task_imports WHERE anima=?", (anima,)).fetchone()
            if prior:
                return json.loads(prior[0])
        with self.transaction() as db:
            prior = db.execute("SELECT report_json FROM task_imports WHERE anima=?", (anima,)).fetchone()
            if prior:
                return json.loads(prior[0])
            entries: dict[str, dict[str, Any]] = {}
            archived_ids: set[str] = set()
            errors = 0
            non_task_rows = 0
            for filename in ("task_queue_archive.jsonl", "task_queue.jsonl"):
                path = anima_dir / "state" / filename
                if not path.exists():
                    continue
                for line in path.read_text(encoding="utf-8").splitlines():
                    if not line.strip():
                        continue
                    try:
                        raw = json.loads(line)
                        if (
                            isinstance(raw, dict)
                            and "task_id" not in raw
                            and raw.get("type") == "heartbeat"
                            and raw.get("status") == "done"
                            and raw.get("id")
                            and raw.get("created_at")
                        ):
                            non_task_rows += 1
                            continue
                        task_id = str(raw["task_id"])
                        if raw.pop("_event", None) == "update":
                            if task_id in entries:
                                entries[task_id].update(raw)
                        else:
                            entries[task_id] = raw
                        if filename == "task_queue_archive.jsonl":
                            archived_ids.add(task_id)
                        else:
                            archived_ids.discard(task_id)
                    except (ValueError, TypeError, KeyError):
                        errors += 1
            descriptors: dict[str, dict[str, Any]] = {}
            ignored_artifacts = 0
            terminal_descriptors = 0
            terminal_ids = {
                task_id
                for task_id, raw in entries.items()
                if raw.get("status") in {"done", "completed", "cancelled"}
                or (task_id in archived_ids and raw.get("status") == "failed")
            }
            pending = anima_dir / "state" / "pending"
            if pending.exists():
                from core.platform.processing_lease import is_processing_lease_live, read_processing_lease

                for path in sorted(pending.rglob("*.json")):
                    # Historical API evidence and arbitrary backup trees are not
                    # scheduler inputs. Preserve them in place without parsing.
                    if path.parent not in {
                        pending,
                        *(pending / name for name in ("processing", "failed", "suppressed")),
                    }:
                        ignored_artifacts += 1
                        continue
                    if path.parent.name == "processing":
                        if read_processing_lease(path) is None:
                            raise RuntimeError(
                                f"Resolve missing or malformed legacy lease before migration: {path.name}"
                            )
                        if is_processing_lease_live(path, expected_anima=anima):
                            raise RuntimeError(f"Drain active legacy task before migration: {path.name}")
                    try:
                        payload = json.loads(path.read_text(encoding="utf-8"))
                        if isinstance(payload, dict) and payload.get("task_type") == "llm" and payload.get("task_id"):
                            payload = dict(payload)
                            payload["_legacy_ready"] = path.parent == pending
                            task_id = str(payload["task_id"])
                            if task_id in terminal_ids:
                                # A terminal ledger wins over stale descriptors.
                                # Do not select one of conflicting historical inputs.
                                terminal_descriptors += 1
                                continue
                            previous = descriptors.get(task_id)
                            if previous is not None:
                                previous_input = {
                                    key: value for key, value in previous.items() if key != "_legacy_ready"
                                }
                                current_input = {key: value for key, value in payload.items() if key != "_legacy_ready"}
                                if previous_input != current_input:
                                    raise RuntimeError(
                                        f"Resolve conflicting legacy task descriptors before migration: {task_id}"
                                    )
                                # Duplicate files do not establish that an old attempt is safe to replay.
                                payload["_legacy_ready"] = payload["_legacy_ready"] and previous["_legacy_ready"]
                            descriptors[task_id] = payload
                    except (ValueError, OSError):
                        errors += 1
            imported = 0
            review_count = 0
            instruction_reconciliations = 0
            for task_id in entries.keys() | descriptors.keys():
                payload = descriptors.get(task_id)
                raw = entries.get(task_id)
                if raw is None and payload is not None:
                    raw = dict(
                        task_id=task_id,
                        ts=now_iso(),
                        updated_at=now_iso(),
                        source="anima",
                        assignee=anima,
                        original_instruction=payload.get("description", ""),
                        summary=payload.get("title", ""),
                        status="pending",
                        meta={"executor": "taskexec"},
                    )
                try:
                    assert raw is not None
                    if raw.get("status") == "completed":
                        raw = {**raw, "status": "done", "meta": {**raw.get("meta", {}), "legacy_status": "completed"}}
                    if (
                        payload is not None
                        and payload.get("description")
                        and raw.get("original_instruction") != payload["description"]
                    ):
                        raw = {
                            **raw,
                            "original_instruction": payload["description"],
                            "meta": {
                                **raw.get("meta", {}),
                                "legacy_original_instruction": raw.get("original_instruction", ""),
                            },
                        }
                        instruction_reconciliations += 1
                    archived_terminal = task_id in archived_ids and task_id in terminal_ids
                    if raw.get("status") in {"blocked", "failed", "in_progress"} and not archived_terminal:
                        raw = {
                            **raw,
                            "status": "pending",
                            "meta": {
                                **raw.get("meta", {}),
                                "migration_review": True,
                                "migration_review_reason": f"legacy_status:{raw.get('status')}",
                            },
                        }
                    elif payload and not payload.get("_legacy_ready") and raw.get("status") == "pending":
                        raw = {
                            **raw,
                            "meta": {
                                **raw.get("meta", {}),
                                "migration_review": True,
                                "migration_review_reason": "legacy_nonready_descriptor",
                            },
                        }
                    entry = TaskEntry(**raw)
                except (ValueError, TypeError):
                    errors += 1
                    continue
                self.apply(anima, entry.model_dump())
                if archived_terminal:
                    db.execute("UPDATE tasks SET archived=1,ready=0 WHERE anima=? AND task_id=?", (anima, task_id))
                if payload is not None and entry.status not in {"done", "cancelled", "delegated"}:
                    ready = bool(payload.pop("_legacy_ready", False)) and not entry.meta.get("migration_review")
                    db.execute(
                        "UPDATE tasks SET input_json=?,ready=? WHERE anima=? AND task_id=?",
                        (_json(payload), int(ready), anima, task_id),
                    )
                if entry.meta.get("migration_review"):
                    review_count += 1
                    db.execute(
                        "INSERT OR IGNORE INTO task_wakeups VALUES(?,?,?,?,?,NULL)",
                        (anima, task_id, f"migration-{task_id}", "migration_review", now_iso()),
                    )
                imported += 1
            report = {
                "tasks": imported,
                "invalid_rows": errors,
                "descriptors": len(descriptors),
                "review_required": review_count,
                "instruction_reconciliations": instruction_reconciliations,
                "non_task_rows": non_task_rows,
                "ignored_artifacts": ignored_artifacts,
                "terminal_descriptors": terminal_descriptors,
            }
            db.execute("INSERT INTO task_imports VALUES(?,?,?,?)", (anima, now_iso(), str(anima_dir), _json(report)))
            return report
