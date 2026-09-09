from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""SQLite health checks for per-anima ChromaDB stores."""

import logging
import os
import sqlite3
import time
from collections.abc import Callable
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger("animaworks.rag.sqlite_health")

CHROMA_SQLITE_NAME = "chroma.sqlite3"
DEFAULT_QUICK_CHECK_TIMEOUT_SECONDS = 10.0
QUICK_CHECK_CORRUPT_RETRY_COUNT = 2
QUICK_CHECK_CORRUPT_RETRY_DELAY_SECONDS = 0.5


@dataclass(frozen=True)
class SQLiteHealthResult:
    """Result of a Chroma SQLite health check."""

    db_path: Path
    ok: bool
    status: str
    details: tuple[str, ...] = ()
    error: str | None = None

    @property
    def corrupt(self) -> bool:
        return not self.ok and self.status == "corrupt"


def chroma_sqlite_path(persist_dir: Path) -> Path:
    """Return the SQLite database path inside a Chroma persistence directory."""
    return persist_dir / CHROMA_SQLITE_NAME


def quick_check_chroma_sqlite(
    persist_dir: Path,
    *,
    timeout_seconds: float = DEFAULT_QUICK_CHECK_TIMEOUT_SECONDS,
    runner: Callable[[Path, float], tuple[str, ...]] | None = None,
) -> SQLiteHealthResult:
    """Run ``PRAGMA quick_check`` against an existing Chroma SQLite DB.

    Missing databases are healthy for preflight purposes because Chroma will
    create them on first use. Possible corruption is retried briefly before it
    is confirmed; timeout and operational availability failures remain
    non-corruption statuses.
    """
    db_path = chroma_sqlite_path(persist_dir)
    if not db_path.exists():
        return SQLiteHealthResult(db_path=db_path, ok=True, status="missing")

    check_runner = runner or _run_quick_check
    result = _quick_check_once(db_path, timeout_seconds, check_runner)
    if not result.corrupt:
        return result

    logger.debug(
        "Chroma SQLite quick_check reported possible corruption; retrying: db=%s status=%s detail=%s",
        db_path,
        result.status,
        result.error or result.details,
    )
    for _attempt in range(QUICK_CHECK_CORRUPT_RETRY_COUNT):
        time.sleep(QUICK_CHECK_CORRUPT_RETRY_DELAY_SECONDS)
        result = _quick_check_once(db_path, timeout_seconds, check_runner)
        if not result.corrupt:
            return result
    return result


def _quick_check_once(
    db_path: Path,
    timeout_seconds: float,
    runner: Callable[[Path, float], tuple[str, ...]],
) -> SQLiteHealthResult:
    """Run one SQLite quick_check attempt and classify its result."""
    try:
        details = runner(db_path, timeout_seconds)
    except TimeoutError as exc:
        return SQLiteHealthResult(db_path=db_path, ok=False, status="timeout", error=str(exc))
    except sqlite3.OperationalError as exc:
        if _sqlite_busy_or_locked(exc):
            return SQLiteHealthResult(db_path=db_path, ok=False, status="busy", error=str(exc))
        if _sqlite_unavailable(exc):
            return SQLiteHealthResult(db_path=db_path, ok=False, status="unavailable", error=str(exc))
        return SQLiteHealthResult(db_path=db_path, ok=False, status="corrupt", error=str(exc))
    except sqlite3.DatabaseError as exc:
        return SQLiteHealthResult(db_path=db_path, ok=False, status="corrupt", error=str(exc))
    except OSError as exc:
        if _sqlite_unavailable(exc):
            return SQLiteHealthResult(db_path=db_path, ok=False, status="unavailable", error=str(exc))
        return SQLiteHealthResult(db_path=db_path, ok=False, status="unreadable", error=str(exc))

    normalized = tuple(item.strip() for item in details if item.strip())
    if normalized == ("ok",):
        return SQLiteHealthResult(db_path=db_path, ok=True, status="ok", details=normalized)
    return SQLiteHealthResult(db_path=db_path, ok=False, status="corrupt", details=normalized)


def bootstrap_chroma_sqlite_wal(
    persist_dir: Path,
    *,
    timeout_seconds: float = DEFAULT_QUICK_CHECK_TIMEOUT_SECONDS,
) -> SQLiteHealthResult:
    """Pre-create an empty Chroma SQLite file in WAL mode before client init.

    Chroma's first PersistentClient open otherwise creates the DB in the
    SQLite default ``delete`` journal mode.  WAL must be set *before* the
    Rust client takes ownership: a post-init Python rw connection that sets
    WAL and then closes will unlink -wal/-shm the live client just created
    (2026-08-11 incident).  An empty pre-created file is accepted by Chroma
    and keeps the durable journal mode for the whole lifetime of the store.
    """
    db_path = chroma_sqlite_path(persist_dir)
    if db_path.exists():
        return SQLiteHealthResult(db_path=db_path, ok=True, status="ok")

    try:
        persist_dir.mkdir(parents=True, exist_ok=True)
        # URI mode=rw refuses to create a missing file; use plain connect for bootstrap.
        # A Connection context manager commits/rolls back but does not close.
        # Finish RW ownership here, before a Rust client can create its WAL.
        with closing(sqlite3.connect(db_path, timeout=timeout_seconds)) as conn, conn:
            journal_mode = str(conn.execute("PRAGMA journal_mode=WAL").fetchone()[0]).lower()
            conn.execute(f"PRAGMA busy_timeout = {int(timeout_seconds * 1000)}")
            conn.execute("PRAGMA synchronous=NORMAL")
            synchronous = str(conn.execute("PRAGMA synchronous").fetchone()[0])
            conn.commit()
    except sqlite3.OperationalError as exc:
        if _sqlite_busy_or_locked(exc):
            return SQLiteHealthResult(db_path=db_path, ok=False, status="busy", error=str(exc))
        if _sqlite_unavailable(exc):
            return SQLiteHealthResult(db_path=db_path, ok=False, status="unavailable", error=str(exc))
        return SQLiteHealthResult(db_path=db_path, ok=False, status="corrupt", error=str(exc))
    except sqlite3.DatabaseError as exc:
        return SQLiteHealthResult(db_path=db_path, ok=False, status="corrupt", error=str(exc))
    except OSError as exc:
        if _sqlite_unavailable(exc):
            return SQLiteHealthResult(db_path=db_path, ok=False, status="unavailable", error=str(exc))
        return SQLiteHealthResult(db_path=db_path, ok=False, status="unreadable", error=str(exc))

    if journal_mode != "wal":
        return SQLiteHealthResult(
            db_path=db_path,
            ok=False,
            status="pragma_failed",
            details=(f"journal_mode={journal_mode}", f"synchronous={synchronous}"),
        )
    logger.info("Bootstrapped empty Chroma SQLite database in WAL mode: %s", db_path)
    return SQLiteHealthResult(
        db_path=db_path,
        ok=True,
        status="ok",
        details=(f"journal_mode={journal_mode}", f"synchronous={synchronous}"),
    )


def configure_chroma_sqlite_pragmas(
    persist_dir: Path,
    *,
    timeout_seconds: float = DEFAULT_QUICK_CHECK_TIMEOUT_SECONDS,
) -> SQLiteHealthResult:
    """Set defensive SQLite pragmas for an existing Chroma database.

    Chroma owns its SQLite connections, but applying persistent
    ``journal_mode=WAL`` before client startup makes the durable mode explicit
    for the database.  ``synchronous=NORMAL`` is connection-local, so this also
    validates the setting for any maintenance connection opened here.
    """
    db_path = chroma_sqlite_path(persist_dir)
    if not db_path.exists():
        return SQLiteHealthResult(db_path=db_path, ok=True, status="missing")

    try:
        with closing(_connect(db_path, timeout_seconds)) as conn, conn:
            current_journal_mode = str(conn.execute("PRAGMA journal_mode").fetchone()[0]).lower()
            conn.execute(f"PRAGMA busy_timeout = {int(timeout_seconds * 1000)}")
            journal_mode = current_journal_mode
            if current_journal_mode != "wal":
                journal_mode = str(conn.execute("PRAGMA journal_mode=WAL").fetchone()[0]).lower()
            conn.execute("PRAGMA synchronous=NORMAL")
            synchronous = str(conn.execute("PRAGMA synchronous").fetchone()[0])
            conn.commit()
    except sqlite3.OperationalError as exc:
        if _sqlite_busy_or_locked(exc):
            return SQLiteHealthResult(db_path=db_path, ok=False, status="busy", error=str(exc))
        if _sqlite_unavailable(exc):
            return SQLiteHealthResult(db_path=db_path, ok=False, status="unavailable", error=str(exc))
        return SQLiteHealthResult(db_path=db_path, ok=False, status="corrupt", error=str(exc))
    except sqlite3.DatabaseError as exc:
        return SQLiteHealthResult(db_path=db_path, ok=False, status="corrupt", error=str(exc))
    except OSError as exc:
        if _sqlite_unavailable(exc):
            return SQLiteHealthResult(db_path=db_path, ok=False, status="unavailable", error=str(exc))
        return SQLiteHealthResult(db_path=db_path, ok=False, status="unreadable", error=str(exc))

    if journal_mode != "wal":
        return SQLiteHealthResult(
            db_path=db_path,
            ok=False,
            status="pragma_failed",
            details=(f"journal_mode={journal_mode}", f"synchronous={synchronous}"),
        )
    if current_journal_mode != "wal":
        logger.info("Enabled WAL journal mode for Chroma SQLite database: %s", db_path)
    return SQLiteHealthResult(
        db_path=db_path,
        ok=True,
        status="ok",
        details=(f"journal_mode={journal_mode}", f"synchronous={synchronous}"),
    )


def prepare_chroma_sqlite_for_startup(persist_dir: Path, *, anima_name: str | None) -> None:
    """Validate and configure a Chroma SQLite database before client startup."""
    health = quick_check_chroma_sqlite(persist_dir)
    if health.corrupt:
        collection = f"{anima_name}_knowledge" if anima_name else "<vectordb>"
        request_repair_for_sqlite_health(
            anima_name=anima_name,
            collection=collection,
            result=health,
            source="startup_quick_check",
        )
        rechecked = quick_check_chroma_sqlite(persist_dir)
        if rechecked.ok:
            logger.warning(
                "Chroma SQLite transient corruption signal cleared on re-check: anima=%s db=%s initial_detail=%s",
                anima_name,
                health.db_path,
                health.error or health.details,
            )
        elif rechecked.corrupt:
            raise RuntimeError(
                f"Chroma SQLite database corrupt before startup: {rechecked.db_path} "
                f"status={rechecked.status} detail={rechecked.error or rechecked.details}"
            )
    pragma = configure_chroma_sqlite_pragmas(persist_dir)
    if pragma.status == "missing":
        # First open: pre-create empty DB in WAL so Chroma does not inherit
        # the SQLite default ``delete`` journal mode.
        pragma = bootstrap_chroma_sqlite_wal(persist_dir)
    if not pragma.ok:
        logger.warning(
            "Failed to configure Chroma SQLite pragmas at %s: status=%s detail=%s",
            pragma.db_path,
            pragma.status,
            pragma.error or pragma.details,
        )


def request_repair_for_sqlite_health(
    *,
    anima_name: str | None,
    collection: str,
    result: SQLiteHealthResult,
    source: str,
) -> bool:
    """Record a health-check corruption result with the supervised repair service."""
    if not result.corrupt:
        return False
    reason_text = result.error or "; ".join(result.details) or result.status
    classified_text = f"Chroma SQLite database corrupt: {reason_text}"
    logger.warning(
        "Chroma SQLite health check failed: anima=%s collection=%s status=%s db=%s detail=%s",
        anima_name,
        collection,
        result.status,
        result.db_path,
        reason_text,
    )
    try:
        from core.memory.rag.repair import record_chroma_error

        return record_chroma_error(
            anima_name=anima_name,
            collection=collection,
            error=classified_text,
            source=source,
        )
    except Exception:
        logger.debug("Failed to record Chroma SQLite health repair signal", exc_info=True)
        return False


def check_anima_vectordb_health(
    anima_name: str,
    *,
    timeout_seconds: float = DEFAULT_QUICK_CHECK_TIMEOUT_SECONDS,
    source: str = "quick_check",
    record_repair: bool = True,
) -> SQLiteHealthResult:
    """Run quick_check for one anima DB and request repair on corruption."""
    from core.paths import get_anima_vectordb_dir

    persist_dir = get_anima_vectordb_dir(anima_name)
    result = quick_check_chroma_sqlite(persist_dir, timeout_seconds=timeout_seconds)
    if result.corrupt and record_repair:
        request_repair_for_sqlite_health(
            anima_name=anima_name,
            collection=f"{anima_name}_knowledge",
            result=result,
            source=source,
        )
    return result


def check_anima_vectordb_health_via_worker_or_direct(
    anima_name: str,
    *,
    timeout_seconds: float = DEFAULT_QUICK_CHECK_TIMEOUT_SECONDS,
    source: str = "quick_check",
    record_repair: bool = True,
) -> SQLiteHealthResult:
    """Run quick_check through the vector worker when one is configured."""
    vector_url = os.environ.get("ANIMAWORKS_VECTOR_URL")
    if vector_url:
        try:
            import httpx

            with httpx.Client(base_url=vector_url.rstrip("/"), timeout=timeout_seconds + 2.0) as client:
                resp = client.post(
                    "/quick-check",
                    json={
                        "anima_name": anima_name,
                        "timeout_seconds": timeout_seconds,
                        "source": source,
                        "record_repair": record_repair,
                    },
                )
                resp.raise_for_status()
                data = resp.json()
            return SQLiteHealthResult(
                db_path=Path(str(data.get("db_path") or "")),
                ok=bool(data.get("ok")),
                status=str(data.get("status") or "unknown"),
                details=tuple(str(item) for item in (data.get("details") or [])),
                error=str(data["error"]) if data.get("error") is not None else None,
            )
        except Exception:
            logger.warning(
                "Vector worker quick_check failed for anima=%s; falling back to direct SQLite check",
                anima_name,
                exc_info=True,
            )
    return check_anima_vectordb_health(
        anima_name,
        timeout_seconds=timeout_seconds,
        source=source,
        record_repair=record_repair,
    )


def _run_quick_check(db_path: Path, timeout_seconds: float) -> tuple[str, ...]:
    # Read-only: an rw maintenance connection deletes the -wal/-shm files on
    # close when it believes it is the last connection (chroma's Rust-side
    # SQLite locks are invisible to it), leaving live chroma connections on
    # stale deleted-WAL fds → "file is not a database" storms (2026-08-11).
    with closing(_connect(db_path, timeout_seconds, mode="ro")) as conn, conn:
        conn.execute(f"PRAGMA busy_timeout = {int(timeout_seconds * 1000)}")
        deadline = time.monotonic() + timeout_seconds
        timed_out = False

        def progress_handler() -> int:
            nonlocal timed_out
            if time.monotonic() > deadline:
                timed_out = True
                return 1
            return 0

        conn.set_progress_handler(progress_handler, 1000)
        try:
            rows = conn.execute("PRAGMA quick_check").fetchall()
        except sqlite3.OperationalError as exc:
            if timed_out:
                raise TimeoutError(f"quick_check exceeded {timeout_seconds:.1f}s for {db_path}") from exc
            raise
        finally:
            conn.set_progress_handler(None, 0)
    return tuple(str(row[0]) for row in rows)


def _connect(db_path: Path, timeout_seconds: float, *, mode: str = "rw") -> sqlite3.Connection:
    uri = f"file:{db_path}?mode={mode}"
    return sqlite3.connect(uri, uri=True, timeout=timeout_seconds)


def _sqlite_busy_or_locked(exc: sqlite3.OperationalError) -> bool:
    lower = str(exc).lower()
    return "locked" in lower or "busy" in lower


def _sqlite_unavailable(exc: Exception) -> bool:
    lower = str(exc).lower()
    return "unable to open database file" in lower or "too many open files" in lower
