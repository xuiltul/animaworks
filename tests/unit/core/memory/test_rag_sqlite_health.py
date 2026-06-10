from __future__ import annotations

import sqlite3
from pathlib import Path
from unittest.mock import patch

from core.memory.rag.sqlite_health import (
    SQLiteHealthResult,
    check_anima_vectordb_health_via_worker_or_direct,
    chroma_sqlite_path,
    configure_chroma_sqlite_pragmas,
    prepare_chroma_sqlite_for_startup,
    quick_check_chroma_sqlite,
    request_repair_for_sqlite_health,
)


def test_quick_check_missing_chroma_db_is_healthy(tmp_path: Path) -> None:
    result = quick_check_chroma_sqlite(tmp_path)

    assert result.ok is True
    assert result.status == "missing"
    assert result.db_path == tmp_path / "chroma.sqlite3"


def test_quick_check_valid_chroma_sqlite_db(tmp_path: Path) -> None:
    db_path = chroma_sqlite_path(tmp_path)
    with sqlite3.connect(db_path) as conn:
        conn.execute("CREATE TABLE t(id INTEGER PRIMARY KEY, value TEXT)")
        conn.execute("INSERT INTO t(value) VALUES ('ok')")

    result = quick_check_chroma_sqlite(tmp_path)

    assert result.ok is True
    assert result.status == "ok"
    assert result.details == ("ok",)


def test_configure_chroma_sqlite_pragmas_sets_wal(tmp_path: Path) -> None:
    db_path = chroma_sqlite_path(tmp_path)
    with sqlite3.connect(db_path) as conn:
        conn.execute("CREATE TABLE t(id INTEGER PRIMARY KEY)")

    result = configure_chroma_sqlite_pragmas(tmp_path)

    assert result.ok is True
    with sqlite3.connect(db_path) as conn:
        assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"


def test_configure_chroma_sqlite_pragmas_missing_db_is_healthy(tmp_path: Path) -> None:
    result = configure_chroma_sqlite_pragmas(tmp_path)

    assert result.ok is True
    assert result.status == "missing"


def test_quick_check_reports_corrupt_result_from_runner(tmp_path: Path) -> None:
    db_path = chroma_sqlite_path(tmp_path)
    db_path.write_bytes(b"not sqlite")

    result = quick_check_chroma_sqlite(
        tmp_path,
        runner=lambda _path, _timeout: ("*** in database main ***", "page 3 missing"),
    )

    assert result.corrupt is True
    assert result.status == "corrupt"
    assert "page 3 missing" in result.details


def test_quick_check_reports_database_error_from_runner(tmp_path: Path) -> None:
    db_path = chroma_sqlite_path(tmp_path)
    db_path.write_bytes(b"not sqlite")

    def failing_runner(_path: Path, _timeout: float) -> tuple[str, ...]:
        raise sqlite3.DatabaseError("database disk image is malformed")

    result = quick_check_chroma_sqlite(tmp_path, runner=failing_runner)

    assert result.corrupt is True
    assert result.status == "corrupt"
    assert result.error == "database disk image is malformed"


def test_quick_check_reports_timeout_from_runner(tmp_path: Path) -> None:
    db_path = chroma_sqlite_path(tmp_path)
    db_path.write_bytes(b"not sqlite")

    def timeout_runner(_path: Path, _timeout: float) -> tuple[str, ...]:
        raise TimeoutError("quick_check exceeded 10s")

    result = quick_check_chroma_sqlite(tmp_path, runner=timeout_runner)

    assert result.corrupt is True
    assert result.status == "timeout"
    assert result.error == "quick_check exceeded 10s"


def test_quick_check_locked_db_is_not_corruption(tmp_path: Path) -> None:
    db_path = chroma_sqlite_path(tmp_path)
    db_path.write_bytes(b"not sqlite")

    def locked_runner(_path: Path, _timeout: float) -> tuple[str, ...]:
        raise sqlite3.OperationalError("database is locked")

    result = quick_check_chroma_sqlite(tmp_path, runner=locked_runner)

    assert result.corrupt is False
    assert result.status == "busy"


def test_prepare_startup_records_repair_and_raises_on_corruption(tmp_path: Path) -> None:
    health = SQLiteHealthResult(
        db_path=tmp_path / "chroma.sqlite3",
        ok=False,
        status="corrupt",
        error="database disk image is malformed",
    )

    with (
        patch("core.memory.rag.sqlite_health.quick_check_chroma_sqlite", return_value=health),
        patch("core.memory.rag.sqlite_health.request_repair_for_sqlite_health", return_value=True) as repair,
    ):
        try:
            prepare_chroma_sqlite_for_startup(tmp_path, anima_name="sora")
        except RuntimeError as exc:
            assert "corrupt before startup" in str(exc)
        else:  # pragma: no cover - defensive assertion
            raise AssertionError("expected startup preparation to fail")

    repair.assert_called_once_with(
        anima_name="sora",
        collection="sora_knowledge",
        result=health,
        source="startup_quick_check",
    )


def test_request_repair_for_sqlite_health_ignores_ok_result(tmp_path: Path) -> None:
    result = SQLiteHealthResult(db_path=tmp_path / "chroma.sqlite3", ok=True, status="ok")

    with patch("core.memory.rag.repair.record_chroma_error") as record:
        assert (
            request_repair_for_sqlite_health(
                anima_name="sora",
                collection="sora_knowledge",
                result=result,
                source="startup_quick_check",
            )
            is False
        )

    record.assert_not_called()


def test_request_repair_for_sqlite_health_records_corruption(tmp_path: Path) -> None:
    result = SQLiteHealthResult(
        db_path=tmp_path / "chroma.sqlite3",
        ok=False,
        status="corrupt",
        error="database disk image is malformed",
    )

    with patch("core.memory.rag.repair.record_chroma_error", return_value=True) as record:
        assert (
            request_repair_for_sqlite_health(
                anima_name="sora",
                collection="sora_knowledge",
                result=result,
                source="startup_quick_check",
            )
            is True
        )

    record.assert_called_once_with(
        anima_name="sora",
        collection="sora_knowledge",
        error="Chroma SQLite database corrupt: database disk image is malformed",
        source="startup_quick_check",
    )


def test_request_repair_for_sqlite_health_swallows_record_failure(tmp_path: Path) -> None:
    result = SQLiteHealthResult(
        db_path=tmp_path / "chroma.sqlite3",
        ok=False,
        status="corrupt",
        details=("page 3 missing",),
    )

    with patch("core.memory.rag.repair.record_chroma_error", side_effect=RuntimeError("boom")):
        assert (
            request_repair_for_sqlite_health(
                anima_name="sora",
                collection="sora_knowledge",
                result=result,
                source="startup_quick_check",
            )
            is False
        )


def test_check_anima_vectordb_health_uses_vector_worker(monkeypatch) -> None:
    requests: list[dict] = []

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {
                "db_path": "/tmp/animas/sora/vectordb/chroma.sqlite3",
                "ok": False,
                "status": "timeout",
                "details": ["quick_check interrupted"],
                "error": None,
            }

    class FakeClient:
        def __init__(self, *, base_url: str, timeout: float) -> None:
            assert base_url == "http://worker"
            assert timeout == 6.0

        def __enter__(self) -> FakeClient:
            return self

        def __exit__(self, *_args) -> None:
            return None

        def post(self, path: str, json: dict) -> FakeResponse:
            assert path == "/quick-check"
            requests.append(json)
            return FakeResponse()

    monkeypatch.setenv("ANIMAWORKS_VECTOR_URL", "http://worker")
    monkeypatch.setattr("httpx.Client", FakeClient)

    result = check_anima_vectordb_health_via_worker_or_direct(
        "sora",
        timeout_seconds=4.0,
        source="daily_indexing_quick_check",
        record_repair=False,
    )

    assert result.corrupt is True
    assert result.status == "timeout"
    assert requests == [
        {
            "anima_name": "sora",
            "timeout_seconds": 4.0,
            "source": "daily_indexing_quick_check",
            "record_repair": False,
        }
    ]


def test_check_anima_vectordb_health_falls_back_when_worker_fails(monkeypatch, tmp_path: Path) -> None:
    expected = SQLiteHealthResult(db_path=tmp_path / "chroma.sqlite3", ok=True, status="ok")

    class FailingClient:
        def __init__(self, **_kwargs) -> None:
            raise RuntimeError("worker unavailable")

    def fake_direct_check(anima_name: str, **kwargs) -> SQLiteHealthResult:
        assert anima_name == "sora"
        assert kwargs == {
            "timeout_seconds": 4.0,
            "source": "daily_indexing_quick_check",
            "record_repair": False,
        }
        return expected

    monkeypatch.setenv("ANIMAWORKS_VECTOR_URL", "http://worker")
    monkeypatch.setattr("httpx.Client", FailingClient)
    monkeypatch.setattr("core.memory.rag.sqlite_health.check_anima_vectordb_health", fake_direct_check)

    result = check_anima_vectordb_health_via_worker_or_direct(
        "sora",
        timeout_seconds=4.0,
        source="daily_indexing_quick_check",
        record_repair=False,
    )

    assert result is expected
