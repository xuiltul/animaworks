from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from core.memory.rag.sqlite_health import SQLiteHealthResult


def test_vector_worker_shutdown_closes_cached_stores(monkeypatch) -> None:
    monkeypatch.delenv("ANIMAWORKS_VECTOR_URL", raising=False)

    from core.memory.rag.vector_worker import create_app

    with (
        patch("core.memory.rag.singleton.close_all_vector_stores") as close_all,
        TestClient(create_app()) as client,
    ):
        assert client.get("/health").json() == {"status": "ok"}

    close_all.assert_called_once()


def test_vector_worker_quick_check_endpoint(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("ANIMAWORKS_VECTOR_URL", raising=False)

    from core.memory.rag.vector_worker import create_app

    check = MagicMock(
        return_value=SQLiteHealthResult(
            db_path=tmp_path / "chroma.sqlite3",
            ok=True,
            status="ok",
            details=("ok",),
        )
    )
    with (
        patch("core.memory.rag.sqlite_health.check_anima_vectordb_health", check),
        TestClient(create_app()) as client,
    ):
        resp = client.post(
            "/quick-check",
            json={
                "anima_name": "sora",
                "timeout_seconds": 3,
                "source": "test_quick_check",
            },
        )

    assert resp.status_code == 200
    assert resp.json()["ok"] is True
    check.assert_called_once_with(
        "sora",
        timeout_seconds=3.0,
        source="test_quick_check",
        record_repair=True,
    )
