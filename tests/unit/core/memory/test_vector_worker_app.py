from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from core.memory.rag.sqlite_health import SQLiteHealthResult


def _doc(doc_id: str = "doc1", content: str = "hello", metadata: dict | None = None) -> SimpleNamespace:
    return SimpleNamespace(id=doc_id, content=content, metadata=metadata or {"kind": "test"})


def test_raise_fd_soft_limit_raises_to_hard(monkeypatch) -> None:
    from core.platform.fd_limits import raise_fd_soft_limit

    calls: list[tuple[int, tuple[int, int]]] = []
    fake_resource = SimpleNamespace(
        RLIMIT_NOFILE=7,
        RLIM_INFINITY=-1,
        getrlimit=lambda _limit: (1024, 4096),
        setrlimit=lambda limit, value: calls.append((limit, value)),
    )
    monkeypatch.setitem(sys.modules, "resource", fake_resource)

    assert raise_fd_soft_limit() == (4096, 4096)
    assert calls == [(7, (4096, 4096))]


def test_raise_fd_soft_limit_skips_when_already_at_target(monkeypatch) -> None:
    from core.platform.fd_limits import raise_fd_soft_limit

    calls: list[tuple[int, tuple[int, int]]] = []
    fake_resource = SimpleNamespace(
        RLIMIT_NOFILE=7,
        RLIM_INFINITY=-1,
        getrlimit=lambda _limit: (4096, 4096),
        setrlimit=lambda limit, value: calls.append((limit, value)),
    )
    monkeypatch.setitem(sys.modules, "resource", fake_resource)

    assert raise_fd_soft_limit() == (4096, 4096)
    assert calls == []


def test_raise_fd_soft_limit_returns_original_when_setrlimit_fails(monkeypatch) -> None:
    from core.platform.fd_limits import raise_fd_soft_limit

    def fail_setrlimit(_limit: int, _value: tuple[int, int]) -> None:
        raise OSError("permission denied")

    fake_resource = SimpleNamespace(
        RLIMIT_NOFILE=7,
        RLIM_INFINITY=-1,
        getrlimit=lambda _limit: (1024, 4096),
        setrlimit=fail_setrlimit,
    )
    monkeypatch.setitem(sys.modules, "resource", fake_resource)

    assert raise_fd_soft_limit() == (1024, 4096)


def test_raise_fd_soft_limit_uses_finite_target_for_unlimited_hard(monkeypatch) -> None:
    from core.platform.fd_limits import _NOFILE_INFINITY_FALLBACK, raise_fd_soft_limit

    calls: list[tuple[int, tuple[int, int]]] = []
    fake_resource = SimpleNamespace(
        RLIMIT_NOFILE=7,
        RLIM_INFINITY=-1,
        getrlimit=lambda _limit: (1024, -1),
        setrlimit=lambda limit, value: calls.append((limit, value)),
    )
    monkeypatch.setitem(sys.modules, "resource", fake_resource)

    assert raise_fd_soft_limit() == (_NOFILE_INFINITY_FALLBACK, -1)
    assert calls == [(7, (_NOFILE_INFINITY_FALLBACK, -1))]


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


def test_vector_worker_upsert_failure_opens_backoff(monkeypatch) -> None:
    monkeypatch.delenv("ANIMAWORKS_VECTOR_URL", raising=False)

    from core.memory.rag import vector_worker
    from core.memory.rag.vector_worker import create_app

    monkeypatch.setattr(vector_worker, "_CIRCUIT_FAILURE_THRESHOLD", 2)
    monkeypatch.setattr(vector_worker, "_CIRCUIT_BACKOFF_BASE_SECONDS", 30.0)
    monkeypatch.setattr(vector_worker, "_CIRCUIT_BACKOFF_MAX_SECONDS", 30.0)
    store = MagicMock()
    store.upsert.return_value = False

    with (
        patch("core.memory.rag.singleton.get_vector_store", return_value=store),
        TestClient(create_app()) as client,
    ):
        payload = {
            "anima_name": None,
            "collection": "shared_common_knowledge",
            "documents": [{"id": "doc1", "content": "hello", "embedding": [0.1], "metadata": {}}],
        }
        first_failed = client.post("/upsert", json=payload)
        second_failed = client.post("/upsert", json=payload)
        blocked = client.post("/upsert", json=payload)
        status = client.get("/status").json()

    assert first_failed.status_code == 500
    assert first_failed.json()["consecutive_failures"] == 1
    assert second_failed.status_code == 500
    assert second_failed.json()["consecutive_failures"] == 2
    assert blocked.status_code == 429
    assert blocked.json()["detail"] == "Vector write circuit breaker open"
    assert blocked.json()["retry_after_seconds"] > 0
    breakers = status["write_circuit_breakers"]
    assert breakers[0]["owner"] == "shared"
    assert breakers[0]["collection"] == "shared_common_knowledge"
    assert breakers[0]["open"] is True
    assert breakers[0]["consecutive_failures"] == 2
    assert breakers[0]["threshold"] == 2
    assert store.upsert.call_count == 2


def test_vector_worker_status_includes_gpu_section(monkeypatch) -> None:
    monkeypatch.delenv("ANIMAWORKS_VECTOR_URL", raising=False)

    from core.memory.rag.vector_worker import create_app

    gpu = {
        "embedding_device": "cpu",
        "degraded": False,
        "last_error": None,
        "detected_at": None,
    }
    with (
        patch("core.gpu.get_gpu_status", return_value=gpu),
        TestClient(create_app()) as client,
    ):
        status = client.get("/status").json()

    assert status["gpu"] == gpu


def test_vector_worker_reset_store_clears_cache_and_owner_breakers(monkeypatch) -> None:
    monkeypatch.delenv("ANIMAWORKS_VECTOR_URL", raising=False)

    from core.memory.rag import vector_worker
    from core.memory.rag.vector_worker import create_app

    reset = MagicMock()
    with (
        patch("core.memory.rag.singleton.reset_vector_store", reset),
        TestClient(create_app()) as client,
    ):
        vector_worker._write_circuit_breakers["sora:knowledge"] = {
            "owner": "sora",
            "collection": "knowledge",
            "operation": "upsert",
            "consecutive_failures": 3,
        }
        vector_worker._write_circuit_breakers["other:knowledge"] = {
            "owner": "other",
            "collection": "knowledge",
            "operation": "upsert",
            "consecutive_failures": 3,
        }

        resp = client.post("/reset-store", json={"anima_name": "sora"})

    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}
    reset.assert_called_once_with("sora")
    assert "sora:knowledge" not in vector_worker._write_circuit_breakers
    assert "other:knowledge" in vector_worker._write_circuit_breakers


def test_vector_worker_read_endpoints(monkeypatch) -> None:
    monkeypatch.delenv("ANIMAWORKS_VECTOR_URL", raising=False)

    from core.memory.rag.vector_worker import create_app

    doc = _doc()
    result = SimpleNamespace(document=doc, score=0.75)
    store = MagicMock()
    store.query.return_value = [result]
    store.get_by_metadata.return_value = [result]
    store.get_by_ids.return_value = [doc]
    store.list_collections.return_value = ["knowledge"]

    with (
        patch("core.memory.rag.singleton.get_vector_store", return_value=store),
        TestClient(create_app()) as client,
    ):
        query = client.post(
            "/query",
            json={"anima_name": "sora", "collection": "knowledge", "embedding": [0.1], "top_k": 1},
        )
        by_meta = client.post(
            "/get-by-metadata",
            json={"anima_name": "sora", "collection": "knowledge", "where": {"kind": "test"}, "limit": 1},
        )
        by_ids = client.post(
            "/get-by-ids",
            json={"anima_name": "sora", "collection": "knowledge", "ids": ["doc1"]},
        )
        collections = client.post("/list-collections", json={"anima_name": "sora"})

    assert query.json()["results"][0]["id"] == "doc1"
    assert by_meta.json()["results"][0]["score"] == 0.75
    assert by_ids.json()["documents"][0]["metadata"] == {"kind": "test"}
    assert collections.json()["collections"] == ["knowledge"]


def test_vector_worker_write_endpoints_success(monkeypatch) -> None:
    monkeypatch.delenv("ANIMAWORKS_VECTOR_URL", raising=False)

    from core.memory.rag.vector_worker import create_app

    store = MagicMock()
    store.update_metadata.return_value = True
    store.delete_documents.return_value = True
    store.create_collection.return_value = True
    store.delete_collection.return_value = True

    with (
        patch("core.memory.rag.singleton.get_vector_store", return_value=store),
        TestClient(create_app()) as client,
    ):
        update = client.post(
            "/update-metadata",
            json={
                "anima_name": "sora",
                "collection": "knowledge",
                "ids": ["doc1"],
                "metadatas": [{"kind": "updated"}],
            },
        )
        delete_docs = client.post(
            "/delete-documents",
            json={"anima_name": "sora", "collection": "knowledge", "ids": ["doc1"]},
        )
        create = client.post(
            "/create-collection",
            json={"anima_name": "sora", "collection": "knowledge"},
        )
        delete_collection = client.post(
            "/delete-collection",
            json={"anima_name": "sora", "collection": "knowledge"},
        )

    assert update.json() == {"status": "ok"}
    assert delete_docs.json() == {"status": "ok"}
    assert create.json() == {"status": "ok"}
    assert delete_collection.json() == {"status": "ok"}
    store.update_metadata.assert_called_once_with("knowledge", ["doc1"], [{"kind": "updated"}])
    store.delete_documents.assert_called_once_with("knowledge", ["doc1"])
    store.create_collection.assert_called_once_with("knowledge")
    store.delete_collection.assert_called_once_with("knowledge")


def test_vector_worker_write_store_unavailable(monkeypatch) -> None:
    monkeypatch.delenv("ANIMAWORKS_VECTOR_URL", raising=False)

    from core.memory.rag.vector_worker import create_app

    with (
        patch("core.memory.rag.singleton.get_vector_store", return_value=None),
        TestClient(create_app()) as client,
    ):
        resp = client.post(
            "/delete-documents",
            json={"anima_name": "sora", "collection": "knowledge", "ids": ["doc1"]},
        )

    assert resp.status_code == 503
    assert resp.json() == {"detail": "Vector store unavailable"}
