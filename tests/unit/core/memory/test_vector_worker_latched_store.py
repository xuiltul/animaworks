from __future__ import annotations

import json
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from core.memory.rag.sqlite_health import SQLiteHealthResult


def _doc(doc_id: str = "doc1", content: str = "hello", metadata: dict | None = None) -> SimpleNamespace:
    return SimpleNamespace(id=doc_id, content=content, metadata=metadata or {"kind": "test"})


def _health_result(
    tmp_path: Path, status: str, *, ok: bool | None = None, error: str | None = None
) -> SQLiteHealthResult:
    return SQLiteHealthResult(
        db_path=tmp_path / status / "chroma.sqlite3",
        ok=status in {"ok", "missing"} if ok is None else ok,
        status=status,
        details=("ok",) if status == "ok" else (),
        error=error,
    )


def _patch_latched_store_recovery(
    stack: ExitStack,
    *,
    get_store: MagicMock,
    health: SQLiteHealthResult,
    init_failed: bool = True,
    global_failed: bool = False,
) -> SimpleNamespace:
    health_check = MagicMock(return_value=health)
    is_init_failed = MagicMock(return_value=init_failed)
    is_global_failed = MagicMock(return_value=global_failed)
    clear_init_failed = MagicMock(return_value=True)
    reset_store = MagicMock()

    stack.enter_context(patch("core.memory.rag.singleton.get_vector_store", get_store))
    stack.enter_context(patch("core.memory.rag.vector_worker.get_vector_store", get_store, create=True))
    stack.enter_context(patch("core.memory.rag.sqlite_health.check_anima_vectordb_health", health_check))
    stack.enter_context(patch("core.memory.rag.vector_worker.check_anima_vectordb_health", health_check, create=True))
    stack.enter_context(patch("core.memory.rag.singleton.is_vector_store_init_failed", is_init_failed, create=True))
    stack.enter_context(patch("core.memory.rag.vector_worker.is_vector_store_init_failed", is_init_failed, create=True))
    stack.enter_context(
        patch("core.memory.rag.singleton.is_global_vector_store_init_failed", is_global_failed, create=True)
    )
    stack.enter_context(
        patch("core.memory.rag.vector_worker.is_global_vector_store_init_failed", is_global_failed, create=True)
    )
    stack.enter_context(
        patch("core.memory.rag.singleton.clear_vector_store_init_failed", clear_init_failed, create=True)
    )
    stack.enter_context(
        patch("core.memory.rag.vector_worker.clear_vector_store_init_failed", clear_init_failed, create=True)
    )
    stack.enter_context(patch("core.memory.rag.singleton.reset_vector_store", reset_store))
    stack.enter_context(patch("core.memory.rag.vector_worker.reset_vector_store", reset_store, create=True))

    return SimpleNamespace(
        clear_init_failed=clear_init_failed,
        health_check=health_check,
        is_global_failed=is_global_failed,
        is_init_failed=is_init_failed,
        reset_store=reset_store,
    )


@pytest.mark.parametrize("status", ["ok", "missing"])
def test_vector_worker_latched_store_write_recovers_after_ok_or_missing_quick_check(
    monkeypatch,
    tmp_path: Path,
    status: str,
) -> None:
    monkeypatch.delenv("ANIMAWORKS_VECTOR_URL", raising=False)

    from core.memory.rag import vector_worker
    from core.memory.rag.vector_worker import create_app

    store = MagicMock()
    store.upsert.return_value = True
    get_store = MagicMock(side_effect=[None, store])

    with ExitStack() as stack:
        mocks = _patch_latched_store_recovery(
            stack,
            get_store=get_store,
            health=_health_result(tmp_path, status),
        )
        with TestClient(create_app()) as client:
            vector_worker._write_circuit_breakers["sora:knowledge"] = {
                "owner": "sora",
                "collection": "knowledge",
                "operation": "upsert",
                "consecutive_failures": 3,
            }
            vector_worker._write_circuit_breakers["rin:knowledge"] = {
                "owner": "rin",
                "collection": "knowledge",
                "operation": "upsert",
                "consecutive_failures": 3,
            }
            resp = client.post(
                "/upsert",
                json={
                    "anima_name": "sora",
                    "collection": "knowledge",
                    "documents": [{"id": "doc1", "content": "hello", "embedding": [0.1], "metadata": {}}],
                },
            )

    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}
    assert get_store.call_count == 2
    mocks.health_check.assert_called_once()
    assert mocks.health_check.call_args.args[0] == "sora"
    assert mocks.health_check.call_args.kwargs.get("record_repair", True) is True
    mocks.clear_init_failed.assert_called_once_with("sora")
    mocks.reset_store.assert_not_called()
    store.upsert.assert_called_once()
    assert "sora:knowledge" not in vector_worker._write_circuit_breakers
    assert "rin:knowledge" in vector_worker._write_circuit_breakers


def test_vector_worker_latched_store_corrupt_health_records_without_retry_or_reset(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.delenv("ANIMAWORKS_VECTOR_URL", raising=False)

    from core.memory.rag.vector_worker import create_app

    get_store = MagicMock(return_value=None)

    with ExitStack() as stack:
        mocks = _patch_latched_store_recovery(
            stack,
            get_store=get_store,
            health=_health_result(tmp_path, "corrupt", ok=False, error="database disk image is malformed"),
        )
        with TestClient(create_app()) as client:
            resp = client.post(
                "/delete-documents",
                json={"anima_name": "sora", "collection": "knowledge", "ids": ["doc1"]},
            )

    assert resp.status_code == 503
    assert resp.json() == {"detail": "Vector store unavailable"}
    get_store.assert_called_once_with("sora")
    mocks.health_check.assert_called_once()
    assert mocks.health_check.call_args.args[0] == "sora"
    assert mocks.health_check.call_args.kwargs.get("record_repair", True) is True
    mocks.clear_init_failed.assert_not_called()
    mocks.reset_store.assert_not_called()


@pytest.mark.parametrize("status", ["busy", "timeout", "unavailable", "unreadable", "unknown"])
def test_vector_worker_latched_store_ambiguous_health_does_not_retry(
    monkeypatch,
    tmp_path: Path,
    status: str,
) -> None:
    monkeypatch.delenv("ANIMAWORKS_VECTOR_URL", raising=False)

    from core.memory.rag.vector_worker import create_app

    get_store = MagicMock(return_value=None)

    with ExitStack() as stack:
        mocks = _patch_latched_store_recovery(
            stack,
            get_store=get_store,
            health=_health_result(tmp_path, status, ok=False, error=f"{status} health"),
        )
        with TestClient(create_app()) as client:
            resp = client.post(
                "/update-metadata",
                json={
                    "anima_name": "sora",
                    "collection": "knowledge",
                    "ids": ["doc1"],
                    "metadatas": [{"kind": "updated"}],
                },
            )
            blocked_by_backoff = client.post(
                "/update-metadata",
                json={
                    "anima_name": "sora",
                    "collection": "knowledge",
                    "ids": ["doc1"],
                    "metadatas": [{"kind": "updated"}],
                },
            )

    assert resp.status_code == 503
    assert resp.json() == {"detail": "Vector store unavailable"}
    assert blocked_by_backoff.status_code == 503
    assert blocked_by_backoff.json() == {"detail": "Vector store unavailable"}
    assert get_store.call_count == 2
    mocks.health_check.assert_called_once()
    mocks.clear_init_failed.assert_not_called()
    mocks.reset_store.assert_not_called()


def test_vector_worker_latched_store_retry_unavailable_uses_backoff(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.delenv("ANIMAWORKS_VECTOR_URL", raising=False)

    from core.memory.rag.vector_worker import create_app

    get_store = MagicMock(return_value=None)
    payload = {
        "anima_name": "sora",
        "collection": "knowledge",
        "documents": [{"id": "doc1", "content": "hello", "embedding": [0.1], "metadata": {}}],
    }

    with ExitStack() as stack:
        mocks = _patch_latched_store_recovery(
            stack,
            get_store=get_store,
            health=_health_result(tmp_path, "ok"),
        )
        with TestClient(create_app()) as client:
            first = client.post("/upsert", json=payload)
            second = client.post("/upsert", json=payload)

    assert first.status_code == 503
    assert first.json() == {"detail": "Vector store unavailable"}
    assert second.status_code == 503
    assert second.json() == {"detail": "Vector store unavailable"}
    mocks.health_check.assert_called_once()
    mocks.clear_init_failed.assert_called_once_with("sora")
    mocks.reset_store.assert_not_called()


@pytest.mark.parametrize("repair_status", ["requested", "stopping", "repairing"])
def test_vector_worker_active_repair_state_skips_latched_store_recovery(
    monkeypatch,
    tmp_path: Path,
    data_dir: Path,
    repair_status: str,
) -> None:
    monkeypatch.delenv("ANIMAWORKS_VECTOR_URL", raising=False)
    state_dir = data_dir / "animas" / "sora" / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / "rag_repair.json").write_text(json.dumps({"status": repair_status}), encoding="utf-8")

    from core.memory.rag.vector_worker import create_app

    get_store = MagicMock(return_value=None)

    with ExitStack() as stack:
        mocks = _patch_latched_store_recovery(
            stack,
            get_store=get_store,
            health=_health_result(tmp_path, "ok"),
        )
        with TestClient(create_app()) as client:
            resp = client.post(
                "/delete-collection",
                json={"anima_name": "sora", "collection": "knowledge"},
            )

    assert resp.status_code == 503
    assert resp.json() == {"detail": "Vector store unavailable"}
    get_store.assert_called_once_with("sora")
    mocks.health_check.assert_not_called()
    mocks.clear_init_failed.assert_not_called()
    mocks.reset_store.assert_not_called()


def test_vector_worker_global_init_failure_skips_latched_store_recovery(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("ANIMAWORKS_VECTOR_URL", raising=False)

    from core.memory.rag.vector_worker import create_app

    get_store = MagicMock(return_value=None)

    with ExitStack() as stack:
        mocks = _patch_latched_store_recovery(
            stack,
            get_store=get_store,
            health=_health_result(tmp_path, "ok"),
            global_failed=True,
        )
        with TestClient(create_app()) as client:
            resp = client.post(
                "/create-collection",
                json={"anima_name": "sora", "collection": "knowledge"},
            )

    assert resp.status_code == 503
    assert resp.json() == {"detail": "Vector store unavailable"}
    get_store.assert_called_once_with("sora")
    mocks.health_check.assert_not_called()
    mocks.clear_init_failed.assert_not_called()
    mocks.reset_store.assert_not_called()


def test_vector_worker_shared_store_skips_latched_store_recovery(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("ANIMAWORKS_VECTOR_URL", raising=False)

    from core.memory.rag.vector_worker import create_app

    get_store = MagicMock(return_value=None)

    with ExitStack() as stack:
        mocks = _patch_latched_store_recovery(
            stack,
            get_store=get_store,
            health=_health_result(tmp_path, "ok"),
        )
        with TestClient(create_app()) as client:
            resp = client.post(
                "/upsert",
                json={
                    "anima_name": None,
                    "collection": "shared_common_knowledge",
                    "documents": [{"id": "doc1", "content": "hello", "embedding": [0.1], "metadata": {}}],
                },
            )

    assert resp.status_code == 503
    assert resp.json() == {"detail": "Vector store unavailable"}
    get_store.assert_called_once_with(None)
    mocks.health_check.assert_not_called()
    mocks.clear_init_failed.assert_not_called()
    mocks.reset_store.assert_not_called()


def test_vector_worker_latched_store_read_recovers_after_ok_quick_check(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.delenv("ANIMAWORKS_VECTOR_URL", raising=False)

    from core.memory.rag.vector_worker import create_app

    doc = _doc()
    result = SimpleNamespace(document=doc, score=0.75)
    store = MagicMock()
    store.query.return_value = [result]
    get_store = MagicMock(side_effect=[None, store])

    with ExitStack() as stack:
        mocks = _patch_latched_store_recovery(
            stack,
            get_store=get_store,
            health=_health_result(tmp_path, "ok"),
        )
        with TestClient(create_app()) as client:
            resp = client.post(
                "/query",
                json={"anima_name": "sora", "collection": "knowledge", "embedding": [0.1], "top_k": 1},
            )

    assert resp.status_code == 200
    assert resp.json()["results"][0]["id"] == "doc1"
    assert get_store.call_count == 2
    mocks.health_check.assert_called_once()
    mocks.clear_init_failed.assert_called_once_with("sora")
    mocks.reset_store.assert_not_called()
    store.query.assert_called_once_with("knowledge", [0.1], 1, None)
