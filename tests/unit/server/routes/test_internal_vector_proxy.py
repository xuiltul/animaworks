from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from core.memory.rag.vector_worker_client import VectorWorkerResponse
from server.routes.internal import create_internal_router


@pytest.fixture(autouse=True)
def english_diagnostics(monkeypatch):
    monkeypatch.setattr("core.paths._get_locale", lambda: "en")


class _FailingVectorWorker:
    enabled = True

    async def post(self, path: str, payload: dict):
        return VectorWorkerResponse(
            status_code=503,
            data={"detail": "worker failed"},
            headers={
                "Content-Length": "9999",
                "Content-Encoding": "gzip",
                "Server": "upstream",
                "Retry-After": "7",
            },
        )


@pytest.mark.asyncio
async def test_vector_proxy_error_allows_only_retry_after_header() -> None:
    app = FastAPI()
    app.state.vector_worker = _FailingVectorWorker()
    app.include_router(create_internal_router(), prefix="/api")

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/internal/vector/query",
            json={
                "anima_name": "sakura",
                "collection": "knowledge",
                "embedding": [0.1, 0.2],
                "top_k": 1,
            },
        )

    assert response.status_code == 503
    assert response.json() == {"detail": "worker failed"}
    assert response.headers["retry-after"] == "7"
    assert response.headers["content-length"] != "9999"
    assert "content-encoding" not in response.headers
    assert "server" not in response.headers


class _RecordingVectorWorker:
    enabled = True

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    async def post(self, path: str, payload: dict):
        self.calls.append((path, payload))
        return VectorWorkerResponse(status_code=200, data={"status": "ok"}, headers={})


@pytest.mark.asyncio
async def test_reset_store_proxy_forwards_to_worker() -> None:
    # Regression: the proxy previously lacked a /reset-store route, so repair's
    # worker-cache reset returned 405 and never reached the worker, leaving
    # stale/corrupt handles latched. The route must forward to the worker.
    worker = _RecordingVectorWorker()
    app = FastAPI()
    app.state.vector_worker = worker
    app.include_router(create_internal_router(), prefix="/api")

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/internal/vector/reset-store",
            json={"anima_name": "mei"},
        )

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    assert worker.calls == [("/reset-store", {"anima_name": "mei"})]


@pytest.mark.asyncio
async def test_vector_proxy_phase3_without_root_fails_closed(tmp_path: Path) -> None:
    anima_dir = tmp_path / "animas" / "sakura"
    anima_dir.mkdir(parents=True)
    (anima_dir / "status.json").write_text('{"process_model":"phase3"}', encoding="utf-8")
    worker = _RecordingVectorWorker()
    app = FastAPI()
    app.state.vector_worker = worker
    app.include_router(create_internal_router(), prefix="/api")

    with patch("core.paths.get_animas_dir", return_value=tmp_path / "animas"):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/api/internal/vector/query",
                json={
                    "anima_name": "sakura",
                    "collection": "knowledge",
                    "embedding": [0.1],
                },
            )

    assert response.status_code == 503
    assert response.json() == {"detail": "Root memory service unavailable"}
    assert worker.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "path,method,payload",
    [
        ("query", "query", {"collection": "sakura_knowledge", "embedding": [0.1], "top_k": 1, "filter_metadata": None}),
        ("get-by-ids", "get_by_ids", {"collection": "sakura_knowledge", "ids": ["doc-1"]}),
        ("get-by-metadata", "get_by_metadata", {"collection": "sakura_knowledge", "where": {}, "limit": 2}),
        ("list-collections", "list_collections_checked", {}),
        ("create-collection", "create_collection", {"collection": "sakura_knowledge"}),
        ("delete-collection", "delete_collection", {"collection": "sakura_knowledge"}),
        ("upsert", "upsert", {"collection": "sakura_knowledge", "documents": []}),
        ("delete-documents", "delete_documents", {"collection": "sakura_knowledge", "ids": []}),
        ("update-metadata", "update_metadata", {"collection": "sakura_knowledge", "ids": [], "metadatas": []}),
    ],
)
async def test_phase3_http_proxy_routes_only_to_named_native_owner(path, method, payload):
    worker = _RecordingVectorWorker()
    supervisor = SimpleNamespace(send_request=AsyncMock(return_value={"ok": True}))
    app = FastAPI()
    app.state.vector_worker = worker
    app.state.supervisor = supervisor
    app.include_router(create_internal_router(), prefix="/api")
    with patch(
        "core.config.resolver.resolve_process_model_config",
        return_value=SimpleNamespace(valid=True, process_model="phase3"),
    ):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post("/api/internal/vector/" + path, json={"anima_name": "sakura", **payload})
    assert response.status_code == 200
    supervisor.send_request.assert_awaited_once_with(
        "sakura",
        "memory",
        {"method": "memory." + method, "params": payload},
        timeout=120.0,
    )
    assert worker.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("result", [{"ok": False}, None, RuntimeError("root unavailable")])
async def test_phase3_proxy_does_not_report_failed_write_as_success(result):
    sender = AsyncMock(side_effect=result) if isinstance(result, Exception) else AsyncMock(return_value=result)
    app = FastAPI()
    app.state.supervisor = SimpleNamespace(send_request=sender)
    app.state.vector_worker = _RecordingVectorWorker()
    app.include_router(create_internal_router(), prefix="/api")
    with patch(
        "core.config.resolver.resolve_process_model_config",
        return_value=SimpleNamespace(valid=True, process_model="phase3"),
    ):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post(
                "/api/internal/vector/upsert",
                json={"anima_name": "sakura", "collection": "sakura_knowledge", "documents": []},
            )
    assert response.status_code == 503
    assert response.headers["retry-after"] == "1"
    assert app.state.vector_worker.calls == []


@pytest.mark.asyncio
async def test_phase3_reset_cannot_open_second_native_owner():
    app = FastAPI()
    app.state.vector_worker = _RecordingVectorWorker()
    app.state.supervisor = SimpleNamespace(send_request=AsyncMock())
    app.include_router(create_internal_router(), prefix="/api")
    with patch(
        "core.config.resolver.resolve_process_model_config",
        return_value=SimpleNamespace(valid=True, process_model="phase3"),
    ):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post("/api/internal/vector/reset-store", json={"anima_name": "sakura"})
    assert response.status_code == 409
    app.state.supervisor.send_request.assert_not_awaited()
    assert app.state.vector_worker.calls == []


@pytest.mark.asyncio
async def test_vector_proxy_rejects_anima_path_traversal_before_resolving():
    app = FastAPI()
    app.include_router(create_internal_router(), prefix="/api")
    with patch("core.config.resolver.resolve_process_model_config") as resolver:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post("/api/internal/vector/list-collections", json={"anima_name": "../other"})
    assert response.status_code == 422
    resolver.assert_not_called()
