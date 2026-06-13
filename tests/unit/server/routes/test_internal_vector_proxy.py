from __future__ import annotations

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from core.memory.rag.vector_worker_client import VectorWorkerResponse
from server.routes.internal import create_internal_router


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
