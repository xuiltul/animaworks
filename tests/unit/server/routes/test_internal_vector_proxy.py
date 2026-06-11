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
