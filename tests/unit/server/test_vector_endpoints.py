# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for server vector endpoints worker-only routing."""

from __future__ import annotations

from unittest.mock import patch

from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from core.memory.rag.vector_worker_client import VectorWorkerResponse, VectorWorkerUnavailable
from server.routes.internal import create_internal_router


def _make_test_app(worker=None) -> FastAPI:
    app = FastAPI()
    app.include_router(create_internal_router(), prefix="/api")
    if worker is not None:
        app.state.vector_worker = worker
    return app


class _FakeVectorWorker:
    enabled = True
    fallback_direct = True
    native_crash_detected = False

    def __init__(self, response: VectorWorkerResponse | None = None, exc: Exception | None = None) -> None:
        self.response = response or VectorWorkerResponse(status_code=200, data={"status": "ok"})
        self.exc = exc
        self.calls: list[tuple[str, dict]] = []

    async def post(self, path: str, payload: dict):
        self.calls.append((path, payload))
        if self.exc:
            raise self.exc
        return self.response


async def _post(path: str, payload: dict, worker=None):
    app = _make_test_app(worker)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        return await client.post(path, json=payload)


async def test_vector_query_proxies_to_worker_without_touching_local_store():
    worker = _FakeVectorWorker(
        VectorWorkerResponse(
            status_code=200,
            data={"results": [{"id": "docw", "content": "worker", "score": 1.0, "metadata": {}}]},
        )
    )

    with patch("core.memory.rag.singleton.get_vector_store") as get_store:
        resp = await _post(
            "/api/internal/vector/query",
            {
                "anima_name": "rin",
                "collection": "rin_knowledge",
                "embedding": [0.1, 0.2],
                "top_k": 5,
            },
            worker,
        )

    assert resp.status_code == 200
    assert resp.json()["results"][0]["id"] == "docw"
    assert worker.calls == [
        (
            "/query",
            {
                "anima_name": "rin",
                "collection": "rin_knowledge",
                "embedding": [0.1, 0.2],
                "top_k": 5,
                "filter_metadata": None,
            },
        )
    ]
    get_store.assert_not_called()


async def test_vector_query_returns_503_when_worker_missing():
    with patch("core.memory.rag.singleton.get_vector_store") as get_store:
        resp = await _post(
            "/api/internal/vector/query",
            {"collection": "rin_knowledge", "embedding": [0.1]},
        )

    assert resp.status_code == 503
    assert resp.json()["detail"] == "Vector worker unavailable"
    get_store.assert_not_called()


async def test_vector_query_returns_503_when_worker_unavailable_even_if_fallback_flag_true():
    worker = _FakeVectorWorker(exc=VectorWorkerUnavailable("down"))
    worker.fallback_direct = True

    with patch("core.memory.rag.singleton.get_vector_store") as get_store:
        resp = await _post(
            "/api/internal/vector/query",
            {"collection": "rin_knowledge", "embedding": [0.1]},
            worker,
        )

    assert resp.status_code == 503
    assert resp.json()["detail"] == "Vector worker unavailable"
    get_store.assert_not_called()


async def test_vector_worker_error_status_is_forwarded():
    worker = _FakeVectorWorker(VectorWorkerResponse(status_code=500, data={"detail": "Vector upsert failed"}))

    resp = await _post(
        "/api/internal/vector/upsert",
        {"collection": "rin_knowledge", "documents": [{"id": "d1", "content": "c1"}]},
        worker,
    )

    assert resp.status_code == 500
    assert resp.json()["detail"] == "Vector upsert failed"


async def test_all_vector_endpoints_proxy_to_worker_paths():
    cases = [
        (
            "/api/internal/vector/upsert",
            {"collection": "c", "documents": [{"id": "d1", "content": "c1"}]},
            "/upsert",
        ),
        (
            "/api/internal/vector/update-metadata",
            {"collection": "c", "ids": ["d1"], "metadatas": [{"k": "v"}]},
            "/update-metadata",
        ),
        (
            "/api/internal/vector/delete-documents",
            {"collection": "c", "ids": ["d1"]},
            "/delete-documents",
        ),
        (
            "/api/internal/vector/get-by-metadata",
            {"collection": "c", "where": {"k": "v"}, "limit": 1},
            "/get-by-metadata",
        ),
        (
            "/api/internal/vector/get-by-ids",
            {"collection": "c", "ids": ["d1"]},
            "/get-by-ids",
        ),
        (
            "/api/internal/vector/create-collection",
            {"collection": "c"},
            "/create-collection",
        ),
        (
            "/api/internal/vector/delete-collection",
            {"collection": "c"},
            "/delete-collection",
        ),
        (
            "/api/internal/vector/list-collections",
            {},
            "/list-collections",
        ),
    ]

    for endpoint, payload, worker_path in cases:
        worker = _FakeVectorWorker(VectorWorkerResponse(status_code=200, data={"status": "ok"}))
        with patch("core.memory.rag.singleton.get_vector_store") as get_store:
            resp = await _post(endpoint, payload, worker)

        assert resp.status_code == 200
        assert worker.calls
        assert worker.calls[0][0] == worker_path
        get_store.assert_not_called()
