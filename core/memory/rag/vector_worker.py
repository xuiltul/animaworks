from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Isolated HTTP worker for native ChromaDB vector operations."""

import argparse
import asyncio
import concurrent.futures
import functools
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from core.platform.fd_limits import raise_fd_soft_limit

logger = logging.getLogger("animaworks.rag.vector_worker")

_CIRCUIT_FAILURE_THRESHOLD = int(os.environ.get("ANIMAWORKS_VECTOR_CIRCUIT_FAILURE_THRESHOLD", "3"))
_CIRCUIT_BACKOFF_BASE_SECONDS = float(os.environ.get("ANIMAWORKS_VECTOR_CIRCUIT_BACKOFF_BASE_SECONDS", "1"))
_CIRCUIT_BACKOFF_MAX_SECONDS = float(os.environ.get("ANIMAWORKS_VECTOR_CIRCUIT_BACKOFF_MAX_SECONDS", "300"))
_write_circuit_breakers: dict[str, dict[str, Any]] = {}

_native_executor = concurrent.futures.ThreadPoolExecutor(
    max_workers=1,
    thread_name_prefix="vector-worker-native",
)


class VectorQueryRequest(BaseModel):
    anima_name: str | None = None
    collection: str
    embedding: list[float]
    top_k: int = 10
    filter_metadata: dict[str, str | int | float] | None = None


class VectorUpsertRequest(BaseModel):
    anima_name: str | None = None
    collection: str
    documents: list[dict[str, Any]]


class VectorUpdateMetadataRequest(BaseModel):
    anima_name: str | None = None
    collection: str
    ids: list[str]
    metadatas: list[dict[str, str | int | float]]


class VectorDeleteDocumentsRequest(BaseModel):
    anima_name: str | None = None
    collection: str
    ids: list[str]


class VectorGetByMetadataRequest(BaseModel):
    anima_name: str | None = None
    collection: str
    where: dict[str, str | int | float] = {}
    limit: int = 20


class VectorGetByIdsRequest(BaseModel):
    anima_name: str | None = None
    collection: str
    ids: list[str]


class VectorCollectionRequest(BaseModel):
    anima_name: str | None = None
    collection: str


class VectorListCollectionsRequest(BaseModel):
    anima_name: str | None = None


class VectorQuickCheckRequest(BaseModel):
    anima_name: str
    timeout_seconds: float = 10.0
    source: str = "worker_quick_check"
    record_repair: bool = True


async def _run_native(fn, *args, **kwargs):
    loop = asyncio.get_running_loop()
    call = functools.partial(fn, *args, **kwargs)
    return await loop.run_in_executor(_native_executor, call)


def _vector_write_failed(operation: str, collection: str) -> JSONResponse:
    logger.warning("Vector worker %s failed for collection '%s'", operation, collection)
    return JSONResponse(
        status_code=500,
        content={
            "detail": f"Vector {operation} failed",
            "collection": collection,
        },
    )


def _breaker_key(anima_name: str | None, collection: str) -> str:
    owner = anima_name or "shared"
    return f"{owner}:{collection}"


def _clear_owner_write_circuit_breakers(anima_name: str | None) -> None:
    owner = anima_name or "shared"
    for key, state in list(_write_circuit_breakers.items()):
        if state.get("owner", "shared") == owner or key.startswith(f"{owner}:"):
            _write_circuit_breakers.pop(key, None)


def _before_vector_write(anima_name: str | None, collection: str) -> JSONResponse | None:
    key = _breaker_key(anima_name, collection)
    state = _write_circuit_breakers.get(key)
    if not state:
        return None
    retry_at = float(state.get("next_retry_at") or 0.0)
    now = time.monotonic()
    if retry_at <= now:
        return None
    retry_after = max(1, int(retry_at - now))
    logger.error(
        "Vector write circuit breaker open: owner=%s collection=%s failures=%s retry_after=%ss",
        anima_name or "shared",
        collection,
        state.get("consecutive_failures", 0),
        retry_after,
    )
    return JSONResponse(
        status_code=429,
        content={
            "detail": "Vector write circuit breaker open",
            "collection": collection,
            "owner": anima_name or "shared",
            "consecutive_failures": state.get("consecutive_failures", 0),
            "retry_after_seconds": retry_after,
        },
        headers={"Retry-After": str(retry_after)},
    )


def _record_vector_write_success(anima_name: str | None, collection: str) -> None:
    _write_circuit_breakers.pop(_breaker_key(anima_name, collection), None)


def _record_vector_write_failure(anima_name: str | None, collection: str, operation: str) -> dict[str, Any]:
    key = _breaker_key(anima_name, collection)
    now = time.monotonic()
    state = dict(_write_circuit_breakers.get(key) or {})
    failures = int(state.get("consecutive_failures") or 0) + 1
    delay = (
        min(_CIRCUIT_BACKOFF_BASE_SECONDS * (2 ** max(0, failures - 1)), _CIRCUIT_BACKOFF_MAX_SECONDS)
        if failures >= _CIRCUIT_FAILURE_THRESHOLD
        else 0.0
    )
    state.update(
        {
            "owner": anima_name or "shared",
            "collection": collection,
            "operation": operation,
            "consecutive_failures": failures,
            "next_retry_at": now + delay if delay > 0 else 0.0,
            "last_failure_monotonic": now,
            "threshold": _CIRCUIT_FAILURE_THRESHOLD,
        }
    )
    _write_circuit_breakers[key] = state
    log = logger.error if failures >= _CIRCUIT_FAILURE_THRESHOLD else logger.warning
    log(
        "Vector write failure recorded: owner=%s collection=%s operation=%s failures=%d next_retry=%.1fs",
        anima_name or "shared",
        collection,
        operation,
        failures,
        delay,
    )
    return state


def _breaker_status() -> list[dict[str, Any]]:
    now = time.monotonic()
    statuses: list[dict[str, Any]] = []
    for state in _write_circuit_breakers.values():
        retry_at = float(state.get("next_retry_at") or 0.0)
        item = {
            "owner": state.get("owner", "shared"),
            "collection": state.get("collection", ""),
            "operation": state.get("operation", ""),
            "consecutive_failures": int(state.get("consecutive_failures") or 0),
            "threshold": int(state.get("threshold") or _CIRCUIT_FAILURE_THRESHOLD),
            "open": retry_at > now,
            "retry_after_seconds": max(0, int(retry_at - now)),
        }
        statuses.append(item)
    return statuses


def _write_success_response(anima_name: str | None, collection: str) -> dict[str, str]:
    _record_vector_write_success(anima_name, collection)
    return {"status": "ok"}


def _write_failure_response(anima_name: str | None, collection: str, operation: str) -> JSONResponse:
    state = _record_vector_write_failure(anima_name, collection, operation)
    return JSONResponse(
        status_code=500,
        content={
            "detail": f"Vector {operation} failed",
            "collection": collection,
            "owner": anima_name or "shared",
            "consecutive_failures": state["consecutive_failures"],
            "circuit_breaker_threshold": state["threshold"],
        },
    )


def _search_results_payload(results) -> dict[str, Any]:
    return {
        "results": [
            {
                "id": r.document.id,
                "content": r.document.content,
                "score": r.score,
                "metadata": r.document.metadata,
            }
            for r in results
        ]
    }


async def _close_native_vector_stores() -> None:
    from core.memory.rag.singleton import close_all_vector_stores

    logger.info("Vector worker shutdown: closing cached vector stores")
    await _run_native(close_all_vector_stores)


def create_app() -> FastAPI:
    os.environ.pop("ANIMAWORKS_VECTOR_URL", None)
    _write_circuit_breakers.clear()
    from core.memory.rag.direct_access import enable_direct_chroma_for_process

    enable_direct_chroma_for_process()

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        try:
            yield
        finally:
            await _close_native_vector_stores()

    app = FastAPI(title="AnimaWorks Vector Worker", lifespan=lifespan)

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/status")
    async def status() -> dict[str, Any]:
        from core.gpu import get_gpu_status

        return {"status": "ok", "write_circuit_breakers": _breaker_status(), "gpu": get_gpu_status()}

    @app.post("/reset-store")
    async def vector_reset_store(body: VectorListCollectionsRequest) -> dict[str, str]:
        from core.memory.rag.singleton import reset_vector_store

        await _run_native(reset_vector_store, body.anima_name)
        _clear_owner_write_circuit_breakers(body.anima_name)
        return {"status": "ok"}

    @app.post("/query")
    async def vector_query(body: VectorQueryRequest):
        from core.memory.rag.singleton import get_vector_store

        store = get_vector_store(body.anima_name)
        if store is None:
            return {"results": []}
        results = await _run_native(
            store.query,
            body.collection,
            body.embedding,
            body.top_k,
            body.filter_metadata,
        )
        return _search_results_payload(results)

    @app.post("/upsert")
    async def vector_upsert(body: VectorUpsertRequest):
        from core.memory.rag.singleton import get_vector_store
        from core.memory.rag.store import Document

        breaker = _before_vector_write(body.anima_name, body.collection)
        if breaker is not None:
            return breaker
        store = get_vector_store(body.anima_name)
        if store is None:
            return JSONResponse(status_code=503, content={"detail": "Vector store unavailable"})
        docs = [
            Document(
                id=d["id"],
                content=d.get("content", ""),
                embedding=d.get("embedding"),
                metadata=d.get("metadata", {}),
            )
            for d in body.documents
        ]
        ok = await _run_native(store.upsert, body.collection, docs)
        if not ok:
            return _write_failure_response(body.anima_name, body.collection, "upsert")
        return _write_success_response(body.anima_name, body.collection)

    @app.post("/update-metadata")
    async def vector_update_metadata(body: VectorUpdateMetadataRequest):
        from core.memory.rag.singleton import get_vector_store

        breaker = _before_vector_write(body.anima_name, body.collection)
        if breaker is not None:
            return breaker
        store = get_vector_store(body.anima_name)
        if store is None:
            return JSONResponse(status_code=503, content={"detail": "Vector store unavailable"})
        ok = await _run_native(
            store.update_metadata,
            body.collection,
            body.ids,
            body.metadatas,
        )
        if not ok:
            return _write_failure_response(body.anima_name, body.collection, "update-metadata")
        return _write_success_response(body.anima_name, body.collection)

    @app.post("/delete-documents")
    async def vector_delete_documents(body: VectorDeleteDocumentsRequest):
        from core.memory.rag.singleton import get_vector_store

        breaker = _before_vector_write(body.anima_name, body.collection)
        if breaker is not None:
            return breaker
        store = get_vector_store(body.anima_name)
        if store is None:
            return JSONResponse(status_code=503, content={"detail": "Vector store unavailable"})
        ok = await _run_native(store.delete_documents, body.collection, body.ids)
        if not ok:
            return _write_failure_response(body.anima_name, body.collection, "delete-documents")
        return _write_success_response(body.anima_name, body.collection)

    @app.post("/get-by-metadata")
    async def vector_get_by_metadata(body: VectorGetByMetadataRequest):
        from core.memory.rag.singleton import get_vector_store

        store = get_vector_store(body.anima_name)
        if store is None:
            return {"results": []}
        results = await _run_native(
            store.get_by_metadata,
            body.collection,
            body.where,
            body.limit,
        )
        return _search_results_payload(results)

    @app.post("/get-by-ids")
    async def vector_get_by_ids(body: VectorGetByIdsRequest):
        from core.memory.rag.singleton import get_vector_store

        store = get_vector_store(body.anima_name)
        if store is None:
            return {"documents": []}
        docs = await _run_native(store.get_by_ids, body.collection, body.ids)
        return {"documents": [{"id": d.id, "content": d.content, "metadata": d.metadata} for d in docs]}

    @app.post("/create-collection")
    async def vector_create_collection(body: VectorCollectionRequest):
        from core.memory.rag.singleton import get_vector_store

        breaker = _before_vector_write(body.anima_name, body.collection)
        if breaker is not None:
            return breaker
        store = get_vector_store(body.anima_name)
        if store is None:
            return JSONResponse(status_code=503, content={"detail": "Vector store unavailable"})
        ok = await _run_native(store.create_collection, body.collection)
        if not ok:
            return _write_failure_response(body.anima_name, body.collection, "create-collection")
        return _write_success_response(body.anima_name, body.collection)

    @app.post("/delete-collection")
    async def vector_delete_collection(body: VectorCollectionRequest):
        from core.memory.rag.singleton import get_vector_store

        breaker = _before_vector_write(body.anima_name, body.collection)
        if breaker is not None:
            return breaker
        store = get_vector_store(body.anima_name)
        if store is None:
            return JSONResponse(status_code=503, content={"detail": "Vector store unavailable"})
        ok = await _run_native(store.delete_collection, body.collection)
        if not ok:
            return _write_failure_response(body.anima_name, body.collection, "delete-collection")
        return _write_success_response(body.anima_name, body.collection)

    @app.post("/list-collections")
    async def vector_list_collections(body: VectorListCollectionsRequest):
        from core.memory.rag.singleton import get_vector_store

        store = get_vector_store(body.anima_name)
        if store is None:
            return {"collections": []}
        collections = await _run_native(store.list_collections)
        return {"collections": collections}

    @app.post("/quick-check")
    async def vector_quick_check(body: VectorQuickCheckRequest):
        from core.memory.rag.sqlite_health import check_anima_vectordb_health

        result = await _run_native(
            check_anima_vectordb_health,
            body.anima_name,
            timeout_seconds=body.timeout_seconds,
            source=body.source,
            record_repair=body.record_repair,
        )
        return {
            "status": result.status,
            "ok": result.ok,
            "corrupt": result.corrupt,
            "db_path": str(result.db_path),
            "details": list(result.details),
            "error": result.error,
        }

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Run isolated AnimaWorks vector worker")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()

    import uvicorn

    raise_fd_soft_limit(logger=logger, process_label="vector worker")
    uvicorn.run(
        create_app(),
        host=args.host,
        port=args.port,
        log_level="info",
        timeout_keep_alive=65,
    )


if __name__ == "__main__":
    main()
