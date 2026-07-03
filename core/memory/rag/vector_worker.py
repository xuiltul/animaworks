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
import threading
import time
from collections.abc import Callable
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
_VECTOR_ACTION_ERROR = object()
_LATCH_RECOVERY_BACKOFF_SECONDS = float(os.environ.get("ANIMAWORKS_VECTOR_LATCH_RECOVERY_BACKOFF_SECONDS", "5"))
_LATCH_RECOVERY_RETRY_STATUSES = {"ok", "missing"}
_ACTIVE_REPAIR_STATUSES = {"requested", "stopping", "repairing"}
_latched_store_recovery_lock = threading.Lock()
_latched_store_recovery_backoff_until: dict[str, float] = {}
_latched_store_recovery_in_progress: set[str] = set()

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


def _has_active_repair_state(anima_name: str) -> bool:
    try:
        from core.memory.rag import repair_state

        return repair_state.read_state(anima_name).get("status") in _ACTIVE_REPAIR_STATUSES
    except Exception:
        logger.debug("Failed to read RAG repair state for owner=%s", anima_name, exc_info=True)
        return True


def _begin_latched_store_recovery(anima_name: str) -> bool:
    now = time.monotonic()
    with _latched_store_recovery_lock:
        retry_at = float(_latched_store_recovery_backoff_until.get(anima_name) or 0.0)
        if retry_at > now:
            return False
        if retry_at:
            _latched_store_recovery_backoff_until.pop(anima_name, None)
        if anima_name in _latched_store_recovery_in_progress:
            return False
        _latched_store_recovery_in_progress.add(anima_name)
        return True


def _end_latched_store_recovery(anima_name: str) -> None:
    with _latched_store_recovery_lock:
        _latched_store_recovery_in_progress.discard(anima_name)


def _set_latched_store_recovery_backoff(anima_name: str) -> None:
    with _latched_store_recovery_lock:
        _latched_store_recovery_backoff_until[anima_name] = time.monotonic() + _LATCH_RECOVERY_BACKOFF_SECONDS


def _clear_latched_store_recovery_backoff(anima_name: str) -> None:
    with _latched_store_recovery_lock:
        _latched_store_recovery_backoff_until.pop(anima_name, None)


def _try_recover_latched_store(anima_name: str | None) -> Any | None:
    if anima_name is None:
        return None

    from core.memory.rag.singleton import (
        clear_vector_store_init_failed,
        get_vector_store,
        is_global_vector_store_init_failed,
        is_vector_store_init_failed,
    )

    if is_global_vector_store_init_failed() or not is_vector_store_init_failed(anima_name):
        return None
    if _has_active_repair_state(anima_name):
        return None
    if not _begin_latched_store_recovery(anima_name):
        return None

    try:
        if is_global_vector_store_init_failed() or not is_vector_store_init_failed(anima_name):
            return None
        if _has_active_repair_state(anima_name):
            return None

        from core.memory.rag.sqlite_health import check_anima_vectordb_health

        health = check_anima_vectordb_health(
            anima_name,
            source="worker_store_unavailable",
            record_repair=True,
        )
        if health.status not in _LATCH_RECOVERY_RETRY_STATUSES:
            _set_latched_store_recovery_backoff(anima_name)
            logger.info(
                "Skipping latched vector-store recovery for owner=%s: health_status=%s; backing off for %.1fs",
                anima_name,
                health.status,
                _LATCH_RECOVERY_BACKOFF_SECONDS,
            )
            return None

        clear_vector_store_init_failed(anima_name)
        store = get_vector_store(anima_name)
        if store is not None:
            _clear_latched_store_recovery_backoff(anima_name)
            _clear_owner_write_circuit_breakers(anima_name)
            logger.info("Recovered latched vector store for owner=%s", anima_name)
            return store

        _set_latched_store_recovery_backoff(anima_name)
        logger.warning(
            "Latched vector-store recovery did not reopen a store for owner=%s; backing off for %.1fs",
            anima_name,
            _LATCH_RECOVERY_BACKOFF_SECONDS,
        )
        return None
    except Exception:
        _set_latched_store_recovery_backoff(anima_name)
        logger.warning("Latched vector-store recovery failed for owner=%s", anima_name, exc_info=True)
        return None
    finally:
        _end_latched_store_recovery(anima_name)


def _call_vector_store(anima_name: str | None, action: Callable[[Any], Any]) -> Any | None:
    from core.memory.rag.singleton import get_vector_store, reset_vector_store_after_error

    try:
        store = get_vector_store(anima_name)
        if store is None:
            store = _try_recover_latched_store(anima_name)
            if store is None:
                return None
        return action(store)
    except Exception:
        logger.warning("Vector worker native store action failed for owner=%s", anima_name or "shared", exc_info=True)
        try:
            reset_vector_store_after_error(anima_name, source="worker_action_failure")
        except Exception:
            logger.debug(
                "Vector worker failed to reset native store after action failure for owner=%s",
                anima_name or "shared",
                exc_info=True,
            )
        return _VECTOR_ACTION_ERROR


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


def _is_vector_action_error(value: Any) -> bool:
    return value is _VECTOR_ACTION_ERROR


def _write_failure_response(anima_name: str | None, collection: str, operation: str) -> JSONResponse:
    state = _record_vector_write_failure(anima_name, collection, operation)
    retry_at = float(state.get("next_retry_at") or 0.0)
    retry_after = max(0, int(retry_at - time.monotonic()))
    content = {
        "detail": f"Vector {operation} failed",
        "collection": collection,
        "owner": anima_name or "shared",
        "consecutive_failures": state["consecutive_failures"],
        "circuit_breaker_threshold": state["threshold"],
    }
    headers = None
    if retry_after > 0:
        content["retry_after_seconds"] = retry_after
        headers = {"Retry-After": str(retry_after)}
    return JSONResponse(
        status_code=500,
        content=content,
        headers=headers,
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
    _latched_store_recovery_backoff_until.clear()
    _latched_store_recovery_in_progress.clear()
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
        results = await _run_native(
            _call_vector_store,
            body.anima_name,
            lambda store: store.query(
                body.collection,
                body.embedding,
                body.top_k,
                body.filter_metadata,
            ),
        )
        if results is None:
            return {"results": []}
        if _is_vector_action_error(results):
            return {"results": []}
        return _search_results_payload(results)

    @app.post("/upsert")
    async def vector_upsert(body: VectorUpsertRequest):
        from core.memory.rag.store import Document

        breaker = _before_vector_write(body.anima_name, body.collection)
        if breaker is not None:
            return breaker
        docs = [
            Document(
                id=d["id"],
                content=d.get("content", ""),
                embedding=d.get("embedding"),
                metadata=d.get("metadata", {}),
            )
            for d in body.documents
        ]
        ok = await _run_native(
            _call_vector_store,
            body.anima_name,
            lambda store: store.upsert(body.collection, docs),
        )
        if ok is None:
            return JSONResponse(status_code=503, content={"detail": "Vector store unavailable"})
        if _is_vector_action_error(ok) or not ok:
            return _write_failure_response(body.anima_name, body.collection, "upsert")
        return _write_success_response(body.anima_name, body.collection)

    @app.post("/update-metadata")
    async def vector_update_metadata(body: VectorUpdateMetadataRequest):
        breaker = _before_vector_write(body.anima_name, body.collection)
        if breaker is not None:
            return breaker
        ok = await _run_native(
            _call_vector_store,
            body.anima_name,
            lambda store: store.update_metadata(
                body.collection,
                body.ids,
                body.metadatas,
            ),
        )
        if ok is None:
            return JSONResponse(status_code=503, content={"detail": "Vector store unavailable"})
        if _is_vector_action_error(ok) or not ok:
            return _write_failure_response(body.anima_name, body.collection, "update-metadata")
        return _write_success_response(body.anima_name, body.collection)

    @app.post("/delete-documents")
    async def vector_delete_documents(body: VectorDeleteDocumentsRequest):
        breaker = _before_vector_write(body.anima_name, body.collection)
        if breaker is not None:
            return breaker
        ok = await _run_native(
            _call_vector_store,
            body.anima_name,
            lambda store: store.delete_documents(body.collection, body.ids),
        )
        if ok is None:
            return JSONResponse(status_code=503, content={"detail": "Vector store unavailable"})
        if _is_vector_action_error(ok) or not ok:
            return _write_failure_response(body.anima_name, body.collection, "delete-documents")
        return _write_success_response(body.anima_name, body.collection)

    @app.post("/get-by-metadata")
    async def vector_get_by_metadata(body: VectorGetByMetadataRequest):
        results = await _run_native(
            _call_vector_store,
            body.anima_name,
            lambda store: store.get_by_metadata(
                body.collection,
                body.where,
                body.limit,
            ),
        )
        if results is None:
            return {"results": []}
        if _is_vector_action_error(results):
            return {"results": []}
        return _search_results_payload(results)

    @app.post("/get-by-ids")
    async def vector_get_by_ids(body: VectorGetByIdsRequest):
        docs = await _run_native(
            _call_vector_store,
            body.anima_name,
            lambda store: store.get_by_ids(body.collection, body.ids),
        )
        if docs is None:
            return {"documents": []}
        if _is_vector_action_error(docs):
            return {"documents": []}
        return {"documents": [{"id": d.id, "content": d.content, "metadata": d.metadata} for d in docs]}

    @app.post("/create-collection")
    async def vector_create_collection(body: VectorCollectionRequest):
        breaker = _before_vector_write(body.anima_name, body.collection)
        if breaker is not None:
            return breaker
        ok = await _run_native(
            _call_vector_store,
            body.anima_name,
            lambda store: store.create_collection(body.collection),
        )
        if ok is None:
            return JSONResponse(status_code=503, content={"detail": "Vector store unavailable"})
        if _is_vector_action_error(ok) or not ok:
            return _write_failure_response(body.anima_name, body.collection, "create-collection")
        return _write_success_response(body.anima_name, body.collection)

    @app.post("/delete-collection")
    async def vector_delete_collection(body: VectorCollectionRequest):
        breaker = _before_vector_write(body.anima_name, body.collection)
        if breaker is not None:
            return breaker
        ok = await _run_native(
            _call_vector_store,
            body.anima_name,
            lambda store: store.delete_collection(body.collection),
        )
        if ok is None:
            return JSONResponse(status_code=503, content={"detail": "Vector store unavailable"})
        if _is_vector_action_error(ok) or not ok:
            return _write_failure_response(body.anima_name, body.collection, "delete-collection")
        return _write_success_response(body.anima_name, body.collection)

    @app.post("/list-collections")
    async def vector_list_collections(body: VectorListCollectionsRequest):
        collections = await _run_native(
            _call_vector_store,
            body.anima_name,
            lambda store: store.list_collections(),
        )
        if collections is None:
            return {"collections": []}
        if _is_vector_action_error(collections):
            return {"collections": []}
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
