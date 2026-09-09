"""VectorStore operations over root IPC."""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from collections.abc import Awaitable, Callable
from typing import Any

from core.memory.rag.http_store import (
    _UPSERT_BATCH_LIMIT,
    HttpVectorStore,
    _parse_documents,
    _parse_search_results,
)
from core.memory.rag.store import Document

logger = logging.getLogger(__name__)

MemoryRequester = Callable[[str, dict[str, Any]], dict[str, Any]]


def root_memory_requester(
    handler: Callable[[str, dict[str, Any]], Awaitable[dict[str, Any]]],
) -> MemoryRequester:
    """Bridge root-local off-loop callers to the existing owner memory queue.

    Capture the owning loop explicitly: tool threads can have their own event
    loops. Never block the owner loop, nor silently switch to a different DB
    when its lifecycle has ended.
    """
    loop = asyncio.get_running_loop()
    loop_thread = threading.get_ident()

    def request(method: str, params: dict[str, Any]) -> dict[str, Any]:
        if threading.get_ident() == loop_thread:
            raise RuntimeError("synchronous memory operation attempted on the root event loop")
        if loop.is_closed() or not loop.is_running():
            raise RuntimeError("root memory event loop is unavailable")
        future = asyncio.run_coroutine_threadsafe(handler(method, params), loop)
        try:
            return future.result(timeout=125.0)
        except TimeoutError:
            future.cancel()
            raise

    return request


# Cap matches plan: never sleep longer than 500ms on a single retry.
_MAX_RETRY_AFTER_MS = 500
_DEFAULT_RETRY_AFTER_MS = 250


def _retry_after_ms(exc: BaseException) -> int | None:
    """Return sleep ms when *exc* is an explicit retryable failure, else None."""
    if getattr(exc, "retryable", None) is True:
        raw = getattr(exc, "retry_after_ms", _DEFAULT_RETRY_AFTER_MS)
        try:
            ms = int(raw)
        except (TypeError, ValueError):
            ms = _DEFAULT_RETRY_AFTER_MS
        return max(0, min(ms, _MAX_RETRY_AFTER_MS))

    # Production path: _MemoryRpcClient raises RuntimeError("UNAVAILABLE: ...")
    # after root returns retryable=True (retry_after_ms is stripped on the wire
    # conversion). Match the known code prefix only.
    message = str(exc)
    if message.startswith("UNAVAILABLE:") or message.startswith("UNAVAILABLE "):
        # Deterministic failures never heal within one retry window — waiting
        # 250ms per query for a permanently missing collection just burns the
        # priming budget (observed fleet-wide with {anima}_entities).
        if "does not exist" in message or "already exists" in message:
            return None
        return _DEFAULT_RETRY_AFTER_MS
    return None


class IpcVectorStore(HttpVectorStore):
    """Root IPC backend for one phase3 anima's vector database."""

    def __init__(self, base_url: str, anima_name: str | None, requester: MemoryRequester) -> None:
        super().__init__(base_url, anima_name)
        self._requester = requester
        self._ipc_unavailable_writes: set[str] = set()

    def _call(self, method: str, params: dict[str, Any], *, allow_retry: bool = False) -> dict[str, Any] | None:
        try:
            return self._requester(method, params)
        except Exception as exc:
            delay_ms = _retry_after_ms(exc) if allow_retry else None
            if delay_ms is not None:
                logger.warning(
                    "IPC vector request retryable: method=%s anima=%s retry_after_ms=%s error=%s",
                    method,
                    self._anima_name,
                    delay_ms,
                    exc,
                )
                if delay_ms:
                    time.sleep(delay_ms / 1000.0)
                try:
                    return self._requester(method, params)
                except Exception as retry_exc:
                    logger.warning(
                        "IPC vector request failed after retry: method=%s anima=%s error=%s",
                        method,
                        self._anima_name,
                        retry_exc,
                    )
                    return None
            logger.warning("IPC vector request failed: method=%s anima=%s error=%s", method, self._anima_name, exc)
            return None

    def _write(self, method: str, params: dict[str, Any]) -> bool:
        data = self._call(method, params)
        collection = params.get("collection")
        if isinstance(collection, str):
            if data is None:
                self._ipc_unavailable_writes.add(collection)
            elif data.get("ok") is True:
                self._ipc_unavailable_writes.discard(collection)
        return data is not None and data.get("ok") is True

    def is_transient_write_failure(self, collection: str) -> bool:
        return collection in self._ipc_unavailable_writes

    def create_collection(self, name: str) -> bool:
        return self._write("memory.create_collection", {"collection": name})

    def delete_collection(self, name: str) -> bool:
        return self._write("memory.delete_collection", {"collection": name})

    def upsert(self, collection: str, documents: list[Document]) -> bool:
        if not documents:
            return True
        for start in range(0, len(documents), _UPSERT_BATCH_LIMIT):
            batch = documents[start : start + _UPSERT_BATCH_LIMIT]
            if not self._write(
                "memory.upsert",
                {
                    "collection": collection,
                    "documents": [
                        {
                            "id": document.id,
                            "content": document.content,
                            "embedding": document.embedding,
                            "metadata": document.metadata,
                        }
                        for document in batch
                    ],
                },
            ):
                return False
        return True

    def delete_documents(self, collection: str, ids: list[str]) -> bool:
        return not ids or self._write("memory.delete_documents", {"collection": collection, "ids": ids})

    def update_metadata(
        self,
        collection: str,
        ids: list[str],
        metadatas: list[dict[str, str | int | float]],
    ) -> bool:
        return not ids or self._write(
            "memory.update_metadata",
            {"collection": collection, "ids": ids, "metadatas": metadatas},
        )

    def enqueue_access_updates(self, operations: list[dict[str, Any]]) -> bool:
        """Queue atomic access-counter deltas in the root memory service."""
        return not operations or self._write("memory.apply_access_updates", {"operations": operations})

    def list_collections(self) -> list[str]:
        data = self._call("memory.list_collections_checked", {}, allow_retry=True)
        collections = data.get("collections") if data else None
        return (
            list(collections)
            if isinstance(collections, list) and all(isinstance(name, str) for name in collections)
            else []
        )

    def list_collections_checked(self) -> list[str] | None:
        data = self._call("memory.list_collections_checked", {}, allow_retry=True)
        collections = data.get("collections") if data else None
        if not isinstance(collections, list) or not all(isinstance(name, str) for name in collections):
            return None
        return list(collections)

    def query(
        self,
        collection: str,
        embedding: list[float],
        top_k: int = 10,
        filter_metadata: dict[str, str | int | float] | None = None,
    ):
        params: dict[str, Any] = {"collection": collection, "embedding": embedding, "top_k": top_k}
        if filter_metadata:
            params["filter_metadata"] = filter_metadata
        data = self._call("memory.query", params, allow_retry=True)
        return _parse_search_results(data["results"]) if data and isinstance(data.get("results"), list) else []

    def get_by_metadata(
        self,
        collection: str,
        where: dict[str, str | int | float],
        limit: int = 20,
    ):
        data = self._call(
            "memory.get_by_metadata",
            {"collection": collection, "where": where, "limit": limit},
            allow_retry=True,
        )
        return _parse_search_results(data["results"]) if data and isinstance(data.get("results"), list) else []

    def get_by_ids(self, collection: str, ids: list[str]):
        if not ids:
            return []
        data = self._call("memory.get_by_ids", {"collection": collection, "ids": ids}, allow_retry=True)
        return _parse_documents(data["documents"]) if data and isinstance(data.get("documents"), list) else []
