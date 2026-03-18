from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""HTTP-based VectorStore proxy for child processes.

Delegates all VectorStore operations to the server's
``/api/internal/vector/*`` endpoints. Used when the
``ANIMAWORKS_VECTOR_URL`` environment variable is set.
"""

import logging
from typing import Any

from core.memory.rag.store import Document, SearchResult, VectorStore

logger = logging.getLogger(__name__)

_UPSERT_BATCH_LIMIT = 500


def _parse_search_results(data: list[dict[str, Any]]) -> list[SearchResult]:
    """Convert JSON dicts to SearchResult objects.

    Supports both nested format (document + score) and flat format
    (id, content, score, metadata at top level) as returned by the server.
    """
    results: list[SearchResult] = []
    for item in data:
        doc_data = item.get("document")
        if doc_data is None:
            doc_data = {k: v for k, v in item.items() if k != "score"}
        doc = Document(
            id=doc_data.get("id", ""),
            content=doc_data.get("content", ""),
            metadata=doc_data.get("metadata") or {},
        )
        score = float(item.get("score", 1.0))
        results.append(SearchResult(document=doc, score=score))
    return results


def _parse_documents(data: list[dict[str, Any]]) -> list[Document]:
    """Convert JSON dicts to Document objects."""
    return [
        Document(
            id=d.get("id", ""),
            content=d.get("content", ""),
            metadata=d.get("metadata") or {},
        )
        for d in data
    ]


class HttpVectorStore(VectorStore):
    """VectorStore that delegates to server via HTTP."""

    def __init__(self, base_url: str, anima_name: str | None = None) -> None:
        """Initialize HTTP vector store.

        Args:
            base_url: Base URL for vector API (e.g., http://localhost:8000/api/internal/vector)
            anima_name: Anima name for server-side routing.
        """
        self._base_url = base_url.rstrip("/")
        self._anima_name = anima_name
        self._client: Any = None

    def _get_client(self):
        """Lazy-initialize httpx client."""
        if self._client is None:
            import httpx

            self._client = httpx.Client(base_url=self._base_url, timeout=30.0)
        return self._client

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any] | None:
        """POST to endpoint and return JSON, or None on error."""
        try:
            resp = self._get_client().post(path, json=payload)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.warning("HTTP vector request failed %s: %s", path, e)
            return None

    def create_collection(self, name: str) -> bool:
        """Create a new collection."""
        return self._post("/create-collection", {"anima_name": self._anima_name, "collection": name}) is not None

    def delete_collection(self, name: str) -> bool:
        """Delete a collection."""
        return self._post("/delete-collection", {"anima_name": self._anima_name, "collection": name}) is not None

    def list_collections(self) -> list[str]:
        """List all collection names."""
        data = self._post("/list-collections", {"anima_name": self._anima_name})
        if data and "collections" in data:
            return list(data["collections"])
        return []

    def upsert(self, collection: str, documents: list[Document]) -> bool:
        """Insert or update documents in a collection."""
        if not documents:
            return True
        ok = True
        for i in range(0, len(documents), _UPSERT_BATCH_LIMIT):
            batch = documents[i : i + _UPSERT_BATCH_LIMIT]
            payload = {
                "anima_name": self._anima_name,
                "collection": collection,
                "documents": [
                    {
                        "id": d.id,
                        "content": d.content,
                        "embedding": d.embedding,
                        "metadata": d.metadata,
                    }
                    for d in batch
                ],
            }
            if self._post("/upsert", payload) is None:
                ok = False
        return ok

    def query(
        self,
        collection: str,
        embedding: list[float],
        top_k: int = 10,
        filter_metadata: dict[str, str | int | float] | None = None,
    ) -> list[SearchResult]:
        """Query collection by embedding similarity."""
        payload: dict[str, Any] = {
            "anima_name": self._anima_name,
            "collection": collection,
            "embedding": embedding,
            "top_k": top_k,
        }
        if filter_metadata:
            payload["filter_metadata"] = filter_metadata
        data = self._post("/query", payload)
        if data and "results" in data:
            return _parse_search_results(data["results"])
        return []

    def delete_documents(self, collection: str, ids: list[str]) -> bool:
        """Delete specific documents by ID."""
        if not ids:
            return True
        return (
            self._post("/delete-documents", {"anima_name": self._anima_name, "collection": collection, "ids": ids})
            is not None
        )

    def update_metadata(self, collection: str, ids: list[str], metadatas: list[dict[str, str | int | float]]) -> bool:
        """Update metadata for existing documents."""
        if not ids:
            return True
        return (
            self._post(
                "/update-metadata",
                {"anima_name": self._anima_name, "collection": collection, "ids": ids, "metadatas": metadatas},
            )
            is not None
        )

    def get_by_metadata(
        self,
        collection: str,
        where: dict[str, str | int | float],
        limit: int = 20,
    ) -> list[SearchResult]:
        """Retrieve documents by metadata filter without embedding search."""
        data = self._post(
            "/get-by-metadata",
            {"anima_name": self._anima_name, "collection": collection, "where": where, "limit": limit},
        )
        if data and "results" in data:
            return _parse_search_results(data["results"])
        return []

    def get_by_ids(self, collection: str, ids: list[str]) -> list[Document]:
        """Fetch documents (with metadata) by their IDs."""
        if not ids:
            return []
        data = self._post("/get-by-ids", {"anima_name": self._anima_name, "collection": collection, "ids": ids})
        if data and "documents" in data:
            return _parse_documents(data["documents"])
        return []

    def close(self) -> None:
        """Close the HTTP client if created."""
        if self._client is not None:
            try:
                self._client.close()
            except Exception:
                pass
            self._client = None
