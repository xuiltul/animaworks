from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Vector store abstraction and ChromaDB implementation.

Provides persistent vector storage for memory embeddings with:
- Anima-isolated collections
- Metadata filtering (date, memory_type, importance, tags)
- Batch upsert and query operations
"""

import logging
import os
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, cast

from core.memory.rag.direct_access import _DirectAccessCapability, require_direct_chroma_allowed
from core.platform.env import CHROMA_THREAD_ENV_VARS

logger = logging.getLogger("animaworks.rag.store")
_persistent_client_init_lock = threading.Lock()
_DEFAULT_CHROMA_NATIVE_THREADS = 4
_missing_collection_warned: set[str] = set()
_missing_collection_warned_lock = threading.Lock()


def _bound_chroma_native_threads() -> None:
    """Limit the Rust thread pools inside chroma, which default to one thread per CPU.

    Each client starts a tokio runtime when it is created, and the first write
    starts a process-wide rayon pool.  Both read these variables when they
    start, and rayon starts long after the client, so the values stay set for
    the life of the process.  Values already in the environment are kept.
    """
    try:
        from core.config import load_config

        limit = int(load_config().rag.chroma_native_threads)
    except Exception:
        limit = _DEFAULT_CHROMA_NATIVE_THREADS
    if limit <= 0:
        return
    value = str(min(limit, os.cpu_count() or limit))
    for name in CHROMA_THREAD_ENV_VARS:
        os.environ.setdefault(name, value)


def _is_missing_collection_error(error: BaseException | str) -> bool:
    text = str(error).lower()
    return "collection" in text and "does not exist" in text


def log_missing_collection_once(collection: str, message: str, *args: Any) -> None:
    """Log a missing-collection warning once per process; later hits are DEBUG."""
    with _missing_collection_warned_lock:
        first = collection not in _missing_collection_warned
        if first:
            _missing_collection_warned.add(collection)
    if first:
        logger.warning(message, *args)
    else:
        logger.debug(message, *args)


def reset_missing_collection_warnings() -> None:
    """Clear the process-local missing-collection warning set (tests only)."""
    with _missing_collection_warned_lock:
        _missing_collection_warned.clear()


# ── Data structures ────────────────────────────────────────────────


@dataclass
class Document:
    """A document chunk with embedding and metadata."""

    id: str
    content: str
    embedding: list[float] | None = None
    metadata: dict[str, str | int | float | bool | list[str]] = field(default_factory=dict)


@dataclass
class SearchResult:
    """A search result with score."""

    document: Document
    score: float


class CollectionExistence(StrEnum):
    """Three-state result for a collection existence check."""

    EXISTS = "exists"
    MISSING = "missing"
    UNAVAILABLE = "unavailable"


# ── VectorStore abstract base class ────────────────────────────────


class VectorStore(ABC):
    """Abstract base class for vector storage backends."""

    @abstractmethod
    def create_collection(self, name: str) -> bool:
        """Create a new collection.

        Args:
            name: Collection name (e.g., "sakura_knowledge")

        Returns:
            True on success, False on failure.
        """

    @abstractmethod
    def delete_collection(self, name: str) -> bool:
        """Delete a collection and all its documents.

        Returns:
            True on success, False on failure.
        """

    @abstractmethod
    def list_collections(self) -> list[str] | None:
        """List collection names, or return ``None`` when the result is unavailable."""

    def collection_exists(self, name: str) -> CollectionExistence:
        """Return whether a collection exists, is missing, or is unavailable."""
        collections = self.list_collections()
        if collections is None:
            return CollectionExistence.UNAVAILABLE
        if name in collections:
            return CollectionExistence.EXISTS
        return CollectionExistence.MISSING

    @abstractmethod
    def upsert(self, collection: str, documents: list[Document]) -> bool:
        """Insert or update documents in a collection.

        Args:
            collection: Collection name
            documents: List of documents with embeddings

        Returns:
            True on success, False on failure.
        """

    @abstractmethod
    def query(
        self,
        collection: str,
        embedding: list[float],
        top_k: int = 10,
        filter_metadata: dict[str, str | int | float] | None = None,
    ) -> list[SearchResult]:
        """Query collection by embedding similarity.

        Args:
            collection: Collection name
            embedding: Query embedding vector
            top_k: Number of results to return
            filter_metadata: Optional metadata filters (exact match)

        Returns:
            List of search results sorted by similarity score (descending)
        """

    @abstractmethod
    def delete_documents(self, collection: str, ids: list[str]) -> bool:
        """Delete specific documents by ID.

        Returns:
            True on success, False on failure.
        """

    @abstractmethod
    def update_metadata(self, collection: str, ids: list[str], metadatas: list[dict[str, str | int | float]]) -> bool:
        """Update metadata for existing documents without re-embedding.

        Returns:
            True on success, False on failure.
        """

    @abstractmethod
    def get_by_metadata(
        self,
        collection: str,
        where: dict[str, str | int | float],
        limit: int = 20,
    ) -> list[SearchResult]:
        """Retrieve documents by metadata filter without embedding search."""

    @abstractmethod
    def get_all(self, collection: str, limit: int = 100_000) -> list[SearchResult]:
        """Retrieve documents without applying a metadata filter."""

    @abstractmethod
    def count(self, collection: str) -> int | None:
        """Return the collection size, or None when it cannot be determined."""

    @abstractmethod
    def get_by_ids(
        self,
        collection: str,
        ids: list[str],
    ) -> list[Document]:
        """Fetch documents (with metadata) by their IDs."""


# ── ChromaDB implementation ─────────────────────────────────────────


def _stop_chroma_client_system(client: Any) -> None:
    """Best-effort fallback for Chroma clients without a working close()."""
    system = getattr(client, "_system", None) or getattr(client, "_server", None)
    stop = getattr(system, "stop", None)
    if callable(stop):
        try:
            stop()
        except Exception:
            logger.debug("Failed to stop ChromaDB system", exc_info=True)


class ChromaVectorStore(VectorStore):
    """ChromaDB-backed vector store.

    Stores embeddings in SQLite at ~/.animaworks/vectordb/chroma.sqlite3
    """

    def __init__(
        self,
        persist_dir: Path,
        anima_name: str | None = None,
        *,
        allow_direct: _DirectAccessCapability | None = None,
    ) -> None:
        """Initialize ChromaDB at an explicit persistent directory.

        Args:
            persist_dir: Explicit per-anima or staging directory for ChromaDB.
            anima_name: Owner anima for repair signal attribution.
        """
        require_direct_chroma_allowed(allow_direct)
        if persist_dir is None:
            raise ValueError("persist_dir is required for direct ChromaDB access")

        import chromadb

        if persist_dir.parent.exists():
            persist_dir.mkdir(exist_ok=True)
        else:
            logger.warning(
                "Parent directory does not exist for vectordb; creating with parents: %s",
                persist_dir,
            )
            persist_dir.mkdir(parents=True, exist_ok=True)

        from core.memory.rag.sqlite_health import prepare_chroma_sqlite_for_startup

        prepare_chroma_sqlite_for_startup(persist_dir, anima_name=anima_name)
        logger.debug("Initializing ChromaDB at %s", persist_dir)
        with _persistent_client_init_lock:
            _bound_chroma_native_threads()
            self.client = chromadb.PersistentClient(path=str(persist_dir))
        self.persist_dir = persist_dir
        self.anima_name = anima_name
        self._closed = False
        # No maintenance connection after PersistentClient init: closing a
        # Python sqlite3 rw connection here deletes the -wal/-shm the live
        # client just created (POSIX locks cannot see the Rust client's locks
        # in the same process), stranding it on deleted-WAL fds → "file is not
        # a database" storms (2026-08-11 incident). WAL mode is already
        # ensured by prepare_chroma_sqlite_for_startup() before client init.

    def close(self) -> None:
        """Close the underlying Chroma client if supported.

        A closed store instance is not reusable; callers should drop it and
        obtain a fresh singleton/store after reset.
        """
        if self._closed:
            return
        self._closed = True
        client = getattr(self, "client", None)
        try:
            if client is None:
                return
            close = getattr(client, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    logger.debug("Failed to close ChromaDB client at %s", self.persist_dir, exc_info=True)
                    _stop_chroma_client_system(client)
            else:
                _stop_chroma_client_system(client)
        finally:
            self.client = None
            _clear_chroma_system_cache()

    def _report_chroma_error(self, collection: str, error: Exception, source: str) -> None:
        """Record Chroma errors that may indicate persistent-index corruption."""
        try:
            from core.memory.rag.repair import record_chroma_error

            record_chroma_error(
                anima_name=self._anima_name(),
                collection=collection,
                error=error,
                source=source,
            )
        except Exception:
            logger.debug("Failed to record RAG repair signal", exc_info=True)

    def _anima_name(self) -> str | None:
        return getattr(self, "anima_name", None)

    def _owner_label(self) -> str:
        return self._anima_name() or "shared"

    def _create_collection_once(self, name: str) -> bool:
        try:
            self.client.create_collection(
                name=name,
                metadata={"hnsw:space": "cosine"},
            )
        except Exception as exc:
            # Root memory service calls this method directly to keep repair
            # ownership centralized. Repeated initialization is still valid.
            if "already exists" in str(exc).lower():
                logger.debug("Collection '%s' already exists", name)
                return True
            raise
        logger.info("Created collection '%s' (space=cosine)", name)
        return True

    def create_collection(self, name: str) -> bool:
        """Create a new collection or get existing one."""
        try:
            return self._create_collection_once(name)
        except Exception as e:
            if "already exists" in str(e).lower():
                logger.debug("Collection '%s' already exists: %s", name, e)
                return True
            self._report_chroma_error(name, e, "create_collection")
            logger.warning(
                "Failed to create collection '%s': owner=%s db_path=%s error=%s",
                name,
                self._owner_label(),
                self.persist_dir,
                e,
            )
            return False

    def _delete_collection_once(self, name: str) -> bool:
        self.client.delete_collection(name=name)
        logger.info("Deleted collection '%s'", name)
        return True

    def delete_collection(self, name: str) -> bool:
        """Delete a collection."""
        try:
            return self._delete_collection_once(name)
        except Exception as e:
            self._report_chroma_error(name, e, "delete_collection")
            logger.warning(
                "Failed to delete collection '%s': owner=%s db_path=%s error=%s",
                name,
                self._owner_label(),
                self.persist_dir,
                e,
            )
            return False

    def _list_collections_once(self) -> list[str]:
        collections = self.client.list_collections()
        return [c.name for c in collections]

    def list_collections(self) -> list[str] | None:
        """List all collections, or return ``None`` when the store is unavailable."""
        try:
            return self._list_collections_once()
        except Exception as e:
            self._report_chroma_error("<list_collections>", e, "list_collections")
            logger.warning(
                "Failed to list collections: owner=%s db_path=%s error=%s",
                self._owner_label(),
                self.persist_dir,
                e,
            )
            return None

    def verify_rebuilt_data(self, *, expected_chunks: int) -> dict[str, int]:
        """Verify counts and query every nonempty freshly reopened collection."""
        collections = self._list_collections_once()
        opened = [(name, self.client.get_collection(name=name)) for name in collections]
        counted = [(name, collection, collection.count()) for name, collection in opened]
        chunks = sum(count for _name, _collection, count in counted)
        if chunks != expected_chunks:
            raise RuntimeError(f"expected {expected_chunks} chunks after rebuild, found {chunks}")
        if expected_chunks <= 0:
            return {"collections": len(collections), "chunks": chunks, "query_results": 0}
        if not collections:
            raise RuntimeError(f"no collections found despite indexing {expected_chunks} chunks")

        query_results = 0
        for name, collection, count in counted:
            if count == 0:
                continue
            sample = collection.get(limit=1, include=["embeddings"])
            ids = sample.get("ids") or []
            embeddings = sample.get("embeddings")
            if not ids or embeddings is None or len(embeddings) == 0:
                raise RuntimeError(f"rebuilt collection '{name}' has {count} documents but no queryable embedding")
            embedding = list(embeddings[0])
            queried = collection.query(query_embeddings=[embedding], n_results=1)
            result_ids = queried.get("ids") or []
            if not result_ids or not result_ids[0]:
                raise RuntimeError(f"rebuilt collection '{name}' returned no similarity query results")
            query_results += len(result_ids[0])

        return {"collections": len(collections), "chunks": chunks, "query_results": query_results}

    def _upsert_once(self, collection: str, documents: list[Document]) -> bool:
        coll = self.client.get_or_create_collection(
            name=collection,
            metadata={"hnsw:space": "cosine"},
        )

        ids = [doc.id for doc in documents]
        contents = [doc.content for doc in documents]
        embeddings = [doc.embedding for doc in documents]
        metadatas = [self._serialize_metadata(doc.metadata) for doc in documents]

        if any(emb is None for emb in embeddings):
            logger.warning(
                "Upsert to '%s' failed: missing embeddings owner=%s db_path=%s",
                collection,
                self._owner_label(),
                self.persist_dir,
            )
            return False

        coll.upsert(
            ids=ids,
            documents=contents,
            embeddings=cast(Any, embeddings),
            metadatas=cast(Any, metadatas),
        )

        logger.debug("Upserted %d documents to collection '%s'", len(documents), collection)
        return True

    def upsert(self, collection: str, documents: list[Document]) -> bool:
        """Upsert documents into collection."""
        if not documents:
            return True

        try:
            return self._upsert_once(collection, documents)
        except Exception as e:
            self._report_chroma_error(collection, e, "upsert")
            msg = "Failed to upsert %d documents to '%s': owner=%s db_path=%s error=%s"
            msg_args = (len(documents), collection, self._owner_label(), self.persist_dir, e)
            if _is_missing_collection_error(e):
                log_missing_collection_once(collection, msg, *msg_args)
            else:
                logger.warning(msg, *msg_args)
            return False

    def _query_once(
        self,
        collection: str,
        embedding: list[float],
        top_k: int = 10,
        filter_metadata: dict[str, str | int | float] | None = None,
    ) -> list[SearchResult]:
        coll = self.client.get_collection(name=collection)

        # Build where clause for metadata filtering
        where = None
        if filter_metadata:
            where = {k: v for k, v in filter_metadata.items()}

        # Query ChromaDB
        results = coll.query(
            query_embeddings=[embedding],
            n_results=top_k,
            where=cast(Any, where),
        )

        # Parse results
        search_results: list[SearchResult] = []
        if results["ids"] and results["ids"][0]:
            for i, doc_id in enumerate(results["ids"][0]):
                meta_raw = results["metadatas"][0][i] if results["metadatas"] else {}
                meta_dict: dict[str, str | int | float | list[str]] = cast(Any, dict(meta_raw)) if meta_raw else {}
                doc = Document(
                    id=doc_id,
                    content=results["documents"][0][i] if results["documents"] else "",
                    metadata=meta_dict,
                )
                distances = results["distances"]
                raw = 1.0 - distances[0][i] if distances else 0.0
                score = max(0.0, min(1.0, raw))
                search_results.append(SearchResult(document=doc, score=score))

        logger.debug(
            "Query returned %d results from collection '%s'",
            len(search_results),
            collection,
        )
        return search_results

    def query(
        self,
        collection: str,
        embedding: list[float],
        top_k: int = 10,
        filter_metadata: dict[str, str | int | float] | None = None,
    ) -> list[SearchResult]:
        """Query by embedding similarity."""
        try:
            return self._query_once(collection, embedding, top_k, filter_metadata)
        except Exception as e:
            self._report_chroma_error(collection, e, "query")
            msg = "ChromaDB query failed for collection '%s': owner=%s db_path=%s error=%s"
            msg_args = (collection, self._owner_label(), self.persist_dir, e)
            if _is_missing_collection_error(e):
                log_missing_collection_once(collection, msg, *msg_args)
            else:
                logger.warning(msg, *msg_args)
            return []

    def _delete_documents_once(self, collection: str, ids: list[str]) -> bool:
        coll = self.client.get_collection(name=collection)
        coll.delete(ids=ids)
        logger.debug("Deleted %d documents from collection '%s'", len(ids), collection)
        return True

    def delete_documents(self, collection: str, ids: list[str]) -> bool:
        """Delete documents by ID."""
        if not ids:
            return True

        try:
            return self._delete_documents_once(collection, ids)
        except Exception as e:
            self._report_chroma_error(collection, e, "delete_documents")
            msg = "Failed to delete documents from '%s': owner=%s db_path=%s error=%s"
            msg_args = (collection, self._owner_label(), self.persist_dir, e)
            if _is_missing_collection_error(e):
                log_missing_collection_once(collection, msg, *msg_args)
            else:
                logger.warning(msg, *msg_args)
            return False

    def _update_metadata_once(
        self,
        collection: str,
        ids: list[str],
        metadatas: list[dict[str, str | int | float]],
    ) -> bool:
        coll = self.client.get_collection(name=collection)
        serialized = [self._serialize_metadata(dict(m)) for m in metadatas]
        access_fields = {
            "access_count",
            "retrieved_count",
            "used_count",
            "last_accessed_at",
            "last_retrieved_at",
            "last_used_at",
        }
        access_only = all(
            all(key in access_fields or key.startswith("ac_") for key in metadata) for metadata in serialized
        )
        if access_only:
            coll._client._update(  # noqa: SLF001 - avoids irrelevant Chroma sparse-schema work
                collection_id=coll.id,
                ids=ids,
                embeddings=None,
                metadatas=cast(Any, serialized),
                documents=None,
                uris=None,
                tenant=coll.tenant,
                database=coll.database,
            )
        else:
            coll.update(ids=ids, metadatas=cast(Any, serialized))
        logger.debug("Updated metadata for %d documents in '%s'", len(ids), collection)
        return True

    def update_metadata(self, collection: str, ids: list[str], metadatas: list[dict[str, str | int | float]]) -> bool:
        """Update metadata for existing documents."""
        if not ids:
            return True
        try:
            return self._update_metadata_once(collection, ids, metadatas)
        except Exception as e:
            self._report_chroma_error(collection, e, "update_metadata")
            msg = "Failed to update metadata in '%s': owner=%s db_path=%s error=%s"
            msg_args = (collection, self._owner_label(), self.persist_dir, e)
            if _is_missing_collection_error(e):
                log_missing_collection_once(collection, msg, *msg_args)
            else:
                logger.warning(msg, *msg_args)
            return False

    def _get_by_metadata_once(
        self,
        collection: str,
        where: dict[str, str | int | float],
        limit: int = 20,
    ) -> list[SearchResult]:
        try:
            coll = self.client.get_collection(name=collection)
        except Exception as exc:
            # A fresh runtime can recall shared memories before their first
            # indexing pass. The root memory service calls this method directly.
            if _is_missing_collection_error(exc):
                logger.debug("Metadata recall before collection '%s' is initialized", collection)
                return []
            raise

        # ChromaDB: empty dict means no filter; pass None for "return all"
        where_arg: Any = where if where else None

        data = coll.get(
            where=where_arg,
            limit=limit,
            include=["documents", "metadatas"],
        )

        search_results: list[SearchResult] = []
        ids = data.get("ids") or []
        documents = data.get("documents") or []
        metadatas = data.get("metadatas") or []

        for i, doc_id in enumerate(ids):
            content = documents[i] if i < len(documents) else ""
            meta_raw = metadatas[i] if i < len(metadatas) else {}
            meta_dict: dict[str, str | int | float | list[str]] = cast(Any, dict(meta_raw)) if meta_raw else {}
            doc = Document(id=doc_id, content=content, metadata=meta_dict)
            search_results.append(SearchResult(document=doc, score=1.0))

        logger.debug(
            "get_by_metadata returned %d results from collection '%s'",
            len(search_results),
            collection,
        )
        return search_results

    def get_by_metadata(
        self,
        collection: str,
        where: dict[str, str | int | float],
        limit: int = 20,
    ) -> list[SearchResult]:
        """Retrieve documents by metadata filter without embedding search."""
        try:
            return self._get_by_metadata_once(collection, where, limit)
        except Exception as e:
            self._report_chroma_error(collection, e, "get_by_metadata")
            logger.debug(
                "get_by_metadata failed for collection '%s': owner=%s db_path=%s error=%s",
                collection,
                self._owner_label(),
                self.persist_dir,
                e,
            )
            return []

    def _get_all_once(self, collection: str, limit: int = 100_000) -> list[SearchResult]:
        """Retrieve collection records without relying on empty-filter semantics."""
        try:
            coll = self.client.get_collection(name=collection)
        except Exception as exc:
            if _is_missing_collection_error(exc):
                logger.debug("All-document recall before collection '%s' is initialized", collection)
                return []
            raise

        data = coll.get(limit=limit, include=["documents", "metadatas"])
        documents = data.get("documents") or []
        metadatas = data.get("metadatas") or []
        results: list[SearchResult] = []
        for i, doc_id in enumerate(data.get("ids") or []):
            content = documents[i] if i < len(documents) else ""
            meta_raw = metadatas[i] if i < len(metadatas) else {}
            metadata = cast(Any, dict(meta_raw)) if meta_raw else {}
            results.append(SearchResult(Document(id=doc_id, content=content, metadata=metadata), score=1.0))
        return results

    def get_all(self, collection: str, limit: int = 100_000) -> list[SearchResult]:
        """Retrieve documents without passing a metadata filter to Chroma."""
        try:
            return self._get_all_once(collection, limit)
        except Exception as exc:
            self._report_chroma_error(collection, exc, "get_all")
            logger.debug(
                "get_all failed for collection '%s': owner=%s db_path=%s error=%s",
                collection,
                self._owner_label(),
                self.persist_dir,
                exc,
            )
            return []

    def _count_once(self, collection: str) -> int:
        try:
            coll = self.client.get_collection(name=collection)
        except Exception as exc:
            if _is_missing_collection_error(exc):
                return 0
            raise
        return int(coll.count())

    def count(self, collection: str) -> int | None:
        """Return the collection size, or None if the store is unavailable."""
        try:
            return self._count_once(collection)
        except Exception as exc:
            self._report_chroma_error(collection, exc, "count")
            logger.debug(
                "count failed for collection '%s': owner=%s db_path=%s error=%s",
                collection,
                self._owner_label(),
                self.persist_dir,
                exc,
            )
            return None

    def _get_by_ids_once(self, collection: str, ids: list[str]) -> list[Document]:
        coll = self.client.get_collection(name=collection)
        data = coll.get(ids=ids, include=["documents", "metadatas"])
        documents: list[Document] = []
        for i, doc_id in enumerate(data["ids"]):
            content = data["documents"][i] if data["documents"] and i < len(data["documents"]) else ""
            meta_raw = data["metadatas"][i] if data["metadatas"] and i < len(data["metadatas"]) else {}
            meta_dict = cast(Any, dict(meta_raw)) if meta_raw else {}
            documents.append(Document(id=doc_id, content=content, metadata=meta_dict))
        return documents

    def get_by_ids(self, collection: str, ids: list[str]) -> list[Document]:
        """Fetch documents by their IDs."""
        if not ids:
            return []
        try:
            return self._get_by_ids_once(collection, ids)
        except Exception as e:
            self._report_chroma_error(collection, e, "get_by_ids")
            logger.debug(
                "get_by_ids failed for collection '%s': owner=%s db_path=%s error=%s",
                collection,
                self._owner_label(),
                self.persist_dir,
                e,
            )
            return []

    @staticmethod
    def _serialize_metadata(
        metadata: dict[str, str | int | float | list[str]],
    ) -> dict[str, str | int | float]:
        """Serialize metadata for ChromaDB (converts lists to JSON strings)."""
        serialized: dict[str, str | int | float] = {}
        for key, value in metadata.items():
            if isinstance(value, list):
                # ChromaDB doesn't support list values directly
                import json

                serialized[key] = json.dumps(value, ensure_ascii=False)
            else:
                serialized[key] = value
        return serialized


def _clear_chroma_system_cache() -> None:
    """Drop stopped PersistentClient systems before a MemoryService reopens."""
    try:
        from chromadb.api.client import SharedSystemClient

        SharedSystemClient.clear_system_cache()
    except Exception:
        logger.debug("Failed to clear ChromaDB system cache", exc_info=True)


def create_chroma_vector_store(
    *,
    persist_dir: Path,
    anima_name: str | None = None,
    allow_direct: _DirectAccessCapability | None = None,
) -> ChromaVectorStore:
    """Create a guarded direct Chroma vector store."""
    return ChromaVectorStore(
        persist_dir=persist_dir,
        anima_name=anima_name,
        allow_direct=allow_direct,
    )
