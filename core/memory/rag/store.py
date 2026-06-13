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
import threading
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeVar, cast

logger = logging.getLogger("animaworks.rag.store")
_T = TypeVar("_T")

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
    def list_collections(self) -> list[str]:
        """List all collection names."""

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
    def get_by_ids(
        self,
        collection: str,
        ids: list[str],
    ) -> list[Document]:
        """Fetch documents (with metadata) by their IDs."""


# ── ChromaDB implementation ─────────────────────────────────────────


class ChromaVectorStore(VectorStore):
    """ChromaDB-backed vector store.

    Stores embeddings in SQLite at ~/.animaworks/vectordb/chroma.sqlite3
    """

    _self_heal_reset_lock = threading.Lock()

    def __init__(self, persist_dir: Path | None = None, anima_name: str | None = None) -> None:
        """Initialize ChromaDB client.

        Args:
            persist_dir: Directory for ChromaDB persistence
                        (defaults to ~/.animaworks/vectordb)
            anima_name: Owner anima for repair signal attribution.
        """
        from core.memory.rag.direct_access import require_direct_chroma_allowed

        require_direct_chroma_allowed()

        import chromadb

        if persist_dir is None:
            from core.paths import get_data_dir

            persist_dir = get_data_dir() / "vectordb"

        if persist_dir.parent.exists():
            persist_dir.mkdir(exist_ok=True)
        else:
            logger.warning(
                "Parent directory does not exist for vectordb; creating with parents: %s",
                persist_dir,
            )
            persist_dir.mkdir(parents=True, exist_ok=True)

        from core.memory.rag.sqlite_health import configure_chroma_sqlite_pragmas, prepare_chroma_sqlite_for_startup

        prepare_chroma_sqlite_for_startup(persist_dir, anima_name=anima_name)
        logger.debug("Initializing ChromaDB at %s", persist_dir)
        self.client = chromadb.PersistentClient(path=str(persist_dir))
        self.persist_dir = persist_dir
        self.anima_name = anima_name
        self._closed = False
        post_init_pragma = configure_chroma_sqlite_pragmas(persist_dir)
        if not post_init_pragma.ok and post_init_pragma.status != "missing":
            logger.warning(
                "Failed to configure Chroma SQLite pragmas after client init at %s: status=%s detail=%s",
                post_init_pragma.db_path,
                post_init_pragma.status,
                post_init_pragma.error or post_init_pragma.details,
            )

    def close(self) -> None:
        """Close the underlying Chroma client if supported.

        A closed store instance is not reusable; callers should drop it and
        obtain a fresh singleton/store after reset.
        """
        if self._closed:
            return
        self._closed = True
        close = getattr(self.client, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                logger.debug("Failed to close ChromaDB client at %s", self.persist_dir, exc_info=True)

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

    def _with_self_heal(
        self,
        operation: str,
        collection: str,
        action: Callable[[ChromaVectorStore], _T],
    ) -> _T:
        try:
            return action(self)
        except Exception as error:
            from core.memory.rag.repair_utils import classify_corruption_error

            reason = classify_corruption_error(error)
            if reason is None:
                raise
            self._report_chroma_error(collection, error, operation)
            fresh_store = self._reset_for_self_heal(operation, collection, reason, error)
            if fresh_store is None:
                raise
            try:
                return action(fresh_store)
            except Exception as retry_error:
                fresh_store._report_chroma_error(collection, retry_error, operation)
                logger.warning(
                    "ChromaDB self-heal retry failed during %s: owner=%s db_path=%s collection=%s error=%s",
                    operation,
                    fresh_store._owner_label(),
                    fresh_store.persist_dir,
                    collection,
                    retry_error,
                )
                raise

    def _reset_for_self_heal(
        self,
        operation: str,
        collection: str,
        reason: str,
        error: Exception,
    ) -> ChromaVectorStore | None:
        with type(self)._self_heal_reset_lock:
            logger.warning(
                "ChromaDB corruption detected during %s; resetting vector store before one retry: "
                "owner=%s db_path=%s collection=%s reason=%s error=%s",
                operation,
                self._owner_label(),
                self.persist_dir,
                collection,
                reason,
                error,
            )
            try:
                from core.memory.rag.singleton import reset_vector_store

                reset_vector_store(self._anima_name())
            except Exception:
                logger.debug(
                    "Failed to reset singleton vector store during self-heal: owner=%s db_path=%s",
                    self._owner_label(),
                    self.persist_dir,
                    exc_info=True,
                )
            self.close()
            try:
                return type(self)(persist_dir=self.persist_dir, anima_name=self._anima_name())
            except Exception as recreate_error:
                logger.warning(
                    "Failed to recreate ChromaDB vector store after reset: owner=%s db_path=%s "
                    "collection=%s operation=%s error=%s",
                    self._owner_label(),
                    self.persist_dir,
                    collection,
                    operation,
                    recreate_error,
                )
                return None

    def _create_collection_once(self, name: str) -> bool:
        self.client.create_collection(
            name=name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("Created collection '%s' (space=cosine)", name)
        return True

    def create_collection(self, name: str) -> bool:
        """Create a new collection or get existing one."""
        try:
            return self._with_self_heal("create_collection", name, lambda store: store._create_collection_once(name))
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
            return self._with_self_heal("delete_collection", name, lambda store: store._delete_collection_once(name))
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

    def list_collections(self) -> list[str]:
        """List all collections."""
        try:
            return self._with_self_heal(
                "list_collections",
                "<list_collections>",
                lambda store: store._list_collections_once(),
            )
        except Exception as e:
            self._report_chroma_error("<list_collections>", e, "list_collections")
            logger.warning(
                "Failed to list collections: owner=%s db_path=%s error=%s",
                self._owner_label(),
                self.persist_dir,
                e,
            )
            raise

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
            return self._with_self_heal("upsert", collection, lambda store: store._upsert_once(collection, documents))
        except Exception as e:
            self._report_chroma_error(collection, e, "upsert")
            logger.warning(
                "Failed to upsert %d documents to '%s': owner=%s db_path=%s error=%s",
                len(documents),
                collection,
                self._owner_label(),
                self.persist_dir,
                e,
            )
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
            return self._with_self_heal(
                "query",
                collection,
                lambda store: store._query_once(collection, embedding, top_k, filter_metadata),
            )
        except Exception as e:
            self._report_chroma_error(collection, e, "query")
            logger.warning(
                "ChromaDB query failed for collection '%s': owner=%s db_path=%s error=%s",
                collection,
                self._owner_label(),
                self.persist_dir,
                e,
            )
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
            return self._with_self_heal(
                "delete_documents",
                collection,
                lambda store: store._delete_documents_once(collection, ids),
            )
        except Exception as e:
            self._report_chroma_error(collection, e, "delete_documents")
            logger.warning(
                "Failed to delete documents from '%s': owner=%s db_path=%s error=%s",
                collection,
                self._owner_label(),
                self.persist_dir,
                e,
            )
            return False

    def _update_metadata_once(
        self,
        collection: str,
        ids: list[str],
        metadatas: list[dict[str, str | int | float]],
    ) -> bool:
        coll = self.client.get_collection(name=collection)
        serialized = [self._serialize_metadata(dict(m)) for m in metadatas]
        coll.update(ids=ids, metadatas=cast(Any, serialized))
        logger.debug("Updated metadata for %d documents in '%s'", len(ids), collection)
        return True

    def update_metadata(self, collection: str, ids: list[str], metadatas: list[dict[str, str | int | float]]) -> bool:
        """Update metadata for existing documents."""
        if not ids:
            return True
        try:
            return self._with_self_heal(
                "update_metadata",
                collection,
                lambda store: store._update_metadata_once(collection, ids, metadatas),
            )
        except Exception as e:
            self._report_chroma_error(collection, e, "update_metadata")
            logger.warning(
                "Failed to update metadata in '%s': owner=%s db_path=%s error=%s",
                collection,
                self._owner_label(),
                self.persist_dir,
                e,
            )
            return False

    def _get_by_metadata_once(
        self,
        collection: str,
        where: dict[str, str | int | float],
        limit: int = 20,
    ) -> list[SearchResult]:
        coll = self.client.get_collection(name=collection)

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
            return self._with_self_heal(
                "get_by_metadata",
                collection,
                lambda store: store._get_by_metadata_once(collection, where, limit),
            )
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
            return self._with_self_heal(
                "get_by_ids",
                collection,
                lambda store: store._get_by_ids_once(collection, ids),
            )
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

    def needs_cosine_migration(self) -> list[str]:
        """Return collection names still using L2 (non-cosine) distance."""
        l2_collections: list[str] = []
        for name in self.list_collections():
            try:
                coll = self.client.get_collection(name=name)
                space = (coll.metadata or {}).get("hnsw:space", "l2")
                if space != "cosine":
                    l2_collections.append(name)
            except Exception:
                logger.debug("Skipping collection '%s' during migration check", name)
        return l2_collections

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


def create_chroma_vector_store(
    *,
    persist_dir: Path | None = None,
    anima_name: str | None = None,
) -> ChromaVectorStore:
    """Create a guarded direct Chroma vector store."""
    if anima_name is None:
        return ChromaVectorStore(persist_dir=persist_dir)
    return ChromaVectorStore(persist_dir=persist_dir, anima_name=anima_name)
