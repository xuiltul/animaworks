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
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

logger = logging.getLogger("animaworks.rag.store")

# ── Data structures ────────────────────────────────────────────────


@dataclass
class Document:
    """A document chunk with embedding and metadata."""

    id: str
    content: str
    embedding: list[float] | None = None
    metadata: dict[str, str | int | float | list[str]] = field(default_factory=dict)


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

    def __init__(self, persist_dir: Path | None = None) -> None:
        """Initialize ChromaDB client.

        Args:
            persist_dir: Directory for ChromaDB persistence
                        (defaults to ~/.animaworks/vectordb)
        """
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

        logger.debug("Initializing ChromaDB at %s", persist_dir)
        self.client = chromadb.PersistentClient(path=str(persist_dir))
        self.persist_dir = persist_dir

    def create_collection(self, name: str) -> bool:
        """Create a new collection or get existing one."""
        try:
            self.client.create_collection(
                name=name,
                metadata={"hnsw:space": "cosine"},
            )
            logger.info("Created collection '%s' (space=cosine)", name)
            return True
        except Exception as e:
            logger.debug("Collection '%s' already exists: %s", name, e)
            return True  # already-exists is not a failure

    def delete_collection(self, name: str) -> bool:
        """Delete a collection."""
        try:
            self.client.delete_collection(name=name)
            logger.info("Deleted collection '%s'", name)
            return True
        except Exception as e:
            logger.warning("Failed to delete collection '%s': %s", name, e)
            return False

    def list_collections(self) -> list[str]:
        """List all collections."""
        collections = self.client.list_collections()
        return [c.name for c in collections]

    def upsert(self, collection: str, documents: list[Document]) -> bool:
        """Upsert documents into collection."""
        if not documents:
            return True

        try:
            coll = self.client.get_or_create_collection(
                name=collection,
                metadata={"hnsw:space": "cosine"},
            )

            ids = [doc.id for doc in documents]
            contents = [doc.content for doc in documents]
            embeddings = [doc.embedding for doc in documents]
            metadatas = [self._serialize_metadata(doc.metadata) for doc in documents]

            if any(emb is None for emb in embeddings):
                logger.warning("Upsert to '%s' failed: missing embeddings", collection)
                return False

            coll.upsert(
                ids=ids,
                documents=contents,
                embeddings=cast(Any, embeddings),
                metadatas=cast(Any, metadatas),
            )

            logger.debug("Upserted %d documents to collection '%s'", len(documents), collection)
            return True
        except Exception as e:
            logger.warning("Failed to upsert %d documents to '%s': %s", len(documents), collection, e)
            return False

    def query(
        self,
        collection: str,
        embedding: list[float],
        top_k: int = 10,
        filter_metadata: dict[str, str | int | float] | None = None,
    ) -> list[SearchResult]:
        """Query by embedding similarity."""
        try:
            coll = self.client.get_collection(name=collection)
        except Exception as e:
            logger.warning("Collection '%s' not found: %s", collection, e)
            return []

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

    def delete_documents(self, collection: str, ids: list[str]) -> bool:
        """Delete documents by ID."""
        if not ids:
            return True

        try:
            coll = self.client.get_collection(name=collection)
            coll.delete(ids=ids)
            logger.debug("Deleted %d documents from collection '%s'", len(ids), collection)
            return True
        except Exception as e:
            logger.warning("Failed to delete documents from '%s': %s", collection, e)
            return False

    def update_metadata(self, collection: str, ids: list[str], metadatas: list[dict[str, str | int | float]]) -> bool:
        """Update metadata for existing documents."""
        if not ids:
            return True
        try:
            coll = self.client.get_collection(name=collection)
            serialized = [self._serialize_metadata(dict(m)) for m in metadatas]
            coll.update(ids=ids, metadatas=cast(Any, serialized))
            logger.debug("Updated metadata for %d documents in '%s'", len(ids), collection)
            return True
        except Exception as e:
            logger.warning("Failed to update metadata in '%s': %s", collection, e)
            return False

    def get_by_metadata(
        self,
        collection: str,
        where: dict[str, str | int | float],
        limit: int = 20,
    ) -> list[SearchResult]:
        """Retrieve documents by metadata filter without embedding search."""
        try:
            coll = self.client.get_collection(name=collection)
        except Exception as e:
            logger.debug("Collection '%s' not found for get_by_metadata: %s", collection, e)
            return []

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

    def get_by_ids(self, collection: str, ids: list[str]) -> list[Document]:
        """Fetch documents by their IDs."""
        if not ids:
            return []
        try:
            coll = self.client.get_collection(name=collection)
        except Exception as e:
            logger.debug("Collection '%s' not found for get_by_ids: %s", collection, e)
            return []
        data = coll.get(ids=ids, include=["documents", "metadatas"])
        documents: list[Document] = []
        for i, doc_id in enumerate(data["ids"]):
            content = data["documents"][i] if data["documents"] and i < len(data["documents"]) else ""
            meta_raw = data["metadatas"][i] if data["metadatas"] and i < len(data["metadatas"]) else {}
            meta_dict = cast(Any, dict(meta_raw)) if meta_raw else {}
            documents.append(Document(id=doc_id, content=content, metadata=meta_dict))
        return documents

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
