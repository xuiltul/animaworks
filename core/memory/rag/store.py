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
from datetime import datetime
from pathlib import Path

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
    def create_collection(self, name: str, dimension: int) -> None:
        """Create a new collection.

        Args:
            name: Collection name (e.g., "sakura_knowledge")
            dimension: Embedding dimension (e.g., 384 for multilingual-e5-small)
        """

    @abstractmethod
    def delete_collection(self, name: str) -> None:
        """Delete a collection and all its documents."""

    @abstractmethod
    def list_collections(self) -> list[str]:
        """List all collection names."""

    @abstractmethod
    def upsert(self, collection: str, documents: list[Document]) -> None:
        """Insert or update documents in a collection.

        Args:
            collection: Collection name
            documents: List of documents with embeddings
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
    def delete_documents(self, collection: str, ids: list[str]) -> None:
        """Delete specific documents by ID."""

    @abstractmethod
    def update_metadata(
        self, collection: str, ids: list[str], metadatas: list[dict[str, str | int | float]]
    ) -> None:
        """Update metadata for existing documents without re-embedding."""


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

        persist_dir.mkdir(parents=True, exist_ok=True)

        logger.debug("Initializing ChromaDB at %s", persist_dir)
        self.client = chromadb.PersistentClient(path=str(persist_dir))
        self.persist_dir = persist_dir

    def create_collection(self, name: str, dimension: int) -> None:
        """Create a new collection or get existing one."""
        try:
            self.client.create_collection(
                name=name,
                metadata={"dimension": dimension},
            )
            logger.info("Created collection '%s' (dimension=%d)", name, dimension)
        except Exception as e:
            # Collection already exists
            logger.debug("Collection '%s' already exists: %s", name, e)

    def delete_collection(self, name: str) -> None:
        """Delete a collection."""
        try:
            self.client.delete_collection(name=name)
            logger.info("Deleted collection '%s'", name)
        except Exception as e:
            logger.warning("Failed to delete collection '%s': %s", name, e)

    def list_collections(self) -> list[str]:
        """List all collections."""
        collections = self.client.list_collections()
        return [c.name for c in collections]

    def upsert(self, collection: str, documents: list[Document]) -> None:
        """Upsert documents into collection."""
        if not documents:
            return

        # Get or create collection
        coll = self.client.get_or_create_collection(name=collection)

        # Prepare batch data
        ids = [doc.id for doc in documents]
        contents = [doc.content for doc in documents]
        embeddings = [doc.embedding for doc in documents]
        metadatas = [self._serialize_metadata(doc.metadata) for doc in documents]

        # Validate embeddings
        if any(emb is None for emb in embeddings):
            raise ValueError("All documents must have embeddings for upsert")

        # Upsert to ChromaDB
        coll.upsert(
            ids=ids,
            documents=contents,
            embeddings=embeddings,
            metadatas=metadatas,
        )

        logger.debug(
            "Upserted %d documents to collection '%s'", len(documents), collection
        )

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
            where=where,
        )

        # Parse results
        search_results: list[SearchResult] = []
        if results["ids"] and results["ids"][0]:
            for i, doc_id in enumerate(results["ids"][0]):
                doc = Document(
                    id=doc_id,
                    content=results["documents"][0][i] if results["documents"] else "",
                    metadata=results["metadatas"][0][i] if results["metadatas"] else {},
                )
                score = 1.0 - results["distances"][0][i]  # Convert distance to similarity
                search_results.append(SearchResult(document=doc, score=score))

        logger.debug(
            "Query returned %d results from collection '%s'",
            len(search_results),
            collection,
        )
        return search_results

    def delete_documents(self, collection: str, ids: list[str]) -> None:
        """Delete documents by ID."""
        if not ids:
            return

        try:
            coll = self.client.get_collection(name=collection)
            coll.delete(ids=ids)
            logger.debug("Deleted %d documents from collection '%s'", len(ids), collection)
        except Exception as e:
            logger.warning("Failed to delete documents from '%s': %s", collection, e)

    def update_metadata(
        self, collection: str, ids: list[str], metadatas: list[dict[str, str | int | float]]
    ) -> None:
        """Update metadata for existing documents."""
        if not ids:
            return
        try:
            coll = self.client.get_collection(name=collection)
            serialized = [self._serialize_metadata(m) for m in metadatas]
            coll.update(ids=ids, metadatas=serialized)
            logger.debug("Updated metadata for %d documents in '%s'", len(ids), collection)
        except Exception as e:
            logger.warning("Failed to update metadata in '%s': %s", collection, e)

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
