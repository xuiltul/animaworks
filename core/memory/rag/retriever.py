from __future__ import annotations
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Dense vector retrieval system with temporal decay and spreading activation.

Implements:
- Dense vector similarity search (semantic)
- Temporal decay scoring (newer documents ranked higher)
- Spreading activation via knowledge graph (optional)
"""

import logging
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from core.time_utils import ensure_aware, now_iso, now_jst

logger = logging.getLogger("animaworks.rag.retriever")

# ── Configuration ───────────────────────────────────────────────────

WEIGHT_RECENCY = 0.2

# Temporal decay half-life (days)
RECENCY_HALF_LIFE_DAYS = 30.0

WEIGHT_FREQUENCY = 0.1


# ── Data structures ─────────────────────────────────────────────────


@dataclass
class RetrievalResult:
    """A retrieved document with combined score."""

    doc_id: str
    content: str
    score: float
    metadata: dict[str, str | int | float | list[str]]
    source_scores: dict[str, float]  # Debug info: individual scores


# ── MemoryRetriever ────────────────────────────────────────────────


class MemoryRetriever:
    """Dense vector search with temporal decay and spreading activation.

    Pipeline:
      Query → Dense Vector Search → Temporal Decay → Sort → Spreading Activation → Results
    """

    def __init__(
        self,
        vector_store,  # VectorStore instance
        indexer,  # MemoryIndexer instance
        knowledge_dir: Path,
    ) -> None:
        """Initialize memory retriever.

        Args:
            vector_store: VectorStore instance
            indexer: MemoryIndexer instance (for embedding generation)
            knowledge_dir: Path to knowledge directory (for spreading activation)
        """
        self.vector_store = vector_store
        self.indexer = indexer
        self.knowledge_dir = knowledge_dir
        self._knowledge_graph = None  # Lazy initialization

    # ── Main search API ─────────────────────────────────────────────

    def search(
        self,
        query: str,
        anima_name: str,
        memory_type: str = "knowledge",
        top_k: int = 3,
        enable_spreading_activation: bool = False,
        *,
        include_shared: bool = False,
        include_superseded: bool = False,
    ) -> list[RetrievalResult]:
        """Perform dense vector search with temporal decay.

        Args:
            query: Search query text
            anima_name: Anima name (for collection selection)
            memory_type: Memory type (knowledge, episodes, etc.)
            top_k: Number of results to return
            enable_spreading_activation: Enable graph-based spreading activation
            include_shared: Also search ``shared_common_knowledge`` collection
                and merge results by score.
            include_superseded: If False (default), exclude knowledge that has
                been superseded (``valid_until`` is non-empty). Set to True
                to include all knowledge regardless of validity.

        Returns:
            List of retrieval results sorted by combined score
        """
        logger.debug(
            "Vector search: query='%s', anima=%s, type=%s, top_k=%d, "
            "spreading=%s, shared=%s",
            query,
            anima_name,
            memory_type,
            top_k,
            enable_spreading_activation,
            include_shared,
        )

        # Build metadata filter for superseded knowledge exclusion
        filter_metadata: dict[str, str | int | float] | None = None
        if not include_superseded and memory_type == "knowledge":
            filter_metadata = {"valid_until": ""}

        # 1. Dense Vector search (personal collection)
        vector_results = self._vector_search(
            query, anima_name, memory_type, top_k * 2,
            filter_metadata=filter_metadata,
        )

        # 1b. Shared collection search (if requested)
        _SHARED_COLLECTION_MAP: dict[str, str] = {
            "knowledge": "shared_common_knowledge",
            "skills": "shared_common_skills",
        }
        if include_shared:
            shared_collection = _SHARED_COLLECTION_MAP.get(memory_type)
            if shared_collection:
                shared_results = self._vector_search_collection(
                    query, shared_collection, top_k * 2,
                    filter_metadata=filter_metadata,
                )
                vector_results.extend(shared_results)

        # 2. Convert to RetrievalResult
        results = [
            RetrievalResult(
                doc_id=doc_id,
                content=content,
                score=score,
                metadata=metadata,
                source_scores={"vector": score},
            )
            for doc_id, content, score, metadata in vector_results
        ]

        # 3. Apply temporal decay and frequency boost
        results = self._apply_score_adjustments(results)

        # 4. Sort & top_k
        results.sort(key=lambda r: r.score, reverse=True)
        initial_results = results[:top_k]

        # 5. Apply spreading activation if enabled
        if enable_spreading_activation and memory_type in ("knowledge", "episodes"):
            try:
                expanded = self._apply_spreading_activation(initial_results, anima_name)
                return expanded
            except Exception as e:
                logger.warning("Spreading activation failed, returning initial results: %s", e)
                return initial_results

        return initial_results

    # ── Search methods ──────────────────────────────────────────────

    def _vector_search(
        self,
        query: str,
        anima_name: str,
        memory_type: str,
        top_k: int,
        filter_metadata: dict[str, str | int | float] | None = None,
    ) -> list[tuple[str, str, float, dict]]:
        """Perform vector similarity search on an anima collection.

        Returns:
            List of (doc_id, content, score, metadata) tuples
        """
        collection_name = f"{anima_name}_{memory_type}"
        return self._vector_search_collection(
            query, collection_name, top_k, filter_metadata=filter_metadata,
        )

    def _vector_search_collection(
        self,
        query: str,
        collection_name: str,
        top_k: int,
        filter_metadata: dict[str, str | int | float] | None = None,
    ) -> list[tuple[str, str, float, dict]]:
        """Perform vector similarity search on a named collection.

        Args:
            query: Search query text
            collection_name: Name of the ChromaDB collection
            top_k: Number of results to return
            filter_metadata: Optional metadata filters (exact match).
                Used to exclude superseded knowledge via ``valid_until``.

        Returns:
            List of (doc_id, content, score, metadata) tuples
        """
        # Generate query embedding
        embedding = self.indexer._generate_embeddings([query])[0]

        # Query vector store
        results = self.vector_store.query(
            collection=collection_name,
            embedding=embedding,
            top_k=top_k,
            filter_metadata=filter_metadata,
        )

        return [
            (r.document.id, r.document.content, r.score, r.document.metadata)
            for r in results
        ]

    # ── Score adjustment ────────────────────────────────────────────

    def _apply_score_adjustments(
        self,
        results: list[RetrievalResult],
    ) -> list[RetrievalResult]:
        """Apply temporal decay and frequency boost to scores.

        - Temporal decay: exponential decay based on document age
        - Frequency boost: log-scaled boost based on access count (Hebbian LTP)
        """
        now = now_jst()

        for result in results:
            # --- Temporal decay (existing) ---
            updated_at_str = result.metadata.get("updated_at")
            if not updated_at_str:
                decay_factor = 0.5
            else:
                try:
                    updated_at = ensure_aware(datetime.fromisoformat(str(updated_at_str)))
                    age_days = (now - updated_at).total_seconds() / 86400.0
                    decay_factor = 0.5 ** (age_days / RECENCY_HALF_LIFE_DAYS)
                except (ValueError, TypeError):
                    decay_factor = 0.5

            recency_score = WEIGHT_RECENCY * decay_factor
            result.score = result.score + recency_score
            result.source_scores["recency"] = recency_score

            # --- Frequency boost (new: Hebbian LTP analog) ---
            access_count = int(result.metadata.get("access_count", 0))
            frequency_boost = WEIGHT_FREQUENCY * math.log1p(access_count)
            result.score += frequency_boost
            result.source_scores["frequency"] = frequency_boost

        return results

    def record_access(self, results: list[RetrievalResult], anima_name: str) -> None:
        """Record access for retrieved results (LTP analog).

        Increments access_count and updates last_accessed_at for each result.
        Called when results are injected into the agent's context.
        """
        if not results:
            return

        now_iso_str = now_iso()
        updates_by_collection: dict[str, tuple[list[str], list[dict]]] = {}

        for r in results:
            memory_type = r.metadata.get("memory_type", "knowledge")
            source = r.metadata.get("anima", anima_name)
            collection = f"{source}_{memory_type}"
            if collection not in updates_by_collection:
                updates_by_collection[collection] = ([], [])
            ids, metas = updates_by_collection[collection]
            ids.append(r.doc_id)
            metas.append({
                "access_count": int(r.metadata.get("access_count", 0)) + 1,
                "last_accessed_at": now_iso_str,
            })

        for collection, (ids, metas) in updates_by_collection.items():
            try:
                self.vector_store.update_metadata(collection, ids, metas)
                logger.debug(
                    "Recorded access for %d chunks in %s", len(ids), collection
                )
            except Exception as e:
                logger.warning("Failed to record access for %s: %s", collection, e)

    # ── Spreading activation ────────────────────────────────────────

    def _apply_spreading_activation(
        self,
        initial_results: list[RetrievalResult],
        anima_name: str,
    ) -> list[RetrievalResult]:
        """Apply spreading activation to expand search results.

        Tries loading cached graph first, then falls back to full build.

        Args:
            initial_results: Initial search results
            anima_name: Anima name

        Returns:
            Expanded results with activated neighbors
        """
        # Lazy initialization of knowledge graph
        if self._knowledge_graph is None:
            try:
                from core.memory.rag.graph import KnowledgeGraph

                self._knowledge_graph = KnowledgeGraph(
                    self.vector_store,
                    self.indexer,
                )

                # Try loading from cache first
                cache_dir = self.knowledge_dir.parent / "vectordb"
                if not self._knowledge_graph.load_graph(cache_dir):
                    # Cache miss: full build and save
                    self._knowledge_graph.build_graph(anima_name, self.knowledge_dir)
                    cache_dir.mkdir(parents=True, exist_ok=True)
                    self._knowledge_graph.save_graph(cache_dir)

            except Exception as e:
                logger.warning("Failed to initialize knowledge graph: %s", e)
                return initial_results

        # Expand results using graph
        return self._knowledge_graph.expand_search_results(initial_results)
