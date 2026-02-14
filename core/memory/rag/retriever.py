from __future__ import annotations
# AnimaWorks - Digital Person Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Hybrid retrieval system combining vector search, BM25, and temporal decay.

Implements:
- Vector similarity search (semantic)
- BM25 keyword search (exact term matching)
- RRF (Reciprocal Rank Fusion) score combination
- Temporal decay scoring (newer documents ranked higher)
"""

import logging
import math
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

logger = logging.getLogger("animaworks.rag.retriever")

# ── Configuration ───────────────────────────────────────────────────

# RRF parameter (standard value from literature)
RRF_K = 60

# Hybrid search weights
WEIGHT_VECTOR = 0.5
WEIGHT_BM25 = 0.3
WEIGHT_RECENCY = 0.2

# Temporal decay half-life (days)
RECENCY_HALF_LIFE_DAYS = 30.0


# ── Data structures ─────────────────────────────────────────────────


@dataclass
class RetrievalResult:
    """A retrieved document with combined score."""

    doc_id: str
    content: str
    score: float
    metadata: dict[str, str | int | float | list[str]]
    source_scores: dict[str, float]  # Debug info: individual scores


# ── HybridRetriever ─────────────────────────────────────────────────


class HybridRetriever:
    """Hybrid search combining vector similarity, BM25, and recency.

    Uses RRF (Reciprocal Rank Fusion) to combine rankings from multiple
    search methods, then applies temporal decay scoring.

    Phase 3: Supports spreading activation via knowledge graph.
    """

    def __init__(
        self,
        vector_store,  # VectorStore instance
        indexer,  # MemoryIndexer instance
        knowledge_dir: Path,
    ) -> None:
        """Initialize hybrid retriever.

        Args:
            vector_store: VectorStore instance
            indexer: MemoryIndexer instance (for embedding generation)
            knowledge_dir: Path to knowledge directory (for BM25 search)
        """
        self.vector_store = vector_store
        self.indexer = indexer
        self.knowledge_dir = knowledge_dir
        self._knowledge_graph = None  # Lazy initialization

    # ── Main search API ─────────────────────────────────────────────

    def search(
        self,
        query: str,
        person_name: str,
        memory_type: str = "knowledge",
        top_k: int = 3,
        enable_spreading_activation: bool = False,
    ) -> list[RetrievalResult]:
        """Perform hybrid search.

        Args:
            query: Search query text
            person_name: Person name (for collection selection)
            memory_type: Memory type (knowledge, episodes, etc.)
            top_k: Number of results to return
            enable_spreading_activation: Enable graph-based spreading activation (Phase 3)

        Returns:
            List of retrieval results sorted by combined score
        """
        logger.debug(
            "Hybrid search: query='%s', person=%s, type=%s, top_k=%d, spreading=%s",
            query,
            person_name,
            memory_type,
            top_k,
            enable_spreading_activation,
        )

        # Execute search methods in parallel (conceptually; Python async optional)
        vector_results = self._vector_search(query, person_name, memory_type, top_k * 2)
        bm25_results = self._bm25_search(query, memory_type, top_k * 2)

        # Combine with RRF
        combined = self._combine_with_rrf(vector_results, bm25_results)

        # Apply temporal decay
        final = self._apply_temporal_decay(combined)

        # Sort by final score and return top K
        final.sort(key=lambda r: r.score, reverse=True)
        initial_results = final[:top_k]

        # Phase 3: Apply spreading activation if enabled
        if enable_spreading_activation and memory_type == "knowledge":
            try:
                expanded = self._apply_spreading_activation(initial_results, person_name)
                return expanded
            except Exception as e:
                logger.warning("Spreading activation failed, returning initial results: %s", e)
                return initial_results

        return initial_results

    # ── Search methods ──────────────────────────────────────────────

    def _vector_search(
        self,
        query: str,
        person_name: str,
        memory_type: str,
        top_k: int,
    ) -> list[tuple[str, float, dict]]:
        """Perform vector similarity search.

        Returns:
            List of (doc_id, score, metadata) tuples
        """
        # Generate query embedding
        embedding = self.indexer._generate_embeddings([query])[0]

        # Query vector store
        collection_name = f"{person_name}_{memory_type}"
        results = self.vector_store.query(
            collection=collection_name,
            embedding=embedding,
            top_k=top_k,
        )

        return [
            (r.document.id, r.score, r.document.metadata)
            for r in results
        ]

    def _bm25_search(
        self,
        query: str,
        memory_type: str,
        top_k: int,
    ) -> list[tuple[str, float, dict]]:
        """Perform BM25 keyword search using ripgrep.

        Returns:
            List of (doc_id, score, metadata) tuples
        """
        if memory_type == "knowledge":
            search_dir = self.knowledge_dir
        else:
            # For other types, use person_dir/{memory_type}
            search_dir = self.knowledge_dir.parent / memory_type

        if not search_dir.is_dir():
            logger.debug("Search directory not found: %s", search_dir)
            return []

        # Extract keywords from query
        keywords = self._extract_keywords(query)
        if not keywords:
            return []

        # Build ripgrep pattern (OR of keywords)
        escaped_keywords = [re.escape(kw) for kw in keywords[:5]]
        pattern = "|".join(escaped_keywords)

        try:
            # Run ripgrep
            result = subprocess.run(
                [
                    "rg",
                    "--ignore-case",
                    "--count",  # Count matches per file
                    "--no-heading",
                    "--with-filename",
                    pattern,
                    str(search_dir),
                ],
                capture_output=True,
                text=True,
                timeout=2.0,
            )

            if result.returncode != 0 or not result.stdout:
                return []

            # Parse results (format: "filename:count")
            file_scores: dict[str, int] = {}
            for line in result.stdout.strip().splitlines():
                parts = line.rsplit(":", 1)
                if len(parts) == 2:
                    filename, count_str = parts
                    try:
                        file_scores[filename] = int(count_str)
                    except ValueError:
                        pass

            # Convert to (doc_id, score, metadata) format
            results: list[tuple[str, float, dict]] = []
            for filename, count in file_scores.items():
                # Normalize score (log scale for BM25-like behavior)
                score = math.log1p(count)  # log(1 + count)

                # Create pseudo doc_id (simplified - real implementation should use chunk IDs)
                doc_id = f"{self.indexer.person_name}/{memory_type}/{Path(filename).name}#0"

                metadata = {
                    "source_file": str(Path(filename).relative_to(search_dir)),
                    "bm25_match_count": count,
                }

                results.append((doc_id, score, metadata))

            # Sort by score and return top K
            results.sort(key=lambda r: r[1], reverse=True)
            return results[:top_k]

        except subprocess.TimeoutExpired:
            logger.warning("BM25 search timeout")
            return []
        except FileNotFoundError:
            logger.warning("ripgrep not found, BM25 search unavailable")
            return []
        except Exception as e:
            logger.warning("BM25 search failed: %s", e)
            return []

    # ── Score combination ───────────────────────────────────────────

    def _combine_with_rrf(
        self,
        vector_results: list[tuple[str, float, dict]],
        bm25_results: list[tuple[str, float, dict]],
    ) -> list[RetrievalResult]:
        """Combine search results using Reciprocal Rank Fusion.

        RRF formula: score(d) = Σ 1 / (k + rank_i(d))
        where k = 60 (standard), rank_i = rank in search method i
        """
        # Build rank dictionaries
        vector_ranks = {doc_id: i + 1 for i, (doc_id, _, _) in enumerate(vector_results)}
        bm25_ranks = {doc_id: i + 1 for i, (doc_id, _, _) in enumerate(bm25_results)}

        # Collect all unique doc IDs
        all_doc_ids = set(vector_ranks.keys()) | set(bm25_ranks.keys())

        # Build metadata map (prefer vector metadata as it's more complete)
        metadata_map: dict[str, dict] = {}
        for doc_id, _, meta in vector_results:
            metadata_map[doc_id] = meta
        for doc_id, _, meta in bm25_results:
            if doc_id not in metadata_map:
                metadata_map[doc_id] = meta

        # Compute RRF scores
        results: list[RetrievalResult] = []
        for doc_id in all_doc_ids:
            vector_rank = vector_ranks.get(doc_id, 0)
            bm25_rank = bm25_ranks.get(doc_id, 0)

            # RRF score components
            vector_score = (1.0 / (RRF_K + vector_rank)) if vector_rank > 0 else 0.0
            bm25_score = (1.0 / (RRF_K + bm25_rank)) if bm25_rank > 0 else 0.0

            # Weighted combination
            combined_score = (
                WEIGHT_VECTOR * vector_score + WEIGHT_BM25 * bm25_score
            )

            # Get content (simplified - real implementation should fetch from store)
            content = metadata_map.get(doc_id, {}).get("content", "")

            results.append(
                RetrievalResult(
                    doc_id=doc_id,
                    content=content,
                    score=combined_score,
                    metadata=metadata_map.get(doc_id, {}),
                    source_scores={
                        "vector": vector_score,
                        "bm25": bm25_score,
                        "combined": combined_score,
                    },
                )
            )

        return results

    def _apply_temporal_decay(
        self,
        results: list[RetrievalResult],
    ) -> list[RetrievalResult]:
        """Apply temporal decay to scores based on document age.

        Uses exponential decay: decay_factor = 0.5 ^ (age_days / half_life)
        """
        now = datetime.now()

        for result in results:
            # Extract update timestamp from metadata
            updated_at_str = result.metadata.get("updated_at")
            if not updated_at_str:
                # No timestamp - use neutral decay (0.5)
                decay_factor = 0.5
            else:
                try:
                    updated_at = datetime.fromisoformat(str(updated_at_str))
                    age_days = (now - updated_at).total_seconds() / 86400.0

                    # Exponential decay
                    decay_factor = 0.5 ** (age_days / RECENCY_HALF_LIFE_DAYS)
                except (ValueError, TypeError):
                    decay_factor = 0.5

            # Apply decay with weight
            recency_score = WEIGHT_RECENCY * decay_factor
            result.score = result.score + recency_score
            result.source_scores["recency"] = recency_score

        return results

    # ── Spreading activation (Phase 3) ──────────────────────────────

    def _apply_spreading_activation(
        self,
        initial_results: list[RetrievalResult],
        person_name: str,
    ) -> list[RetrievalResult]:
        """Apply spreading activation to expand search results.

        Args:
            initial_results: Initial hybrid search results
            person_name: Person name

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
                self._knowledge_graph.build_graph(person_name, self.knowledge_dir)
            except Exception as e:
                logger.warning("Failed to build knowledge graph: %s", e)
                return initial_results

        # Expand results using graph
        return self._knowledge_graph.expand_search_results(initial_results)

    # ── Helpers ─────────────────────────────────────────────────────

    @staticmethod
    def _extract_keywords(query: str) -> list[str]:
        """Extract keywords from query (same as priming.py)."""
        stopwords = {
            "の", "に", "は", "を", "が", "で", "と", "から", "まで",
            "も", "や", "へ", "より", "など", "について",
            "the", "a", "an", "and", "or", "but", "in", "on", "at",
            "to", "for", "of", "with", "by", "from", "up", "about",
            "into", "through", "during", "it", "is", "are", "was",
            "were", "be", "been", "being", "have", "has", "had",
            "do", "does", "did", "will", "would", "should", "could",
        }

        # Split on whitespace and punctuation
        words = re.findall(r"[\w\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF]+", query)

        # Filter stopwords and short words
        keywords = [
            w for w in words
            if len(w) >= 2 and w.lower() not in stopwords
        ]

        return keywords[:10]
