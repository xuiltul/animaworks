from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Hybrid search — BM25 + Vector + BFS + Cross-encoder reranking."""

import asyncio
import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.memory.graph.driver import Neo4jDriver

logger = logging.getLogger(__name__)


# ── HybridSearch ──────────────────────────────────────────────────────────


class HybridSearch:
    """4-source hybrid search with RRF merge and cross-encoder reranking."""

    def __init__(
        self,
        driver: Neo4jDriver,
        group_id: str,
        *,
        cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-12-v2",
    ) -> None:
        self._driver = driver
        self._group_id = group_id
        self._ce_model = cross_encoder_model

    # ── Public API ────────────────────────────────────────────────────────

    async def search(
        self,
        query: str,
        *,
        scope: str = "fact",
        limit: int = 10,
        as_of_time: str | None = None,
        query_embedding: list[float] | None = None,
    ) -> list[dict]:
        """Execute hybrid search across 4 sources.

        Args:
            query: Natural language query.
            scope: "fact", "entity", or "episode".
            limit: Max results to return.
            as_of_time: ISO datetime for temporal filter (default: now).
            query_embedding: Pre-computed query embedding for vector search.

        Returns:
            List of result dicts sorted by relevance.
        """
        if not query or not query.strip():
            raise ValueError("Search query must not be empty")

        if as_of_time is None:
            as_of_time = datetime.now(tz=UTC).isoformat()

        results = await asyncio.gather(
            self._vector_search(query, scope, as_of_time, query_embedding),
            self._fulltext_search(query, scope, as_of_time),
            self._bfs_search(query, scope, as_of_time, query_embedding),
            return_exceptions=True,
        )

        result_lists: list[list[dict]] = []
        source_names = ["vector", "fulltext", "bfs"]
        for i, r in enumerate(results):
            if isinstance(r, Exception):
                logger.warning("Search source %s failed: %s", source_names[i], r)
                continue
            if r:
                result_lists.append(r)

        if not result_lists:
            return []

        from core.memory.graph.rrf import rrf_merge

        merged = rrf_merge(result_lists, top_k=min(30, limit * 3))

        if not merged:
            return []

        try:
            from core.memory.graph.reranker import get_reranker

            reranker = get_reranker(self._ce_model)
            text_field = "fact" if scope == "fact" else "content" if scope == "episode" else "name"
            return await reranker.rerank(query, merged, text_field=text_field, top_k=limit)
        except Exception:
            logger.warning("Cross-encoder rerank failed, using RRF order", exc_info=True)
            return merged[:limit]

    # ── Private search sources ────────────────────────────────────────────

    async def _vector_search(
        self,
        query: str,
        scope: str,
        as_of_time: str,
        embedding: list[float] | None,
    ) -> list[dict]:
        """Vector similarity search on fact/entity embeddings."""
        if not embedding:
            return []

        from core.memory.graph.queries import VECTOR_SEARCH_ENTITIES, VECTOR_SEARCH_FACTS

        if scope in ("fact", "all"):
            return await self._driver.execute_query(
                VECTOR_SEARCH_FACTS,
                {
                    "embedding": embedding,
                    "group_id": self._group_id,
                    "as_of_time": as_of_time,
                    "top_k": 20,
                },
            )
        if scope == "entity":
            return await self._driver.execute_query(
                VECTOR_SEARCH_ENTITIES,
                {
                    "embedding": embedding,
                    "group_id": self._group_id,
                    "top_k": 20,
                },
            )
        return []

    async def _fulltext_search(
        self,
        query: str,
        scope: str,
        as_of_time: str,
    ) -> list[dict]:
        """BM25 fulltext search."""
        from core.memory.graph.queries import FULLTEXT_SEARCH_ENTITIES, FULLTEXT_SEARCH_FACTS

        if scope in ("fact", "all"):
            try:
                return await self._driver.execute_query(
                    FULLTEXT_SEARCH_FACTS,
                    {
                        "query": query,
                        "group_id": self._group_id,
                        "as_of_time": as_of_time,
                        "top_k": 20,
                    },
                )
            except Exception:
                logger.debug("Fulltext search on facts failed (index may not exist)", exc_info=True)
                return []
        if scope == "entity":
            try:
                return await self._driver.execute_query(
                    FULLTEXT_SEARCH_ENTITIES,
                    {
                        "query": query,
                        "group_id": self._group_id,
                        "top_k": 20,
                    },
                )
            except Exception:
                logger.debug("Fulltext search on entities failed", exc_info=True)
                return []
        return []

    async def _bfs_search(
        self,
        query: str,
        scope: str,
        as_of_time: str,
        embedding: list[float] | None,
    ) -> list[dict]:
        """Graph BFS from seed entities."""
        if not embedding:
            return []

        from core.memory.graph.queries import BFS_FACTS_FROM_ENTITY, FIND_ENTITIES_BY_VECTOR

        try:
            seeds = await self._driver.execute_query(
                FIND_ENTITIES_BY_VECTOR,
                {
                    "embedding": embedding,
                    "group_id": self._group_id,
                    "top_k": 5,
                    "min_score": 0.3,
                    "entity_type": "",
                },
            )

            if not seeds:
                return []

            all_facts: list[dict] = []
            for seed in seeds[:5]:
                seed_uuid = seed.get("uuid")
                if not seed_uuid:
                    continue
                facts = await self._driver.execute_query(
                    BFS_FACTS_FROM_ENTITY,
                    {
                        "entity_uuid": seed_uuid,
                        "group_id": self._group_id,
                        "as_of_time": as_of_time,
                        "max_depth": 2,
                        "limit": 10,
                    },
                )
                all_facts.extend(facts)

            seen: set[str] = set()
            deduped: list[dict] = []
            for f in all_facts:
                uid = f.get("uuid", "")
                if uid and uid not in seen:
                    seen.add(uid)
                    deduped.append(f)

            return deduped[:20]
        except Exception:
            logger.debug("BFS search failed", exc_info=True)
            return []
