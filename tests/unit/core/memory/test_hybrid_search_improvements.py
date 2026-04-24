"""Tests for Issue #20 — Hybrid search improvements.

Covers:
- BFS max_depth dynamic query generation
- Episode vector search query
- RRF k parameter passthrough
- Episode embedding in ingest
- Schema version bump
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest


# ── TestBfsFactsQuery ────────────────────────────────────


class TestBfsFactsQuery:
    def test_default_depth_2(self) -> None:
        from core.memory.graph.queries import bfs_facts_query

        q = bfs_facts_query()
        assert "*1..2" in q

    def test_depth_3(self) -> None:
        from core.memory.graph.queries import bfs_facts_query

        q = bfs_facts_query(3)
        assert "*1..3" in q

    def test_depth_clamped_min(self) -> None:
        from core.memory.graph.queries import bfs_facts_query

        q = bfs_facts_query(0)
        assert "*1..1" in q

    def test_depth_clamped_max(self) -> None:
        from core.memory.graph.queries import bfs_facts_query

        q = bfs_facts_query(10)
        assert "*1..5" in q

    def test_contains_deleted_at_filter(self) -> None:
        from core.memory.graph.queries import bfs_facts_query

        q = bfs_facts_query()
        assert "deleted_at IS NULL" in q

    def test_contains_temporal_filters(self) -> None:
        from core.memory.graph.queries import bfs_facts_query

        q = bfs_facts_query()
        assert "invalid_at" in q
        assert "expired_at" in q


# ── TestVectorSearchEpisodes ─────────────────────────────


class TestVectorSearchEpisodes:
    def test_query_exists(self) -> None:
        from core.memory.graph.queries import VECTOR_SEARCH_EPISODES

        assert "episode_content_embedding" in VECTOR_SEARCH_EPISODES
        assert "$embedding" in VECTOR_SEARCH_EPISODES
        assert "deleted_at IS NULL" in VECTOR_SEARCH_EPISODES

    def test_returns_content_field(self) -> None:
        from core.memory.graph.queries import VECTOR_SEARCH_EPISODES

        assert "content" in VECTOR_SEARCH_EPISODES
        assert "source" in VECTOR_SEARCH_EPISODES


# ── TestSchemaVersion ────────────────────────────────────


class TestSchemaVersion:
    def test_version_bumped(self) -> None:
        from core.memory.graph.schema import SCHEMA_VERSION

        assert SCHEMA_VERSION >= 3

    def test_episode_vector_index_exists(self) -> None:
        from core.memory.graph.schema import VECTOR_INDEXES

        names = [vi["name"] for vi in VECTOR_INDEXES]
        assert "episode_content_embedding" in names


# ── TestHybridSearchInit ─────────────────────────────────


class TestHybridSearchInit:
    def test_max_depth_default(self) -> None:
        from core.memory.graph.search import HybridSearch

        hs = HybridSearch(MagicMock(), "group")
        assert hs._max_depth == 2

    def test_max_depth_custom(self) -> None:
        from core.memory.graph.search import HybridSearch

        hs = HybridSearch(MagicMock(), "group", max_depth=4)
        assert hs._max_depth == 4

    def test_rrf_k_default(self) -> None:
        from core.memory.graph.search import HybridSearch

        hs = HybridSearch(MagicMock(), "group")
        assert hs._rrf_k == 60

    def test_rrf_k_custom(self) -> None:
        from core.memory.graph.search import HybridSearch

        hs = HybridSearch(MagicMock(), "group", rrf_k=30)
        assert hs._rrf_k == 30


# ── TestEpisodeVectorSearch ──────────────────────────────


class TestEpisodeVectorSearch:
    @pytest.mark.asyncio
    async def test_episode_scope_calls_vector_search_episodes(self) -> None:
        from core.memory.graph.search import HybridSearch

        mock_driver = AsyncMock()
        mock_driver.execute_query = AsyncMock(return_value=[{"uuid": "ep1", "content": "test", "score": 0.9}])

        hs = HybridSearch(mock_driver, "group")
        results = await hs._vector_search("query", "episode", "2026-01-01", [0.1] * 384)
        assert len(results) == 1
        assert results[0]["uuid"] == "ep1"

    @pytest.mark.asyncio
    async def test_episode_scope_without_embedding_returns_empty(self) -> None:
        from core.memory.graph.search import HybridSearch

        hs = HybridSearch(AsyncMock(), "group")
        results = await hs._vector_search("query", "episode", "2026-01-01", None)
        assert results == []


# ── TestBfsUsesMaxDepth ──────────────────────────────────


class TestBfsUsesMaxDepth:
    @pytest.mark.asyncio
    async def test_bfs_uses_configured_depth(self) -> None:
        from core.memory.graph.search import HybridSearch

        mock_driver = AsyncMock()
        mock_driver.execute_query = AsyncMock(
            side_effect=[
                [{"uuid": "seed1", "name": "A", "summary": "s", "entity_type": "Person", "score": 0.8}],
                [{"uuid": "f1", "fact": "test", "source_name": "A", "target_name": "B", "valid_at": "2026-01-01"}],
            ]
        )

        hs = HybridSearch(mock_driver, "group", max_depth=3)
        results = await hs._bfs_search("query", "fact", "2026-01-01", [0.1] * 384)

        bfs_call = mock_driver.execute_query.call_args_list[1]
        bfs_query = bfs_call[0][0]
        assert "*1..3" in bfs_query


# ── TestEpisodeEmbeddingInIngest ─────────────────────────


class TestEpisodeEmbeddingInIngest:
    @pytest.mark.asyncio
    async def test_episode_gets_embedding(self, tmp_path) -> None:
        from core.memory.backend.neo4j_graph import Neo4jGraphBackend

        anima_dir = tmp_path / "animas" / "test"
        anima_dir.mkdir(parents=True)
        backend = Neo4jGraphBackend(anima_dir, group_id="test")

        mock_driver = AsyncMock()
        mock_driver.execute_write = AsyncMock()
        mock_driver.execute_query = AsyncMock(return_value=[])
        backend._driver = mock_driver
        backend._schema_ensured = True

        mock_extractor = MagicMock()
        mock_extractor.extract_entities = AsyncMock(return_value=[])
        mock_extractor.extract_facts = AsyncMock(return_value=[])
        backend._extractor = mock_extractor

        backend._embedding_available = True

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(
                backend,
                "_embed_texts",
                AsyncMock(return_value=[[0.1] * 384]),
            )
            await backend.ingest_text("test content", source="test")

        embedding_calls = [
            c for c in mock_driver.execute_write.call_args_list
            if "content_embedding" in str(c)
        ]
        assert len(embedding_calls) >= 1


# ── TestRrfKPassthrough ──────────────────────────────────


class TestRrfKPassthrough:
    def test_rrf_merge_accepts_custom_k(self) -> None:
        from core.memory.graph.rrf import rrf_merge

        lists = [
            [{"uuid": "a", "fact": "1"}, {"uuid": "b", "fact": "2"}],
            [{"uuid": "b", "fact": "2"}, {"uuid": "a", "fact": "1"}],
        ]
        result_k60 = rrf_merge(lists, k=60)
        result_k10 = rrf_merge(lists, k=10)

        assert len(result_k60) == 2
        assert len(result_k10) == 2
        assert result_k60[0]["rrf_score"] != result_k10[0]["rrf_score"]
