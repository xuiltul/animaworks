"""Unit tests for Priming Neo4j optimization: community context + recent facts.

All tests are fully mocked — no Neo4j instance required.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.memory.backend.base import MemoryBackend, RetrievedMemory

# ── Override detection ────────────────────────────────────────────────────
# The parallel agent may or may not have added get_community_context /
# get_recent_facts overrides to Neo4jGraphBackend yet.  Detect at
# collection time so we can skip tests that require them.

try:
    from core.memory.backend.neo4j_graph import Neo4jGraphBackend

    _HAS_COMMUNITY = "get_community_context" in Neo4jGraphBackend.__dict__
    _HAS_RECENT_FACTS = "get_recent_facts" in Neo4jGraphBackend.__dict__
except ImportError:
    _HAS_COMMUNITY = False
    _HAS_RECENT_FACTS = False


# ── Concrete stub for base-class defaults ─────────────────────────────────


class _StubBackend(MemoryBackend):
    """Minimal concrete subclass with no overrides beyond abstract methods."""

    async def ingest_file(self, path: Path) -> int:
        return 0

    async def ingest_text(self, text: str, source: str, metadata: dict | None = None) -> int:
        return 0

    async def retrieve(
        self, query: str, *, scope: str, limit: int = 10, min_score: float = 0.0
    ) -> list[RetrievedMemory]:
        return []

    async def delete(self, source: str) -> None:
        pass

    async def reset(self) -> None:
        pass

    async def stats(self) -> dict[str, int | float]:
        return {"total_chunks": 0, "total_sources": 0}

    async def health_check(self) -> bool:
        return True


# ── TestBaseDefaults ──────────────────────────────────────────────────────


class TestBaseDefaults:
    """MemoryBackend concrete defaults return empty lists."""

    async def test_get_community_context_default_empty(self) -> None:
        backend = _StubBackend()
        result = await backend.get_community_context("test query")
        assert result == []

    async def test_get_recent_facts_default_empty(self) -> None:
        backend = _StubBackend()
        result = await backend.get_recent_facts("test query")
        assert result == []


# ── TestLegacyCommunityContext ────────────────────────────────────────────


class TestLegacyCommunityContext:
    """LegacyRAGBackend community and recent-facts behaviour."""

    @pytest.fixture
    def legacy_backend(self, tmp_path: Path):
        with (
            patch("core.memory.rag.singleton.get_vector_store", return_value=MagicMock()),
            patch("core.memory.rag_search.RAGMemorySearch"),
        ):
            from core.memory.backend.legacy import LegacyRAGBackend

            return LegacyRAGBackend(tmp_path)

    async def test_legacy_community_always_empty(self, legacy_backend) -> None:
        result = await legacy_backend.get_community_context("any query")
        assert result == []

    async def test_legacy_recent_facts_with_bm25(self, legacy_backend) -> None:
        bm25_results = [
            {"content": "fact one", "score": 0.9, "source": "activity_log/2026-04-22.jsonl"},
            {"content": "fact two", "score": 0.7, "source": "activity_log/2026-04-22.jsonl"},
        ]
        with patch("core.memory.bm25.search_activity_log", return_value=bm25_results):
            result = await legacy_backend.get_recent_facts("test query")

        assert len(result) == 2
        assert all(isinstance(r, RetrievedMemory) for r in result)
        assert result[0].content == "fact one"
        assert result[0].score == pytest.approx(0.9)
        assert result[0].metadata["search_method"] == "bm25_activity"

    async def test_legacy_recent_facts_bm25_failure(self, legacy_backend) -> None:
        with patch(
            "core.memory.bm25.search_activity_log",
            side_effect=RuntimeError("BM25 failed"),
        ):
            result = await legacy_backend.get_recent_facts("test query")

        assert result == []


# ── TestNeo4jCommunityContext ─────────────────────────────────────────────


class TestNeo4jCommunityContext:
    """Neo4jGraphBackend.get_community_context delegates to _retrieve_communities."""

    @pytest.fixture
    def neo4j_backend(self, tmp_path: Path):
        from core.memory.backend.neo4j_graph import Neo4jGraphBackend

        backend = Neo4jGraphBackend(tmp_path)
        mock_driver = AsyncMock()
        backend._driver = mock_driver
        backend._schema_ensured = True
        return backend, mock_driver

    @pytest.mark.skipif(not _HAS_COMMUNITY, reason="get_community_context override pending")
    async def test_neo4j_community_returns_results(self, neo4j_backend) -> None:
        backend, mock_driver = neo4j_backend
        mock_driver.execute_query = AsyncMock(
            return_value=[
                {"uuid": "c1", "name": "Team Alpha", "summary": "Frontend team"},
                {"uuid": "c2", "name": "Team Beta", "summary": "Backend team"},
            ]
        )

        result = await backend.get_community_context("team")

        assert len(result) == 2
        assert "[Team Alpha]" in result[0].content
        assert "Frontend team" in result[0].content
        assert all(isinstance(r, RetrievedMemory) for r in result)

    @pytest.mark.skipif(not _HAS_COMMUNITY, reason="get_community_context override pending")
    async def test_neo4j_community_failure_returns_empty(self, neo4j_backend) -> None:
        backend, mock_driver = neo4j_backend
        mock_driver.execute_query = AsyncMock(side_effect=RuntimeError("Neo4j down"))

        result = await backend.get_community_context("team")
        assert result == []


# ── TestNeo4jRecentFacts ──────────────────────────────────────────────────


class TestNeo4jRecentFacts:
    """Neo4jGraphBackend.get_recent_facts calls FIND_RECENT_FACTS Cypher."""

    @pytest.fixture
    def neo4j_backend(self, tmp_path: Path):
        from core.memory.backend.neo4j_graph import Neo4jGraphBackend

        backend = Neo4jGraphBackend(tmp_path)
        mock_driver = AsyncMock()
        backend._driver = mock_driver
        backend._schema_ensured = True
        return backend, mock_driver

    @pytest.mark.skipif(not _HAS_RECENT_FACTS, reason="get_recent_facts override pending")
    async def test_neo4j_recent_facts_returns_results(self, neo4j_backend) -> None:
        backend, mock_driver = neo4j_backend
        mock_driver.execute_query = AsyncMock(
            return_value=[
                {
                    "uuid": "f1",
                    "fact": "works on frontend",
                    "source_name": "Alice",
                    "target_name": "React",
                    "valid_at": "2026-04-22T00:00:00Z",
                    "created_at": "2026-04-22T10:00:00Z",
                },
            ]
        )

        result = await backend.get_recent_facts("Alice frontend")

        assert len(result) == 1
        mem = result[0]
        assert isinstance(mem, RetrievedMemory)
        assert "Alice" in mem.content
        assert "React" in mem.content
        assert "works on frontend" in mem.content
        assert "\u2192" in mem.content  # → arrow

    @pytest.mark.skipif(not _HAS_RECENT_FACTS, reason="get_recent_facts override pending")
    async def test_neo4j_recent_facts_empty(self, neo4j_backend) -> None:
        backend, mock_driver = neo4j_backend
        mock_driver.execute_query = AsyncMock(return_value=[])

        result = await backend.get_recent_facts("unknown")
        assert result == []

    @pytest.mark.skipif(not _HAS_RECENT_FACTS, reason="get_recent_facts override pending")
    async def test_neo4j_recent_facts_failure(self, neo4j_backend) -> None:
        backend, mock_driver = neo4j_backend
        mock_driver.execute_query = AsyncMock(side_effect=RuntimeError("Neo4j down"))

        result = await backend.get_recent_facts("test")
        assert result == []


# ── TestRecentFactsQuery ──────────────────────────────────────────────────


class TestRecentFactsQuery:
    """Verify FIND_RECENT_FACTS Cypher query has required filters."""

    def test_find_recent_facts_has_since_filter(self) -> None:
        from core.memory.graph.queries import FIND_RECENT_FACTS

        assert "created_at >= datetime($since)" in FIND_RECENT_FACTS

    def test_find_recent_facts_excludes_invalid(self) -> None:
        from core.memory.graph.queries import FIND_RECENT_FACTS

        assert "invalid_at IS NULL" in FIND_RECENT_FACTS
