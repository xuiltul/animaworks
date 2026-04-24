"""Unit tests for hybrid search: RRF, cross-encoder reranker, HybridSearch."""

from __future__ import annotations

import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ── TestRRFMerge ──────────────────────────────────────────────────────────


class TestRRFMerge:
    """Tests for reciprocal rank fusion merge."""

    def _rrf(self, *args, **kwargs):
        from core.memory.graph.rrf import rrf_merge

        return rrf_merge(*args, **kwargs)

    def test_single_list(self):
        results = self._rrf([[{"uuid": "a", "fact": "x"}, {"uuid": "b", "fact": "y"}]])
        assert len(results) == 2
        assert all("rrf_score" in r for r in results)
        assert results[0]["rrf_score"] >= results[1]["rrf_score"]

    def test_two_lists_overlap(self):
        list1 = [{"uuid": "a", "fact": "f1"}, {"uuid": "b", "fact": "f2"}]
        list2 = [{"uuid": "a", "fact": "f1"}, {"uuid": "c", "fact": "f3"}]
        results = self._rrf([list1, list2])

        scores = {r["uuid"]: r["rrf_score"] for r in results}
        assert scores["a"] > scores["b"]
        assert scores["a"] > scores["c"]

    def test_deduplication(self):
        list1 = [{"uuid": "a"}, {"uuid": "b"}]
        list2 = [{"uuid": "a"}, {"uuid": "c"}]
        results = self._rrf([list1, list2])
        uuids = [r["uuid"] for r in results]
        assert uuids.count("a") == 1

    def test_empty_input(self):
        assert self._rrf([]) == []
        assert self._rrf([[]]) == []

    def test_top_k_limit(self):
        items = [{"uuid": str(i)} for i in range(20)]
        results = self._rrf([items], top_k=5)
        assert len(results) == 5

    def test_custom_k_constant(self):
        items = [{"uuid": "a"}, {"uuid": "b"}]
        r_k1 = self._rrf([items], k=1)
        r_k60 = self._rrf([items], k=60)
        assert r_k1[0]["rrf_score"] != r_k60[0]["rrf_score"]
        assert r_k1[0]["rrf_score"] > r_k60[0]["rrf_score"]

    def test_key_field(self):
        items = [{"id": "x", "fact": "f1"}, {"id": "y", "fact": "f2"}]
        results = self._rrf([items], key_field="id")
        uuids = {r["id"] for r in results}
        assert uuids == {"x", "y"}


# ── TestCrossEncoderReranker ──────────────────────────────────────────────


def _make_reranker_with_mock(scores: list[float]):
    """Create a CrossEncoderReranker with a pre-injected mock model."""
    from core.memory.graph.reranker import CrossEncoderReranker

    reranker = CrossEncoderReranker()
    mock_model = MagicMock()
    mock_model.predict.return_value = scores
    reranker._model = mock_model
    reranker._available = True
    return reranker


class TestCrossEncoderReranker:
    """Tests for cross-encoder reranker with mocked model."""

    @pytest.mark.asyncio
    async def test_rerank_with_mock_model(self):
        reranker = _make_reranker_with_mock([0.1, 0.9, 0.5])
        items = [{"fact": "a"}, {"fact": "b"}, {"fact": "c"}]
        result = await reranker.rerank("query", items)

        assert result[0]["fact"] == "b"
        assert result[1]["fact"] == "c"
        assert result[2]["fact"] == "a"

    @pytest.mark.asyncio
    async def test_rerank_fallback_on_import_error(self):
        from core.memory.graph.reranker import CrossEncoderReranker

        reranker = CrossEncoderReranker()
        reranker._available = True
        reranker._model = None

        mock_st = MagicMock()
        mock_st.CrossEncoder = MagicMock(side_effect=ImportError("no module"))
        with patch.dict(sys.modules, {"sentence_transformers": mock_st}):
            items = [{"fact": "a"}, {"fact": "b"}, {"fact": "c"}]
            result = await reranker.rerank("query", items)

        assert len(result) == 3
        assert all("ce_score" in r for r in result)
        assert all(r["ce_score"] == 0.0 for r in result)

    @pytest.mark.asyncio
    async def test_rerank_empty_items(self):
        from core.memory.graph.reranker import CrossEncoderReranker

        reranker = CrossEncoderReranker()
        result = await reranker.rerank("query", [])
        assert result == []

    @pytest.mark.asyncio
    async def test_rerank_adds_ce_score(self):
        reranker = _make_reranker_with_mock([0.7, 0.3])
        result = await reranker.rerank("q", [{"fact": "x"}, {"fact": "y"}])

        assert all("ce_score" in r for r in result)
        assert result[0]["ce_score"] == pytest.approx(0.7)
        assert result[1]["ce_score"] == pytest.approx(0.3)


# ── TestHybridSearch ──────────────────────────────────────────────────────


def _make_fact(uuid: str, fact: str = "test") -> dict:
    return {
        "uuid": uuid,
        "fact": fact,
        "source_name": "A",
        "target_name": "B",
        "valid_at": "2024-01-01T00:00:00Z",
    }


def _patch_reranker():
    """Patch the reranker singleton to return items with ce_score."""

    mock_reranker = AsyncMock()

    async def _passthrough(query, items, **kw):
        return [{**it, "ce_score": 0.5} for it in items]

    mock_reranker.rerank = _passthrough
    return patch("core.memory.graph.reranker.get_reranker", return_value=mock_reranker)


class TestHybridSearch:
    """Tests for HybridSearch with mocked Neo4j driver."""

    def _make_search(self, driver: AsyncMock):
        from core.memory.graph.search import HybridSearch

        return HybridSearch(driver, "test_group")

    @pytest.mark.asyncio
    async def test_search_runs_parallel_sources(self):
        original_facts = [_make_fact("v1"), _make_fact("v2")]

        async def _mock_execute(query, params=None, **kw):
            q = query.strip()
            if "queryRelationships" in q:
                return list(original_facts)
            if "queryNodes" in q:
                return [{"uuid": "seed1", "name": "Entity1"}]
            if "fulltext" in q.lower():
                return [_make_fact("ft1")]
            return [_make_fact("bfs1")]

        driver = AsyncMock()
        driver.execute_query = AsyncMock(side_effect=_mock_execute)

        search = self._make_search(driver)
        with _patch_reranker():
            await search.search("test query", query_embedding=[0.1] * 384)

        assert driver.execute_query.call_count >= 2

    @pytest.mark.asyncio
    async def test_search_handles_source_failure(self):
        async def _mock_execute(query, params=None, **kw):
            q = query.strip()
            if "queryRelationships" in q:
                raise RuntimeError("Vector search down")
            if "fulltext" in q.lower():
                return [_make_fact("ft1")]
            return []

        driver = AsyncMock()
        driver.execute_query = AsyncMock(side_effect=_mock_execute)

        search = self._make_search(driver)
        with _patch_reranker():
            result = await search.search("query", query_embedding=[0.1] * 384)

        assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_search_all_sources_fail(self):
        driver = AsyncMock()
        driver.execute_query = AsyncMock(side_effect=RuntimeError("DB down"))

        search = self._make_search(driver)
        result = await search.search("query", query_embedding=[0.1] * 384)
        assert result == []

    @pytest.mark.asyncio
    async def test_search_empty_query_raises(self):
        driver = AsyncMock()
        search = self._make_search(driver)
        with pytest.raises(ValueError, match="empty"):
            await search.search("")
        with pytest.raises(ValueError, match="empty"):
            await search.search("   ")

    @pytest.mark.asyncio
    async def test_search_applies_temporal_filter(self):
        captured_params: list[dict] = []

        async def _capture(query, params=None, **kw):
            if params:
                captured_params.append(dict(params))
            return []

        driver = AsyncMock()
        driver.execute_query = AsyncMock(side_effect=_capture)

        search = self._make_search(driver)
        await search.search(
            "query",
            as_of_time="2025-06-01T00:00:00Z",
            query_embedding=[0.1] * 384,
        )

        as_of_values = [p.get("as_of_time") for p in captured_params if "as_of_time" in p]
        assert all(v == "2025-06-01T00:00:00Z" for v in as_of_values)

    @pytest.mark.asyncio
    async def test_search_without_embedding(self):
        queries_run: list[str] = []

        async def _track(query, params=None, **kw):
            queries_run.append(query.strip()[:80])
            if "fulltext" in query.lower():
                return [_make_fact("ft1")]
            return []

        driver = AsyncMock()
        driver.execute_query = AsyncMock(side_effect=_track)

        search = self._make_search(driver)
        with _patch_reranker():
            await search.search("query", query_embedding=None)

        has_vector = any("vector.queryRelationships" in q for q in queries_run)
        assert not has_vector, "Vector search should not run without embedding"


# ── TestNeo4jRetrieve ─────────────────────────────────────────────────────


class TestNeo4jRetrieve:
    """Tests for Neo4jGraphBackend.retrieve() with mocked search."""

    def _make_backend(self, tmp_path):
        from core.memory.backend.neo4j_graph import Neo4jGraphBackend

        return Neo4jGraphBackend(tmp_path)

    def _patch_hybrid_search(self, return_value=None, side_effect=None):
        mock_hs = AsyncMock()
        if side_effect:
            mock_hs.search = AsyncMock(side_effect=side_effect)
        else:
            mock_hs.search = AsyncMock(return_value=return_value or [])
        return patch("core.memory.graph.search.HybridSearch", return_value=mock_hs)

    @pytest.mark.asyncio
    async def test_retrieve_returns_retrieved_memory(self, tmp_path):
        from core.memory.backend.base import RetrievedMemory

        backend = self._make_backend(tmp_path)
        mock_results = [
            {
                "uuid": "f1",
                "fact": "likes coffee",
                "source_name": "Alice",
                "target_name": "Coffee",
                "valid_at": "2024-01-01",
                "rrf_score": 0.5,
            }
        ]

        with (
            patch.object(backend, "_ensure_driver", new_callable=AsyncMock),
            self._patch_hybrid_search(return_value=mock_results),
        ):
            result = await backend.retrieve("coffee", scope="fact")

        assert len(result) == 1
        assert isinstance(result[0], RetrievedMemory)
        assert result[0].score == pytest.approx(0.5)

    @pytest.mark.asyncio
    async def test_retrieve_fact_scope_formatting(self, tmp_path):
        backend = self._make_backend(tmp_path)
        mock_results = [
            {
                "uuid": "f1",
                "fact": "likes coffee",
                "source_name": "Alice",
                "target_name": "Coffee",
                "valid_at": "2024-01-01",
                "ce_score": 0.8,
            }
        ]

        with (
            patch.object(backend, "_ensure_driver", new_callable=AsyncMock),
            self._patch_hybrid_search(return_value=mock_results),
        ):
            result = await backend.retrieve("coffee", scope="fact")

        assert "Alice" in result[0].content
        assert "Coffee" in result[0].content
        assert "likes coffee" in result[0].content
        assert "-[RELATES_TO]->" in result[0].content

    @pytest.mark.asyncio
    async def test_retrieve_entity_scope_formatting(self, tmp_path):
        backend = self._make_backend(tmp_path)
        mock_results = [
            {
                "uuid": "e1",
                "name": "Alice",
                "summary": "A software engineer",
                "entity_type": "person",
                "ce_score": 0.9,
            }
        ]

        with (
            patch.object(backend, "_ensure_driver", new_callable=AsyncMock),
            self._patch_hybrid_search(return_value=mock_results),
        ):
            result = await backend.retrieve("Alice", scope="entity")

        assert "Alice" in result[0].content
        assert "A software engineer" in result[0].content
        assert result[0].source.startswith("entity:")

    @pytest.mark.asyncio
    async def test_retrieve_min_score_filter(self, tmp_path):
        backend = self._make_backend(tmp_path)
        mock_results = [
            {**_make_fact("f1"), "ce_score": 0.9},
            {**_make_fact("f2"), "ce_score": 0.1},
        ]

        with (
            patch.object(backend, "_ensure_driver", new_callable=AsyncMock),
            self._patch_hybrid_search(return_value=mock_results),
        ):
            result = await backend.retrieve("test", scope="fact", min_score=0.5)

        assert len(result) == 1
        assert result[0].score == pytest.approx(0.9)

    @pytest.mark.asyncio
    async def test_retrieve_handles_search_failure(self, tmp_path):
        backend = self._make_backend(tmp_path)

        with (
            patch.object(backend, "_ensure_driver", new_callable=AsyncMock),
            self._patch_hybrid_search(side_effect=RuntimeError("DB error")),
        ):
            result = await backend.retrieve("test", scope="fact")

        assert result == []


# ── TestSearchQueries ─────────────────────────────────────────────────────


class TestSearchQueries:
    """Verify Cypher query templates contain expected clauses."""

    def test_vector_search_facts_has_temporal_filter(self):
        from core.memory.graph.queries import VECTOR_SEARCH_FACTS

        assert "invalid_at" in VECTOR_SEARCH_FACTS
        assert "$as_of_time" in VECTOR_SEARCH_FACTS

    def test_bfs_has_temporal_filter(self):
        from core.memory.graph.queries import BFS_FACTS_FROM_ENTITY

        assert "invalid_at" in BFS_FACTS_FROM_ENTITY
        assert "$as_of_time" in BFS_FACTS_FROM_ENTITY

    def test_fulltext_search_facts_exists(self):
        from core.memory.graph import queries

        assert hasattr(queries, "FULLTEXT_SEARCH_FACTS")
        assert len(queries.FULLTEXT_SEARCH_FACTS.strip()) > 0
