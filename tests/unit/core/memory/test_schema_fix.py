"""Tests for Issue #16 — Neo4j schema inconsistencies fix.

Covers:
- fact_fulltext index in schema.py
- entity_type in CREATE_ENTITY
- expired_at in search queries
- EXPIRE_FACT query and EdgeInvalidator.expire_fact()
- SCHEMA_VERSION constant
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest


# ── TestSchemaVersion ────────────────────────────────────────


class TestSchemaVersion:
    def test_schema_version_exists(self) -> None:
        from core.memory.graph.schema import SCHEMA_VERSION

        assert isinstance(SCHEMA_VERSION, int)
        assert SCHEMA_VERSION >= 2


# ── TestFactFulltextIndex ────────────────────────────────────


class TestFactFulltextIndex:
    def test_advanced_indexes_has_fact_fulltext(self) -> None:
        from core.memory.graph.schema import ADVANCED_INDEXES

        fulltext_stmts = [s for s in ADVANCED_INDEXES if "fact_fulltext" in s]
        assert len(fulltext_stmts) == 1

    def test_fact_fulltext_targets_relates_to(self) -> None:
        from core.memory.graph.schema import ADVANCED_INDEXES

        stmt = next(s for s in ADVANCED_INDEXES if "fact_fulltext" in s)
        assert "RELATES_TO" in stmt
        assert "r.fact" in stmt

    def test_entity_fulltext_still_present(self) -> None:
        from core.memory.graph.schema import ADVANCED_INDEXES

        entity_stmts = [s for s in ADVANCED_INDEXES if "entity_name_fulltext" in s]
        assert len(entity_stmts) == 1

    @pytest.mark.asyncio
    async def test_ensure_schema_runs_advanced_indexes(self) -> None:
        from core.memory.graph.schema import ensure_schema

        mock_driver = AsyncMock()
        mock_driver.execute_write = AsyncMock()
        counts = await ensure_schema(mock_driver)
        assert counts["advanced"] >= 2


# ── TestEntityTypeInQueries ──────────────────────────────────


class TestEntityTypeInQueries:
    def test_create_entity_has_entity_type_param(self) -> None:
        from core.memory.graph.queries import CREATE_ENTITY

        assert "$entity_type" in CREATE_ENTITY
        assert "entity_type:" in CREATE_ENTITY

    def test_find_entities_return_entity_type(self) -> None:
        from core.memory.graph.queries import FIND_ENTITIES_BY_NAME, FIND_ENTITIES_BY_VECTOR

        for q in (FIND_ENTITIES_BY_NAME, FIND_ENTITIES_BY_VECTOR):
            assert "entity_type" in q


class TestEntityTypeInIngest:
    @pytest.mark.asyncio
    async def test_ingest_passes_entity_type(self, tmp_path) -> None:
        from core.memory.backend.neo4j_graph import Neo4jGraphBackend

        anima_dir = tmp_path / "animas" / "test"
        anima_dir.mkdir(parents=True)
        backend = Neo4jGraphBackend(anima_dir, group_id="test")

        mock_driver = AsyncMock()
        mock_driver.execute_write = AsyncMock()
        mock_driver.execute_query = AsyncMock(return_value=[])
        backend._driver = mock_driver
        backend._schema_ensured = True

        mock_entity = MagicMock()
        mock_entity.name = "Tokyo"
        mock_entity.entity_type = "Place"
        mock_entity.summary = "Capital of Japan"

        mock_extractor = MagicMock()
        mock_extractor.extract_entities = AsyncMock(return_value=[mock_entity])
        mock_extractor.extract_facts = AsyncMock(return_value=[])

        mock_resolved = MagicMock()
        mock_resolved.uuid = "ent-1"
        mock_resolved.name = "Tokyo"
        mock_resolved.summary = "Capital of Japan"
        mock_resolved.is_new = True

        mock_resolver = MagicMock()
        mock_resolver.resolve = AsyncMock(return_value=mock_resolved)

        backend._extractor = mock_extractor
        backend._resolver = mock_resolver

        await backend.ingest_text("Tokyo is the capital", source="test")

        create_entity_calls = [
            c
            for c in mock_driver.execute_write.call_args_list
            if c[0] and "entity_type" in str(c[0][0])
        ]
        assert len(create_entity_calls) >= 1
        params = create_entity_calls[0][0][1]
        assert params["entity_type"] == "Place"


# ── TestExpiredAtInQueries ───────────────────────────────────


class TestExpiredAtInQueries:
    def test_vector_search_facts_filters_expired(self) -> None:
        from core.memory.graph.queries import VECTOR_SEARCH_FACTS

        assert "expired_at" in VECTOR_SEARCH_FACTS

    def test_fulltext_search_facts_filters_expired(self) -> None:
        from core.memory.graph.queries import FULLTEXT_SEARCH_FACTS

        assert "expired_at" in FULLTEXT_SEARCH_FACTS

    def test_find_valid_facts_filters_expired(self) -> None:
        from core.memory.graph.queries import FIND_VALID_FACTS_BY_GROUP

        assert "expired_at" in FIND_VALID_FACTS_BY_GROUP

    def test_find_recent_facts_filters_expired(self) -> None:
        from core.memory.graph.queries import FIND_RECENT_FACTS

        assert "expired_at IS NULL" in FIND_RECENT_FACTS

    def test_invalidate_uses_group_id(self) -> None:
        from core.memory.graph.queries import INVALIDATE_FACT

        assert "invalid_at" in INVALIDATE_FACT
        assert "$group_id" in INVALIDATE_FACT

    def test_bfs_filters_expired(self) -> None:
        from core.memory.graph.queries import BFS_FACTS_FROM_ENTITY

        assert "expired_at" in BFS_FACTS_FROM_ENTITY

    def test_active_facts_pair_filters_expired(self) -> None:
        from core.memory.graph.queries import FIND_ACTIVE_FACTS_FOR_PAIR

        assert "expired_at IS NULL" in FIND_ACTIVE_FACTS_FOR_PAIR

    def test_active_facts_pair_reverse_filters_expired(self) -> None:
        from core.memory.graph.queries import FIND_ACTIVE_FACTS_FOR_PAIR_REVERSE

        assert "expired_at IS NULL" in FIND_ACTIVE_FACTS_FOR_PAIR_REVERSE


# ── TestExpireFactQuery ──────────────────────────────────────


class TestExpireFactQuery:
    def test_expire_fact_query_exists(self) -> None:
        from core.memory.graph.queries import EXPIRE_FACT

        assert "expired_at" in EXPIRE_FACT
        assert "$uuid" in EXPIRE_FACT
        assert "$expired_at" in EXPIRE_FACT

    def test_expire_and_invalidate_are_separate(self) -> None:
        from core.memory.graph.queries import EXPIRE_FACT, INVALIDATE_FACT

        assert "expired_at" in EXPIRE_FACT
        assert "invalid_at" in INVALIDATE_FACT

    def test_expire_fact_has_group_id(self) -> None:
        from core.memory.graph.queries import EXPIRE_FACT

        assert "$group_id" in EXPIRE_FACT


# ── TestEdgeInvalidatorExpireFact ────────────────────────────


class TestEdgeInvalidatorExpireFact:
    @pytest.fixture
    def mock_driver(self) -> AsyncMock:
        driver = AsyncMock()
        driver.execute_write = AsyncMock()
        return driver

    @pytest.fixture
    def invalidator(self, mock_driver: AsyncMock):
        from core.memory.extraction.invalidator import EdgeInvalidator

        return EdgeInvalidator(mock_driver, "test_group", model="test-model")

    @pytest.mark.asyncio
    async def test_expire_fact_success(self, invalidator, mock_driver: AsyncMock) -> None:
        result = await invalidator.expire_fact("fact-123", "2026-06-01T00:00:00")
        assert result is True
        mock_driver.execute_write.assert_called_once()
        args = mock_driver.execute_write.call_args[0]
        assert args[1]["uuid"] == "fact-123"
        assert args[1]["expired_at"] == "2026-06-01T00:00:00"
        assert args[1]["group_id"] == "test_group"

    @pytest.mark.asyncio
    async def test_expire_fact_failure_returns_false(self, invalidator, mock_driver: AsyncMock) -> None:
        mock_driver.execute_write = AsyncMock(side_effect=RuntimeError("Neo4j down"))
        result = await invalidator.expire_fact("fact-123", "2026-06-01T00:00:00")
        assert result is False

    @pytest.mark.asyncio
    async def test_expire_fact_uses_expire_query(self, invalidator, mock_driver: AsyncMock) -> None:
        from core.memory.graph.queries import EXPIRE_FACT

        await invalidator.expire_fact("fact-x", "2026-01-01T00:00:00")
        call_args = mock_driver.execute_write.call_args[0]
        assert call_args[0] is EXPIRE_FACT


# ── TestSchemaIdempotency ────────────────────────────────────


class TestSchemaIdempotency:
    @pytest.mark.asyncio
    async def test_ensure_schema_twice_no_error(self) -> None:
        """Running ensure_schema twice should not raise."""
        from core.memory.graph.schema import ensure_schema

        mock_driver = AsyncMock()
        mock_driver.execute_write = AsyncMock()

        counts1 = await ensure_schema(mock_driver)
        counts2 = await ensure_schema(mock_driver)
        assert counts1["errors"] == 0
        assert counts2["errors"] == 0

    @pytest.mark.asyncio
    async def test_ensure_schema_handles_advanced_index_failure(self) -> None:
        from core.memory.graph.schema import ADVANCED_INDEXES, ensure_schema

        call_count = 0

        async def _flaky_write(stmt, params=None):
            nonlocal call_count
            call_count += 1
            if "fact_fulltext" in stmt:
                raise RuntimeError("Fulltext not supported")

        mock_driver = AsyncMock()
        mock_driver.execute_write = _flaky_write
        counts = await ensure_schema(mock_driver)
        assert counts["errors"] >= 1
        assert counts["advanced"] >= 1
