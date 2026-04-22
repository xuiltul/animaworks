"""Unit tests for Neo4j schema management."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest


class TestSchemaConstants:
    """Verify schema definitions are properly structured."""

    def test_constraints_are_list(self):
        from core.memory.graph.schema import CONSTRAINTS

        assert isinstance(CONSTRAINTS, list)
        assert len(CONSTRAINTS) >= 3  # Entity, Episode, Community

    def test_constraints_are_idempotent(self):
        from core.memory.graph.schema import CONSTRAINTS

        for c in CONSTRAINTS:
            assert "IF NOT EXISTS" in c

    def test_indexes_are_idempotent(self):
        from core.memory.graph.schema import INDEXES

        for i in INDEXES:
            assert "IF NOT EXISTS" in i

    def test_advanced_indexes_are_idempotent(self):
        from core.memory.graph.schema import ADVANCED_INDEXES

        for i in ADVANCED_INDEXES:
            assert "IF NOT EXISTS" in i

    def test_vector_indexes_have_name_and_query(self):
        from core.memory.graph.schema import VECTOR_INDEXES

        for vi in VECTOR_INDEXES:
            assert "name" in vi
            assert "query" in vi


class TestEnsureSchema:
    """Test ensure_schema with mocked driver."""

    @pytest.mark.asyncio
    async def test_executes_all_statements(self):
        from core.memory.graph.schema import (
            ADVANCED_INDEXES,
            CONSTRAINTS,
            INDEXES,
            VECTOR_INDEXES,
            ensure_schema,
        )

        mock_driver = AsyncMock()
        mock_driver.execute_write = AsyncMock()

        result = await ensure_schema(mock_driver)

        assert "constraints" in result
        assert "indexes" in result
        assert "advanced" in result
        assert "vector" in result
        expected_calls = len(CONSTRAINTS) + len(INDEXES) + len(ADVANCED_INDEXES) + len(VECTOR_INDEXES)
        assert mock_driver.execute_write.call_count == expected_calls

    @pytest.mark.asyncio
    async def test_counts_successes(self):
        from core.memory.graph.schema import (
            ADVANCED_INDEXES,
            CONSTRAINTS,
            INDEXES,
            VECTOR_INDEXES,
            ensure_schema,
        )

        mock_driver = AsyncMock()
        mock_driver.execute_write = AsyncMock()

        result = await ensure_schema(mock_driver)

        assert result["constraints"] == len(CONSTRAINTS)
        assert result["indexes"] == len(INDEXES)
        assert result["advanced"] == len(ADVANCED_INDEXES)
        assert result["vector"] == len(VECTOR_INDEXES)
        assert result["errors"] == 0

    @pytest.mark.asyncio
    async def test_continues_on_error(self):
        """Schema creation should not abort on individual statement failures."""
        from core.memory.graph.schema import (
            ADVANCED_INDEXES,
            CONSTRAINTS,
            INDEXES,
            VECTOR_INDEXES,
            ensure_schema,
        )

        total = len(CONSTRAINTS) + len(INDEXES) + len(ADVANCED_INDEXES) + len(VECTOR_INDEXES)
        effects = [None] + [Exception("fail")] + [None] * (total - 2)

        mock_driver = AsyncMock()
        mock_driver.execute_write = AsyncMock(side_effect=effects)

        result = await ensure_schema(mock_driver)
        assert result["errors"] == 1
        assert mock_driver.execute_write.call_count == total

    @pytest.mark.asyncio
    async def test_idempotent_double_call(self):
        """Calling ensure_schema twice should not fail."""
        from core.memory.graph.schema import ensure_schema

        mock_driver = AsyncMock()
        mock_driver.execute_write = AsyncMock()

        r1 = await ensure_schema(mock_driver)
        r2 = await ensure_schema(mock_driver)
        assert isinstance(r1, dict)
        assert isinstance(r2, dict)
