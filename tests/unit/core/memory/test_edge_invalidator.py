from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for EdgeInvalidator — temporal fact invalidation via LLM contradiction check."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.memory.extraction.invalidator import EdgeInvalidator
from core.memory.graph.queries import (
    FIND_ACTIVE_FACTS_FOR_PAIR,
    FIND_VALID_FACTS_BY_GROUP,
    INVALIDATE_FACT,
)

# ── Fixtures ────────────────────────────────────────────────


@pytest.fixture
def mock_driver() -> AsyncMock:
    driver = AsyncMock()
    driver.execute_query = AsyncMock(return_value=[])
    driver.execute_write = AsyncMock()
    return driver


@pytest.fixture
def invalidator(mock_driver: AsyncMock) -> EdgeInvalidator:
    return EdgeInvalidator(mock_driver, "test_group", model="test-model")


COMMON_KWARGS = {
    "new_fact_uuid": "new-fact-1",
    "source_entity_uuid": "entity-src",
    "target_entity_uuid": "entity-tgt",
    "new_fact_text": "Alice lives in Osaka",
    "new_valid_at": "2025-06-01T00:00:00",
}


def _make_candidate(uuid: str = "old-fact-1", fact: str = "Alice lives in Tokyo") -> dict:
    return {"uuid": uuid, "fact": fact, "valid_at": "2024-01-01T00:00:00"}


# ── TestEdgeInvalidatorInit ─────────────────────────────────


class TestEdgeInvalidatorInit:
    def test_creates_with_driver(self, mock_driver: AsyncMock) -> None:
        inv = EdgeInvalidator(mock_driver, "group1")
        assert inv._group_id == "group1"
        assert inv._driver is mock_driver

    def test_default_model(self, mock_driver: AsyncMock) -> None:
        inv = EdgeInvalidator(mock_driver, "g")
        assert inv._model == "claude-sonnet-4-6"


# ── TestFindAndInvalidate ───────────────────────────────────


class TestFindAndInvalidate:
    @pytest.mark.asyncio
    async def test_no_candidates_returns_empty(self, invalidator: EdgeInvalidator, mock_driver: AsyncMock) -> None:
        mock_driver.execute_query = AsyncMock(return_value=[])
        result = await invalidator.find_and_invalidate(**COMMON_KWARGS)
        assert result == []
        mock_driver.execute_write.assert_not_called()

    @pytest.mark.asyncio
    async def test_candidates_found_llm_says_contradicted(
        self, invalidator: EdgeInvalidator, mock_driver: AsyncMock
    ) -> None:
        mock_driver.execute_query = AsyncMock(
            side_effect=[
                [_make_candidate("existing-uuid")],
                [],
            ]
        )

        llm_resp = MagicMock()
        llm_resp.choices = [MagicMock()]
        llm_resp.choices[0].message.content = json.dumps({"contradicted_uuids": ["existing-uuid"]})

        with patch("litellm.acompletion", new_callable=AsyncMock, return_value=llm_resp):
            result = await invalidator.find_and_invalidate(**COMMON_KWARGS)

        assert result == ["existing-uuid"]
        mock_driver.execute_write.assert_called_once()
        call_args = mock_driver.execute_write.call_args
        assert call_args[0][1]["uuid"] == "existing-uuid"
        assert call_args[0][1]["invalid_at"] == "2025-06-01T00:00:00"

    @pytest.mark.asyncio
    async def test_candidates_found_llm_says_no_contradiction(
        self, invalidator: EdgeInvalidator, mock_driver: AsyncMock
    ) -> None:
        mock_driver.execute_query = AsyncMock(
            side_effect=[
                [_make_candidate()],
                [],
            ]
        )

        llm_resp = MagicMock()
        llm_resp.choices = [MagicMock()]
        llm_resp.choices[0].message.content = json.dumps({"contradicted_uuids": []})

        with patch("litellm.acompletion", new_callable=AsyncMock, return_value=llm_resp):
            result = await invalidator.find_and_invalidate(**COMMON_KWARGS)

        assert result == []
        mock_driver.execute_write.assert_not_called()

    @pytest.mark.asyncio
    async def test_llm_failure_safe_fallback(self, invalidator: EdgeInvalidator, mock_driver: AsyncMock) -> None:
        mock_driver.execute_query = AsyncMock(
            side_effect=[
                [_make_candidate()],
                [],
            ]
        )

        with patch("litellm.acompletion", new_callable=AsyncMock, side_effect=RuntimeError("LLM down")):
            result = await invalidator.find_and_invalidate(**COMMON_KWARGS)

        assert result == []
        mock_driver.execute_write.assert_not_called()

    @pytest.mark.asyncio
    async def test_multiple_candidates_partial_contradiction(
        self, invalidator: EdgeInvalidator, mock_driver: AsyncMock
    ) -> None:
        mock_driver.execute_query = AsyncMock(
            side_effect=[
                [
                    _make_candidate("fact-a", "Alice lives in Tokyo"),
                    _make_candidate("fact-b", "Alice likes sushi"),
                ],
                [],
            ]
        )

        llm_resp = MagicMock()
        llm_resp.choices = [MagicMock()]
        llm_resp.choices[0].message.content = json.dumps({"contradicted_uuids": ["fact-a"]})

        with patch("litellm.acompletion", new_callable=AsyncMock, return_value=llm_resp):
            result = await invalidator.find_and_invalidate(**COMMON_KWARGS)

        assert result == ["fact-a"]
        assert mock_driver.execute_write.call_count == 1
        assert mock_driver.execute_write.call_args[0][1]["uuid"] == "fact-a"

    @pytest.mark.asyncio
    async def test_reverse_direction_candidates(self, invalidator: EdgeInvalidator, mock_driver: AsyncMock) -> None:
        mock_driver.execute_query = AsyncMock(
            side_effect=[
                [],
                [_make_candidate("rev-fact-1", "Tokyo is home to Alice")],
            ]
        )

        llm_resp = MagicMock()
        llm_resp.choices = [MagicMock()]
        llm_resp.choices[0].message.content = json.dumps({"contradicted_uuids": ["rev-fact-1"]})

        with patch("litellm.acompletion", new_callable=AsyncMock, return_value=llm_resp):
            result = await invalidator.find_and_invalidate(**COMMON_KWARGS)

        assert result == ["rev-fact-1"]
        assert mock_driver.execute_query.call_count == 2
        mock_driver.execute_write.assert_called_once()


# ── TestParseInvalidationResponse ───────────────────────────


class TestParseInvalidationResponse:
    def test_parse_valid_json(self) -> None:
        text = '{"contradicted_uuids": ["abc"]}'
        assert EdgeInvalidator._parse_invalidation_response(text) == ["abc"]

    def test_parse_json_in_fence(self) -> None:
        text = '```json\n{"contradicted_uuids": ["abc"]}\n```'
        assert EdgeInvalidator._parse_invalidation_response(text) == ["abc"]

    def test_parse_empty_list(self) -> None:
        text = '{"contradicted_uuids": []}'
        assert EdgeInvalidator._parse_invalidation_response(text) == []

    def test_parse_invalid_json(self) -> None:
        text = "this is not json at all"
        assert EdgeInvalidator._parse_invalidation_response(text) == []

    def test_parse_list_format(self) -> None:
        text = '["abc", "def"]'
        assert EdgeInvalidator._parse_invalidation_response(text) == ["abc", "def"]


# ── TestTemporalQueries ─────────────────────────────────────


class TestTemporalQueries:
    def test_find_active_facts_query_has_null_check(self) -> None:
        assert "invalid_at IS NULL" in FIND_ACTIVE_FACTS_FOR_PAIR

    def test_invalidate_fact_query_sets_invalid_at(self) -> None:
        assert "SET r.invalid_at" in INVALIDATE_FACT

    def test_find_valid_facts_has_temporal_filter(self) -> None:
        assert "invalid_at IS NULL OR r.invalid_at >" in FIND_VALID_FACTS_BY_GROUP


# ── TestIngestIntegration ───────────────────────────────────


class TestIngestIntegration:
    @pytest.mark.asyncio
    async def test_ingest_text_calls_invalidator(self, tmp_path) -> None:
        from core.memory.backend.neo4j_graph import Neo4jGraphBackend

        anima_dir = tmp_path / "animas" / "test-anima"
        anima_dir.mkdir(parents=True)

        backend = Neo4jGraphBackend(anima_dir, group_id="test_group")

        mock_driver = AsyncMock()
        mock_driver.execute_write = AsyncMock()
        mock_driver.execute_query = AsyncMock(return_value=[])
        backend._driver = mock_driver
        backend._schema_ensured = True

        mock_entity = MagicMock(name="Alice", summary="A person")
        mock_entity.name = "Alice"
        mock_fact = MagicMock()
        mock_fact.source_entity = "Alice"
        mock_fact.target_entity = "Alice"
        mock_fact.fact = "Alice lives in Osaka"
        mock_fact.valid_at = "2025-06-01T00:00:00"

        mock_extractor = MagicMock()
        mock_extractor.extract_entities = AsyncMock(return_value=[mock_entity])
        mock_extractor.extract_facts = AsyncMock(return_value=[mock_fact])

        mock_resolver_result = MagicMock()
        mock_resolver_result.uuid = "entity-alice"
        mock_resolver_result.name = "Alice"
        mock_resolver_result.summary = "A person"
        mock_resolver_result.is_new = True

        mock_resolver = MagicMock()
        mock_resolver.resolve = AsyncMock(return_value=mock_resolver_result)

        mock_invalidator = MagicMock()
        mock_invalidator.find_and_invalidate = AsyncMock(return_value=[])

        backend._extractor = mock_extractor
        backend._resolver = mock_resolver
        backend._invalidator = mock_invalidator

        result = await backend.ingest_text("Alice moved to Osaka", source="test")
        assert result >= 1
        mock_invalidator.find_and_invalidate.assert_called_once()

    @pytest.mark.asyncio
    async def test_ingest_text_invalidation_failure_doesnt_break_ingest(self, tmp_path) -> None:
        from core.memory.backend.neo4j_graph import Neo4jGraphBackend

        anima_dir = tmp_path / "animas" / "test-anima"
        anima_dir.mkdir(parents=True)

        backend = Neo4jGraphBackend(anima_dir, group_id="test_group")

        mock_driver = AsyncMock()
        mock_driver.execute_write = AsyncMock()
        mock_driver.execute_query = AsyncMock(return_value=[])
        backend._driver = mock_driver
        backend._schema_ensured = True

        mock_entity = MagicMock()
        mock_entity.name = "Bob"
        mock_fact = MagicMock()
        mock_fact.source_entity = "Bob"
        mock_fact.target_entity = "Bob"
        mock_fact.fact = "Bob works at Acme"
        mock_fact.valid_at = "2025-07-01T00:00:00"

        mock_extractor = MagicMock()
        mock_extractor.extract_entities = AsyncMock(return_value=[mock_entity])
        mock_extractor.extract_facts = AsyncMock(return_value=[mock_fact])

        mock_resolver_result = MagicMock()
        mock_resolver_result.uuid = "entity-bob"
        mock_resolver_result.name = "Bob"
        mock_resolver_result.summary = "A worker"
        mock_resolver_result.is_new = True

        mock_resolver = MagicMock()
        mock_resolver.resolve = AsyncMock(return_value=mock_resolver_result)

        mock_invalidator = MagicMock()
        mock_invalidator.find_and_invalidate = AsyncMock(side_effect=RuntimeError("invalidation exploded"))

        backend._extractor = mock_extractor
        backend._resolver = mock_resolver
        backend._invalidator = mock_invalidator

        result = await backend.ingest_text("Bob joined Acme", source="test")
        assert result == 3  # 1 episode + 1 entity + 1 fact
