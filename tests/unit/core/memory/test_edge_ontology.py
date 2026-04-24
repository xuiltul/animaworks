# Copyright 2026 AnimaWorks
# Licensed under the Apache License, Version 2.0
"""Tests for Issue #21 — Edge Ontology Extension."""

from __future__ import annotations

import json
from typing import get_args
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.memory.ontology.default import (
    DEFAULT_EDGE_TYPE,
    EDGE_TYPE_DESCRIPTIONS,
    EDGE_TYPES,
    ExtractedFact,
)


# ── Ontology definition tests ─────────────────────────────────────


class TestEdgeTypeDefinitions:
    """Verify EDGE_TYPES Literal and companion data."""

    def test_edge_types_contains_required_types(self) -> None:
        valid = frozenset(get_args(EDGE_TYPES))
        expected = {
            "WORKS_AT",
            "LIVES_IN",
            "KNOWS",
            "PREFERS",
            "SKILLED_IN",
            "PARTICIPATED_IN",
            "CREATED",
            "REPORTED",
            "DEPENDS_ON",
            "RELATES_TO",
        }
        assert expected == valid

    def test_default_edge_type_is_relates_to(self) -> None:
        assert DEFAULT_EDGE_TYPE == "RELATES_TO"

    def test_descriptions_cover_all_types(self) -> None:
        valid = frozenset(get_args(EDGE_TYPES))
        assert set(EDGE_TYPE_DESCRIPTIONS.keys()) == valid

    def test_descriptions_are_non_empty(self) -> None:
        for k, v in EDGE_TYPE_DESCRIPTIONS.items():
            assert v.strip(), f"Description for {k} is empty"


# ── ExtractedFact model tests ─────────────────────────────────────


class TestExtractedFactEdgeType:
    """Verify ExtractedFact edge_type field."""

    def test_default_edge_type(self) -> None:
        fact = ExtractedFact(
            source_entity="Alice",
            target_entity="Bob",
            fact="Alice knows Bob",
        )
        assert fact.edge_type == "RELATES_TO"

    def test_explicit_edge_type(self) -> None:
        fact = ExtractedFact(
            source_entity="Alice",
            target_entity="Acme",
            fact="Alice works at Acme",
            edge_type="WORKS_AT",
        )
        assert fact.edge_type == "WORKS_AT"

    def test_edge_type_in_serialization(self) -> None:
        fact = ExtractedFact(
            source_entity="Alice",
            target_entity="Tokyo",
            fact="Alice lives in Tokyo",
            edge_type="LIVES_IN",
        )
        data = fact.model_dump(mode="json")
        assert data["edge_type"] == "LIVES_IN"

    def test_edge_type_from_json(self) -> None:
        raw = {
            "source_entity": "Alice",
            "target_entity": "Python",
            "fact": "Alice is skilled in Python",
            "edge_type": "SKILLED_IN",
        }
        fact = ExtractedFact.model_validate(raw)
        assert fact.edge_type == "SKILLED_IN"

    def test_missing_edge_type_defaults_to_relates_to(self) -> None:
        raw = {
            "source_entity": "A",
            "target_entity": "B",
            "fact": "A relates to B",
        }
        fact = ExtractedFact.model_validate(raw)
        assert fact.edge_type == "RELATES_TO"


# ── Extractor validation tests ────────────────────────────────────


class TestExtractorEdgeTypeValidation:
    """Verify edge_type validation in FactExtractor."""

    @pytest.mark.asyncio
    async def test_invalid_edge_type_falls_back_to_default(self) -> None:
        from core.memory.extraction.extractor import FactExtractor

        extractor = FactExtractor(model="test-model")
        llm_response = json.dumps(
            {
                "facts": [
                    {
                        "source_entity": "Alice",
                        "target_entity": "Bob",
                        "fact": "Alice mentors Bob",
                        "edge_type": "MENTORS",
                    }
                ]
            }
        )

        with patch.object(extractor, "_call_llm", new_callable=AsyncMock, return_value=llm_response):
            entities = [
                MagicMock(name="Alice", spec=["name", "model_dump"]),
                MagicMock(name="Bob", spec=["name", "model_dump"]),
            ]
            entities[0].name = "Alice"
            entities[1].name = "Bob"
            entities[0].model_dump = lambda mode="json": {"name": "Alice", "entity_type": "Person", "summary": ""}
            entities[1].model_dump = lambda mode="json": {"name": "Bob", "entity_type": "Person", "summary": ""}

            facts = await extractor.extract_facts("Alice mentors Bob", entities)

        assert len(facts) == 1
        assert facts[0].edge_type == "RELATES_TO"

    @pytest.mark.asyncio
    async def test_valid_edge_type_preserved(self) -> None:
        from core.memory.extraction.extractor import FactExtractor

        extractor = FactExtractor(model="test-model")
        llm_response = json.dumps(
            {
                "facts": [
                    {
                        "source_entity": "Alice",
                        "target_entity": "Acme",
                        "fact": "Alice works at Acme",
                        "edge_type": "WORKS_AT",
                    }
                ]
            }
        )

        with patch.object(extractor, "_call_llm", new_callable=AsyncMock, return_value=llm_response):
            entities = [
                MagicMock(name="Alice", spec=["name", "model_dump"]),
                MagicMock(name="Acme", spec=["name", "model_dump"]),
            ]
            entities[0].name = "Alice"
            entities[1].name = "Acme"
            entities[0].model_dump = lambda mode="json": {"name": "Alice", "entity_type": "Person", "summary": ""}
            entities[1].model_dump = lambda mode="json": {"name": "Acme", "entity_type": "Organization", "summary": ""}

            facts = await extractor.extract_facts("Alice works at Acme", entities)

        assert len(facts) == 1
        assert facts[0].edge_type == "WORKS_AT"


# ── Prompt template tests ─────────────────────────────────────────


class TestPromptEdgeTypes:
    """Verify prompt templates include edge type instructions."""

    def test_ja_prompt_has_edge_types_placeholder(self) -> None:
        from core.memory.extraction.prompts import ja

        assert "{edge_types_list}" in ja.FACT_USER
        assert "edge_type" in ja.FACT_USER

    def test_en_prompt_has_edge_types_placeholder(self) -> None:
        from core.memory.extraction.prompts import en

        assert "{edge_types_list}" in en.FACT_USER
        assert "edge_type" in en.FACT_USER

    def test_ja_prompt_formats_correctly(self) -> None:
        from core.memory.extraction.prompts import ja

        edge_types_list = "\n".join(f"- `{k}`: {v}" for k, v in EDGE_TYPE_DESCRIPTIONS.items())
        result = ja.FACT_USER.format(
            content="test content",
            entities_json="[]",
            edge_types_list=edge_types_list,
        )
        assert "WORKS_AT" in result
        assert "RELATES_TO" in result

    def test_en_prompt_formats_correctly(self) -> None:
        from core.memory.extraction.prompts import en

        edge_types_list = "\n".join(f"- `{k}`: {v}" for k, v in EDGE_TYPE_DESCRIPTIONS.items())
        result = en.FACT_USER.format(
            content="test content",
            entities_json="[]",
            edge_types_list=edge_types_list,
        )
        assert "WORKS_AT" in result
        assert "RELATES_TO" in result


# ── Cypher query tests ────────────────────────────────────────────


class TestQueriesEdgeType:
    """Verify Cypher queries include edge_type."""

    def test_create_fact_has_edge_type(self) -> None:
        from core.memory.graph.queries import CREATE_FACT

        assert "edge_type" in CREATE_FACT
        assert "$edge_type" in CREATE_FACT

    def test_vector_search_facts_returns_edge_type(self) -> None:
        from core.memory.graph.queries import VECTOR_SEARCH_FACTS

        assert "edge_type" in VECTOR_SEARCH_FACTS
        assert "coalesce(r.edge_type, 'RELATES_TO')" in VECTOR_SEARCH_FACTS

    def test_fulltext_search_facts_returns_edge_type(self) -> None:
        from core.memory.graph.queries import FULLTEXT_SEARCH_FACTS

        assert "edge_type" in FULLTEXT_SEARCH_FACTS
        assert "coalesce(r.edge_type, 'RELATES_TO')" in FULLTEXT_SEARCH_FACTS

    def test_bfs_facts_returns_edge_type(self) -> None:
        from core.memory.graph.queries import BFS_FACTS_FROM_ENTITY

        assert "edge_type" in BFS_FACTS_FROM_ENTITY

    def test_bfs_facts_query_returns_edge_type(self) -> None:
        from core.memory.graph.queries import bfs_facts_query

        query = bfs_facts_query(3)
        assert "edge_type" in query
        assert "coalesce(r.edge_type, 'RELATES_TO')" in query

    def test_find_valid_facts_returns_edge_type(self) -> None:
        from core.memory.graph.queries import FIND_VALID_FACTS_BY_GROUP

        assert "edge_type" in FIND_VALID_FACTS_BY_GROUP

    def test_find_recent_facts_returns_edge_type(self) -> None:
        from core.memory.graph.queries import FIND_RECENT_FACTS

        assert "edge_type" in FIND_RECENT_FACTS

    def test_redirect_outgoing_preserves_edge_type(self) -> None:
        from core.memory.graph.queries import REDIRECT_OUTGOING_FACTS

        assert "edge_type" in REDIRECT_OUTGOING_FACTS
        assert "coalesce(old.edge_type, 'RELATES_TO')" in REDIRECT_OUTGOING_FACTS

    def test_redirect_incoming_preserves_edge_type(self) -> None:
        from core.memory.graph.queries import REDIRECT_INCOMING_FACTS

        assert "edge_type" in REDIRECT_INCOMING_FACTS
        assert "coalesce(old.edge_type, 'RELATES_TO')" in REDIRECT_INCOMING_FACTS

    def test_fetch_edges_for_community_returns_edge_type(self) -> None:
        from core.memory.graph.queries import FETCH_EDGES_FOR_COMMUNITY

        assert "edge_type" in FETCH_EDGES_FOR_COMMUNITY


# ── HybridSearch edge_type_filter tests ───────────────────────────


class TestHybridSearchEdgeTypeFilter:
    """Verify HybridSearch.search filters by edge_type."""

    @pytest.mark.asyncio
    async def test_edge_type_filter_removes_non_matching(self) -> None:
        from core.memory.graph.search import HybridSearch

        mock_driver = MagicMock()
        mock_driver.execute_query = AsyncMock(return_value=[])
        search = HybridSearch(mock_driver, "test-group")

        results_with_types = [
            {"uuid": "1", "fact": "A works at B", "edge_type": "WORKS_AT", "rrf_score": 0.5},
            {"uuid": "2", "fact": "A knows C", "edge_type": "KNOWS", "rrf_score": 0.4},
            {"uuid": "3", "fact": "A relates to D", "edge_type": "RELATES_TO", "rrf_score": 0.3},
        ]

        with (
            patch("core.memory.graph.search.asyncio.gather", new_callable=AsyncMock) as mock_gather,
            patch("core.memory.graph.rrf.rrf_merge", return_value=results_with_types),
        ):
            mock_gather.return_value = [results_with_types, [], []]

            results = await search.search(
                "test query",
                scope="fact",
                query_embedding=[0.1] * 384,
                edge_type_filter="WORKS_AT",
            )

        assert all(r.get("edge_type") == "WORKS_AT" for r in results)

    @pytest.mark.asyncio
    async def test_no_filter_returns_all_types(self) -> None:
        from core.memory.graph.search import HybridSearch

        mock_driver = MagicMock()
        mock_driver.execute_query = AsyncMock(return_value=[])
        search = HybridSearch(mock_driver, "test-group")

        results_with_types = [
            {"uuid": "1", "fact": "A works at B", "edge_type": "WORKS_AT", "rrf_score": 0.5},
            {"uuid": "2", "fact": "A knows C", "edge_type": "KNOWS", "rrf_score": 0.4},
        ]

        with (
            patch("core.memory.graph.search.asyncio.gather", new_callable=AsyncMock) as mock_gather,
            patch("core.memory.graph.rrf.rrf_merge", return_value=results_with_types),
        ):
            mock_gather.return_value = [results_with_types, [], []]

            results = await search.search(
                "test query",
                scope="fact",
                query_embedding=[0.1] * 384,
            )

        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_edge_type_filter_ignored_for_entity_scope(self) -> None:
        from core.memory.graph.search import HybridSearch

        mock_driver = MagicMock()
        mock_driver.execute_query = AsyncMock(return_value=[])
        search = HybridSearch(mock_driver, "test-group")

        entity_results = [
            {"uuid": "1", "name": "Alice", "summary": "A person", "rrf_score": 0.5},
        ]

        with (
            patch("core.memory.graph.search.asyncio.gather", new_callable=AsyncMock) as mock_gather,
            patch("core.memory.graph.rrf.rrf_merge", return_value=entity_results),
        ):
            mock_gather.return_value = [entity_results, [], []]

            results = await search.search(
                "test query",
                scope="entity",
                query_embedding=[0.1] * 384,
                edge_type_filter="WORKS_AT",
            )

        assert len(results) == 1


# ── Backend edge_type passthrough tests ───────────────────────────


class TestBackendEdgeType:
    """Verify Neo4jGraphBackend passes edge_type to CREATE_FACT."""

    @pytest.mark.asyncio
    async def test_ingest_passes_edge_type_to_create_fact(self) -> None:
        from pathlib import Path

        from core.memory.backend.neo4j_graph import Neo4jGraphBackend

        backend = Neo4jGraphBackend(Path("/tmp/test-anima"), group_id="test")

        mock_driver = AsyncMock()
        mock_driver.execute_query = AsyncMock(return_value=[])
        mock_driver.execute_write = AsyncMock(return_value=None)
        mock_driver.health_check = AsyncMock(return_value=True)
        backend._driver = mock_driver
        backend._schema_ensured = True

        mock_extractor = AsyncMock()
        mock_entity = MagicMock()
        mock_entity.name = "Alice"
        mock_entity.entity_type = "Person"
        mock_entity.summary = "A person"
        mock_entity.model_dump = lambda mode="json": {"name": "Alice", "entity_type": "Person", "summary": "A person"}

        mock_fact = MagicMock()
        mock_fact.source_entity = "Alice"
        mock_fact.target_entity = "Alice"
        mock_fact.fact = "Alice is a person"
        mock_fact.valid_at = None
        mock_fact.edge_type = "RELATES_TO"

        mock_extractor.extract_entities = AsyncMock(return_value=[mock_entity])
        mock_extractor.extract_facts = AsyncMock(return_value=[mock_fact])
        backend._extractor = mock_extractor

        mock_resolver = MagicMock()
        resolved = MagicMock()
        resolved.uuid = "entity-uuid-1"
        resolved.name = "Alice"
        resolved.summary = "A person"
        resolved.is_new = True
        mock_resolver.resolve = AsyncMock(return_value=resolved)
        backend._resolver = mock_resolver

        backend._embedding_available = False

        await backend.ingest_text("Alice is a person", source="test")

        create_fact_calls = [
            call
            for call in mock_driver.execute_write.call_args_list
            if call.args and "edge_type" in str(call.args[0])
        ]
        assert len(create_fact_calls) >= 1
        params = create_fact_calls[0].args[1]
        assert "edge_type" in params
        assert params["edge_type"] == "RELATES_TO"

    def test_retrieve_content_includes_edge_type_label(self) -> None:
        """Verify fact content format includes edge type."""
        result_dict = {
            "uuid": "fact-1",
            "fact": "Alice works at Acme",
            "source_name": "Alice",
            "target_name": "Acme",
            "edge_type": "WORKS_AT",
            "valid_at": "2026-01-01T00:00:00",
            "rrf_score": 0.8,
        }

        edge_label = result_dict.get("edge_type", "RELATES_TO")
        content = f"{result_dict.get('source_name', '')} -[{edge_label}]-> {result_dict.get('target_name', '')}: {result_dict.get('fact', '')}"

        assert "WORKS_AT" in content
        assert "-[WORKS_AT]->" in content

    def test_backward_compat_missing_edge_type(self) -> None:
        """Facts without edge_type should be treated as RELATES_TO."""
        result_dict = {
            "uuid": "old-fact",
            "fact": "old relationship",
            "source_name": "A",
            "target_name": "B",
            "valid_at": "2026-01-01",
            "rrf_score": 0.5,
        }
        edge_label = result_dict.get("edge_type", "RELATES_TO")
        assert edge_label == "RELATES_TO"


# ── Import validation ─────────────────────────────────────────────


class TestOntologyImports:
    """Verify re-exports from __init__.py."""

    def test_import_edge_types_from_init(self) -> None:
        from core.memory.ontology import EDGE_TYPES

        assert EDGE_TYPES is not None

    def test_import_edge_type_descriptions_from_init(self) -> None:
        from core.memory.ontology import EDGE_TYPE_DESCRIPTIONS

        assert isinstance(EDGE_TYPE_DESCRIPTIONS, dict)

    def test_import_default_edge_type_from_init(self) -> None:
        from core.memory.ontology import DEFAULT_EDGE_TYPE

        assert DEFAULT_EDGE_TYPE == "RELATES_TO"
