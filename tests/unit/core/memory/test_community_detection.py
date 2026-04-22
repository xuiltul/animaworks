"""Unit tests for community detection (CommunityDetector + Community dataclass)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ── TestCommunity ──────────────────────────────────────────────────────────


class TestCommunity:
    def test_dataclass_fields(self):
        from core.memory.graph.community import Community

        c = Community(
            uuid="c1",
            name="Team Alpha",
            summary="A team of entities",
            member_uuids=["e1", "e2", "e3"],
            member_names=["Entity1", "Entity2", "Entity3"],
        )
        assert c.uuid == "c1"
        assert c.name == "Team Alpha"
        assert c.summary == "A team of entities"
        assert c.member_uuids == ["e1", "e2", "e3"]
        assert c.member_names == ["Entity1", "Entity2", "Entity3"]


# ── TestCommunityDetectorInit ──────────────────────────────────────────────


class TestCommunityDetectorInit:
    def test_creates_with_driver(self):
        from core.memory.graph.community import CommunityDetector

        driver = AsyncMock()
        detector = CommunityDetector(driver, "group")
        assert detector is not None

    def test_default_min_members(self):
        from core.memory.graph.community import CommunityDetector

        driver = AsyncMock()
        detector = CommunityDetector(driver, "group")
        assert detector._min_members == 3


# ── TestDetectAndStore ─────────────────────────────────────────────────────


class TestDetectAndStore:
    @pytest.mark.asyncio
    async def test_too_few_entities(self):
        from core.memory.graph.community import CommunityDetector

        mock_driver = AsyncMock()
        mock_driver.execute_query = AsyncMock(
            side_effect=[
                [
                    {"uuid": "e1", "name": "Entity1", "summary": "S1"},
                    {"uuid": "e2", "name": "Entity2", "summary": "S2"},
                ],
                [],
            ]
        )

        detector = CommunityDetector(mock_driver, "group", min_members=3)
        result = await detector.detect_and_store()
        assert result == []

    @pytest.mark.asyncio
    @patch("litellm.acompletion")
    async def test_detects_community(self, mock_llm):
        from core.memory.graph.community import CommunityDetector

        entities = [{"uuid": f"e{i}", "name": f"Entity{i}", "summary": f"Summary{i}"} for i in range(5)]
        edges = [
            {"source_uuid": "e0", "target_uuid": "e1", "edge_uuid": "r1"},
            {"source_uuid": "e1", "target_uuid": "e2", "edge_uuid": "r2"},
            {"source_uuid": "e2", "target_uuid": "e3", "edge_uuid": "r3"},
            {"source_uuid": "e3", "target_uuid": "e4", "edge_uuid": "r4"},
            {"source_uuid": "e0", "target_uuid": "e2", "edge_uuid": "r5"},
        ]

        mock_driver = AsyncMock()
        mock_driver.execute_query = AsyncMock(side_effect=[entities, edges])
        mock_driver.execute_write = AsyncMock()

        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock()]
        mock_resp.choices[0].message.content = '{"name": "Team Alpha", "summary": "A group of related entities"}'
        mock_llm.return_value = mock_resp

        detector = CommunityDetector(mock_driver, "group", min_members=3)
        result = await detector.detect_and_store()

        assert len(result) >= 1
        assert result[0].name == "Team Alpha"
        assert len(result[0].member_uuids) >= 3

    @pytest.mark.asyncio
    @patch("litellm.acompletion")
    async def test_filters_small_communities(self, mock_llm):
        """LPA returns a community with 2 members → filtered out by min_members=3."""
        from core.memory.graph.community import CommunityDetector

        entities = [{"uuid": f"e{i}", "name": f"Entity{i}", "summary": f"Summary{i}"} for i in range(4)]
        edges = [
            {"source_uuid": "e0", "target_uuid": "e1", "edge_uuid": "r1"},
            {"source_uuid": "e2", "target_uuid": "e3", "edge_uuid": "r2"},
        ]

        mock_driver = AsyncMock()
        mock_driver.execute_query = AsyncMock(side_effect=[entities, edges])
        mock_driver.execute_write = AsyncMock()

        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock()]
        mock_resp.choices[0].message.content = '{"name": "Duo", "summary": "Two"}'
        mock_llm.return_value = mock_resp

        detector = CommunityDetector(mock_driver, "group", min_members=3)
        result = await detector.detect_and_store()
        assert all(len(c.member_uuids) >= 3 for c in result)

    @pytest.mark.asyncio
    @patch("litellm.acompletion")
    async def test_llm_summary_failure_fallback(self, mock_llm):
        from core.memory.graph.community import CommunityDetector

        entities = [{"uuid": f"e{i}", "name": f"Entity{i}", "summary": f"Summary{i}"} for i in range(5)]
        edges = [
            {"source_uuid": "e0", "target_uuid": "e1", "edge_uuid": "r1"},
            {"source_uuid": "e1", "target_uuid": "e2", "edge_uuid": "r2"},
            {"source_uuid": "e2", "target_uuid": "e3", "edge_uuid": "r3"},
            {"source_uuid": "e3", "target_uuid": "e4", "edge_uuid": "r4"},
            {"source_uuid": "e0", "target_uuid": "e2", "edge_uuid": "r5"},
        ]

        mock_driver = AsyncMock()
        mock_driver.execute_query = AsyncMock(side_effect=[entities, edges])
        mock_driver.execute_write = AsyncMock()

        mock_llm.side_effect = Exception("LLM unavailable")

        detector = CommunityDetector(mock_driver, "group", min_members=3)
        result = await detector.detect_and_store()

        assert len(result) >= 1
        for c in result:
            assert c.name
            assert c.summary

    @pytest.mark.asyncio
    @patch("litellm.acompletion")
    async def test_deletes_old_communities(self, mock_llm):
        from core.memory.graph.community import CommunityDetector
        from core.memory.graph.queries import DELETE_COMMUNITIES_BY_GROUP

        entities = [{"uuid": f"e{i}", "name": f"Entity{i}", "summary": f"Summary{i}"} for i in range(5)]
        edges = [
            {"source_uuid": "e0", "target_uuid": "e1", "edge_uuid": "r1"},
            {"source_uuid": "e1", "target_uuid": "e2", "edge_uuid": "r2"},
            {"source_uuid": "e2", "target_uuid": "e3", "edge_uuid": "r3"},
            {"source_uuid": "e3", "target_uuid": "e4", "edge_uuid": "r4"},
            {"source_uuid": "e0", "target_uuid": "e2", "edge_uuid": "r5"},
        ]

        mock_driver = AsyncMock()
        mock_driver.execute_query = AsyncMock(side_effect=[entities, edges])
        mock_driver.execute_write = AsyncMock()

        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock()]
        mock_resp.choices[0].message.content = '{"name": "X", "summary": "Y"}'
        mock_llm.return_value = mock_resp

        detector = CommunityDetector(mock_driver, "group", min_members=3)
        await detector.detect_and_store()

        delete_calls = [
            c for c in mock_driver.execute_write.call_args_list if c.args and DELETE_COMMUNITIES_BY_GROUP in c.args[0]
        ]
        assert len(delete_calls) >= 1

    @pytest.mark.asyncio
    @patch("litellm.acompletion")
    async def test_creates_has_member_edges(self, mock_llm):
        from core.memory.graph.community import CommunityDetector
        from core.memory.graph.queries import CREATE_HAS_MEMBER

        entities = [{"uuid": f"e{i}", "name": f"Entity{i}", "summary": f"Summary{i}"} for i in range(5)]
        edges = [
            {"source_uuid": "e0", "target_uuid": "e1", "edge_uuid": "r1"},
            {"source_uuid": "e1", "target_uuid": "e2", "edge_uuid": "r2"},
            {"source_uuid": "e2", "target_uuid": "e3", "edge_uuid": "r3"},
            {"source_uuid": "e3", "target_uuid": "e4", "edge_uuid": "r4"},
            {"source_uuid": "e0", "target_uuid": "e2", "edge_uuid": "r5"},
        ]

        mock_driver = AsyncMock()
        mock_driver.execute_query = AsyncMock(side_effect=[entities, edges])
        mock_driver.execute_write = AsyncMock()

        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock()]
        mock_resp.choices[0].message.content = '{"name": "X", "summary": "Y"}'
        mock_llm.return_value = mock_resp

        detector = CommunityDetector(mock_driver, "group", min_members=3)
        result = await detector.detect_and_store()

        member_calls = [
            c for c in mock_driver.execute_write.call_args_list if c.args and CREATE_HAS_MEMBER in c.args[0]
        ]
        total_members = sum(len(c.member_uuids) for c in result)
        assert len(member_calls) == total_members


# ── TestDynamicUpdate ──────────────────────────────────────────────────────


class TestDynamicUpdate:
    @pytest.mark.asyncio
    async def test_assigns_to_majority_community(self):
        from core.memory.graph.community import CommunityDetector

        mock_driver = AsyncMock()
        mock_driver.execute_query = AsyncMock(
            side_effect=[
                [
                    {"community_uuid": "cA", "name": "ComA", "summary": "A"},
                ],
                [
                    {"community_uuid": "cA", "name": "ComA", "summary": "A"},
                ],
                [
                    {"community_uuid": "cB", "name": "ComB", "summary": "B"},
                ],
            ]
        )
        mock_driver.execute_write = AsyncMock()

        detector = CommunityDetector(mock_driver, "group")
        result = await detector.dynamic_update("e_new", ["n1", "n2", "n3"])
        assert result == "cA"

    @pytest.mark.asyncio
    async def test_no_neighbors_returns_none(self):
        from core.memory.graph.community import CommunityDetector

        mock_driver = AsyncMock()
        detector = CommunityDetector(mock_driver, "group")
        result = await detector.dynamic_update("e_new", [])
        assert result is None

    @pytest.mark.asyncio
    async def test_no_community_neighbors_returns_none(self):
        from core.memory.graph.community import CommunityDetector

        mock_driver = AsyncMock()
        mock_driver.execute_query = AsyncMock(return_value=[])

        detector = CommunityDetector(mock_driver, "group")
        result = await detector.dynamic_update("e_new", ["n1", "n2"])
        assert result is None

    @pytest.mark.asyncio
    async def test_creates_has_member(self):
        from core.memory.graph.community import CommunityDetector
        from core.memory.graph.queries import CREATE_HAS_MEMBER

        mock_driver = AsyncMock()
        mock_driver.execute_query = AsyncMock(
            return_value=[
                {"community_uuid": "cA", "name": "ComA", "summary": "A"},
            ]
        )
        mock_driver.execute_write = AsyncMock()

        detector = CommunityDetector(mock_driver, "group")
        result = await detector.dynamic_update("e_new", ["n1"])

        assert result == "cA"
        member_calls = [
            c for c in mock_driver.execute_write.call_args_list if c.args and CREATE_HAS_MEMBER in c.args[0]
        ]
        assert len(member_calls) == 1
        params = (
            member_calls[0].args[1] if len(member_calls[0].args) > 1 else member_calls[0].kwargs.get("parameters", {})
        )
        assert params.get("entity_uuid") == "e_new"


# ── TestParseCommunityResponse ─────────────────────────────────────────────


class TestParseCommunityResponse:
    def test_parse_valid_json(self):
        from core.memory.graph.community import CommunityDetector

        name, summary = CommunityDetector._parse_community_response('{"name": "X", "summary": "Y"}')
        assert name == "X"
        assert summary == "Y"

    def test_parse_json_in_fence(self):
        from core.memory.graph.community import CommunityDetector

        text = '```json\n{"name": "Fenced", "summary": "In a fence"}\n```'
        name, summary = CommunityDetector._parse_community_response(text)
        assert name == "Fenced"
        assert summary == "In a fence"

    def test_parse_plain_text_fallback(self):
        from core.memory.graph.community import CommunityDetector

        text = "My Community Name\nThis is the description of the community."
        name, summary = CommunityDetector._parse_community_response(text)
        assert name == "My Community Name"
        assert "description" in summary.lower()

    def test_parse_empty(self):
        from core.memory.graph.community import CommunityDetector

        name, summary = CommunityDetector._parse_community_response("")
        assert isinstance(name, str)
        assert isinstance(summary, str)


# ── TestRetrieveCommunity ──────────────────────────────────────────────────


class TestRetrieveCommunity:
    @pytest.mark.asyncio
    async def test_retrieve_community_scope(self):
        """Mock backend with community data, verify content format."""
        from core.memory.backend.neo4j_graph import Neo4jGraphBackend

        backend = Neo4jGraphBackend(
            anima_dir=MagicMock(name="test_anima"),
        )

        community_rows = [
            {"uuid": "c1", "name": "Team Alpha", "summary": "A team"},
            {"uuid": "c2", "name": "Team Beta", "summary": "B team"},
        ]

        mock_driver = AsyncMock()
        mock_driver.execute_query = AsyncMock(return_value=community_rows)
        backend._driver = mock_driver

        results = await backend.retrieve("test query", scope="community", limit=10)

        assert len(results) == 2
        assert "[Team Alpha] A team" in results[0].content
        assert "[Team Beta] B team" in results[1].content
        assert results[0].source.startswith("community:")
