"""Tests for Issue #17 — Community Detection scheduling.

Covers:
- Batch community detection in system consolidation
- Dynamic community update in ingest_text
- Backend check (legacy vs neo4j)
- Error resilience
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ── TestDynamicCommunityUpdateInIngest ────────────────────


class TestDynamicCommunityUpdateInIngest:
    """Test that ingest_text calls dynamic_update for new entities."""

    @pytest.fixture
    def _setup_backend(self, tmp_path):
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
        mock_entity.name = "Alice"
        mock_entity.entity_type = "Person"
        mock_entity.summary = "A person"

        mock_extractor = MagicMock()
        mock_extractor.extract_entities = AsyncMock(return_value=[mock_entity])
        mock_extractor.extract_facts = AsyncMock(return_value=[])

        mock_resolved = MagicMock()
        mock_resolved.uuid = "ent-alice"
        mock_resolved.name = "Alice"
        mock_resolved.summary = "A person"
        mock_resolved.is_new = True

        mock_resolver = MagicMock()
        mock_resolver.resolve = AsyncMock(return_value=mock_resolved)

        backend._extractor = mock_extractor
        backend._resolver = mock_resolver

        return backend, mock_driver

    @pytest.mark.asyncio
    async def test_ingest_calls_dynamic_update(self, _setup_backend) -> None:
        backend, mock_driver = _setup_backend

        with patch.object(backend, "_dynamic_community_update", new_callable=AsyncMock) as mock_update:
            await backend.ingest_text("Alice is a person", source="test")
            mock_update.assert_called_once()
            args = mock_update.call_args[0]
            assert args[0] is mock_driver
            assert "ent-alice" in args[1]

    @pytest.mark.asyncio
    async def test_ingest_skips_update_when_no_new_entities(self, tmp_path) -> None:
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
        mock_entity.name = "Alice"
        mock_entity.entity_type = "Person"
        mock_entity.summary = "A person"

        mock_extractor = MagicMock()
        mock_extractor.extract_entities = AsyncMock(return_value=[mock_entity])
        mock_extractor.extract_facts = AsyncMock(return_value=[])

        mock_resolved = MagicMock()
        mock_resolved.uuid = "ent-alice"
        mock_resolved.name = "Alice"
        mock_resolved.summary = "A person"
        mock_resolved.is_new = False

        mock_resolver = MagicMock()
        mock_resolver.resolve = AsyncMock(return_value=mock_resolved)

        backend._extractor = mock_extractor
        backend._resolver = mock_resolver

        with patch.object(backend, "_dynamic_community_update", new_callable=AsyncMock) as mock_update:
            await backend.ingest_text("Alice is a person", source="test")
            mock_update.assert_not_called()

    @pytest.mark.asyncio
    async def test_dynamic_update_failure_doesnt_break_ingest(self, _setup_backend) -> None:
        backend, _ = _setup_backend

        with patch.object(
            backend,
            "_dynamic_community_update",
            new_callable=AsyncMock,
            side_effect=RuntimeError("community error"),
        ):
            result = await backend.ingest_text("Alice is here", source="test")
            assert result >= 1


# ── TestDynamicCommunityUpdateMethod ─────────────────────


class TestDynamicCommunityUpdateMethod:
    """Test _dynamic_community_update method directly."""

    @pytest.mark.asyncio
    async def test_finds_neighbors_and_calls_detector(self, tmp_path) -> None:
        from core.memory.backend.neo4j_graph import Neo4jGraphBackend

        anima_dir = tmp_path / "animas" / "test"
        anima_dir.mkdir(parents=True)
        backend = Neo4jGraphBackend(anima_dir, group_id="test")

        mock_driver = AsyncMock()
        mock_driver.execute_query = AsyncMock(return_value=[{"uuid": "neighbor-1"}])
        mock_driver.execute_write = AsyncMock()

        with patch("core.memory.graph.community.CommunityDetector") as MockDetector:
            mock_detector = MagicMock()
            mock_detector.dynamic_update = AsyncMock(return_value="comm-1")
            MockDetector.return_value = mock_detector

            await backend._dynamic_community_update(mock_driver, ["ent-new"])

            mock_detector.dynamic_update.assert_called_once_with("ent-new", ["neighbor-1"])

    @pytest.mark.asyncio
    async def test_no_neighbors_skips_update(self, tmp_path) -> None:
        from core.memory.backend.neo4j_graph import Neo4jGraphBackend

        anima_dir = tmp_path / "animas" / "test"
        anima_dir.mkdir(parents=True)
        backend = Neo4jGraphBackend(anima_dir, group_id="test")

        mock_driver = AsyncMock()
        mock_driver.execute_query = AsyncMock(return_value=[])

        with patch("core.memory.graph.community.CommunityDetector") as MockDetector:
            mock_detector = MagicMock()
            mock_detector.dynamic_update = AsyncMock()
            MockDetector.return_value = mock_detector

            await backend._dynamic_community_update(mock_driver, ["ent-new"])

            mock_detector.dynamic_update.assert_not_called()


# ── TestFindEntityNeighborsQuery ─────────────────────────


class TestFindEntityNeighborsQuery:
    def test_query_exists(self) -> None:
        from core.memory.graph.queries import FIND_ENTITY_NEIGHBORS

        assert "$entity_uuid" in FIND_ENTITY_NEIGHBORS
        assert "$group_id" in FIND_ENTITY_NEIGHBORS
        assert "RELATES_TO" in FIND_ENTITY_NEIGHBORS


# ── TestDetectCommunitiesIfNeo4j ─────────────────────────


class TestDetectCommunitiesIfNeo4j:
    """Test _detect_communities_if_neo4j in SystemConsolidationMixin."""

    @pytest.mark.asyncio
    async def test_skips_when_legacy_backend(self) -> None:
        from core.lifecycle.system_consolidation import SystemConsolidationMixin

        mock_anima = MagicMock()

        with patch("core.config.models.load_config") as mock_cfg:
            cfg = MagicMock()
            cfg.memory = MagicMock()
            cfg.memory.backend = "legacy"
            mock_cfg.return_value = cfg

            await SystemConsolidationMixin._detect_communities_if_neo4j(mock_anima)

    @pytest.mark.asyncio
    async def test_skips_when_no_memory_config(self) -> None:
        from core.lifecycle.system_consolidation import SystemConsolidationMixin

        mock_anima = MagicMock()

        with patch("core.config.models.load_config") as mock_cfg:
            cfg = MagicMock(spec=[])
            mock_cfg.return_value = cfg

            await SystemConsolidationMixin._detect_communities_if_neo4j(mock_anima)

    @pytest.mark.asyncio
    async def test_runs_when_neo4j_backend(self) -> None:
        from core.lifecycle.system_consolidation import SystemConsolidationMixin

        mock_anima = MagicMock()
        mock_anima.name = "test"
        mock_anima.memory.anima_dir = "/tmp/test"

        mock_backend = AsyncMock()
        mock_backend._group_id = "test"
        mock_backend._resolve_background_model.return_value = "test-model"
        mock_backend._resolve_locale.return_value = "ja"
        mock_backend._ensure_driver = AsyncMock()
        mock_backend.close = AsyncMock()

        with (
            patch("core.config.models.load_config") as mock_cfg,
            patch("core.memory.backend.registry.get_backend", return_value=mock_backend),
            patch("core.memory.graph.community.CommunityDetector") as MockDetector,
        ):
            cfg = MagicMock()
            cfg.memory = MagicMock()
            cfg.memory.backend = "neo4j"
            mock_cfg.return_value = cfg

            mock_detector = MagicMock()
            mock_detector.detect_and_store = AsyncMock(return_value=[])
            MockDetector.return_value = mock_detector

            await SystemConsolidationMixin._detect_communities_if_neo4j(mock_anima)

            mock_detector.detect_and_store.assert_called_once()
            mock_backend.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_failure_doesnt_propagate(self) -> None:
        from core.lifecycle.system_consolidation import SystemConsolidationMixin

        mock_anima = MagicMock()
        mock_anima.name = "test"

        with patch("core.config.models.load_config", side_effect=RuntimeError("cfg error")):
            await SystemConsolidationMixin._detect_communities_if_neo4j(mock_anima)
