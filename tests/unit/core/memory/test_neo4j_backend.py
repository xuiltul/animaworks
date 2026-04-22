"""Unit tests for Neo4jGraphBackend skeleton."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest


class TestNeo4jGraphBackendInit:
    """Test initialization without Neo4j connection."""

    def test_can_instantiate(self, tmp_path):
        from core.memory.backend.neo4j_graph import Neo4jGraphBackend

        backend = Neo4jGraphBackend(tmp_path)
        assert backend._anima_name == tmp_path.name
        assert backend._group_id == tmp_path.name

    def test_custom_group_id(self, tmp_path):
        from core.memory.backend.neo4j_graph import Neo4jGraphBackend

        backend = Neo4jGraphBackend(tmp_path, group_id="custom")
        assert backend._group_id == "custom"

    def test_custom_connection_params(self, tmp_path):
        from core.memory.backend.neo4j_graph import Neo4jGraphBackend

        backend = Neo4jGraphBackend(
            tmp_path,
            uri="bolt://custom:7687",
            user="admin",
            password="secret",
            database="mydb",
        )
        assert backend._uri == "bolt://custom:7687"
        assert backend._user == "admin"
        assert backend._database == "mydb"

    def test_default_connection_params(self, tmp_path):
        from core.memory.backend.neo4j_graph import Neo4jGraphBackend

        backend = Neo4jGraphBackend(tmp_path)
        assert backend._uri == "bolt://localhost:7687"
        assert backend._user == "neo4j"
        assert backend._password == "animaworks"
        assert backend._database == "neo4j"

    def test_driver_initially_none(self, tmp_path):
        from core.memory.backend.neo4j_graph import Neo4jGraphBackend

        backend = Neo4jGraphBackend(tmp_path)
        assert backend._driver is None
        assert backend._schema_ensured is False


class TestNeo4jGraphBackendStubs:
    """Test NotImplementedError stubs."""

    @pytest.mark.asyncio
    async def test_ingest_file_returns_zero_on_missing(self, tmp_path):
        from core.memory.backend.neo4j_graph import Neo4jGraphBackend

        backend = Neo4jGraphBackend(tmp_path)
        result = await backend.ingest_file(tmp_path / "nonexistent.md")
        assert result == 0

    @pytest.mark.asyncio
    async def test_ingest_text_needs_driver(self, tmp_path):
        from core.memory.backend.neo4j_graph import Neo4jGraphBackend

        backend = Neo4jGraphBackend(tmp_path)
        with pytest.raises(Exception):
            await backend.ingest_text("hello", "test")

    @pytest.mark.asyncio
    async def test_retrieve_empty_query_returns_empty(self, tmp_path):
        from core.memory.backend.neo4j_graph import Neo4jGraphBackend

        backend = Neo4jGraphBackend(tmp_path)
        backend._driver = AsyncMock()
        backend._schema_ensured = True
        result = await backend.retrieve("", scope="knowledge")
        assert result == []

    @pytest.mark.asyncio
    async def test_delete_raises(self, tmp_path):
        from core.memory.backend.neo4j_graph import Neo4jGraphBackend

        backend = Neo4jGraphBackend(tmp_path)
        with pytest.raises(NotImplementedError, match="Issue #3"):
            await backend.delete("source")


class TestNeo4jGraphBackendWithMockedDriver:
    """Test implemented methods with mocked Neo4j driver."""

    @pytest.mark.asyncio
    async def test_health_check_success(self, tmp_path):
        from core.memory.backend.neo4j_graph import Neo4jGraphBackend

        backend = Neo4jGraphBackend(tmp_path)

        mock_driver = AsyncMock()
        mock_driver.health_check = AsyncMock(return_value=True)
        backend._driver = mock_driver
        backend._schema_ensured = True

        assert await backend.health_check() is True

    @pytest.mark.asyncio
    async def test_health_check_failure(self, tmp_path):
        from core.memory.backend.neo4j_graph import Neo4jGraphBackend

        backend = Neo4jGraphBackend(tmp_path)

        mock_driver = AsyncMock()
        mock_driver.health_check = AsyncMock(side_effect=Exception("down"))
        backend._driver = mock_driver
        backend._schema_ensured = True

        assert await backend.health_check() is False

    @pytest.mark.asyncio
    async def test_reset(self, tmp_path):
        from core.memory.backend.neo4j_graph import Neo4jGraphBackend

        backend = Neo4jGraphBackend(tmp_path)

        mock_driver = AsyncMock()
        mock_driver.execute_write = AsyncMock()
        backend._driver = mock_driver
        backend._schema_ensured = True

        await backend.reset()
        mock_driver.execute_write.assert_called_once()
        call_args = mock_driver.execute_write.call_args
        assert call_args[0][1] == {"group_id": tmp_path.name}

    @pytest.mark.asyncio
    async def test_stats_empty(self, tmp_path):
        from core.memory.backend.neo4j_graph import Neo4jGraphBackend

        backend = Neo4jGraphBackend(tmp_path)

        mock_driver = AsyncMock()
        mock_driver.execute_query = AsyncMock(return_value=[])
        backend._driver = mock_driver
        backend._schema_ensured = True

        result = await backend.stats()
        assert result["total_chunks"] == 0
        assert result["total_sources"] == 0

    @pytest.mark.asyncio
    async def test_stats_with_data(self, tmp_path):
        from core.memory.backend.neo4j_graph import Neo4jGraphBackend

        backend = Neo4jGraphBackend(tmp_path)

        mock_driver = AsyncMock()
        mock_driver.execute_query = AsyncMock(
            side_effect=[
                [{"label": "Entity", "cnt": 10}, {"label": "Episode", "cnt": 5}],
                [{"rel_type": "RELATES_TO", "cnt": 8}],
            ]
        )
        backend._driver = mock_driver
        backend._schema_ensured = True

        result = await backend.stats()
        assert result["total_chunks"] == 15
        assert result["nodes_Entity"] == 10
        assert result["nodes_Episode"] == 5
        assert result["edges_RELATES_TO"] == 8

    @pytest.mark.asyncio
    async def test_stats_on_driver_error(self, tmp_path):
        from core.memory.backend.neo4j_graph import Neo4jGraphBackend

        backend = Neo4jGraphBackend(tmp_path)

        mock_driver = AsyncMock()
        mock_driver.execute_query = AsyncMock(side_effect=Exception("db down"))
        backend._driver = mock_driver
        backend._schema_ensured = True

        result = await backend.stats()
        assert result["total_chunks"] == 0
        assert result["total_sources"] == 0

    @pytest.mark.asyncio
    async def test_close(self, tmp_path):
        from core.memory.backend.neo4j_graph import Neo4jGraphBackend

        backend = Neo4jGraphBackend(tmp_path)

        mock_driver = AsyncMock()
        mock_driver.close = AsyncMock()
        backend._driver = mock_driver

        await backend.close()
        mock_driver.close.assert_called_once()
        assert backend._driver is None

    @pytest.mark.asyncio
    async def test_close_noop_when_no_driver(self, tmp_path):
        from core.memory.backend.neo4j_graph import Neo4jGraphBackend

        backend = Neo4jGraphBackend(tmp_path)
        assert backend._driver is None
        await backend.close()
        assert backend._driver is None


class TestRegistryNeo4j:
    """Test that registry creates Neo4jGraphBackend."""

    def test_registry_returns_neo4j_backend(self, tmp_path):
        from core.memory.backend.neo4j_graph import Neo4jGraphBackend
        from core.memory.backend.registry import get_backend

        backend = get_backend("neo4j", tmp_path)
        assert isinstance(backend, Neo4jGraphBackend)

    def test_registry_passes_kwargs(self, tmp_path):
        from core.memory.backend.registry import get_backend

        backend = get_backend("neo4j", tmp_path, uri="bolt://custom:7687", group_id="test")
        assert backend._uri == "bolt://custom:7687"
        assert backend._group_id == "test"

    def test_registry_unknown_backend_raises(self, tmp_path):
        from core.memory.backend.registry import get_backend

        with pytest.raises(ValueError, match="Unknown memory backend"):
            get_backend("nonexistent", tmp_path)


class TestNeo4jConfig:
    """Test Neo4j config settings."""

    def test_default_config(self):
        from core.config.schemas import MemoryConfig

        cfg = MemoryConfig()
        assert cfg.backend == "legacy"
        assert cfg.neo4j.uri == "bolt://localhost:7687"
        assert cfg.neo4j.user == "neo4j"
        assert cfg.neo4j.database == "neo4j"

    def test_custom_neo4j_config(self):
        from core.config.schemas import MemoryConfig, Neo4jConfig

        cfg = MemoryConfig(
            backend="neo4j",
            neo4j=Neo4jConfig(uri="bolt://prod:7687", password="secret"),
        )
        assert cfg.backend == "neo4j"
        assert cfg.neo4j.uri == "bolt://prod:7687"
        assert cfg.neo4j.password == "secret"

    def test_neo4j_config_defaults(self):
        from core.config.schemas import Neo4jConfig

        cfg = Neo4jConfig()
        assert cfg.uri == "bolt://localhost:7687"
        assert cfg.user == "neo4j"
        assert cfg.password == "animaworks"
        assert cfg.database == "neo4j"
