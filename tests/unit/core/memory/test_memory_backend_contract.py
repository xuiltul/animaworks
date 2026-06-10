from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Contract tests for MemoryBackend implementations.

Every concrete MemoryBackend must pass all of these tests.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from core.memory.backend.base import MemoryBackend, RetrievedMemory
from core.memory.backend.legacy import LegacyRAGBackend
from core.memory.backend.registry import get_backend

# ── Fixtures ───────────────────────────────────────────────────────────────


@pytest.fixture()
def legacy_backend(tmp_path: Path) -> LegacyRAGBackend:
    """Create a LegacyRAGBackend with all heavy internals mocked."""
    with patch("core.paths.get_data_dir", return_value=tmp_path):
        backend = LegacyRAGBackend(tmp_path)

    mock_vs = MagicMock()
    mock_vs.list_collections.return_value = []
    backend._vector_store = mock_vs

    mock_rag = MagicMock()
    mock_rag.search_memory_text.return_value = []
    mock_rag.index_file.return_value = 0
    backend._rag_search = mock_rag

    mock_retriever = MagicMock()
    mock_retriever.search.return_value = []
    mock_retriever.get_important_chunks.return_value = []
    mock_retriever.record_access.return_value = None
    backend._retriever = mock_retriever

    backend._indexer = MagicMock()

    return backend


# ── ABC contract ───────────────────────────────────────────────────────────


def test_abc_cannot_be_instantiated() -> None:
    with pytest.raises(TypeError):
        MemoryBackend()  # type: ignore[abstract]


def test_legacy_implements_all_abstract_methods(tmp_path: Path) -> None:
    with patch("core.paths.get_data_dir", return_value=tmp_path):
        backend = LegacyRAGBackend(tmp_path)
    assert isinstance(backend, MemoryBackend)


# ── Retrieve ───────────────────────────────────────────────────────────────


async def test_retrieve_returns_list_of_retrieved_memory(
    legacy_backend: LegacyRAGBackend,
) -> None:
    result = await legacy_backend.retrieve("test query", scope="knowledge")
    assert isinstance(result, list)
    for item in result:
        assert isinstance(item, RetrievedMemory)


async def test_retrieve_with_invalid_scope(
    legacy_backend: LegacyRAGBackend,
) -> None:
    result = await legacy_backend.retrieve("test query", scope="nonexistent_scope")
    assert isinstance(result, list)


async def test_legacy_retrieve_episodes_uses_rag_search_path(
    legacy_backend: LegacyRAGBackend,
) -> None:
    mock_unified = MagicMock()
    mock_unified.search.return_value = [
        {
            "content": "Caroline recommended Becoming Nicole.",
            "score": 0.9,
            "source_file": "episodes/2026-06-03.md",
            "memory_type": "episodes",
            "search_method": "vector",
        }
    ]
    legacy_backend._unified_search = mock_unified

    result = await legacy_backend.retrieve("Caroline", scope="episodes", limit=3)

    legacy_backend._retriever.search.assert_not_called()
    legacy_backend._rag_search.search_memory_text.assert_not_called()
    mock_unified.search.assert_called_once()
    assert mock_unified.search.call_args.kwargs["limit"] == 3
    assert mock_unified.search.call_args.kwargs["scope"] == "episodes"
    assert result[0].content == "Caroline recommended Becoming Nicole."
    assert result[0].metadata["memory_type"] == "episodes"


# ── Health check ───────────────────────────────────────────────────────────


async def test_health_check_returns_bool(
    legacy_backend: LegacyRAGBackend,
) -> None:
    result = await legacy_backend.health_check()
    assert isinstance(result, bool)


# ── Stats ──────────────────────────────────────────────────────────────────


async def test_stats_returns_dict(
    legacy_backend: LegacyRAGBackend,
) -> None:
    result = await legacy_backend.stats()
    assert isinstance(result, dict)
    assert "total_chunks" in result
    assert "total_sources" in result


async def test_stats_excludes_entity_helper_collection(
    legacy_backend: LegacyRAGBackend,
) -> None:
    prefix = legacy_backend._anima_name
    legacy_backend._vector_store.list_collections.return_value = [
        f"{prefix}_knowledge",
        f"{prefix}_entities",
    ]
    memory_doc = MagicMock()
    memory_doc.document.metadata = {"source_file": "knowledge/a.md"}
    entity_doc = MagicMock()
    entity_doc.document.metadata = {"canonical": "Caroline"}

    def fake_get_by_metadata(collection: str, where: dict, limit: int):
        if collection.endswith("_entities"):
            return [entity_doc]
        return [memory_doc]

    legacy_backend._vector_store.get_by_metadata.side_effect = fake_get_by_metadata

    result = await legacy_backend.stats()

    assert result == {"total_chunks": 1, "total_sources": 1}


# ── Ingest ─────────────────────────────────────────────────────────────────


async def test_ingest_file_returns_int(
    legacy_backend: LegacyRAGBackend,
    tmp_path: Path,
) -> None:
    test_file = tmp_path / "knowledge" / "test.md"
    test_file.parent.mkdir(parents=True, exist_ok=True)
    test_file.write_text("# Test\nHello world")

    result = await legacy_backend.ingest_file(test_file)
    assert isinstance(result, int)


# ── Reset ──────────────────────────────────────────────────────────────────


async def test_reset_does_not_raise(
    legacy_backend: LegacyRAGBackend,
) -> None:
    await legacy_backend.reset()


# ── Optional methods ───────────────────────────────────────────────────────


async def test_get_important_chunks_returns_list(
    legacy_backend: LegacyRAGBackend,
) -> None:
    base_result = await MemoryBackend.get_important_chunks(legacy_backend)
    assert isinstance(base_result, list)

    result = await legacy_backend.get_important_chunks()
    assert isinstance(result, list)


async def test_record_access_accepts_empty_list(
    legacy_backend: LegacyRAGBackend,
) -> None:
    await legacy_backend.record_access([])


async def test_legacy_rebuild_index_rebuilds_longterm_bm25(
    legacy_backend: LegacyRAGBackend,
) -> None:
    (legacy_backend._anima_dir / "knowledge").mkdir(parents=True)
    legacy_backend._indexer.index_directory.return_value = 3

    with patch("core.memory.bm25.rebuild_longterm_bm25_index") as mock_rebuild:
        mock_rebuild.return_value = MagicMock(documents=1)
        total = await legacy_backend.rebuild_index("knowledge")

    assert total == 3
    mock_rebuild.assert_called_once_with(legacy_backend._anima_dir)


# ── Registry ───────────────────────────────────────────────────────────────


def test_registry_legacy(tmp_path: Path) -> None:
    with patch("core.paths.get_data_dir", return_value=tmp_path):
        backend = get_backend("legacy", tmp_path)
    assert isinstance(backend, LegacyRAGBackend)


def test_registry_neo4j_returns_backend(tmp_path: Path) -> None:
    backend = get_backend("neo4j", tmp_path)
    from core.memory.backend.neo4j_graph import Neo4jGraphBackend

    assert isinstance(backend, Neo4jGraphBackend)


def test_registry_unknown_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unknown memory backend"):
        get_backend("unknown", tmp_path)


# ── RetrievedMemory dataclass ──────────────────────────────────────────────


def test_retrieved_memory_dataclass() -> None:
    mem = RetrievedMemory(content="hello", score=0.95, source="knowledge/test.md")
    assert mem.content == "hello"
    assert mem.score == 0.95
    assert mem.source == "knowledge/test.md"
    assert mem.metadata == {}
    assert mem.trust == "medium"

    mem_full = RetrievedMemory(
        content="world",
        score=0.5,
        source="episodes/day.md",
        metadata={"key": "value"},
        trust="trusted",
    )
    assert mem_full.metadata == {"key": "value"}
    assert mem_full.trust == "trusted"
