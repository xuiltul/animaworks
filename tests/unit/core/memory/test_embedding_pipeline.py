"""Unit tests for Neo4j embedding generation pipeline.

Verifies that Neo4jGraphBackend generates and stores embeddings
during ingest, and passes query embeddings during retrieval.
All tests are fully mocked — no real Neo4j, sentence-transformers, or LLM calls.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from core.memory.backend.neo4j_graph import Neo4jGraphBackend
from core.memory.ontology.default import ExtractedEntity, ExtractedFact

# ── Helpers ────────────────────────────────────────────────────────────────

_FAKE_384 = [0.1] * 384


def _make_backend(tmp_path: Path) -> tuple[Neo4jGraphBackend, AsyncMock]:
    """Create a backend with mocked driver and schema ensured."""
    backend = Neo4jGraphBackend(tmp_path)
    mock_driver = AsyncMock()
    mock_driver.execute_write = AsyncMock()
    mock_driver.execute_query = AsyncMock(return_value=[])
    backend._driver = mock_driver
    backend._schema_ensured = True
    return backend, mock_driver


def _make_extractor(entities: list[ExtractedEntity], facts: list[ExtractedFact]) -> MagicMock:
    """Create a mock FactExtractor returning the given entities/facts."""
    ext = AsyncMock()
    ext.extract_entities = AsyncMock(return_value=entities)
    ext.extract_facts = AsyncMock(return_value=facts)
    return ext


# ── TestEmbedTexts ─────────────────────────────────────────────────────────


class TestEmbedTexts:
    """Test the _embed_texts() helper method."""

    async def test_returns_embeddings_from_singleton(self, tmp_path: Path) -> None:
        backend, _ = _make_backend(tmp_path)
        fake_embeddings = [_FAKE_384, _FAKE_384]

        with patch(
            "core.memory.backend.neo4j_graph.asyncio.to_thread",
            new_callable=AsyncMock,
            return_value=fake_embeddings,
        ):
            result = await backend._embed_texts(["hello", "world"])

        assert len(result) == 2
        assert len(result[0]) == 384
        assert backend._embedding_available is True

    async def test_empty_input_returns_empty(self, tmp_path: Path) -> None:
        backend, _ = _make_backend(tmp_path)
        result = await backend._embed_texts([])
        assert result == []

    async def test_fallback_on_import_error(self, tmp_path: Path) -> None:
        backend, _ = _make_backend(tmp_path)

        with patch(
            "core.memory.backend.neo4j_graph.asyncio.to_thread",
            new_callable=AsyncMock,
            side_effect=ImportError("no sentence_transformers"),
        ):
            result = await backend._embed_texts(["test"])

        assert len(result) == 1
        assert result[0] == []
        assert backend._embedding_available is False

    async def test_fallback_on_runtime_error(self, tmp_path: Path) -> None:
        backend, _ = _make_backend(tmp_path)

        with patch(
            "core.memory.backend.neo4j_graph.asyncio.to_thread",
            new_callable=AsyncMock,
            side_effect=RuntimeError("GPU OOM"),
        ):
            result = await backend._embed_texts(["test"])

        assert result == [[]]
        assert backend._embedding_available is False

    async def test_cached_unavailability_skips_call(self, tmp_path: Path) -> None:
        backend, _ = _make_backend(tmp_path)
        backend._embedding_available = False

        result = await backend._embed_texts(["should not call"])
        assert result == [[]]


# ── TestIngestWithEmbeddings ───────────────────────────────────────────────


class TestIngestWithEmbeddings:
    """Verify ingest_text generates and stores embeddings."""

    async def test_entity_embedding_stored_in_neo4j(self, tmp_path: Path) -> None:
        backend, mock_driver = _make_backend(tmp_path)
        backend._extractor = _make_extractor(
            entities=[ExtractedEntity(name="Alice", entity_type="Person", summary="Engineer")],
            facts=[],
        )

        fake_emb = [_FAKE_384]
        with patch.object(backend, "_embed_texts", new_callable=AsyncMock, return_value=fake_emb):
            await backend.ingest_text("Alice is an engineer", "test")

        write_calls = mock_driver.execute_write.call_args_list
        entity_call = next(
            (c for c in write_calls if c.args and "name_embedding" in str(c.args)),
            None,
        )
        if entity_call is None:
            entity_call = next(
                (c for c in write_calls if c.kwargs and "name_embedding" in str(c.kwargs)),
                None,
            )

        entity_params = None
        for call in write_calls:
            params = call.args[1] if len(call.args) > 1 else call.kwargs.get("parameters")
            if params and "name_embedding" in params:
                entity_params = params
                break

        assert entity_params is not None, "CREATE_ENTITY call not found"
        assert entity_params["name_embedding"] == _FAKE_384

    async def test_fact_embedding_stored_in_neo4j(self, tmp_path: Path) -> None:
        backend, mock_driver = _make_backend(tmp_path)
        backend._extractor = _make_extractor(
            entities=[
                ExtractedEntity(name="Alice", entity_type="Person", summary="Engineer"),
                ExtractedEntity(name="Bob", entity_type="Person", summary="Designer"),
            ],
            facts=[
                ExtractedFact(source_entity="Alice", target_entity="Bob", fact="works with"),
            ],
        )

        entity_embs = [_FAKE_384, _FAKE_384]
        fact_embs = [[0.2] * 384]

        call_count = 0

        async def side_effect(texts: list[str]) -> list[list[float]]:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return entity_embs
            return fact_embs

        with patch.object(backend, "_embed_texts", side_effect=side_effect):
            await backend.ingest_text("Alice works with Bob", "test")

        fact_params = None
        for call in mock_driver.execute_write.call_args_list:
            params = call.args[1] if len(call.args) > 1 else call.kwargs.get("parameters")
            if params and "fact_embedding" in params:
                fact_params = params
                break

        assert fact_params is not None, "CREATE_FACT call not found"
        assert fact_params["fact_embedding"] == [0.2] * 384

    async def test_batch_embed_called_for_entities_and_facts(self, tmp_path: Path) -> None:
        backend, _ = _make_backend(tmp_path)
        backend._extractor = _make_extractor(
            entities=[
                ExtractedEntity(name="A", entity_type="Concept", summary="a"),
                ExtractedEntity(name="B", entity_type="Concept", summary="b"),
                ExtractedEntity(name="C", entity_type="Concept", summary="c"),
            ],
            facts=[
                ExtractedFact(source_entity="A", target_entity="B", fact="r1"),
                ExtractedFact(source_entity="B", target_entity="C", fact="r2"),
            ],
        )

        embed_calls: list[list[str]] = []

        async def track_embed(texts: list[str]) -> list[list[float]]:
            embed_calls.append(texts)
            return [_FAKE_384] * len(texts)

        with patch.object(backend, "_embed_texts", side_effect=track_embed):
            await backend.ingest_text("A B C relationships", "test")

        assert len(embed_calls) == 3, f"Expected 3 batch calls, got {len(embed_calls)}"
        assert len(embed_calls[0]) == 1  # 1 episode content
        assert len(embed_calls[1]) == 3  # 3 entities
        assert len(embed_calls[2]) == 2  # 2 facts

    async def test_embedding_failure_does_not_break_ingest(self, tmp_path: Path) -> None:
        backend, mock_driver = _make_backend(tmp_path)
        backend._extractor = _make_extractor(
            entities=[ExtractedEntity(name="X", entity_type="Concept", summary="x")],
            facts=[],
        )

        with patch.object(backend, "_embed_texts", new_callable=AsyncMock, return_value=[[]]):
            count = await backend.ingest_text("test", "test")

        assert count >= 1
        mock_driver.execute_write.assert_called()


# ── TestRetrieveWithEmbeddings ─────────────────────────────────────────────


class TestRetrieveWithEmbeddings:
    """Verify retrieve() generates query embeddings."""

    async def test_retrieve_passes_query_embedding_to_search(self, tmp_path: Path) -> None:
        backend, mock_driver = _make_backend(tmp_path)

        with (
            patch.object(
                backend,
                "_embed_texts",
                new_callable=AsyncMock,
                return_value=[_FAKE_384],
            ) as mock_embed,
            patch("core.memory.graph.search.HybridSearch", autospec=False) as MockSearch,
        ):
            mock_search_inst = MagicMock()
            mock_search_inst.search = AsyncMock(return_value=[])
            MockSearch.return_value = mock_search_inst

            await backend.retrieve("test query", scope="fact", limit=5)

            mock_embed.assert_awaited_once_with(["test query"])
            mock_search_inst.search.assert_awaited_once()
            call_kwargs = mock_search_inst.search.call_args
            assert call_kwargs.kwargs.get("query_embedding") == _FAKE_384

    async def test_retrieve_fallback_with_empty_embedding(self, tmp_path: Path) -> None:
        backend, mock_driver = _make_backend(tmp_path)
        backend._embedding_available = False

        with patch("core.memory.graph.search.HybridSearch", autospec=False) as MockSearch:
            mock_search_inst = MagicMock()
            mock_search_inst.search = AsyncMock(return_value=[])
            MockSearch.return_value = mock_search_inst

            await backend.retrieve("test", scope="fact")

            call_kwargs = mock_search_inst.search.call_args
            assert call_kwargs.kwargs.get("query_embedding") == []

    async def test_community_scope_skips_embedding(self, tmp_path: Path) -> None:
        backend, mock_driver = _make_backend(tmp_path)
        mock_driver.execute_query = AsyncMock(return_value=[])

        with patch.object(backend, "_embed_texts", new_callable=AsyncMock) as mock_embed:
            await backend.retrieve("test", scope="community")
            mock_embed.assert_not_awaited()


# ── TestResolverEmbedding ──────────────────────────────────────────────────


class TestResolverEmbedding:
    """Verify entity resolution receives actual embeddings."""

    async def test_resolver_receives_embedding(self, tmp_path: Path) -> None:
        backend, mock_driver = _make_backend(tmp_path)
        backend._extractor = _make_extractor(
            entities=[ExtractedEntity(name="Test", entity_type="Concept", summary="test entity")],
            facts=[],
        )

        from core.memory.extraction.resolver import ResolvedEntity

        mock_resolver = AsyncMock()
        mock_resolver.resolve = AsyncMock(
            return_value=ResolvedEntity(
                uuid="test-uuid",
                name="Test",
                summary="test entity",
                entity_type="Concept",
                is_new=True,
            )
        )
        backend._resolver = mock_resolver

        with patch.object(
            backend,
            "_embed_texts",
            new_callable=AsyncMock,
            return_value=[_FAKE_384],
        ):
            await backend.ingest_text("test content", "test")

        mock_resolver.resolve.assert_awaited_once()
        call_kwargs = mock_resolver.resolve.call_args
        assert call_kwargs.kwargs.get("name_embedding") == _FAKE_384
