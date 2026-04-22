"""Unit tests for the fact extraction pipeline.

Covers ontology models, FactExtractor, and Neo4jGraphBackend ingestion
with fully mocked LLM + driver — no real API or DB calls.
"""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.memory.ontology.default import (
    EntityExtractionResult,
    ExtractedEntity,
    ExtractedFact,
    FactExtractionResult,
)

# ── Helpers ────────────────────────────────────────────────────────────────


def _make_llm_response(content: str) -> MagicMock:
    """Build a fake LiteLLM acompletion response."""
    msg = MagicMock()
    msg.content = content
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    return resp


_LLM_KWARGS_PATCH = patch(
    "core.memory._llm_utils.get_llm_kwargs_for_model",
    return_value={"model": "test-model"},
)


# ── TestExtractedModels ────────────────────────────────────────────────────


class TestExtractedModels:
    def test_entity_default_type(self):
        ent = ExtractedEntity(name="foo")
        assert ent.entity_type == "Concept"

    def test_entity_all_fields(self):
        ent = ExtractedEntity(name="Alice", entity_type="Person", summary="An engineer")
        assert ent.name == "Alice"
        assert ent.entity_type == "Person"
        assert ent.summary == "An engineer"

    def test_fact_valid_at_optional(self):
        fact = ExtractedFact(source_entity="A", target_entity="B", fact="knows")
        assert fact.valid_at is None

    def test_extraction_result_empty(self):
        result = EntityExtractionResult()
        assert result.entities == []

    def test_fact_extraction_result_empty(self):
        result = FactExtractionResult()
        assert result.facts == []


# ── TestFactExtractorInit ──────────────────────────────────────────────────


class TestFactExtractorInit:
    def test_default_locale(self):
        from core.memory.extraction.extractor import FactExtractor

        ext = FactExtractor(model="m")
        assert ext._locale == "ja"

    def test_custom_locale(self):
        from core.memory.extraction.extractor import FactExtractor

        ext = FactExtractor(model="m", locale="en")
        assert ext._locale == "en"

    def test_model_stored(self):
        from core.memory.extraction.extractor import FactExtractor

        ext = FactExtractor(model="claude-sonnet-4-6")
        assert ext._model == "claude-sonnet-4-6"


# ── TestFactExtractorExtractEntities ───────────────────────────────────────


class TestFactExtractorExtractEntities:
    @pytest.mark.asyncio
    @_LLM_KWARGS_PATCH
    @patch("litellm.acompletion", new_callable=AsyncMock)
    async def test_extract_entities_success(self, mock_acompletion, _mock_kwargs):
        from core.memory.extraction.extractor import FactExtractor

        payload = {
            "entities": [
                {"name": "田中", "entity_type": "Person", "summary": "A person"},
                {"name": "東京", "entity_type": "Place", "summary": "A city"},
            ]
        }
        mock_acompletion.return_value = _make_llm_response(json.dumps(payload, ensure_ascii=False))

        ext = FactExtractor(model="test-model", max_retries=1)
        entities = await ext.extract_entities("田中さんは東京に住んでいる")

        assert len(entities) == 2
        assert entities[0].name == "田中"
        assert entities[0].entity_type == "Person"
        assert entities[1].name == "東京"
        assert entities[1].entity_type == "Place"

    @pytest.mark.asyncio
    @_LLM_KWARGS_PATCH
    @patch("litellm.acompletion", new_callable=AsyncMock)
    async def test_extract_entities_empty_content(self, mock_acompletion, _mock_kwargs):
        from core.memory.extraction.extractor import FactExtractor

        mock_acompletion.return_value = _make_llm_response("")

        ext = FactExtractor(model="test-model", max_retries=1)
        entities = await ext.extract_entities("テスト")
        assert entities == []

    @pytest.mark.asyncio
    @_LLM_KWARGS_PATCH
    @patch("litellm.acompletion", new_callable=AsyncMock)
    async def test_extract_entities_llm_failure(self, mock_acompletion, _mock_kwargs):
        from core.memory.extraction.extractor import FactExtractor

        mock_acompletion.side_effect = RuntimeError("API down")

        ext = FactExtractor(model="test-model", max_retries=1)
        entities = await ext.extract_entities("何かテキスト")
        assert entities == []

    @pytest.mark.asyncio
    @_LLM_KWARGS_PATCH
    @patch("litellm.acompletion", new_callable=AsyncMock)
    async def test_extract_entities_invalid_json(self, mock_acompletion, _mock_kwargs):
        from core.memory.extraction.extractor import FactExtractor

        mock_acompletion.return_value = _make_llm_response("NOT VALID JSON {{{")

        ext = FactExtractor(model="test-model", max_retries=1)
        entities = await ext.extract_entities("テスト")
        assert entities == []

    @pytest.mark.asyncio
    @_LLM_KWARGS_PATCH
    @patch("litellm.acompletion", new_callable=AsyncMock)
    async def test_extract_entities_filters_empty_names(self, mock_acompletion, _mock_kwargs):
        from core.memory.extraction.extractor import FactExtractor

        payload = {
            "entities": [
                {"name": "", "entity_type": "Person", "summary": "empty"},
                {"name": "  ", "entity_type": "Person", "summary": "whitespace"},
                {"name": "Valid", "entity_type": "Concept", "summary": "ok"},
            ]
        }
        mock_acompletion.return_value = _make_llm_response(json.dumps(payload))

        ext = FactExtractor(model="test-model", max_retries=1)
        entities = await ext.extract_entities("テスト")

        assert len(entities) == 1
        assert entities[0].name == "Valid"


# ── TestFactExtractorExtractFacts ──────────────────────────────────────────


class TestFactExtractorExtractFacts:
    @pytest.mark.asyncio
    @_LLM_KWARGS_PATCH
    @patch("litellm.acompletion", new_callable=AsyncMock)
    async def test_extract_facts_success(self, mock_acompletion, _mock_kwargs):
        from core.memory.extraction.extractor import FactExtractor

        payload = {
            "facts": [
                {
                    "source_entity": "田中",
                    "target_entity": "東京",
                    "fact": "田中は東京に住んでいる",
                    "valid_at": None,
                }
            ]
        }
        mock_acompletion.return_value = _make_llm_response(json.dumps(payload, ensure_ascii=False))

        ext = FactExtractor(model="test-model", max_retries=1)
        entities = [
            ExtractedEntity(name="田中", entity_type="Person", summary="person"),
            ExtractedEntity(name="東京", entity_type="Place", summary="city"),
        ]
        facts = await ext.extract_facts("田中さんは東京に住んでいる", entities)

        assert len(facts) == 1
        assert facts[0].source_entity == "田中"
        assert facts[0].target_entity == "東京"
        assert facts[0].fact == "田中は東京に住んでいる"

    @pytest.mark.asyncio
    async def test_extract_facts_no_entities(self):
        from core.memory.extraction.extractor import FactExtractor

        ext = FactExtractor(model="test-model")
        facts = await ext.extract_facts("テスト", [])
        assert facts == []

    @pytest.mark.asyncio
    @_LLM_KWARGS_PATCH
    @patch("litellm.acompletion", new_callable=AsyncMock)
    async def test_extract_facts_filters_invalid_refs(self, mock_acompletion, _mock_kwargs):
        from core.memory.extraction.extractor import FactExtractor

        payload = {
            "facts": [
                {
                    "source_entity": "田中",
                    "target_entity": "大阪",
                    "fact": "visited",
                    "valid_at": None,
                },
                {
                    "source_entity": "田中",
                    "target_entity": "東京",
                    "fact": "lives in",
                    "valid_at": None,
                },
            ]
        }
        mock_acompletion.return_value = _make_llm_response(json.dumps(payload, ensure_ascii=False))

        ext = FactExtractor(model="test-model", max_retries=1)
        entities = [
            ExtractedEntity(name="田中", entity_type="Person", summary="person"),
            ExtractedEntity(name="東京", entity_type="Place", summary="city"),
        ]
        facts = await ext.extract_facts("テスト", entities)

        assert len(facts) == 1
        assert facts[0].target_entity == "東京"

    @pytest.mark.asyncio
    @_LLM_KWARGS_PATCH
    @patch("litellm.acompletion", new_callable=AsyncMock)
    async def test_extract_facts_llm_failure(self, mock_acompletion, _mock_kwargs):
        from core.memory.extraction.extractor import FactExtractor

        mock_acompletion.side_effect = RuntimeError("API down")

        ext = FactExtractor(model="test-model", max_retries=1)
        entities = [ExtractedEntity(name="X", entity_type="Concept", summary="x")]
        facts = await ext.extract_facts("テスト", entities)
        assert facts == []


# ── TestNeo4jGraphBackendIngest ────────────────────────────────────────────


class TestNeo4jGraphBackendIngest:
    @pytest.mark.asyncio
    async def test_ingest_text_creates_episode_and_entities(self, tmp_path):
        from core.memory.backend.neo4j_graph import Neo4jGraphBackend

        backend = Neo4jGraphBackend(tmp_path)
        mock_driver = AsyncMock()
        mock_driver.execute_write = AsyncMock()
        backend._driver = mock_driver
        backend._schema_ensured = True

        mock_extractor = AsyncMock()
        mock_extractor.extract_entities = AsyncMock(
            return_value=[
                ExtractedEntity(name="田中", entity_type="Person", summary="person"),
            ]
        )
        mock_extractor.extract_facts = AsyncMock(
            return_value=[
                ExtractedFact(
                    source_entity="田中",
                    target_entity="田中",
                    fact="self-ref",
                ),
            ]
        )
        backend._extractor = mock_extractor

        count = await backend.ingest_text("田中さんの話", "test")

        assert count >= 3  # 1 episode + 1 entity + 1 fact
        calls = mock_driver.execute_write.call_args_list
        assert len(calls) >= 3  # Episode + Entity + Mention + Fact

    @pytest.mark.asyncio
    async def test_ingest_text_fallback_on_extraction_failure(self, tmp_path):
        from core.memory.backend.neo4j_graph import Neo4jGraphBackend

        backend = Neo4jGraphBackend(tmp_path)
        mock_driver = AsyncMock()
        mock_driver.execute_write = AsyncMock()
        backend._driver = mock_driver
        backend._schema_ensured = True

        mock_extractor = MagicMock()
        mock_extractor.extract_entities = AsyncMock(side_effect=RuntimeError("boom"))
        backend._extractor = mock_extractor

        count = await backend.ingest_text("テスト", "test")
        assert count == 1  # Episode only

    @pytest.mark.asyncio
    async def test_ingest_text_semaphore_limits_concurrency(self, tmp_path):
        from core.memory.backend.neo4j_graph import Neo4jGraphBackend

        backend = Neo4jGraphBackend(tmp_path)
        mock_driver = AsyncMock()
        mock_driver.execute_write = AsyncMock()
        backend._driver = mock_driver
        backend._schema_ensured = True

        mock_extractor = AsyncMock()
        mock_extractor.extract_entities = AsyncMock(return_value=[])
        mock_extractor.extract_facts = AsyncMock(return_value=[])
        backend._extractor = mock_extractor

        peak_concurrent = 0
        current_concurrent = 0
        original_sem = backend._ingest_semaphore
        original_acquire = original_sem.acquire
        original_release = original_sem.release

        async def tracking_acquire():
            nonlocal current_concurrent, peak_concurrent
            await original_acquire()
            current_concurrent += 1
            peak_concurrent = max(peak_concurrent, current_concurrent)
            await asyncio.sleep(0.05)

        def tracking_release():
            nonlocal current_concurrent
            current_concurrent -= 1
            original_release()

        original_sem.acquire = tracking_acquire
        original_sem.release = tracking_release

        await asyncio.gather(
            backend.ingest_text("text1", "s1"),
            backend.ingest_text("text2", "s2"),
            backend.ingest_text("text3", "s3"),
        )

        assert peak_concurrent <= 2

    @pytest.mark.asyncio
    async def test_ingest_file_reads_and_ingests(self, tmp_path):
        from core.memory.backend.neo4j_graph import Neo4jGraphBackend

        test_file = tmp_path / "test.md"
        test_file.write_text("# Hello\nSome content", encoding="utf-8")

        backend = Neo4jGraphBackend(tmp_path)
        mock_driver = AsyncMock()
        mock_driver.execute_write = AsyncMock()
        backend._driver = mock_driver
        backend._schema_ensured = True

        mock_extractor = AsyncMock()
        mock_extractor.extract_entities = AsyncMock(return_value=[])
        mock_extractor.extract_facts = AsyncMock(return_value=[])
        backend._extractor = mock_extractor

        count = await backend.ingest_file(test_file)
        assert count >= 1
        mock_driver.execute_write.assert_called()

    @pytest.mark.asyncio
    async def test_ingest_file_empty_file(self, tmp_path):
        from core.memory.backend.neo4j_graph import Neo4jGraphBackend

        empty_file = tmp_path / "empty.md"
        empty_file.write_text("", encoding="utf-8")

        backend = Neo4jGraphBackend(tmp_path)
        count = await backend.ingest_file(empty_file)
        assert count == 0

    def test_split_sections(self):
        from core.memory.backend.neo4j_graph import Neo4jGraphBackend

        content = "intro\n## Section A\ntext a\n## Section B\ntext b"
        sections = Neo4jGraphBackend._split_sections(content)
        assert len(sections) == 3
        assert "intro" in sections[0]
        assert "Section A" in sections[1]
        assert "Section B" in sections[2]

    def test_split_sections_long_content(self):
        from core.memory.backend.neo4j_graph import Neo4jGraphBackend

        long_text = "x" * 20000
        sections = Neo4jGraphBackend._split_sections(long_text, max_chars=8000)
        assert len(sections) >= 3
        for s in sections:
            assert len(s) <= 8000
