from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from core.memory.entity_index import upsert_entities_from_facts
from core.memory.facts import FactRecord
from core.memory.rag_search import RAGMemorySearch
from core.memory.retrieval.entity import EntityBoostConfig, apply_entity_boost
from core.memory.retrieval.pipeline import PipelineResult


@pytest.fixture
def rag_search(tmp_path: Path) -> RAGMemorySearch:
    anima_dir = tmp_path / "animas" / "alice"
    for subdir in ("knowledge", "episodes", "procedures", "facts", "state"):
        (anima_dir / subdir).mkdir(parents=True)
    common_knowledge = tmp_path / "common_knowledge"
    common_skills = tmp_path / "common_skills"
    common_knowledge.mkdir()
    common_skills.mkdir()
    return RAGMemorySearch(anima_dir, common_knowledge, common_skills)


def _settings(*, enabled: bool, registry_enabled: bool = True) -> dict[str, object]:
    return {
        "rerank_enabled": False,
        "rerank_candidate_pool": 20,
        "cross_encoder_model": "dummy",
        "abstain_on_low_confidence": False,
        "confidence_threshold": 0.35,
        "rrf_confidence_threshold": 0.02,
        "entity_registry_enabled": registry_enabled,
        "entity_boost_enabled": enabled,
        "entity_boost": 0.25,
        "entity_boost_cap": 0.40,
    }


@pytest.mark.unit
def test_entity_boost_prefers_candidate_metadata_list_and_json() -> None:
    candidates = [
        {
            "content": "No explicit name here.",
            "score": 0.2,
            "entities": json.dumps(["Caroline", "Becoming Nicole"]),
        },
        {
            "content": "Caroline appears in content but metadata should win.",
            "score": 0.3,
            "entities": ["Unrelated"],
        },
    ]

    boosted = apply_entity_boost(
        "What did Caroline recommend?",
        candidates,
        EntityBoostConfig(enabled=True, category=None, boost=0.2, max_boost=0.2, query_entities=("caroline",)),
    )

    assert boosted[0]["content"] == "No explicit name here."
    assert boosted[0]["entity_boost"] == 0.2
    assert boosted[0]["candidate_entities"] == ["becoming nicole", "caroline"]
    assert "entity_boost" not in boosted[1]


@pytest.mark.unit
def test_entity_boost_requires_registry_query_entities_when_configured() -> None:
    candidates = [
        {
            "content": "Caroline recommended Becoming Nicole.",
            "score": 0.2,
            "entities": ["Caroline", "Becoming Nicole"],
        }
    ]

    boosted = apply_entity_boost(
        "What did Caroline recommend?",
        candidates,
        EntityBoostConfig(enabled=True, category=None, require_query_entities=True),
    )

    assert boosted == candidates


@pytest.mark.unit
def test_production_config_default_disabled_does_not_pass_entity_boost(rag_search: RAGMemorySearch) -> None:
    rag_search._indexer = MagicMock()
    rag_search._indexer_initialized = True
    mock_pipeline = MagicMock()
    mock_pipeline.run.return_value = PipelineResult(items=[{"content": "x", "score": 1.0}], abstain=False)

    with (
        patch.object(rag_search, "_load_rag_pipeline_settings", return_value=_settings(enabled=False)),
        patch.object(rag_search, "_vector_search_primary", return_value=[{"content": "x", "score": 1.0}]),
        patch.object(rag_search, "_graph_episodes_search", return_value=[]),
        patch("core.memory.rag_search.search_activity_log", return_value=[]),
        patch("core.memory.retrieval.pipeline.RetrievalPipeline", return_value=mock_pipeline),
    ):
        rag_search.search_memory_text(
            "Caroline",
            scope="all",
            knowledge_dir=rag_search._anima_dir / "knowledge",
            episodes_dir=rag_search._anima_dir / "episodes",
            procedures_dir=rag_search._anima_dir / "procedures",
            common_knowledge_dir=rag_search._common_knowledge_dir,
        )

    assert mock_pipeline.run.call_args.kwargs["entity_boost"] is None


@pytest.mark.unit
def test_production_config_enabled_passes_registry_entities_to_hybrid(rag_search: RAGMemorySearch) -> None:
    upsert_entities_from_facts(
        rag_search._anima_dir,
        [
            FactRecord(
                fact_id="fact-1",
                text="Caroline recommended Becoming Nicole.",
                source_entity="Caroline",
                target_entity="Becoming Nicole",
                recorded_at="2026-06-03T10:00:00+09:00",
            )
        ],
    )
    rag_search._indexer = MagicMock()
    rag_search._indexer_initialized = True
    mock_pipeline = MagicMock()
    mock_pipeline.run.return_value = PipelineResult(items=[{"content": "x", "score": 1.0}], abstain=False)

    with (
        patch.object(rag_search, "_load_rag_pipeline_settings", return_value=_settings(enabled=True)),
        patch.object(rag_search, "_vector_search_primary", return_value=[{"content": "x", "score": 1.0}]),
        patch.object(rag_search, "_graph_episodes_search", return_value=[]),
        patch("core.memory.rag_search.search_activity_log", return_value=[]),
        patch("core.memory.retrieval.pipeline.RetrievalPipeline", return_value=mock_pipeline),
    ):
        rag_search.search_memory_text(
            "What did Caroline recommend?",
            scope="all",
            knowledge_dir=rag_search._anima_dir / "knowledge",
            episodes_dir=rag_search._anima_dir / "episodes",
            procedures_dir=rag_search._anima_dir / "procedures",
            common_knowledge_dir=rag_search._common_knowledge_dir,
        )

    config = mock_pipeline.run.call_args.kwargs["entity_boost"]
    assert config.enabled is True
    assert config.category is None
    assert config.boost == 0.25
    assert config.max_boost == 0.40
    assert "caroline" in config.query_entities
    assert config.require_query_entities is True


@pytest.mark.unit
def test_non_all_vector_scope_receives_entity_boost_config(rag_search: RAGMemorySearch) -> None:
    rag_search._indexer = MagicMock()
    rag_search._indexer_initialized = True
    captured: dict[str, object] = {}

    def fake_vector(*args, **kwargs):
        captured["entity_boost"] = kwargs["entity_boost"]
        return []

    with (
        patch.object(rag_search, "_load_rag_pipeline_settings", return_value=_settings(enabled=True)),
        patch.object(rag_search, "_vector_search_primary", side_effect=fake_vector),
    ):
        rag_search.search_memory_text(
            "Caroline",
            scope="facts",
            knowledge_dir=rag_search._anima_dir / "knowledge",
            episodes_dir=rag_search._anima_dir / "episodes",
            procedures_dir=rag_search._anima_dir / "procedures",
            common_knowledge_dir=rag_search._common_knowledge_dir,
        )

    assert captured["entity_boost"].enabled is True
    assert captured["entity_boost"].require_query_entities is True
