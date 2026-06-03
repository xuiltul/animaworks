from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from core.memory.rag_search import RAGMemorySearch
from core.memory.retrieval.pipeline import PipelineResult


@pytest.fixture
def rag_search(tmp_path: Path) -> RAGMemorySearch:
    anima_dir = tmp_path / "animas" / "test"
    for sub in ("knowledge", "episodes", "procedures", "common_knowledge"):
        (anima_dir / sub).mkdir(parents=True)
    common_knowledge = tmp_path / "common_knowledge"
    common_skills = tmp_path / "common_skills"
    common_knowledge.mkdir()
    common_skills.mkdir()
    return RAGMemorySearch(anima_dir, common_knowledge, common_skills)


class TestRAGSearchScopeAllPipeline:
    def test_hybrid_applies_rerank_order(self, rag_search: RAGMemorySearch) -> None:
        vector_hits = [
            {"content": "alpha", "score": 0.9, "source_file": "a.md", "chunk_index": 0},
            {"content": "beta", "score": 0.8, "source_file": "b.md", "chunk_index": 0},
        ]
        reranked = [
            {"content": "beta", "score": 0.99, "search_method": "cross_encoder"},
            {"content": "alpha", "score": 0.11, "search_method": "cross_encoder"},
        ]

        mock_pipeline = MagicMock()
        mock_pipeline.run.return_value = PipelineResult(items=reranked, abstain=False)

        with (
            patch.object(rag_search, "_vector_search_primary", return_value=vector_hits),
            patch.object(rag_search, "_graph_episodes_search", return_value=[]),
            patch.object(rag_search, "_keyword_search_fallback", return_value=[]),
            patch(
                "core.memory.rag_search.search_activity_log",
                return_value=[],
            ),
            patch(
                "core.memory.retrieval.pipeline.RetrievalPipeline",
                return_value=mock_pipeline,
            ),
        ):
            results = rag_search.search_memory_text(
                "query",
                scope="all",
                knowledge_dir=rag_search._anima_dir / "knowledge",
                episodes_dir=rag_search._anima_dir / "episodes",
                procedures_dir=rag_search._anima_dir / "procedures",
                common_knowledge_dir=rag_search._anima_dir / "common_knowledge",
            )

        assert results[0]["content"] == "beta"
        assert results[0]["search_method"] == "cross_encoder"
        mock_pipeline.run.assert_called_once()

    def test_abstain_sets_last_search_meta(self, rag_search: RAGMemorySearch) -> None:
        mock_pipeline = MagicMock()
        mock_pipeline.run.return_value = PipelineResult(
            items=[],
            abstain=True,
            abstain_reason="low_confidence",
        )

        with (
            patch.object(rag_search, "_vector_search_primary", return_value=[{"content": "x", "score": 0.01}]),
            patch.object(rag_search, "_graph_episodes_search", return_value=[]),
            patch.object(rag_search, "_keyword_search_fallback", return_value=[]),
            patch("core.memory.rag_search.search_activity_log", return_value=[]),
            patch("core.memory.retrieval.pipeline.RetrievalPipeline", return_value=mock_pipeline),
        ):
            results = rag_search.search_memory_text(
                "query",
                scope="all",
                knowledge_dir=rag_search._anima_dir / "knowledge",
                episodes_dir=rag_search._anima_dir / "episodes",
                procedures_dir=rag_search._anima_dir / "procedures",
                common_knowledge_dir=rag_search._anima_dir / "common_knowledge",
            )

        assert results == []
        assert rag_search.last_search_meta["abstain"] is True
        assert rag_search.last_search_meta["abstain_reason"] == "low_confidence"

    def test_hybrid_all_returns_keyword_fallback_when_vector_sources_empty(
        self,
        rag_search: RAGMemorySearch,
    ) -> None:
        state_dir = rag_search._anima_dir / "state"
        state_dir.mkdir(parents=True)
        (state_dir / "conversation.json").write_text(
            json.dumps(
                {
                    "turns": [],
                    "compressed_summary": "Reviewed memory consolidation pipeline.",
                }
            ),
            encoding="utf-8",
        )

        with (
            patch.object(rag_search, "_vector_search_primary", return_value=[]),
            patch.object(rag_search, "_graph_episodes_search", return_value=[]),
            patch("core.memory.rag_search.search_activity_log", return_value=[]),
            patch("core.memory.retrieval.pipeline.RetrievalPipeline") as pipeline_cls,
        ):
            results = rag_search.search_memory_text(
                "consolidation",
                scope="all",
                knowledge_dir=rag_search._anima_dir / "knowledge",
                episodes_dir=rag_search._anima_dir / "episodes",
                procedures_dir=rag_search._anima_dir / "procedures",
                common_knowledge_dir=rag_search._anima_dir / "common_knowledge",
            )

        assert results[0]["memory_type"] == "conversation_summary"
        assert "consolidation" in results[0]["content"].lower()
        assert rag_search.last_search_meta["abstain"] is False
        pipeline_cls.assert_not_called()

    def test_hybrid_keyword_fallback_receives_entity_boost_before_slicing(
        self,
        rag_search: RAGMemorySearch,
    ) -> None:
        captured: dict[str, object] = {}

        def fake_keyword(*args, **kwargs):
            captured["entity_boost"] = kwargs["entity_boost"]
            captured["result_limit"] = kwargs["result_limit"]
            return [{"content": "Caroline hit", "score": 1.0, "entities": ["Caroline"]}]

        with (
            patch.object(
                rag_search,
                "_load_rag_pipeline_settings",
                return_value={
                    "rerank_enabled": False,
                    "rerank_candidate_pool": 20,
                    "cross_encoder_model": "dummy",
                    "abstain_on_low_confidence": False,
                    "confidence_threshold": 0.35,
                    "rrf_confidence_threshold": 0.02,
                    "entity_registry_enabled": False,
                    "entity_boost_enabled": True,
                    "entity_boost": 0.25,
                    "entity_boost_cap": 0.40,
                },
            ),
            patch.object(rag_search, "_vector_search_primary", return_value=[]),
            patch.object(rag_search, "_graph_episodes_search", return_value=[]),
            patch.object(rag_search, "_keyword_search_fallback", side_effect=fake_keyword),
            patch("core.memory.rag_search.search_activity_log", return_value=[]),
        ):
            rag_search.search_memory_text(
                "Caroline",
                scope="all",
                knowledge_dir=rag_search._anima_dir / "knowledge",
                episodes_dir=rag_search._anima_dir / "episodes",
                procedures_dir=rag_search._anima_dir / "procedures",
                common_knowledge_dir=rag_search._anima_dir / "common_knowledge",
            )

        assert captured["entity_boost"].enabled is True
        assert captured["result_limit"] == 50

    def test_non_all_scope_clears_meta(self, rag_search: RAGMemorySearch) -> None:
        rag_search._last_search_meta = {"abstain": True}
        with patch.object(rag_search, "_vector_search_primary", return_value=[]):
            rag_search.search_memory_text(
                "query",
                scope="knowledge",
                knowledge_dir=rag_search._anima_dir / "knowledge",
                episodes_dir=rag_search._anima_dir / "episodes",
                procedures_dir=rag_search._anima_dir / "procedures",
                common_knowledge_dir=rag_search._anima_dir / "common_knowledge",
            )
        assert rag_search.last_search_meta["abstain"] is False
