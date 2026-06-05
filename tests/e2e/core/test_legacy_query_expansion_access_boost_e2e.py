from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from core.memory.retrieval.access_boost import AccessBoostConfig
from core.memory.retrieval.unified_search import UnifiedMemorySearch


class StaticRAGSearch:
    def __init__(self, anima_dir: Path) -> None:
        self._anima_dir = anima_dir
        self.queries: list[str] = []

    def _load_rag_pipeline_settings(self) -> dict[str, object]:
        return {
            "rerank_enabled": False,
            "rerank_candidate_pool": 10,
            "cross_encoder_model": "dummy",
            "abstain_on_low_confidence": False,
            "confidence_threshold": 0.35,
            "rrf_confidence_threshold": 0.02,
            "access_boost_enabled": True,
            "access_boost_weight": 0.05,
            "access_boost_cap": 0.25,
            "access_boost_half_life_days": 30.0,
        }

    def _build_entity_boost_config(self, query: str, settings: dict[str, object] | None = None) -> None:
        return None

    def _build_access_boost_config(self, settings: dict[str, object]) -> AccessBoostConfig:
        return AccessBoostConfig(
            enabled=True,
            weight=float(settings["access_boost_weight"]),
            cap=float(settings["access_boost_cap"]),
            half_life_days=float(settings["access_boost_half_life_days"]),
        )

    def _get_indexer(self) -> object:
        return object()

    def _vector_search_primary(self, query: str, scope: str, *args: Any, **kwargs: Any) -> list[dict[str, Any]]:
        self.queries.append(query)
        if scope != "episodes":
            return []
        return [
            {
                "doc_id": "outside",
                "source_file": "outside.md",
                "chunk_index": 0,
                "memory_type": "episodes",
                "content": "Caroline discussed an unrelated trip.",
                "score": 0.99,
                "event_time_iso": "2023-04-20T10:00:00+00:00",
                "access_count": 100,
            },
            {
                "doc_id": "inside-low",
                "source_file": "inside-low.md",
                "chunk_index": 0,
                "memory_type": "episodes",
                "content": "Caroline visited the library.",
                "score": 0.7,
                "event_time_iso": "2023-05-07T10:00:00+00:00",
                "access_count": 0,
            },
            {
                "doc_id": "inside-high",
                "source_file": "inside-high.md",
                "chunk_index": 0,
                "memory_type": "episodes",
                "content": "Caroline visited the bookstore.",
                "score": 0.7,
                "event_time_iso": "2023-05-07T12:00:00+00:00",
                "access_count": 20,
            },
        ]

    def _graph_episodes_search(self, query: str, pool_k: int, knowledge_dir: Path) -> list[dict[str, Any]]:
        return []

    def _keyword_search_fallback(self, query: str, scope: str, *args: Any, **kwargs: Any) -> list[dict[str, Any]]:
        return []


@pytest.mark.e2e
def test_unified_legacy_retrieval_expands_temporal_query_filters_and_access_boosts(tmp_path: Path) -> None:
    anima_dir = tmp_path / "alice"
    anima_dir.mkdir(parents=True)
    rag = StaticRAGSearch(anima_dir)
    searcher = UnifiedMemorySearch(anima_dir, rag_search=rag)

    results = searcher.search(
        "What did Caroline do yesterday?",
        scope="episodes",
        limit=2,
        trigger="chat",
        reference_time=datetime(2023, 5, 8, 12, 0, tzinfo=UTC),
    )

    assert "2023-05-07" in rag.queries[0]
    assert [item["doc_id"] for item in results] == ["inside-high", "inside-low"]
    assert all(item["doc_id"] != "outside" for item in results)
    assert results[0]["access_boost"] > 0.0
