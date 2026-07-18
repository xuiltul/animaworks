from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from core.memory.retrieval.pipeline import PipelineResult
from core.memory.retrieval.unified_search import UnifiedMemorySearch


class FakeRAGSearch:
    def __init__(self, anima_dir: Path) -> None:
        self._anima_dir = anima_dir
        self._common_knowledge_dir = anima_dir.parent.parent / "common_knowledge"
        self._common_skills_dir = anima_dir.parent.parent / "common_skills"
        self.vector_returns: dict[str, list[dict[str, Any]]] = {}
        self.keyword_returns: dict[str, list[dict[str, Any]]] = {}
        self.graph_returns: list[dict[str, Any]] = []
        self.vector_scopes: list[str] = []
        self.vector_queries: list[str] = []
        self.keyword_scopes: list[str] = []
        self.keyword_queries: list[str] = []
        self.graph_calls = 0

    def _load_rag_pipeline_settings(self) -> dict[str, object]:
        return {
            "rerank_enabled": True,
            "rerank_candidate_pool": 50,
            "cross_encoder_model": "dummy",
            "abstain_on_low_confidence": True,
            "confidence_threshold": 0.35,
            "rrf_confidence_threshold": 0.02,
        }

    def _build_entity_boost_config(self, query: str, settings: dict[str, object] | None = None) -> None:
        return None

    def _get_indexer(self) -> object:
        return object()

    def _vector_search_primary(self, query: str, scope: str, *args, **kwargs) -> list[dict[str, Any]]:
        self.vector_queries.append(query)
        self.vector_scopes.append(scope)
        return self.vector_returns.get(scope, [])

    def _graph_episodes_search(self, query: str, pool_k: int, knowledge_dir: Path) -> list[dict[str, Any]]:
        self.graph_calls += 1
        return self.graph_returns

    def _keyword_search_fallback(self, query: str, scope: str, *args, **kwargs) -> list[dict[str, Any]]:
        self.keyword_queries.append(query)
        self.keyword_scopes.append(scope)
        return self.keyword_returns.get(scope, [])


class CapturingPipeline:
    calls: list[dict[str, Any]] = []
    return_items: list[dict[str, Any]] | None = None
    abstain = False
    abstain_reason = ""

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs

    def run(self, query: str, ranked_lists: list[list[dict[str, Any]]], **kwargs) -> PipelineResult:
        self.__class__.calls.append({"query": query, "ranked_lists": ranked_lists, **kwargs})
        if self.__class__.return_items is None:
            items = [item for ranked_list in ranked_lists for item in ranked_list]
        else:
            items = self.__class__.return_items
        return PipelineResult(
            items=items[: int(kwargs.get("limit", len(items)))],
            abstain=self.__class__.abstain,
            abstain_reason=self.__class__.abstain_reason,
        )


@pytest.fixture(autouse=True)
def reset_pipeline() -> None:
    CapturingPipeline.calls = []
    CapturingPipeline.return_items = None
    CapturingPipeline.abstain = False
    CapturingPipeline.abstain_reason = ""


@pytest.fixture
def fake_rag(tmp_path: Path) -> FakeRAGSearch:
    anima_dir = tmp_path / "animas" / "test"
    anima_dir.mkdir(parents=True)
    return FakeRAGSearch(anima_dir)


def _searcher(fake_rag: FakeRAGSearch) -> UnifiedMemorySearch:
    return UnifiedMemorySearch(
        fake_rag._anima_dir,
        common_knowledge_dir=fake_rag._common_knowledge_dir,
        common_skills_dir=fake_rag._common_skills_dir,
        rag_search=fake_rag,
    )


def test_heartbeat_policy_disables_rerank(fake_rag: FakeRAGSearch, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("core.memory.retrieval.pipeline.RetrievalPipeline", CapturingPipeline)
    monkeypatch.setattr("core.memory.retrieval.unified_search.search_activity_log", lambda *args, **kwargs: [])
    fake_rag.vector_returns["episodes"] = [{"doc_id": "episode-1", "content": "episode", "score": 0.8}]

    results = _searcher(fake_rag).search("query", scope="all", limit=3, trigger="heartbeat")

    assert results[0]["doc_id"] == "episode-1"
    assert CapturingPipeline.calls[0]["pool_k"] == 20
    assert CapturingPipeline.calls[0]["rerank_enabled"] is False


def test_explicit_scope_restricts_trigger_scopes(fake_rag: FakeRAGSearch, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("core.memory.retrieval.pipeline.RetrievalPipeline", CapturingPipeline)
    monkeypatch.setattr("core.memory.retrieval.unified_search.search_activity_log", lambda *args, **kwargs: [])
    fake_rag.vector_returns["facts"] = [{"doc_id": "fact-1", "content": "fact", "score": 0.9}]

    _searcher(fake_rag).search("query", scope="facts", limit=3, trigger="chat")

    assert fake_rag.vector_scopes == ["facts"]
    assert fake_rag.keyword_scopes == ["facts"]
    assert fake_rag.graph_calls == 0


def test_tool_offset_applies_after_final_ranking(fake_rag: FakeRAGSearch, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("core.memory.retrieval.pipeline.RetrievalPipeline", CapturingPipeline)
    monkeypatch.setattr("core.memory.retrieval.unified_search.search_activity_log", lambda *args, **kwargs: [])
    fake_rag.vector_returns["episodes"] = [{"doc_id": "seed", "content": "seed", "score": 0.1}]
    CapturingPipeline.return_items = [
        {"doc_id": "a", "content": "a", "score": 0.9},
        {"doc_id": "b", "content": "b", "score": 0.8},
        {"doc_id": "c", "content": "c", "score": 0.7},
    ]

    tool_results = _searcher(fake_rag).search("query", scope="episodes", limit=1, trigger="tool", offset=1)
    chat_results = _searcher(fake_rag).search("query", scope="episodes", limit=1, trigger="chat", offset=1)

    assert tool_results[0]["doc_id"] == "b"
    assert chat_results[0]["doc_id"] == "a"


def test_abstain_propagates_last_search_meta(fake_rag: FakeRAGSearch, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("core.memory.retrieval.pipeline.RetrievalPipeline", CapturingPipeline)
    monkeypatch.setattr("core.memory.retrieval.unified_search.search_activity_log", lambda *args, **kwargs: [])
    fake_rag.vector_returns["knowledge"] = [{"doc_id": "k", "content": "knowledge", "score": 0.01}]
    CapturingPipeline.return_items = []
    CapturingPipeline.abstain = True
    CapturingPipeline.abstain_reason = "low_confidence"
    searcher = _searcher(fake_rag)

    assert searcher.search("query", scope="knowledge", limit=3, trigger="chat") == []
    assert searcher.last_search_meta["abstain"] is True
    assert searcher.last_search_meta["abstain_reason"] == "low_confidence"


def test_missing_fact_index_continues_other_scopes(fake_rag: FakeRAGSearch, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("core.memory.retrieval.pipeline.RetrievalPipeline", CapturingPipeline)
    monkeypatch.setattr("core.memory.retrieval.unified_search.search_activity_log", lambda *args, **kwargs: [])
    fake_rag.vector_returns["facts"] = []
    fake_rag.vector_returns["episodes"] = [{"doc_id": "episode-1", "content": "episode", "score": 0.8}]

    results = _searcher(fake_rag).search("query", scope="all", limit=3, trigger="chat")

    assert "facts" in fake_rag.vector_scopes
    assert results[0]["doc_id"] == "episode-1"


def test_query_expansion_uses_reference_time_and_filters_event_time(
    fake_rag: FakeRAGSearch,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("core.memory.retrieval.pipeline.RetrievalPipeline", CapturingPipeline)
    monkeypatch.setattr("core.memory.retrieval.unified_search.search_activity_log", lambda *args, **kwargs: [])
    fake_rag.vector_returns["episodes"] = [
        {
            "doc_id": "outside",
            "content": "outside",
            "score": 0.9,
            "source_file": "outside.md",
            "chunk_index": 0,
            "memory_type": "episodes",
            "event_time_iso": "2023-04-20T00:00:00+00:00",
        },
        {
            "doc_id": "inside",
            "content": "inside",
            "score": 0.8,
            "source_file": "inside.md",
            "chunk_index": 0,
            "memory_type": "episodes",
            "event_time_iso": "2023-05-07T00:00:00+00:00",
        },
    ]

    results = _searcher(fake_rag).search(
        "What did Caroline do yesterday?",
        scope="episodes",
        limit=3,
        trigger="chat",
        reference_time=datetime(2023, 5, 8, 12, 0, tzinfo=UTC),
    )

    # F19: the expanded ISO date is a sparse-only signal. It reaches the
    # keyword (BM25) query but must stay out of the dense vector query and the
    # cross-encoder query the pipeline reranks against.
    assert "2023-05-07" not in fake_rag.vector_queries[0]
    assert "2023-05-07" in fake_rag.keyword_queries[0]
    assert "2023-05-07" not in CapturingPipeline.calls[0]["query"]
    assert [item["doc_id"] for item in results] == ["inside"]
    assert CapturingPipeline.calls[0]["ranked_lists"][0][0]["doc_id"] == "inside"


def test_unified_search_passes_access_boost_config_to_pipeline(
    fake_rag: FakeRAGSearch,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class AccessFakeRAG(FakeRAGSearch):
        def _build_access_boost_config(self, settings: dict[str, object]) -> str:
            return "access-config"

    access_rag = AccessFakeRAG(fake_rag._anima_dir)
    access_rag.vector_returns["knowledge"] = [{"doc_id": "k", "content": "knowledge", "score": 0.8}]
    monkeypatch.setattr("core.memory.retrieval.pipeline.RetrievalPipeline", CapturingPipeline)
    monkeypatch.setattr("core.memory.retrieval.unified_search.search_activity_log", lambda *args, **kwargs: [])

    _searcher(access_rag).search("query", scope="knowledge", limit=3, trigger="chat")

    assert CapturingPipeline.calls[0]["access_boost"] == "access-config"


def test_unified_search_builds_temporal_boost_from_yesterday_query(
    fake_rag: FakeRAGSearch,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("core.memory.retrieval.pipeline.RetrievalPipeline", CapturingPipeline)
    monkeypatch.setattr("core.memory.retrieval.unified_search.search_activity_log", lambda *args, **kwargs: [])
    fake_rag.vector_returns["episodes"] = [
        {"doc_id": "yesterday", "content": "meeting", "score": 0.8, "source_file": "2026-07-17.md"}
    ]

    _searcher(fake_rag).search(
        "昨日のミーティング",
        scope="episodes",
        limit=3,
        trigger="chat",
        reference_time=datetime(2026, 7, 18, 12, 0),
    )

    temporal = CapturingPipeline.calls[0]["temporal_boost"]
    assert temporal is not None
    assert temporal.time_range.start == datetime(2026, 7, 17)
    assert temporal.time_range.end == datetime(2026, 7, 17, 23, 59, 59, 999999)
    assert temporal.boost == 0.05
    assert temporal.max_boost == 0.10
    assert temporal.half_life_days == 7.0


def test_unified_search_temporal_boost_config_gate_disables_auto_wiring(
    fake_rag: FakeRAGSearch,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_settings = fake_rag._load_rag_pipeline_settings

    def disabled_settings() -> dict[str, object]:
        settings = original_settings()
        settings["temporal_boost_enabled"] = False
        return settings

    monkeypatch.setattr(fake_rag, "_load_rag_pipeline_settings", disabled_settings)
    monkeypatch.setattr("core.memory.retrieval.pipeline.RetrievalPipeline", CapturingPipeline)
    monkeypatch.setattr("core.memory.retrieval.unified_search.search_activity_log", lambda *args, **kwargs: [])
    fake_rag.vector_returns["episodes"] = [
        {"doc_id": "yesterday", "content": "meeting", "score": 0.8, "source_file": "2026-07-17.md"}
    ]

    _searcher(fake_rag).search(
        "昨日のミーティング",
        scope="episodes",
        limit=3,
        trigger="chat",
        reference_time=datetime(2026, 7, 18, 12, 0),
    )

    assert CapturingPipeline.calls[0]["temporal_boost"] is None


def test_unified_search_explicit_range_overrides_query_expression(
    fake_rag: FakeRAGSearch,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("core.memory.retrieval.pipeline.RetrievalPipeline", CapturingPipeline)
    monkeypatch.setattr("core.memory.retrieval.unified_search.search_activity_log", lambda *args, **kwargs: [])
    fake_rag.vector_returns["episodes"] = [
        {"doc_id": "explicit", "content": "meeting", "score": 0.8, "source_file": "2026-07-01.md"}
    ]

    _searcher(fake_rag).search(
        "昨日のミーティング",
        scope="episodes",
        limit=3,
        trigger="tool",
        time_start="2026-07-01",
        time_end="2026-07-02",
        reference_time=datetime(2026, 7, 18, 12, 0),
    )

    temporal = CapturingPipeline.calls[0]["temporal_boost"]
    assert temporal.time_range.start == datetime(2026, 7, 1)
    assert temporal.time_range.end == datetime(2026, 7, 2, 23, 59, 59, 999999)


def test_search_many_automatically_builds_temporal_boost_per_query(
    fake_rag: FakeRAGSearch,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("core.memory.retrieval.pipeline.RetrievalPipeline", CapturingPipeline)
    monkeypatch.setattr("core.memory.retrieval.unified_search.search_activity_log", lambda *args, **kwargs: [])
    fake_rag.vector_returns["episodes"] = [
        {"doc_id": "dated", "content": "meeting", "score": 0.8, "source_file": "2026-07-18.md"}
    ]

    _searcher(fake_rag).search_many(
        ["2026-07-18 meeting"],
        scope="episodes",
        limit=3,
        trigger="chat",
    )

    temporal = CapturingPipeline.calls[0]["temporal_boost"]
    assert temporal.time_range.start == datetime(2026, 7, 18)
    assert temporal.time_range.end == datetime(2026, 7, 18, 23, 59, 59, 999999)


def test_knowledge_bm25_keyword_list_is_merged_in_pipeline(
    fake_rag: FakeRAGSearch,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("core.memory.retrieval.pipeline.RetrievalPipeline", CapturingPipeline)
    monkeypatch.setattr("core.memory.retrieval.unified_search.search_activity_log", lambda *args, **kwargs: [])
    fake_rag.vector_returns["knowledge"] = [
        {"doc_id": "vector-k", "content": "semantic hit", "score": 0.8, "search_method": "vector"},
    ]
    fake_rag.keyword_returns["knowledge"] = [
        {"doc_id": "bm25-k", "content": "ZephyrNova exact hit", "score": 3.0, "search_method": "bm25"},
    ]

    results = _searcher(fake_rag).search("ZephyrNova", scope="knowledge", limit=3, trigger="chat")

    assert [item["doc_id"] for item in results] == ["vector-k", "bm25-k"]
    ranked_lists = CapturingPipeline.calls[0]["ranked_lists"]
    assert any(item.get("search_method") == "bm25" for ranked_list in ranked_lists for item in ranked_list)
    assert fake_rag.keyword_scopes == ["knowledge"]


def test_min_score_skipped_for_rrf_order_results(
    fake_rag: FakeRAGSearch,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # RRF-order results carry tiny fusion scores (~0.03 max). min_score=0.3 must
    # not wipe them out when rerank did not run (F2).
    monkeypatch.setattr("core.memory.retrieval.pipeline.RetrievalPipeline", CapturingPipeline)
    monkeypatch.setattr("core.memory.retrieval.unified_search.search_activity_log", lambda *args, **kwargs: [])
    fake_rag.vector_returns["episodes"] = [{"doc_id": "seed", "content": "seed", "score": 0.1}]
    CapturingPipeline.return_items = [
        {"doc_id": "rrf-1", "content": "a", "score": 0.03, "search_method": "rrf"},
        {"doc_id": "rrf-2", "content": "b", "score": 0.02, "search_method": "rrf"},
    ]

    results = _searcher(fake_rag).search(
        "query", scope="episodes", limit=5, trigger="heartbeat", min_score=0.3
    )

    assert [item["doc_id"] for item in results] == ["rrf-1", "rrf-2"]


def test_min_score_applied_to_reranked_results(
    fake_rag: FakeRAGSearch,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # After cross-encoder rerank the score is a CE logit that min_score is
    # calibrated against, so the filter should still drop low-scoring rows (F2).
    monkeypatch.setattr("core.memory.retrieval.pipeline.RetrievalPipeline", CapturingPipeline)
    monkeypatch.setattr("core.memory.retrieval.unified_search.search_activity_log", lambda *args, **kwargs: [])
    fake_rag.vector_returns["episodes"] = [{"doc_id": "seed", "content": "seed", "score": 0.1}]
    CapturingPipeline.return_items = [
        {"doc_id": "ce-high", "content": "a", "score": 4.5, "search_method": "cross_encoder"},
        {"doc_id": "ce-low", "content": "b", "score": 0.1, "search_method": "cross_encoder"},
    ]

    results = _searcher(fake_rag).search(
        "query", scope="episodes", limit=5, trigger="chat", min_score=0.3
    )

    assert [item["doc_id"] for item in results] == ["ce-high"]


def test_tool_and_priming_overlap_share_top_doc_ids(
    fake_rag: FakeRAGSearch,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("core.memory.retrieval.pipeline.RetrievalPipeline", CapturingPipeline)
    monkeypatch.setattr("core.memory.retrieval.unified_search.search_activity_log", lambda *args, **kwargs: [])
    fake_rag.vector_returns["episodes"] = [
        {"doc_id": "episode-a", "content": "a", "score": 0.9},
        {"doc_id": "episode-b", "content": "b", "score": 0.8},
    ]
    searcher = _searcher(fake_rag)

    tool_doc_ids = [item["doc_id"] for item in searcher.search("query", scope="episodes", limit=2, trigger="tool")]
    priming_doc_ids = [
        item["doc_id"] for item in searcher.search_many(["query"], scope="episodes", limit=2, trigger="chat")
    ]

    assert priming_doc_ids == tool_doc_ids
