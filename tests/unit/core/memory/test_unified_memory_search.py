from __future__ import annotations

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
        self.keyword_scopes: list[str] = []
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
        self.vector_scopes.append(scope)
        return self.vector_returns.get(scope, [])

    def _graph_episodes_search(self, query: str, pool_k: int, knowledge_dir: Path) -> list[dict[str, Any]]:
        self.graph_calls += 1
        return self.graph_returns

    def _keyword_search_fallback(self, query: str, scope: str, *args, **kwargs) -> list[dict[str, Any]]:
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
