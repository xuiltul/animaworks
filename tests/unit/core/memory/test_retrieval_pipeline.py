from __future__ import annotations

from unittest.mock import MagicMock

from core.memory.retrieval.pipeline import RetrievalPipeline


def test_pipeline_rerank_disabled_keeps_rrf_order() -> None:
    ranked = [
        [
            {"content": "first", "score": 0.5, "source_file": "a.md", "chunk_index": 0},
            {"content": "second", "score": 0.4, "source_file": "b.md", "chunk_index": 0},
        ],
    ]
    pipeline = RetrievalPipeline(reranker=MagicMock())
    result = pipeline.run("q", ranked, limit=2, rerank_enabled=False, abstain_on_low_confidence=False)
    assert [c["content"] for c in result.items] == ["first", "second"]
    pipeline._reranker.rerank_sync.assert_not_called()


def test_pipeline_empty_lists_abstains() -> None:
    pipeline = RetrievalPipeline(reranker=MagicMock())
    result = pipeline.run("q", [[]], abstain_on_low_confidence=True, confidence_threshold=0.35)
    assert result.abstain is True
    assert result.items == []


def test_pipeline_uses_reranker_when_enabled() -> None:
    reranker = MagicMock()
    reranker.rerank_sync.return_value = [
        {"content": "b", "score": 0.9, "search_method": "cross_encoder"},
    ]
    ranked = [
        [
            {"content": "a", "score": 0.5, "source_file": "a.md", "chunk_index": 0},
            {"content": "b", "score": 0.4, "source_file": "b.md", "chunk_index": 0},
        ],
    ]
    pipeline = RetrievalPipeline(reranker=reranker)
    result = pipeline.run(
        "q",
        ranked,
        limit=1,
        rerank_enabled=True,
        abstain_on_low_confidence=False,
    )
    assert result.items[0]["content"] == "b"
    reranker.rerank_sync.assert_called_once()
