from __future__ import annotations

from unittest.mock import MagicMock

from core.memory.retrieval.entity import EntityBoostConfig, extract_entities
from core.memory.retrieval.pipeline import RetrievalPipeline
from core.memory.retrieval.temporal import TemporalBoostConfig


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


def test_pipeline_temporal_boost_absent_keeps_default_order() -> None:
    ranked = [
        [
            {"content": "older", "score": 0.5, "source_file": "a.md", "chunk_index": 0},
            {
                "content": "newer 2023",
                "score": 0.4,
                "source_file": "b.md",
                "chunk_index": 0,
                "event_time_iso": "2023-07-20T20:56:00+09:00",
            },
        ],
    ]
    pipeline = RetrievalPipeline(reranker=MagicMock())

    result = pipeline.run("what happened in 2023?", ranked, limit=2, rerank_enabled=False, abstain_on_low_confidence=False)

    assert [item["content"] for item in result.items] == ["older", "newer 2023"]
    assert "temporal_boost" not in result.items[1]


def test_pipeline_temporal_boost_is_category_2_only() -> None:
    ranked = [
        [
            {
                "content": "newer 2023",
                "score": 0.4,
                "source_file": "b.md",
                "chunk_index": 0,
                "event_time_iso": "2023-07-20T20:56:00+09:00",
            },
        ],
    ]
    pipeline = RetrievalPipeline(reranker=MagicMock())

    result = pipeline.run(
        "what happened in 2023?",
        ranked,
        limit=1,
        rerank_enabled=False,
        abstain_on_low_confidence=False,
        temporal_boost=TemporalBoostConfig(enabled=True, category=4),
    )

    assert "temporal_boost" not in result.items[0]


def test_pipeline_temporal_boost_adds_score_for_temporal_year_match() -> None:
    ranked = [
        [
            {
                "content": "newer event in 2023",
                "score": 0.4,
                "source_file": "b.md",
                "chunk_index": 0,
                "event_time_iso": "2023-07-20T20:56:00+09:00",
            },
        ],
    ]
    pipeline = RetrievalPipeline(reranker=MagicMock())

    result = pipeline.run(
        "what happened in 2023?",
        ranked,
        limit=1,
        rerank_enabled=False,
        abstain_on_low_confidence=False,
        temporal_boost=TemporalBoostConfig(enabled=True, category=2),
    )

    item = result.items[0]
    assert item["base_score"] == item["rrf_score"]
    assert item["temporal_boost"] == 0.10
    assert item["score"] == item["base_score"] + 0.10


def test_extract_entities_deduplicates_phrases_and_ignores_speakers() -> None:
    entities = extract_entities(
        'What book did Melanie read from Caroline\'s suggestion, "Becoming Nicole"?',
        ignored_entities=("Melanie", "Caroline"),
    )

    assert "melanie" not in entities
    assert "caroline" not in entities
    assert "becoming nicole" in entities
    assert "book" in entities
    assert "suggestion" in entities


def test_pipeline_entity_boost_absent_keeps_default_order() -> None:
    ranked = [
        [
            {"content": "generic memory", "score": 0.5, "source_file": "a.md", "chunk_index": 0},
            {
                "content": "Melanie read Becoming Nicole after Caroline suggested the book.",
                "score": 0.4,
                "source_file": "b.md",
                "chunk_index": 0,
            },
        ],
    ]
    pipeline = RetrievalPipeline(reranker=MagicMock())

    result = pipeline.run(
        "What book did Melanie read from Caroline's suggestion?",
        ranked,
        limit=2,
        rerank_enabled=False,
        abstain_on_low_confidence=False,
    )

    assert [item["content"] for item in result.items] == [
        "generic memory",
        "Melanie read Becoming Nicole after Caroline suggested the book.",
    ]
    assert "entity_boost" not in result.items[1]


def test_pipeline_entity_boost_reorders_category_1_overlap() -> None:
    ranked = [
        [
            {"content": "generic memory", "score": 0.5, "source_file": "a.md", "chunk_index": 0},
            {
                "content": "Melanie read Becoming Nicole after Caroline suggested the book.",
                "score": 0.4,
                "source_file": "b.md",
                "chunk_index": 0,
            },
        ],
    ]
    pipeline = RetrievalPipeline(reranker=MagicMock())

    result = pipeline.run(
        "What book did Melanie read from Caroline's suggestion?",
        ranked,
        limit=2,
        rerank_enabled=False,
        abstain_on_low_confidence=False,
        entity_boost=EntityBoostConfig(
            enabled=True,
            category=1,
            ignored_entities=("Melanie", "Caroline"),
        ),
    )

    item = result.items[0]
    assert item["content"] == "Melanie read Becoming Nicole after Caroline suggested the book."
    assert item["entity_boost"] > 0.0
    assert "book" in item["entity_overlap"]
    assert "melanie" not in item["query_entities"]


def test_pipeline_entity_boost_is_category_1_and_4_only() -> None:
    ranked = [
        [
            {
                "content": "The adoption agency supports LGBTQ+ individuals.",
                "score": 0.4,
                "source_file": "b.md",
                "chunk_index": 0,
            },
        ],
    ]
    pipeline = RetrievalPipeline(reranker=MagicMock())

    for category in (2, 3, 5):
        result = pipeline.run(
            "What type of individuals does the adoption agency support?",
            ranked,
            limit=1,
            rerank_enabled=False,
            abstain_on_low_confidence=False,
            entity_boost=EntityBoostConfig(enabled=True, category=category),
        )

        assert "entity_boost" not in result.items[0]


def test_pipeline_entity_boost_applies_to_category_4() -> None:
    ranked = [
        [
            {
                "content": "The adoption agency supports LGBTQ+ individuals.",
                "score": 0.4,
                "source_file": "b.md",
                "chunk_index": 0,
            },
        ],
    ]
    pipeline = RetrievalPipeline(reranker=MagicMock())

    result = pipeline.run(
        "What type of individuals does the adoption agency support?",
        ranked,
        limit=1,
        rerank_enabled=False,
        abstain_on_low_confidence=False,
        entity_boost=EntityBoostConfig(enabled=True, category=4),
    )

    assert result.items[0]["entity_boost"] > 0.0
    assert "adoption agency" in result.items[0]["entity_overlap"]
