from __future__ import annotations

from pathlib import Path

from core.memory.entity_index import upsert_entities_from_facts
from core.memory.facts import FactRecord, append_fact_records
from core.memory.rag.graph import KnowledgeGraph
from core.memory.rag.retriever import RetrievalResult


class EmptyVectorStore:
    def query(self, collection, embedding, top_k, filter_metadata=None):
        return []


class MockIndexer:
    def __init__(self, anima_dir: Path):
        self.anima_name = "e2e_anima"
        self.anima_dir = anima_dir

    def _generate_embeddings(self, texts):
        return [[0.1] * 8 for _ in texts]


def test_legacy_entity_aware_graph_expands_episode_to_active_atomic_fact(tmp_path: Path) -> None:
    anima_dir = tmp_path / "e2e_anima"
    knowledge_dir = anima_dir / "knowledge"
    episodes_dir = anima_dir / "episodes"
    knowledge_dir.mkdir(parents=True)
    episodes_dir.mkdir(parents=True)
    (knowledge_dir / "profile.md").write_text("Alice tracks LoCoMo memory scores.", encoding="utf-8")
    (episodes_dir / "2026-06-03.md").write_text("Alice prefers LoCoMo score deltas.", encoding="utf-8")

    active = FactRecord(
        text="Alice prefers LoCoMo score deltas.",
        source_entity="Alice",
        target_entity="LoCoMo",
        edge_type="PREFERS",
        valid_at="2026-06-03T10:00:00+09:00",
        recorded_at="2026-06-03T10:05:00+09:00",
        source_episode="episodes/2026-06-03.md",
    )
    expired = FactRecord(
        text="Alice previously preferred verbose LoCoMo reports.",
        source_entity="Alice",
        target_entity="LoCoMo",
        edge_type="PREFERS",
        valid_at="2026-05-01T10:00:00+09:00",
        recorded_at="2026-05-01T10:05:00+09:00",
        valid_until="2000-01-01T00:00:00+00:00",
        source_episode="episodes/2026-06-03.md",
    )
    append_fact_records(anima_dir, [active, expired])
    upsert_entities_from_facts(anima_dir, [active, expired])

    graph_builder = KnowledgeGraph(EmptyVectorStore(), MockIndexer(anima_dir))
    graph = graph_builder.build_graph(
        "e2e_anima",
        knowledge_dir,
        memory_dirs={"episodes": episodes_dir},
        entity_aware_graph_enabled=True,
        graph_recency_weight_enabled=False,
    )

    assert f"fact:{active.fact_id}" in graph
    assert f"fact:{expired.fact_id}" not in graph

    expanded = graph_builder.expand_search_results(
        [
            RetrievalResult(
                doc_id="e2e_anima/episodes/2026-06-03.md#0",
                content="Alice prefers LoCoMo score deltas.",
                score=0.9,
                metadata={"source_file": "episodes/2026-06-03.md", "memory_type": "episodes"},
                source_scores={"vector": 0.9},
            )
        ],
        max_hops=2,
    )

    fact_results = [result for result in expanded if result.metadata.get("memory_type") == "facts"]
    assert fact_results
    assert fact_results[0].metadata["fact_id"] == active.fact_id
    assert active.text in fact_results[0].content
