from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.memory.entity_index import upsert_entities_from_facts
from core.memory.facts import FactRecord, append_fact_records
from core.memory.rag.graph import GRAPH_SCHEMA_VERSION, KnowledgeGraph
from core.memory.rag.retriever import RetrievalResult


class EmptyVectorStore:
    def query(self, collection, embedding, top_k, filter_metadata=None):
        return []


class MockIndexer:
    def __init__(self, anima_dir: Path):
        self.anima_name = "test_anima"
        self.anima_dir = anima_dir

    def _generate_embeddings(self, texts):
        return [[0.1] * 8 for _ in texts]


def _fixture_anima(tmp_path: Path) -> tuple[Path, FactRecord, FactRecord]:
    anima_dir = tmp_path / "test_anima"
    knowledge_dir = anima_dir / "knowledge"
    episodes_dir = anima_dir / "episodes"
    knowledge_dir.mkdir(parents=True)
    episodes_dir.mkdir(parents=True)

    (knowledge_dir / "profile.md").write_text(
        "# Profile\n\nAlice keeps LoCoMo score notes.",
        encoding="utf-8",
    )
    (episodes_dir / "session-1.md").write_text(
        "# Session 1\n\nAlice discussed LoCoMo score deltas.",
        encoding="utf-8",
    )

    active = FactRecord(
        text="Alice prefers LoCoMo score deltas.",
        source_entity="Alice",
        target_entity="LoCoMo",
        edge_type="PREFERS",
        valid_at="2026-06-03T09:00:00+09:00",
        recorded_at="2026-06-03T09:05:00+09:00",
        source_episode="episodes/session-1.md",
    )
    expired = FactRecord(
        text="Bob uses an obsolete LoCoMo rubric.",
        source_entity="Bob",
        target_entity="LoCoMo",
        edge_type="USES",
        valid_at="2026-06-01T09:00:00+09:00",
        recorded_at="2026-06-01T09:05:00+09:00",
        valid_until="2000-01-01T00:00:00+00:00",
        source_episode="episodes/session-1.md",
    )
    append_fact_records(anima_dir, [active, expired])
    upsert_entities_from_facts(anima_dir, [active, expired])
    return anima_dir, active, expired


def _build_entity_graph(
    anima_dir: Path,
    *,
    edge_cap: int = 8,
    inverse_fan: bool = True,
) -> tuple[KnowledgeGraph, object]:
    builder = KnowledgeGraph(EmptyVectorStore(), MockIndexer(anima_dir))
    graph = builder.build_graph(
        "test_anima",
        anima_dir / "knowledge",
        memory_dirs={"episodes": anima_dir / "episodes"},
        implicit_link_threshold=0.99,
        entity_aware_graph_enabled=True,
        graph_entity_edge_cap=edge_cap,
        graph_inverse_fan_enabled=inverse_fan,
        graph_recency_weight_enabled=False,
    )
    return builder, graph


def test_entity_aware_graph_adds_active_fact_entity_edges_and_diagnostics(tmp_path: Path) -> None:
    anima_dir, active, expired = _fixture_anima(tmp_path)
    _builder, graph = _build_entity_graph(anima_dir)

    assert "entity:alice" in graph
    assert "entity:locomo" in graph
    assert f"fact:{active.fact_id}" in graph
    assert f"fact:{expired.fact_id}" not in graph

    assert graph.has_edge("episodes:session-1", "entity:alice")
    assert graph["episodes:session-1"]["entity:alice"]["link_type"] == "mentions_entity"
    assert graph.has_edge(f"fact:{active.fact_id}", "episodes:session-1")
    assert graph[f"fact:{active.fact_id}"]["episodes:session-1"]["link_type"] == "fact_source"
    assert graph.has_edge(f"fact:{active.fact_id}", "entity:alice")
    assert graph[f"fact:{active.fact_id}"]["entity:alice"]["link_type"] == "fact_entity"
    assert any(data.get("link_type") == "co_mention" for _u, _v, data in graph.edges(data=True))

    diagnostics = graph.graph["diagnostics"]
    assert diagnostics["node_types"]["memory_file"] == 2
    assert diagnostics["node_types"]["fact"] == 1
    assert diagnostics["node_types"]["entity"] >= 2
    assert diagnostics["edge_types"]["fact_entity"] >= 2


def test_entity_edge_cap_limits_co_mentions(tmp_path: Path) -> None:
    anima_dir, _active, _expired = _fixture_anima(tmp_path)
    _builder, graph = _build_entity_graph(anima_dir, edge_cap=1)

    assert [data for _u, _v, data in graph.edges(data=True) if data.get("link_type") == "co_mention"] == []


def test_inverse_fan_reduces_high_fanout_entity_weight(tmp_path: Path) -> None:
    anima_dir, _active, _expired = _fixture_anima(tmp_path)
    _builder, graph = _build_entity_graph(anima_dir, inverse_fan=True)
    _builder_no_inverse, graph_no_inverse = _build_entity_graph(anima_dir, inverse_fan=False)

    weighted = graph["episodes:session-1"]["entity:alice"]
    unweighted = graph_no_inverse["episodes:session-1"]["entity:alice"]
    assert weighted["fanout"] > 1
    assert weighted["similarity"] < unweighted["similarity"]
    assert unweighted["similarity"] == pytest.approx(0.45)


def test_entity_aware_graph_cache_rejects_schema_and_mode_mismatch(tmp_path: Path) -> None:
    anima_dir, _active, _expired = _fixture_anima(tmp_path)
    builder, _graph = _build_entity_graph(anima_dir)
    cache_dir = anima_dir / "vectordb"
    builder.save_graph(cache_dir)

    disabled_loader = KnowledgeGraph(EmptyVectorStore(), MockIndexer(anima_dir))
    assert disabled_loader.load_graph(
        cache_dir,
        expected_schema_version=GRAPH_SCHEMA_VERSION,
        entity_aware_graph_enabled=False,
    ) is False

    cache_path = cache_dir / "knowledge_graph.json"
    payload = json.loads(cache_path.read_text(encoding="utf-8"))
    payload["graph"].pop("schema_version", None)
    cache_path.write_text(json.dumps(payload), encoding="utf-8")

    stale_loader = KnowledgeGraph(EmptyVectorStore(), MockIndexer(anima_dir))
    assert stale_loader.load_graph(
        cache_dir,
        expected_schema_version=GRAPH_SCHEMA_VERSION,
        entity_aware_graph_enabled=True,
    ) is False


def test_entity_activation_maps_to_readable_results_not_bare_entity(tmp_path: Path) -> None:
    anima_dir, active, _expired = _fixture_anima(tmp_path)
    builder, _graph = _build_entity_graph(anima_dir)

    entity_results = builder._results_for_activated_node("entity:alice", 0.5, max_results=5)

    assert entity_results
    assert all(result.metadata.get("memory_type") != "entities" for result in entity_results)
    assert any(result.metadata.get("memory_type") in {"facts", "episodes", "knowledge"} for result in entity_results)

    expanded = builder.expand_search_results(
        [
            RetrievalResult(
                doc_id=f"test_anima/facts/{active.fact_id}#0",
                content=active.text,
                score=0.9,
                metadata={"fact_id": active.fact_id, "memory_type": "facts"},
                source_scores={"vector": 0.9},
            )
        ],
        max_hops=2,
    )
    assert any(result.metadata.get("memory_type") == "episodes" for result in expanded)
