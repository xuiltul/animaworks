from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import networkx as nx
import pytest

from core.memory.rag.graph import GRAPH_CACHE_FILE, GRAPH_SCHEMA_VERSION, KnowledgeGraph
from core.memory.rag.retriever import MemoryRetriever, RetrievalResult


@pytest.fixture
def retrieval(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    anima_dir = tmp_path / "animas" / "test_anima"
    knowledge = anima_dir / "knowledge"
    knowledge.mkdir(parents=True)
    for name in ("seed", "neighbor"):
        (knowledge / f"{name}.md").write_text(f"# {name}\n\nExisting knowledge content.")
    vector_store = MagicMock()
    indexer = MagicMock(anima_name="test_anima", anima_dir=anima_dir)
    retriever = MemoryRetriever(vector_store, indexer, knowledge)
    config = SimpleNamespace(
        rag=SimpleNamespace(
            graph_cache_enabled=True,
            entity_aware_graph_enabled=False,
            spreading_memory_types=("knowledge", "episodes"),
            max_graph_hops=2,
        )
    )
    monkeypatch.setattr(retriever, "_safe_load_config", lambda: config)
    monkeypatch.setattr(retriever, "_load_config", lambda: config)
    seeds = [
        RetrievalResult(
            doc_id="test_anima/knowledge/seed.md#0",
            content="seed evidence",
            score=0.9,
            metadata={"source_file": "knowledge/seed.md"},
            source_scores={"vector": 0.9},
        )
    ]
    return retriever, config, seeds


def save_cache(retriever: MemoryRetriever, *, schema=GRAPH_SCHEMA_VERSION, entity_aware=False):
    graph = KnowledgeGraph(retriever.vector_store, retriever.indexer)
    graph.graph = nx.DiGraph(schema_version=schema, entity_aware_graph_enabled=entity_aware)
    for name in ("seed", "neighbor"):
        graph.graph.add_node(
            name, path=str(retriever.knowledge_dir / f"{name}.md"), memory_type="knowledge", stem=name, rel_key=name
        )
    graph.graph.add_edge("seed", "neighbor", similarity=1.0, link_type="explicit")
    cache_dir = retriever.knowledge_dir.parent / "vectordb"
    cache_dir.mkdir(exist_ok=True)
    graph.save_graph(cache_dir)


@pytest.mark.parametrize("cache_state", ["absent", "corrupt", "old_schema", "wrong_entity_policy", "disabled"])
def test_unusable_graph_cache_keeps_seeds_without_building(retrieval, monkeypatch, cache_state):
    retriever, config, seeds = retrieval
    if cache_state == "corrupt":
        cache_dir = retriever.knowledge_dir.parent / "vectordb"
        cache_dir.mkdir()
        (cache_dir / GRAPH_CACHE_FILE).write_text("not json")
    elif cache_state == "old_schema":
        save_cache(retriever, schema=-1)
    elif cache_state == "wrong_entity_policy":
        save_cache(retriever, entity_aware=True)
    elif cache_state == "disabled":
        save_cache(retriever)
        config.rag.graph_cache_enabled = False
    build = MagicMock(side_effect=AssertionError("Full graph rebuild in request path"))
    save = MagicMock(side_effect=AssertionError("Graph cache write in request path"))
    monkeypatch.setattr(KnowledgeGraph, "build_graph", build)
    monkeypatch.setattr(KnowledgeGraph, "save_graph", save)
    before = sorted(str(p) for p in retriever.knowledge_dir.parent.rglob("*"))

    # A dual-query request and another concurrent request must all return;
    # none may leave a full-build worker running after the caller's timeout.
    with ThreadPoolExecutor(max_workers=3) as pool:
        pending = [pool.submit(retriever._apply_spreading_activation, seeds, "test_anima") for _ in range(3)]
        assert all(future.result(timeout=2) is seeds for future in pending)

    build.assert_not_called()
    save.assert_not_called()
    retriever.indexer._generate_embeddings.assert_not_called()
    retriever.vector_store.query.assert_not_called()
    assert retriever._knowledge_graph is None
    assert sorted(str(p) for p in retriever.knowledge_dir.parent.rglob("*")) == before


def test_valid_cached_graph_still_expands_existing_results(retrieval, monkeypatch):
    retriever, _, seeds = retrieval
    save_cache(retriever)
    build = MagicMock(side_effect=AssertionError("Unexpected rebuild"))
    monkeypatch.setattr(KnowledgeGraph, "build_graph", build)

    result = retriever._apply_spreading_activation(seeds, "test_anima")

    assert result[0] is seeds[0]
    assert any("neighbor" in row.doc_id for row in result)
    build.assert_not_called()
    retriever.indexer._generate_embeddings.assert_not_called()
    retriever.vector_store.query.assert_not_called()


def test_cache_published_after_miss_is_used_on_next_search(retrieval, monkeypatch):
    retriever, _, seeds = retrieval
    assert retriever._apply_spreading_activation(seeds, "test_anima") is seeds
    save_cache(retriever)
    build = MagicMock()
    monkeypatch.setattr(KnowledgeGraph, "build_graph", build)

    result = retriever._apply_spreading_activation(seeds, "test_anima")

    assert any("neighbor" in row.doc_id for row in result)
    build.assert_not_called()


def test_normal_vector_search_returns_evidence_on_cold_graph(retrieval, monkeypatch):
    retriever, _, seeds = retrieval
    seed = seeds[0]
    monkeypatch.setattr(
        retriever,
        "_vector_search",
        lambda *args, **kwargs: [
            (seed.doc_id, seed.content, seed.score, seed.metadata),
        ],
    )
    monkeypatch.setattr(retriever, "_apply_score_adjustments", lambda results, *args: results)
    build = MagicMock()
    monkeypatch.setattr(KnowledgeGraph, "build_graph", build)

    result = retriever.search("question", "test_anima", enable_spreading_activation=True, embedding=[0.1])

    assert len(result) == 1
    assert result[0].doc_id == seed.doc_id
    assert result[0].content == seed.content
    build.assert_not_called()
    retriever.indexer._generate_embeddings.assert_not_called()
