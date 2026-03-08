from __future__ import annotations
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for knowledge graph and spreading activation."""

import tempfile
from pathlib import Path

import pytest

pytest.importorskip("scipy")

from core.memory.rag.graph import KnowledgeGraph


class MockVectorStore:
    """Mock vector store for testing."""

    def query(self, collection, embedding, top_k, filter_metadata=None):
        """Mock query method."""
        from core.memory.rag.store import Document, SearchResult

        # Determine memory type from collection name
        if "_episodes" in collection:
            prefix = "episodes"
        else:
            prefix = "knowledge"

        return [
            SearchResult(
                document=Document(
                    id=f"anima/{prefix}/related{i}.md#0",
                    content=f"Related content {i}",
                    embedding=embedding,
                    metadata={"source_file": f"{prefix}/related{i}.md"},
                ),
                score=0.8 - i * 0.1,
            )
            for i in range(min(3, top_k))
        ]


class MockVectorStoreMatchingNodes:
    """Mock vector store that returns doc_ids matching actual node stems."""

    def __init__(self, stems: list[str] | None = None):
        self._stems = stems or ["file1", "file2", "file3"]

    def query(self, collection, embedding, top_k, filter_metadata=None):
        from core.memory.rag.store import Document, SearchResult

        if "_episodes" in collection:
            prefix = "episodes"
        else:
            prefix = "knowledge"

        results = []
        for i, stem in enumerate(self._stems[:top_k]):
            results.append(
                SearchResult(
                    document=Document(
                        id=f"anima/{prefix}/{stem}.md#0",
                        content=f"Content of {stem}",
                        embedding=embedding,
                        metadata={"source_file": f"{prefix}/{stem}.md"},
                    ),
                    score=0.85 - i * 0.02,
                )
            )
        return results


class MockIndexer:
    """Mock indexer for testing."""

    def __init__(self, anima_dir: Path | None = None):
        self.anima_name = "test_anima"
        self.anima_dir = anima_dir or Path("/tmp/nonexistent")

    def _generate_embeddings(self, texts):
        """Mock embedding generation."""
        return [[0.1] * 384 for _ in texts]


@pytest.fixture
def temp_knowledge_dir():
    """Create temporary knowledge directory with sample files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        knowledge_dir = Path(tmpdir) / "knowledge"
        knowledge_dir.mkdir()

        # Create sample knowledge files
        (knowledge_dir / "file1.md").write_text(
            "# File 1\n\nThis links to [[file2]].\n\nSome content."
        )
        (knowledge_dir / "file2.md").write_text(
            "# File 2\n\nThis links to [[file3]] and [[file1]].\n"
        )
        (knowledge_dir / "file3.md").write_text(
            "# File 3\n\nNo explicit links here.\n"
        )

        yield knowledge_dir


@pytest.fixture
def temp_anima_dir():
    """Create temp dir with both knowledge and episodes."""
    with tempfile.TemporaryDirectory() as tmpdir:
        base = Path(tmpdir)
        knowledge_dir = base / "knowledge"
        knowledge_dir.mkdir()
        episodes_dir = base / "episodes"
        episodes_dir.mkdir()

        (knowledge_dir / "file1.md").write_text(
            "# File 1\n\nThis links to [[file2]] and [[2026-03-01]].\n"
        )
        (knowledge_dir / "file2.md").write_text(
            "# File 2\n\nRelated knowledge.\n"
        )

        (episodes_dir / "2026-03-01.md").write_text(
            "# 2026-03-01\n\n## 09:30 Meeting\n\nDiscussed file1 topic.\n"
        )
        (episodes_dir / "2026-03-02.md").write_text(
            "# 2026-03-02\n\n## 14:00 Review\n\nReviewed file2.\n"
        )

        yield base


def test_graph_construction(temp_knowledge_dir):
    """Test knowledge graph construction from markdown files."""
    vector_store = MockVectorStore()
    indexer = MockIndexer()

    graph_builder = KnowledgeGraph(vector_store, indexer)
    graph = graph_builder.build_graph("test_anima", temp_knowledge_dir)

    # Check nodes
    assert graph.number_of_nodes() == 3
    assert "file1" in graph
    assert "file2" in graph
    assert "file3" in graph

    # Check explicit edges
    assert graph.has_edge("file1", "file2")
    assert graph.has_edge("file2", "file3")
    assert graph.has_edge("file2", "file1")

    # Check edge attributes
    assert graph["file1"]["file2"]["link_type"] == "explicit"
    assert graph["file1"]["file2"]["similarity"] == 1.0

    # Verify explicit edges exist (implicit links depend on mock doc_ids matching nodes)


def test_personalized_pagerank(temp_knowledge_dir):
    """Test Personalized PageRank computation."""
    vector_store = MockVectorStore()
    indexer = MockIndexer()

    graph_builder = KnowledgeGraph(vector_store, indexer)
    graph_builder.build_graph("test_anima", temp_knowledge_dir)

    # Compute PageRank from file1
    scores = graph_builder.personalized_pagerank(["file1"])

    # Check that scores are returned
    assert len(scores) > 0
    assert "file1" in scores
    assert "file2" in scores
    assert "file3" in scores

    # file1 should have highest score (starting point)
    assert scores["file1"] > scores["file3"]


def test_spreading_activation(temp_knowledge_dir):
    """Test spreading activation expansion."""
    from core.memory.rag.retriever import RetrievalResult

    vector_store = MockVectorStore()
    indexer = MockIndexer()

    graph_builder = KnowledgeGraph(vector_store, indexer)
    graph_builder.build_graph("test_anima", temp_knowledge_dir)

    # Create initial results
    initial_results = [
        RetrievalResult(
            doc_id="test_anima/knowledge/file1.md#0",
            content="File 1 content",
            score=0.9,
            metadata={"source_file": "knowledge/file1.md"},
            source_scores={"vector": 0.9},
        )
    ]

    # Expand with spreading activation
    expanded = graph_builder.expand_search_results(initial_results, max_hops=2)

    # Check that results were expanded
    assert len(expanded) > len(initial_results)

    # Original result should still be present
    assert expanded[0].doc_id == initial_results[0].doc_id


def test_empty_graph():
    """Test handling of empty knowledge directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        empty_dir = Path(tmpdir) / "empty"
        empty_dir.mkdir()

        vector_store = MockVectorStore()
        indexer = MockIndexer()

        graph_builder = KnowledgeGraph(vector_store, indexer)
        graph = graph_builder.build_graph("test_anima", empty_dir)

        assert graph.number_of_nodes() == 0
        assert graph.number_of_edges() == 0

        # PageRank on empty graph should return empty dict
        scores = graph_builder.personalized_pagerank(["nonexistent"])
        assert scores == {}


# ── Phase 2 tests ──────────────────────────────────────────────────


def test_expand_results_real_content(temp_knowledge_dir):
    """Test that spreading activation returns real file content instead of placeholder."""
    from core.memory.rag.retriever import RetrievalResult

    vector_store = MockVectorStore()
    indexer = MockIndexer()

    graph_builder = KnowledgeGraph(vector_store, indexer)
    graph_builder.build_graph("test_anima", temp_knowledge_dir)

    # Create initial results referencing file1
    initial_results = [
        RetrievalResult(
            doc_id="test_anima/knowledge/file1.md#0",
            content="File 1 content",
            score=0.9,
            metadata={"source_file": "knowledge/file1.md"},
            source_scores={"vector": 0.9},
        )
    ]

    expanded = graph_builder.expand_search_results(initial_results, max_hops=2)

    # Activated nodes should contain real file content, not placeholder
    for result in expanded[1:]:  # Skip first (original)
        assert not result.content.startswith("[Activated node:")
        # Should contain actual markdown content
        assert "#" in result.content or "content" in result.content.lower() or "link" in result.content.lower()


def test_graph_save_and_load(temp_knowledge_dir):
    """Test graph JSON persistence (save and load)."""
    vector_store = MockVectorStore()
    indexer = MockIndexer()

    # Build and save
    graph_builder = KnowledgeGraph(vector_store, indexer)
    graph_builder.build_graph("test_anima", temp_knowledge_dir)

    with tempfile.TemporaryDirectory() as cache_dir:
        cache_path = Path(cache_dir)
        graph_builder.save_graph(cache_path)

        # Verify cache file exists
        assert (cache_path / "knowledge_graph.json").exists()

        # Load into new instance
        graph_loader = KnowledgeGraph(vector_store, indexer)
        loaded = graph_loader.load_graph(cache_path)

        assert loaded is True
        assert graph_loader.graph is not None
        assert graph_loader.graph.number_of_nodes() == graph_builder.graph.number_of_nodes()
        assert graph_loader.graph.number_of_edges() == graph_builder.graph.number_of_edges()

        # Check node content preserved
        assert "file1" in graph_loader.graph
        assert "file2" in graph_loader.graph
        assert "file3" in graph_loader.graph

        # Check edge attributes preserved
        assert graph_loader.graph.has_edge("file1", "file2")
        edge_data = graph_loader.graph["file1"]["file2"]
        assert edge_data["link_type"] == "explicit"
        assert edge_data["similarity"] == 1.0


def test_graph_load_missing_cache():
    """Test that load_graph returns False when no cache exists."""
    vector_store = MockVectorStore()
    indexer = MockIndexer()

    graph_loader = KnowledgeGraph(vector_store, indexer)

    with tempfile.TemporaryDirectory() as cache_dir:
        loaded = graph_loader.load_graph(Path(cache_dir))
        assert loaded is False
        assert graph_loader.graph is None


def test_graph_incremental_update(temp_knowledge_dir):
    """Test incremental graph update when files change."""
    vector_store = MockVectorStore()
    indexer = MockIndexer()

    # Build initial graph
    graph_builder = KnowledgeGraph(vector_store, indexer)
    graph_builder.build_graph("test_anima", temp_knowledge_dir)

    initial_nodes = graph_builder.graph.number_of_nodes()
    assert initial_nodes == 3

    # Add a new file that links to file1
    new_file = temp_knowledge_dir / "file4.md"
    new_file.write_text("# File 4\n\nThis links to [[file1]].\n")

    # Also need to add it as a node first (simulate watcher detecting new file)
    graph_builder.graph.add_node("file4", path=str(new_file), memory_type="knowledge", stem="file4")

    # Incremental update
    graph_builder.update_graph_incremental([new_file], "test_anima")

    # Should now have 4 nodes
    assert graph_builder.graph.number_of_nodes() == 4
    assert "file4" in graph_builder.graph

    # Should have explicit link from file4 -> file1
    assert graph_builder.graph.has_edge("file4", "file1")


def test_graph_incremental_update_file_deletion(temp_knowledge_dir):
    """Test incremental graph update when a file is deleted."""
    vector_store = MockVectorStore()
    indexer = MockIndexer()

    # Build initial graph
    graph_builder = KnowledgeGraph(vector_store, indexer)
    graph_builder.build_graph("test_anima", temp_knowledge_dir)

    assert "file3" in graph_builder.graph

    # Delete file3
    deleted_file = temp_knowledge_dir / "file3.md"
    deleted_file.unlink()

    # Incremental update for deleted file
    graph_builder.update_graph_incremental([deleted_file], "test_anima")

    # file3 should be removed from graph
    assert "file3" not in graph_builder.graph

    # Remaining nodes should still be there
    assert "file1" in graph_builder.graph
    assert "file2" in graph_builder.graph


def test_pagerank_with_edge_weights(temp_knowledge_dir):
    """Test that PageRank uses edge similarity weights."""
    vector_store = MockVectorStore()
    indexer = MockIndexer()

    graph_builder = KnowledgeGraph(vector_store, indexer)
    graph_builder.build_graph("test_anima", temp_knowledge_dir)

    # Verify explicit edges have similarity=1.0
    for u, v, data in graph_builder.graph.edges(data=True):
        if data["link_type"] == "explicit":
            assert data["similarity"] == 1.0
        elif data["link_type"] == "implicit":
            assert 0.0 < data["similarity"] <= 1.0

    # Compute PageRank - should use weights
    scores = graph_builder.personalized_pagerank(["file1"])

    # All nodes should have scores
    assert len(scores) == graph_builder.graph.number_of_nodes()

    # Scores should sum to ~1.0
    total = sum(scores.values())
    assert 0.99 < total < 1.01


def test_graph_invalidation_on_file_change(temp_knowledge_dir):
    """Test that graph is properly rebuilt when file content changes."""
    vector_store = MockVectorStore()
    indexer = MockIndexer()

    # Build initial graph
    graph_builder = KnowledgeGraph(vector_store, indexer)
    graph_builder.build_graph("test_anima", temp_knowledge_dir)

    # file1 initially links to file2
    assert graph_builder.graph.has_edge("file1", "file2")

    # Modify file1 to link to file3 instead
    file1 = temp_knowledge_dir / "file1.md"
    file1.write_text("# File 1 Updated\n\nNow links to [[file3]].\n")

    # Incremental update
    graph_builder.update_graph_incremental([file1], "test_anima")

    # file1 -> file2 explicit link should be gone (only old link)
    # file1 -> file3 should now exist
    assert graph_builder.graph.has_edge("file1", "file3")

    # file2 -> file1 explicit link should be re-established (from file2's content)
    assert graph_builder.graph.has_edge("file2", "file1")


# ── Episodes support tests ─────────────────────────────────────────


def test_build_graph_with_episodes(temp_anima_dir):
    """Test graph construction from knowledge + episodes."""
    vector_store = MockVectorStore()
    indexer = MockIndexer()

    knowledge_dir = temp_anima_dir / "knowledge"
    episodes_dir = temp_anima_dir / "episodes"

    graph_builder = KnowledgeGraph(vector_store, indexer)
    graph = graph_builder.build_graph(
        "test_anima", knowledge_dir,
        memory_dirs={"episodes": episodes_dir},
    )

    # Knowledge nodes: bare stem
    assert "file1" in graph
    assert "file2" in graph

    # Episode nodes: prefixed with memory type
    assert "episodes:2026-03-01" in graph
    assert "episodes:2026-03-02" in graph

    # Total: 2 knowledge + 2 episodes = 4
    assert graph.number_of_nodes() == 4


def test_cross_type_explicit_links(temp_anima_dir):
    """Test explicit [[links]] from knowledge to episodes."""
    vector_store = MockVectorStore()
    indexer = MockIndexer()

    knowledge_dir = temp_anima_dir / "knowledge"
    episodes_dir = temp_anima_dir / "episodes"

    graph_builder = KnowledgeGraph(vector_store, indexer)
    graph_builder.build_graph(
        "test_anima", knowledge_dir,
        memory_dirs={"episodes": episodes_dir},
    )

    # file1 links to [[file2]] and [[2026-03-01]]
    assert graph_builder.graph.has_edge("file1", "file2")
    assert graph_builder.graph.has_edge("file1", "episodes:2026-03-01")


def test_node_memory_type_attribute(temp_anima_dir):
    """Test that nodes store correct memory_type attribute."""
    vector_store = MockVectorStore()
    indexer = MockIndexer()

    knowledge_dir = temp_anima_dir / "knowledge"
    episodes_dir = temp_anima_dir / "episodes"

    graph_builder = KnowledgeGraph(vector_store, indexer)
    graph_builder.build_graph(
        "test_anima", knowledge_dir,
        memory_dirs={"episodes": episodes_dir},
    )

    assert graph_builder.graph.nodes["file1"]["memory_type"] == "knowledge"
    assert graph_builder.graph.nodes["file2"]["memory_type"] == "knowledge"
    assert graph_builder.graph.nodes["episodes:2026-03-01"]["memory_type"] == "episodes"
    assert graph_builder.graph.nodes["episodes:2026-03-02"]["memory_type"] == "episodes"


def test_spreading_activation_with_episodes(temp_anima_dir):
    """Test spreading activation expands across knowledge and episodes."""
    from core.memory.rag.retriever import RetrievalResult

    vector_store = MockVectorStore()
    indexer = MockIndexer()

    knowledge_dir = temp_anima_dir / "knowledge"
    episodes_dir = temp_anima_dir / "episodes"

    graph_builder = KnowledgeGraph(vector_store, indexer)
    graph_builder.build_graph(
        "test_anima", knowledge_dir,
        memory_dirs={"episodes": episodes_dir},
    )

    initial_results = [
        RetrievalResult(
            doc_id="test_anima/knowledge/file1.md#0",
            content="File 1 content",
            score=0.9,
            metadata={"source_file": "knowledge/file1.md"},
            source_scores={"vector": 0.9},
        )
    ]

    expanded = graph_builder.expand_search_results(initial_results, max_hops=2)

    # Should have more results than initial
    assert len(expanded) > len(initial_results)

    # Check that expanded results include correct memory_type in metadata
    for result in expanded[1:]:
        assert "memory_type" in result.metadata
        mt = result.metadata["memory_type"]
        assert mt in ("knowledge", "episodes")
        # doc_id should use the correct memory_type
        assert f"/{mt}/" in result.doc_id


def test_episode_doc_id_in_expansion(temp_anima_dir):
    """Test that episode-sourced activated nodes produce correct doc_ids."""
    from core.memory.rag.retriever import RetrievalResult

    vector_store = MockVectorStore()
    indexer = MockIndexer()

    knowledge_dir = temp_anima_dir / "knowledge"
    episodes_dir = temp_anima_dir / "episodes"

    graph_builder = KnowledgeGraph(vector_store, indexer)
    graph_builder.build_graph(
        "test_anima", knowledge_dir,
        memory_dirs={"episodes": episodes_dir},
    )

    initial_results = [
        RetrievalResult(
            doc_id="test_anima/episodes/2026-03-01.md#0",
            content="Episode content",
            score=0.9,
            metadata={"source_file": "episodes/2026-03-01.md"},
            source_scores={"vector": 0.9},
        )
    ]

    expanded = graph_builder.expand_search_results(initial_results, max_hops=2)

    for result in expanded[1:]:
        mt = result.metadata.get("memory_type", "")
        if mt == "episodes":
            assert result.doc_id.startswith("test_anima/episodes/")
        elif mt == "knowledge":
            assert result.doc_id.startswith("test_anima/knowledge/")


def test_pagerank_across_types(temp_anima_dir):
    """Test PageRank scores propagate across knowledge and episodes."""
    vector_store = MockVectorStore()
    indexer = MockIndexer()

    knowledge_dir = temp_anima_dir / "knowledge"
    episodes_dir = temp_anima_dir / "episodes"

    graph_builder = KnowledgeGraph(vector_store, indexer)
    graph_builder.build_graph(
        "test_anima", knowledge_dir,
        memory_dirs={"episodes": episodes_dir},
    )

    scores = graph_builder.personalized_pagerank(["file1"])

    # All nodes should have scores
    assert "file1" in scores
    assert "file2" in scores
    assert "episodes:2026-03-01" in scores
    assert "episodes:2026-03-02" in scores

    # file1 (seed) should have highest score
    assert scores["file1"] > scores["episodes:2026-03-02"]


def test_incremental_update_episodes(temp_anima_dir):
    """Test incremental graph update for episode files."""
    vector_store = MockVectorStore()
    indexer = MockIndexer()

    knowledge_dir = temp_anima_dir / "knowledge"
    episodes_dir = temp_anima_dir / "episodes"

    graph_builder = KnowledgeGraph(vector_store, indexer)
    graph_builder.build_graph(
        "test_anima", knowledge_dir,
        memory_dirs={"episodes": episodes_dir},
    )

    initial_nodes = graph_builder.graph.number_of_nodes()

    # Add new episode
    new_episode = episodes_dir / "2026-03-03.md"
    new_episode.write_text("# 2026-03-03\n\n## 10:00 Stand-up\n\nLinks to [[file1]].\n")

    # Pre-add node (simulate watcher)
    graph_builder.graph.add_node(
        "episodes:2026-03-03",
        path=str(new_episode),
        memory_type="episodes",
        stem="2026-03-03",
    )

    graph_builder.update_graph_incremental(
        [new_episode], "test_anima", memory_type="episodes",
    )

    assert graph_builder.graph.number_of_nodes() == initial_nodes + 1
    assert "episodes:2026-03-03" in graph_builder.graph

    # Cross-type link: episode -> knowledge
    assert graph_builder.graph.has_edge("episodes:2026-03-03", "file1")


def test_make_node_id():
    """Test node ID generation for different memory types."""
    assert KnowledgeGraph._make_node_id("file1", "knowledge") == "file1"
    assert KnowledgeGraph._make_node_id("2026-03-01", "episodes") == "episodes:2026-03-01"
    assert KnowledgeGraph._make_node_id("proc1", "procedures") == "procedures:proc1"


def test_knowledge_only_backward_compat(temp_knowledge_dir):
    """Test that calling build_graph without memory_dirs still works."""
    vector_store = MockVectorStore()
    indexer = MockIndexer()

    graph_builder = KnowledgeGraph(vector_store, indexer)
    graph = graph_builder.build_graph("test_anima", temp_knowledge_dir)

    # Should work exactly as before
    assert graph.number_of_nodes() == 3
    assert "file1" in graph
    assert "file2" in graph
    assert "file3" in graph
    assert graph.nodes["file1"]["memory_type"] == "knowledge"


def test_implicit_links_created(temp_knowledge_dir):
    """Test that implicit links are created when vector results match nodes."""
    vector_store = MockVectorStoreMatchingNodes(["file1", "file2", "file3"])
    indexer = MockIndexer()

    graph_builder = KnowledgeGraph(vector_store, indexer)
    graph_builder.build_graph("test_anima", temp_knowledge_dir)

    implicit_edges = [
        (u, v, d)
        for u, v, d in graph_builder.graph.edges(data=True)
        if d["link_type"] == "implicit"
    ]
    assert len(implicit_edges) > 0, (
        "Expected at least one implicit link from vector similarity"
    )


def test_implicit_links_in_multi_type_graph(temp_anima_dir):
    """Test that implicit links are created across memory types."""
    knowledge_dir = temp_anima_dir / "knowledge"
    episodes_dir = temp_anima_dir / "episodes"

    vector_store = MockVectorStoreMatchingNodes(["file1", "file2", "2026-03-01", "2026-03-02"])
    indexer = MockIndexer()

    graph_builder = KnowledgeGraph(vector_store, indexer)
    graph_builder.build_graph(
        "test_anima", knowledge_dir,
        memory_dirs={"episodes": episodes_dir},
    )

    implicit_edges = [
        (u, v, d)
        for u, v, d in graph_builder.graph.edges(data=True)
        if d["link_type"] == "implicit"
    ]
    assert len(implicit_edges) > 0, (
        "Expected implicit links from vector similarity in multi-type graph"
    )


def test_build_graph_with_custom_threshold(temp_knowledge_dir):
    """Test that implicit_link_threshold parameter controls link creation."""
    vector_store = MockVectorStore()
    indexer = MockIndexer()

    # Very high threshold — should create fewer (or zero) implicit links
    graph_builder_strict = KnowledgeGraph(vector_store, indexer)
    graph_strict = graph_builder_strict.build_graph(
        "test_anima", temp_knowledge_dir,
        implicit_link_threshold=0.99,
    )
    implicit_strict = sum(
        1 for _, _, d in graph_strict.edges(data=True) if d["link_type"] == "implicit"
    )

    # Low threshold — should create more implicit links
    graph_builder_lax = KnowledgeGraph(vector_store, indexer)
    graph_lax = graph_builder_lax.build_graph(
        "test_anima", temp_knowledge_dir,
        implicit_link_threshold=0.50,
    )
    implicit_lax = sum(
        1 for _, _, d in graph_lax.edges(data=True) if d["link_type"] == "implicit"
    )

    assert implicit_lax >= implicit_strict


# ── Retriever C方式 tests ──────────────────────────────────────────


class _MockConfig:
    """Minimal mock for AnimaWorksConfig."""

    class _RAG:
        enable_spreading_activation = True
        spreading_memory_types = ["knowledge", "episodes"]
        implicit_link_threshold = 0.75
        max_graph_hops = 2

    rag = _RAG()


class _MockConfigDisabled:
    """Config with spreading activation disabled."""

    class _RAG:
        enable_spreading_activation = False
        spreading_memory_types = ["knowledge", "episodes"]
        implicit_link_threshold = 0.75
        max_graph_hops = 2

    rag = _RAG()


def test_retriever_none_reads_config_enabled(monkeypatch, temp_knowledge_dir):
    """C方式: enable_spreading_activation=None reads from config (enabled)."""
    from core.memory.rag.retriever import MemoryRetriever

    monkeypatch.setattr(
        "core.memory.rag.retriever.MemoryRetriever._load_config",
        staticmethod(lambda: _MockConfig()),
    )

    retriever = MemoryRetriever(MockVectorStore(), MockIndexer(), temp_knowledge_dir)

    results = retriever.search(
        "test query", "test_anima", memory_type="knowledge", top_k=3,
        enable_spreading_activation=None,
    )
    assert isinstance(results, list)


def test_retriever_none_reads_config_disabled(monkeypatch, temp_knowledge_dir):
    """C方式: enable_spreading_activation=None reads from config (disabled)."""
    from core.memory.rag.retriever import MemoryRetriever

    monkeypatch.setattr(
        "core.memory.rag.retriever.MemoryRetriever._load_config",
        staticmethod(lambda: _MockConfigDisabled()),
    )

    retriever = MemoryRetriever(MockVectorStore(), MockIndexer(), temp_knowledge_dir)

    results = retriever.search(
        "test query", "test_anima", memory_type="knowledge", top_k=3,
        enable_spreading_activation=None,
    )
    assert isinstance(results, list)
    # With spreading disabled, no spreading expansion metadata should be present
    for r in results:
        assert r.metadata.get("activation") != "spreading"


def test_retriever_explicit_true_overrides_config(monkeypatch, temp_knowledge_dir):
    """C方式: enable_spreading_activation=True overrides config."""
    from core.memory.rag.retriever import MemoryRetriever

    monkeypatch.setattr(
        "core.memory.rag.retriever.MemoryRetriever._load_config",
        staticmethod(lambda: _MockConfigDisabled()),
    )

    retriever = MemoryRetriever(MockVectorStore(), MockIndexer(), temp_knowledge_dir)

    results = retriever.search(
        "test query", "test_anima", memory_type="knowledge", top_k=3,
        enable_spreading_activation=True,
    )
    assert isinstance(results, list)


def test_retriever_explicit_false_overrides_config(monkeypatch, temp_knowledge_dir):
    """C方式: enable_spreading_activation=False overrides config."""
    from core.memory.rag.retriever import MemoryRetriever

    monkeypatch.setattr(
        "core.memory.rag.retriever.MemoryRetriever._load_config",
        staticmethod(lambda: _MockConfig()),
    )

    retriever = MemoryRetriever(MockVectorStore(), MockIndexer(), temp_knowledge_dir)

    results = retriever.search(
        "test query", "test_anima", memory_type="knowledge", top_k=3,
        enable_spreading_activation=False,
    )
    assert isinstance(results, list)
    for r in results:
        assert r.metadata.get("activation") != "spreading"


# ── _fetch_node_content fallback tests ─────────────────────────────


class MockVectorStoreWithSourceFilter:
    """Mock vector store that respects filter_metadata for source_file."""

    def __init__(self, source_contents: dict[str, str] | None = None):
        self._source_contents = source_contents or {}

    def query(self, collection, embedding, top_k, filter_metadata=None):
        from core.memory.rag.store import Document, SearchResult

        if filter_metadata and "source_file" in filter_metadata:
            sf = filter_metadata["source_file"]
            if sf in self._source_contents:
                return [
                    SearchResult(
                        document=Document(
                            id=f"anima/{sf}#0",
                            content=self._source_contents[sf],
                            embedding=embedding,
                            metadata={"source_file": sf},
                        ),
                        score=0.95,
                    )
                ]
            return []

        return [
            SearchResult(
                document=Document(
                    id=f"anima/knowledge/fallback.md#0",
                    content="Fallback content from vector search",
                    embedding=embedding,
                    metadata={"source_file": "knowledge/fallback.md"},
                ),
                score=0.7,
            )
        ]


def test_fetch_node_content_file_exists(temp_knowledge_dir):
    """_fetch_node_content returns file content when file exists."""
    vector_store = MockVectorStore()
    indexer = MockIndexer()

    graph_builder = KnowledgeGraph(vector_store, indexer)
    file_path = temp_knowledge_dir / "file1.md"

    content = graph_builder._fetch_node_content("file1", file_path, "knowledge")
    assert "File 1" in content
    assert "links to [[file2]]" in content


def test_fetch_node_content_missing_file_uses_source_filter():
    """_fetch_node_content uses source_file filter when file is missing."""
    source_contents = {"knowledge/missing.md": "Content from ChromaDB for missing file"}
    vector_store = MockVectorStoreWithSourceFilter(source_contents)
    indexer = MockIndexer()

    graph_builder = KnowledgeGraph(vector_store, indexer)
    missing_path = Path("/nonexistent/knowledge/missing.md")

    content = graph_builder._fetch_node_content("missing", missing_path, "knowledge")
    assert content == "Content from ChromaDB for missing file"


def test_fetch_node_content_missing_file_stem_fallback():
    """_fetch_node_content falls back to stem-based search when source_file filter returns nothing."""
    vector_store = MockVectorStoreWithSourceFilter({})
    indexer = MockIndexer()

    graph_builder = KnowledgeGraph(vector_store, indexer)
    missing_path = Path("/nonexistent/episodes/2026-03-05.md")

    content = graph_builder._fetch_node_content(
        "episodes:2026-03-05", missing_path, "episodes",
    )
    assert content == "Fallback content from vector search"


def test_fetch_node_content_no_node_id_in_embedding():
    """_fetch_node_content must NOT use node_id as embedding input."""
    calls: list[list[str]] = []
    vector_store = MockVectorStoreWithSourceFilter(
        {"episodes/2026-03-05.md": "Episode content"},
    )

    class TrackingIndexer:
        anima_name = "test_anima"

        def _generate_embeddings(self, texts):
            calls.append(texts)
            return [[0.1] * 384 for _ in texts]

    indexer = TrackingIndexer()
    graph_builder = KnowledgeGraph(vector_store, indexer)
    missing_path = Path("/nonexistent/episodes/2026-03-05.md")

    graph_builder._fetch_node_content("episodes:2026-03-05", missing_path, "episodes")

    for call_texts in calls:
        for text in call_texts:
            assert "episodes:2026-03-05" not in text, (
                f"node_id should not be used as embedding input, got: {text}"
            )


def test_fetch_node_content_prefixed_node_id_with_empty_path():
    """_fetch_node_content handles prefixed node_id (episodes:stem) when path is empty."""
    source_contents = {"episodes/2026-03-05.md": "Episode content via prefix"}
    vector_store = MockVectorStoreWithSourceFilter(source_contents)
    indexer = MockIndexer()

    graph_builder = KnowledgeGraph(vector_store, indexer)
    # Simulate a node whose path attribute was lost
    empty_path = Path("")

    content = graph_builder._fetch_node_content(
        "episodes:2026-03-05", empty_path, "episodes",
    )
    assert content == "Episode content via prefix"


def test_fetch_node_content_unavailable_when_all_fail():
    """_fetch_node_content returns [Content unavailable] when all methods fail."""
    class FailingVectorStore:
        def query(self, **kwargs):
            raise RuntimeError("store unavailable")

    indexer = MockIndexer()
    graph_builder = KnowledgeGraph(FailingVectorStore(), indexer)
    missing_path = Path("/nonexistent/knowledge/gone.md")

    content = graph_builder._fetch_node_content("gone", missing_path, "knowledge")
    assert content.startswith("[Content unavailable:")


def test_expand_results_no_content_unavailable(temp_anima_dir):
    """expand_search_results should not produce [Content unavailable] when data exists."""
    from core.memory.rag.retriever import RetrievalResult

    vector_store = MockVectorStore()
    indexer = MockIndexer()

    knowledge_dir = temp_anima_dir / "knowledge"
    episodes_dir = temp_anima_dir / "episodes"

    graph_builder = KnowledgeGraph(vector_store, indexer)
    graph_builder.build_graph(
        "test_anima", knowledge_dir,
        memory_dirs={"episodes": episodes_dir},
    )

    initial_results = [
        RetrievalResult(
            doc_id="test_anima/knowledge/file1.md#0",
            content="File 1 content",
            score=0.9,
            metadata={"source_file": "knowledge/file1.md"},
            source_scores={"vector": 0.9},
        )
    ]

    expanded = graph_builder.expand_search_results(initial_results, max_hops=2)

    for result in expanded:
        assert "[Content unavailable" not in result.content


# ── _infer_memory_type tests ───────────────────────────────────────


def test_infer_memory_type_knowledge():
    """_infer_memory_type correctly identifies knowledge files."""
    anima_dir = Path("/home/user/.animaworks/animas/alice")
    file_path = anima_dir / "knowledge" / "topic.md"
    assert KnowledgeGraph._infer_memory_type(file_path, anima_dir) == "knowledge"


def test_infer_memory_type_episodes():
    """_infer_memory_type correctly identifies episodes files."""
    anima_dir = Path("/home/user/.animaworks/animas/alice")
    file_path = anima_dir / "episodes" / "2026-03-05.md"
    assert KnowledgeGraph._infer_memory_type(file_path, anima_dir) == "episodes"


def test_infer_memory_type_procedures():
    """_infer_memory_type correctly identifies procedures files."""
    anima_dir = Path("/home/user/.animaworks/animas/alice")
    file_path = anima_dir / "procedures" / "deploy.md"
    assert KnowledgeGraph._infer_memory_type(file_path, anima_dir) == "procedures"


def test_infer_memory_type_outside_dir():
    """_infer_memory_type defaults to knowledge for paths outside anima_dir."""
    anima_dir = Path("/home/user/.animaworks/animas/alice")
    file_path = Path("/tmp/random/file.md")
    assert KnowledgeGraph._infer_memory_type(file_path, anima_dir) == "knowledge"


# ── update_graph_incremental auto-detect tests ─────────────────────


def test_incremental_update_auto_detect_from_anima_dir(temp_anima_dir):
    """update_graph_incremental infers memory_type from anima_dir when memory_type=None."""
    vector_store = MockVectorStore()
    indexer = MockIndexer()

    knowledge_dir = temp_anima_dir / "knowledge"
    episodes_dir = temp_anima_dir / "episodes"

    graph_builder = KnowledgeGraph(vector_store, indexer)
    graph_builder.build_graph(
        "test_anima", knowledge_dir,
        memory_dirs={"episodes": episodes_dir},
    )

    # Add new episode
    new_episode = episodes_dir / "2026-03-03.md"
    new_episode.write_text("# 2026-03-03\n\nNew episode content.\n")

    graph_builder.update_graph_incremental(
        [new_episode], "test_anima",
        memory_type=None, anima_dir=temp_anima_dir,
    )

    assert "episodes:2026-03-03" in graph_builder.graph
    attrs = graph_builder.graph.nodes["episodes:2026-03-03"]
    assert attrs["memory_type"] == "episodes"


def test_incremental_update_auto_detect_from_graph_attrs(temp_anima_dir):
    """update_graph_incremental infers type from existing node attrs when anima_dir is None."""
    vector_store = MockVectorStore()
    indexer = MockIndexer()

    knowledge_dir = temp_anima_dir / "knowledge"
    episodes_dir = temp_anima_dir / "episodes"

    graph_builder = KnowledgeGraph(vector_store, indexer)
    graph_builder.build_graph(
        "test_anima", knowledge_dir,
        memory_dirs={"episodes": episodes_dir},
    )

    # Modify existing episode file
    episode_file = episodes_dir / "2026-03-01.md"
    episode_file.write_text("# 2026-03-01 Updated\n\nUpdated content.\n")

    graph_builder.update_graph_incremental(
        [episode_file], "test_anima",
        memory_type=None, anima_dir=None,
    )

    assert "episodes:2026-03-01" in graph_builder.graph
    attrs = graph_builder.graph.nodes["episodes:2026-03-01"]
    assert attrs["memory_type"] == "episodes"


def test_incremental_update_backward_compat(temp_knowledge_dir):
    """update_graph_incremental with explicit memory_type still works (backward compat)."""
    vector_store = MockVectorStore()
    indexer = MockIndexer()

    graph_builder = KnowledgeGraph(vector_store, indexer)
    graph_builder.build_graph("test_anima", temp_knowledge_dir)

    new_file = temp_knowledge_dir / "file4.md"
    new_file.write_text("# File 4\n\nNew knowledge.\n")

    graph_builder.update_graph_incremental(
        [new_file], "test_anima", memory_type="knowledge",
    )

    assert "file4" in graph_builder.graph
    assert graph_builder.graph.nodes["file4"]["memory_type"] == "knowledge"


def test_incremental_update_mixed_types(temp_anima_dir):
    """update_graph_incremental handles mixed-type files in a single call."""
    vector_store = MockVectorStore()
    indexer = MockIndexer()

    knowledge_dir = temp_anima_dir / "knowledge"
    episodes_dir = temp_anima_dir / "episodes"

    graph_builder = KnowledgeGraph(vector_store, indexer)
    graph_builder.build_graph(
        "test_anima", knowledge_dir,
        memory_dirs={"episodes": episodes_dir},
    )

    # Create one new file per type
    new_knowledge = knowledge_dir / "new_topic.md"
    new_knowledge.write_text("# New Topic\n\nNew knowledge content.\n")
    new_episode = episodes_dir / "2026-03-04.md"
    new_episode.write_text("# 2026-03-04\n\nNew episode.\n")

    graph_builder.update_graph_incremental(
        [new_knowledge, new_episode], "test_anima",
        memory_type=None, anima_dir=temp_anima_dir,
    )

    assert "new_topic" in graph_builder.graph
    assert graph_builder.graph.nodes["new_topic"]["memory_type"] == "knowledge"
    assert "episodes:2026-03-04" in graph_builder.graph
    assert graph_builder.graph.nodes["episodes:2026-03-04"]["memory_type"] == "episodes"
