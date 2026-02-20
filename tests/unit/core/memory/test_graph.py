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

    def query(self, collection, embedding, top_k):
        """Mock query method."""
        from core.memory.rag.store import Document, SearchResult

        # Return mock results
        return [
            SearchResult(
                document=Document(
                    id=f"anima/knowledge/related{i}.md#0",
                    content=f"Related content {i}",
                    embedding=embedding,
                    metadata={"source_file": f"knowledge/related{i}.md"},
                ),
                score=0.8 - i * 0.1,
            )
            for i in range(min(3, top_k))
        ]


class MockIndexer:
    """Mock indexer for testing."""

    def __init__(self):
        self.anima_name = "test_anima"

    def _generate_embeddings(self, texts):
        """Mock embedding generation."""
        # Return dummy embeddings
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
    graph_builder.graph.add_node("file4", path=str(new_file))

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
