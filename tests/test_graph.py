from __future__ import annotations
# AnimaWorks - Digital Person Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for knowledge graph and spreading activation."""

import tempfile
from pathlib import Path

import pytest

from core.memory.rag.graph import KnowledgeGraph


class MockVectorStore:
    """Mock vector store for testing."""

    def query(self, collection, embedding, top_k):
        """Mock query method."""
        from core.memory.rag.store import Document, QueryResult

        # Return mock results
        return [
            QueryResult(
                document=Document(
                    id=f"person/knowledge/related{i}.md#0",
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
        self.person_name = "test_person"

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
    graph = graph_builder.build_graph("test_person", temp_knowledge_dir)

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


def test_personalized_pagerank(temp_knowledge_dir):
    """Test Personalized PageRank computation."""
    vector_store = MockVectorStore()
    indexer = MockIndexer()

    graph_builder = KnowledgeGraph(vector_store, indexer)
    graph_builder.build_graph("test_person", temp_knowledge_dir)

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
    graph_builder.build_graph("test_person", temp_knowledge_dir)

    # Create initial results
    initial_results = [
        RetrievalResult(
            doc_id="test_person/knowledge/file1.md#0",
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
        graph = graph_builder.build_graph("test_person", empty_dir)

        assert graph.number_of_nodes() == 0
        assert graph.number_of_edges() == 0

        # PageRank on empty graph should return empty dict
        scores = graph_builder.personalized_pagerank(["nonexistent"])
        assert scores == {}
