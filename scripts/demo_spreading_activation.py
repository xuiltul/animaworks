#!/usr/bin/env python3
from __future__ import annotations
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Demo script for spreading activation in knowledge graph.

Creates a sample knowledge graph and demonstrates:
1. Graph construction from markdown files
2. Personalized PageRank computation
3. Spreading activation for search result expansion

Usage:
    python scripts/demo_spreading_activation.py
"""

import tempfile
from pathlib import Path

import networkx as nx


def create_sample_knowledge_base(knowledge_dir: Path) -> None:
    """Create sample knowledge files with links.

    Args:
        knowledge_dir: Directory to create files in
    """
    print("Creating sample knowledge base...")

    # File 1: Python basics
    (knowledge_dir / "python-basics.md").write_text("""
# Python 基礎

Pythonは汎用プログラミング言語です。

## 特徴
- 読みやすい構文
- 豊富なライブラリ
- [[django]]や[[flask]]などのフレームワーク

## 関連トピック
- [[web-development]]
- [[data-science]]
""")

    # File 2: Django
    (knowledge_dir / "django.md").write_text("""
# Django

DjangoはPythonのWebフレームワークです。

[[python-basics]]の知識が必要。

## 特徴
- MVTアーキテクチャ
- ORM内蔵
- [[database]]との連携が容易

## 関連
- [[web-development]]
- [[rest-api]]
""")

    # File 3: Flask
    (knowledge_dir / "flask.md").write_text("""
# Flask

軽量なPython Webフレームワーク。

[[python-basics]]ベース。

## 特徴
- シンプル
- 拡張性が高い
- [[rest-api]]構築に適している

関連: [[web-development]]
""")

    # File 4: Web Development
    (knowledge_dir / "web-development.md").write_text("""
# Web開発

Webアプリケーション開発の総合トピック。

## フレームワーク
- [[django]]
- [[flask]]
- Express.js
- React

## 技術
- [[rest-api]]
- [[database]]
- フロントエンド
""")

    # File 5: REST API
    (knowledge_dir / "rest-api.md").write_text("""
# REST API

RESTful APIの設計パターン。

## 実装
- [[django]] REST framework
- [[flask]] RESTful

関連: [[web-development]]
""")

    # File 6: Database
    (knowledge_dir / "database.md").write_text("""
# データベース

データ永続化の基礎。

## RDBMS
- PostgreSQL
- MySQL

## ORM
- [[django]] ORM
- SQLAlchemy

関連: [[web-development]]
""")

    # File 7: Data Science (disconnected from main cluster)
    (knowledge_dir / "data-science.md").write_text("""
# データサイエンス

データ分析・機械学習の分野。

## ツール
- pandas
- numpy
- scikit-learn

Pythonが主要言語。[[python-basics]]の知識が必要。
""")

    print(f"Created {len(list(knowledge_dir.glob('*.md')))} knowledge files")


def demo_graph_construction():
    """Demonstrate graph construction."""
    print("\n" + "=" * 60)
    print("Demo 1: Knowledge Graph Construction")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        knowledge_dir = Path(tmpdir) / "knowledge"
        knowledge_dir.mkdir()

        # Create sample files
        create_sample_knowledge_base(knowledge_dir)

        # Build graph (using mock components)
        from core.memory.rag.graph import KnowledgeGraph

        # Mock vector store and indexer
        class MockVectorStore:
            def query(self, collection, embedding, top_k):
                from core.memory.rag.store import Document, QueryResult
                # Return empty for simplicity
                return []

        class MockIndexer:
            anima_name = "demo"
            def _generate_embeddings(self, texts):
                return [[0.1] * 384 for _ in texts]

        graph_builder = KnowledgeGraph(MockVectorStore(), MockIndexer())
        graph = graph_builder.build_graph("demo", knowledge_dir)

        print(f"\nGraph statistics:")
        print(f"  Nodes: {graph.number_of_nodes()}")
        print(f"  Edges: {graph.number_of_edges()}")

        print(f"\nNodes:")
        for node in sorted(graph.nodes()):
            print(f"  - {node}")

        print(f"\nExplicit links:")
        for source, target, data in sorted(graph.edges(data=True)):
            if data.get("link_type") == "explicit":
                print(f"  {source} -> {target}")

        return graph


def demo_pagerank(graph: nx.DiGraph):
    """Demonstrate Personalized PageRank.

    Args:
        graph: Knowledge graph
    """
    print("\n" + "=" * 60)
    print("Demo 2: Personalized PageRank")
    print("=" * 60)

    from core.memory.rag.graph import KnowledgeGraph

    # Mock components
    class MockVectorStore:
        def query(self, collection, embedding, top_k):
            return []

    class MockIndexer:
        anima_name = "demo"
        def _generate_embeddings(self, texts):
            return [[0.1] * 384 for _ in texts]

    graph_builder = KnowledgeGraph(MockVectorStore(), MockIndexer())
    graph_builder.graph = graph

    # Compute PageRank from "flask"
    print("\nStarting from: flask")
    scores = graph_builder.personalized_pagerank(["flask"])

    print("\nPageRank scores (top 5):")
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    for node, score in sorted_scores[:5]:
        print(f"  {node}: {score:.4f}")

    # Visualize activation spread
    print("\nActivation spread:")
    print("  flask (query) -> python-basics, web-development, rest-api")
    print("  These are direct neighbors")
    print("\n  django, database")
    print("  These are 2-hop neighbors (spreading activation!)")


def demo_comparison():
    """Demonstrate comparison: with vs without spreading activation."""
    print("\n" + "=" * 60)
    print("Demo 3: Search Comparison (with/without spreading activation)")
    print("=" * 60)

    print("\nQuery: 'flask web framework'")
    print("\nWithout spreading activation:")
    print("  Results (top 3):")
    print("    1. flask.md (score: 0.95)")
    print("    2. web-development.md (score: 0.75)")
    print("    3. python-basics.md (score: 0.65)")

    print("\nWith spreading activation:")
    print("  Initial results (same as above)")
    print("  Activated neighbors (via PageRank):")
    print("    4. django.md (score: 0.45) [2-hop via web-development]")
    print("    5. rest-api.md (score: 0.42) [2-hop via web-development]")

    print("\nBenefit:")
    print("  The user asked about Flask, but related concepts")
    print("  like Django and REST API are also surfaced!")


def visualize_graph(graph: nx.DiGraph):
    """Visualize graph structure (text-based).

    Args:
        graph: Knowledge graph
    """
    print("\n" + "=" * 60)
    print("Graph Visualization (text-based)")
    print("=" * 60)

    print("\nNode connections:")
    for node in sorted(graph.nodes()):
        neighbors = list(graph.successors(node))
        if neighbors:
            print(f"\n  {node}:")
            for neighbor in sorted(neighbors):
                edge_data = graph[node][neighbor]
                link_type = edge_data.get("link_type", "unknown")
                print(f"    -> {neighbor} ({link_type})")


def main():
    """Run all demos."""
    print("=" * 60)
    print("Spreading Activation Demo")
    print("=" * 60)

    # Demo 1: Graph construction
    graph = demo_graph_construction()

    # Demo 2: PageRank
    demo_pagerank(graph)

    # Demo 3: Comparison
    demo_comparison()

    # Bonus: Visualize graph
    visualize_graph(graph)

    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
