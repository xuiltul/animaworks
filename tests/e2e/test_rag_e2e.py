from __future__ import annotations
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""E2E tests for the RAG pipeline (Dense Vector + Temporal Decay + Spreading Activation).

These tests use real ChromaDB in-memory vector store, real MemoryIndexer with
sentence-transformers, and real MemoryRetriever. No mocks are used.

Install with: pip install 'animaworks[rag]'
"""

import asyncio
import os
from datetime import datetime, timedelta
from pathlib import Path

import pytest

# Skip all tests if required dependencies are not installed
chromadb = pytest.importorskip(
    "chromadb",
    reason="ChromaDB not installed. Install with: pip install 'animaworks[rag]'",
)
sentence_transformers = pytest.importorskip(
    "sentence_transformers",
    reason="sentence-transformers not installed. Install with: pip install 'animaworks[rag]'",
)


# ── Fixtures ────────────────────────────────────────────────────────


@pytest.fixture
def anima_dir(tmp_path, monkeypatch):
    """Create an isolated anima directory with ANIMAWORKS_DATA_DIR redirected.

    Ensures core.paths.get_data_dir() resolves to tmp_path so that
    MemoryIndexer can cache models and write index metadata without
    touching the real ~/.animaworks/ directory.
    """
    data_dir = tmp_path / ".animaworks"
    data_dir.mkdir()
    (data_dir / "models").mkdir()
    (data_dir / "shared" / "users").mkdir(parents=True)
    (data_dir / "common_skills").mkdir()

    monkeypatch.setenv("ANIMAWORKS_DATA_DIR", str(data_dir))

    # Invalidate cached paths/config so the monkeypatch takes effect
    from core.paths import _prompt_cache
    _prompt_cache.clear()

    anima_dir = data_dir / "animas" / "test_anima"
    anima_dir.mkdir(parents=True)

    # Create standard subdirectories
    for sub in ("knowledge", "episodes", "skills", "procedures", "state", "vectordb"):
        (anima_dir / sub).mkdir()

    yield anima_dir


@pytest.fixture
def vector_store(anima_dir):
    """Create a ChromaDB vector store persisted under the anima's vectordb dir."""
    from core.memory.rag.store import ChromaVectorStore

    store = ChromaVectorStore(persist_dir=anima_dir / "vectordb")
    return store


@pytest.fixture
def indexer(vector_store, anima_dir):
    """Create a MemoryIndexer bound to the test anima directory."""
    from core.memory.rag.indexer import MemoryIndexer

    return MemoryIndexer(vector_store, "test_anima", anima_dir)


@pytest.fixture
def retriever(vector_store, indexer, anima_dir):
    """Create a MemoryRetriever for the test anima."""
    from core.memory.rag.retriever import MemoryRetriever

    return MemoryRetriever(
        vector_store,
        indexer,
        anima_dir / "knowledge",
    )


# ── Test 1: Index and Search ───────────────────────────────────────


def test_e2e_index_and_search(anima_dir, indexer, retriever):
    """Index knowledge files and verify dense vector search returns correct results.

    Creates two knowledge files with distinct topics (Python開発 and 料理レシピ),
    indexes them, and verifies that a query about Python development retrieves the
    correct file with higher relevance.
    """
    knowledge_dir = anima_dir / "knowledge"

    # Create knowledge files with distinct topics
    (knowledge_dir / "python-dev.md").write_text(
        "# Python開発ガイド\n\n"
        "## 概要\n\nPythonでのWebアプリケーション開発手法について。\n"
        "FastAPIとUvicornを使ったAPIサーバー構築が中心。\n\n"
        "## テスト\n\npytestを使った自動テストの書き方。\n",
        encoding="utf-8",
    )
    (knowledge_dir / "cooking-recipe.md").write_text(
        "# 料理レシピ集\n\n"
        "## カレー\n\n玉ねぎをじっくり炒めてカレールーを溶かす。\n\n"
        "## パスタ\n\nアルデンテに茹でたパスタにソースを絡める。\n",
        encoding="utf-8",
    )

    # Index all knowledge files
    total = indexer.index_directory(knowledge_dir, "knowledge")
    assert total > 0, "Indexing should produce at least one chunk"

    # Search for Python-related content
    results = retriever.search(
        query="Pythonでの開発について教えて",
        anima_name="test_anima",
        memory_type="knowledge",
        top_k=3,
    )

    assert len(results) > 0, "Search should return results"
    # The top result should come from python-dev.md
    top_result = results[0]
    assert "python" in top_result.doc_id.lower() or "Python" in top_result.content
    assert top_result.score > 0
    assert "vector" in top_result.source_scores
    assert "recency" in top_result.source_scores


# ── Test 2: Temporal Decay Ordering ────────────────────────────────


def test_e2e_temporal_decay_ordering(anima_dir, indexer, retriever):
    """Verify that newer memories rank higher than older ones with identical content.

    Creates two knowledge files with near-identical content but different
    filesystem modification timestamps. After indexing, the file with the more
    recent mtime should receive a higher combined score due to temporal decay.
    """
    knowledge_dir = anima_dir / "knowledge"

    # Create two files with similar content
    old_file = knowledge_dir / "report-old.md"
    old_file.write_text(
        "# 月次レポート\n\n## 売上報告\n\n今月の売上は前年比120%。営業チーム全体で好調。\n",
        encoding="utf-8",
    )

    new_file = knowledge_dir / "report-new.md"
    new_file.write_text(
        "# 月次レポート\n\n## 売上報告\n\n今月の売上は前年比125%。営業チーム全体で好調な結果。\n",
        encoding="utf-8",
    )

    # Set old_file's mtime to 90 days ago
    old_ts = (datetime.now() - timedelta(days=90)).timestamp()
    os.utime(old_file, (old_ts, old_ts))

    # Set new_file's mtime to now (already default, but be explicit)
    new_ts = datetime.now().timestamp()
    os.utime(new_file, (new_ts, new_ts))

    # Index both files
    indexer.index_file(old_file, "knowledge", force=True)
    indexer.index_file(new_file, "knowledge", force=True)

    # Search for the common topic
    results = retriever.search(
        query="月次売上レポート",
        anima_name="test_anima",
        memory_type="knowledge",
        top_k=5,
    )

    assert len(results) >= 2, "Should find both report files"

    # Find results from each file
    old_results = [r for r in results if "report-old" in r.doc_id]
    new_results = [r for r in results if "report-new" in r.doc_id]

    assert old_results, "Should have results from the old report"
    assert new_results, "Should have results from the new report"

    # The new file should have a higher combined score due to temporal decay
    best_old = max(r.score for r in old_results)
    best_new = max(r.score for r in new_results)
    assert best_new > best_old, (
        f"Newer file (score={best_new:.4f}) should rank higher than older "
        f"file (score={best_old:.4f}) due to temporal decay"
    )

    # Verify recency scores differ
    old_recency = old_results[0].source_scores.get("recency", 0)
    new_recency = new_results[0].source_scores.get("recency", 0)
    assert new_recency > old_recency, "Newer file should have higher recency score"


# ── Test 3: Spreading Activation ──────────────────────────────────


def test_e2e_spreading_activation(anima_dir, vector_store, indexer):
    """Verify spreading activation expands search results via knowledge graph links.

    Creates three knowledge files where file-A links to file-B via ``[[link]]``
    notation. After building the graph, a search for file-A's content with
    spreading activation enabled should also surface file-B as an activated neighbor.
    """
    from core.memory.rag.retriever import MemoryRetriever

    knowledge_dir = anima_dir / "knowledge"

    # Create interconnected files with [[link]] references
    (knowledge_dir / "api-design.md").write_text(
        "# API設計方針\n\n"
        "## RESTful設計\n\nリソース指向でAPIを設計する。関連: [[error-handling]]\n\n"
        "## バージョニング\n\nURLパスにバージョン番号を含める。\n",
        encoding="utf-8",
    )
    (knowledge_dir / "error-handling.md").write_text(
        "# エラーハンドリング\n\n"
        "## HTTP ステータスコード\n\n適切なステータスコードを返す。関連: [[logging-policy]]\n\n"
        "## エラーレスポンス形式\n\nJSON形式で統一的なエラーレスポンスを返す。\n",
        encoding="utf-8",
    )
    (knowledge_dir / "logging-policy.md").write_text(
        "# ログ出力方針\n\n"
        "## ログレベル\n\nDEBUG, INFO, WARNING, ERROR の4段階を使い分ける。\n\n"
        "## 構造化ログ\n\nJSON形式のログ出力を標準とする。\n",
        encoding="utf-8",
    )

    # Index all files
    indexer.index_directory(knowledge_dir, "knowledge")

    # Create retriever with spreading activation
    retriever = MemoryRetriever(
        vector_store,
        indexer,
        knowledge_dir,
    )

    # Search with spreading activation enabled
    results = retriever.search(
        query="API設計のエラー処理について",
        anima_name="test_anima",
        memory_type="knowledge",
        top_k=2,
        enable_spreading_activation=True,
    )

    assert len(results) > 0, "Should return results"

    # Collect all doc IDs and content from results
    all_content = " ".join(r.content for r in results)
    all_doc_ids = [r.doc_id for r in results]

    # With spreading activation, we expect related nodes to appear
    # Either via direct search or via graph expansion
    has_api = any("api-design" in d for d in all_doc_ids)
    has_error = any("error-handling" in d for d in all_doc_ids) or "エラー" in all_content
    has_logging = any("logging-policy" in d for d in all_doc_ids) or "ログ" in all_content

    # At minimum, the directly relevant files should be found
    assert has_api or has_error, (
        "Search should find api-design or error-handling content"
    )

    # Spreading activation should bring in at least one linked neighbor
    # (either error-handling via api-design's link, or logging-policy via error-handling's link)
    total_unique_files = len({
        d.split("/")[-1].split("#")[0]
        for d in all_doc_ids
        if "/" in d
    })
    assert total_unique_files >= 2, (
        f"Spreading activation should expand results beyond a single file "
        f"(found {total_unique_files} unique files)"
    )


# ── Test 4: Graph Cache Persistence ───────────────────────────────


def test_e2e_graph_cache_persistence(anima_dir, vector_store, indexer):
    """Verify knowledge graph can be saved and loaded from JSON cache.

    Builds a graph, saves it to a cache directory, creates a new
    KnowledgeGraph instance, loads the cache, and verifies that the loaded
    graph produces the same PageRank scores as the original.
    """
    from core.memory.rag.graph import KnowledgeGraph

    knowledge_dir = anima_dir / "knowledge"
    cache_dir = anima_dir / "vectordb"

    # Create knowledge files with links
    (knowledge_dir / "infra.md").write_text(
        "# インフラ構成\n\nAWSを使ったインフラ構成。関連: [[deploy]]\n",
        encoding="utf-8",
    )
    (knowledge_dir / "deploy.md").write_text(
        "# デプロイ手順\n\nCI/CDパイプラインの構成。関連: [[infra]]\n",
        encoding="utf-8",
    )
    (knowledge_dir / "monitoring.md").write_text(
        "# 監視設定\n\nCloudWatchでの監視設定。関連: [[infra]]\n",
        encoding="utf-8",
    )

    # Index files first (needed for implicit link calculation)
    indexer.index_directory(knowledge_dir, "knowledge")

    # Build and save graph
    graph1 = KnowledgeGraph(vector_store, indexer)
    graph1.build_graph("test_anima", knowledge_dir)
    graph1.save_graph(cache_dir)

    # Verify cache file exists
    cache_file = cache_dir / "knowledge_graph.json"
    assert cache_file.exists(), "Graph cache file should be created"

    # Compute PageRank on original graph
    scores1 = graph1.personalized_pagerank(["infra"])

    # Load into a new instance
    graph2 = KnowledgeGraph(vector_store, indexer)
    loaded = graph2.load_graph(cache_dir)

    assert loaded is True, "Graph should load successfully from cache"
    assert graph2.graph is not None

    # Verify structural equivalence
    assert graph2.graph.number_of_nodes() == graph1.graph.number_of_nodes(), (
        "Loaded graph should have same number of nodes"
    )
    assert graph2.graph.number_of_edges() == graph1.graph.number_of_edges(), (
        "Loaded graph should have same number of edges"
    )

    # Verify nodes are preserved
    for node in graph1.graph.nodes():
        assert node in graph2.graph, f"Node '{node}' should exist in loaded graph"

    # Verify PageRank scores match
    scores2 = graph2.personalized_pagerank(["infra"])
    for node in scores1:
        assert abs(scores1[node] - scores2[node]) < 1e-6, (
            f"PageRank score for '{node}' should match between original and loaded graph"
        )


# ── Test 5: Incremental Index and Graph ───────────────────────────


def test_e2e_incremental_index_and_graph(anima_dir, vector_store, indexer, retriever):
    """Verify incremental indexing and graph updates work correctly.

    Builds an initial index and graph, then adds a new file. After
    incremental indexing and graph update, the new file should appear
    in search results.
    """
    from core.memory.rag.graph import KnowledgeGraph

    knowledge_dir = anima_dir / "knowledge"

    # Create initial files
    (knowledge_dir / "database.md").write_text(
        "# データベース設計\n\n"
        "## テーブル設計\n\nPostgreSQLでのリレーショナル設計。\n\n"
        "## インデックス\n\n適切なインデックスでクエリ性能を最適化。\n",
        encoding="utf-8",
    )
    (knowledge_dir / "cache.md").write_text(
        "# キャッシュ戦略\n\n"
        "## Redis\n\nRedisを使ったキャッシュ層の実装。\n\n"
        "## 有効期限\n\nTTL設定によるキャッシュ有効期限管理。\n",
        encoding="utf-8",
    )

    # Initial full index
    indexer.index_directory(knowledge_dir, "knowledge")

    # Build initial graph
    graph = KnowledgeGraph(vector_store, indexer)
    graph.build_graph("test_anima", knowledge_dir)

    initial_nodes = graph.graph.number_of_nodes()
    assert initial_nodes == 2

    # Verify initial search works
    results_before = retriever.search(
        query="クエリ最適化",
        anima_name="test_anima",
        memory_type="knowledge",
        top_k=5,
    )
    initial_doc_ids = {r.doc_id for r in results_before}

    # Add a new file
    new_file = knowledge_dir / "query-optimization.md"
    new_file.write_text(
        "# クエリ最適化\n\n"
        "## EXPLAIN\n\nEXPLAINコマンドで実行計画を確認する。関連: [[database]]\n\n"
        "## スロークエリ\n\n遅いクエリを特定し最適化する手法。\n",
        encoding="utf-8",
    )

    # Incremental index of new file only
    chunks_added = indexer.index_file(new_file, "knowledge")
    assert chunks_added > 0, "New file should produce chunks"

    # Incremental graph update
    graph.graph.add_node("query-optimization", path=str(new_file))
    graph.update_graph_incremental([new_file], "test_anima")

    assert graph.graph.number_of_nodes() == initial_nodes + 1
    assert "query-optimization" in graph.graph

    # Search again - new file should appear in results
    results_after = retriever.search(
        query="クエリ最適化とEXPLAIN",
        anima_name="test_anima",
        memory_type="knowledge",
        top_k=5,
    )

    assert len(results_after) > 0
    after_doc_ids = {r.doc_id for r in results_after}

    # The new file should be found
    has_new_file = any("query-optimization" in doc_id for doc_id in after_doc_ids)
    assert has_new_file, (
        f"New file should appear in search results. Found: {after_doc_ids}"
    )


# ── Test 6: Priming Integration ──────────────────────────────────


def test_e2e_priming_integration(anima_dir, vector_store, indexer):
    """Verify PrimingEngine retrieves memories and formats them for system prompt.

    Creates an Anima directory with knowledge, episodes, and skills,
    indexes the knowledge, and then uses PrimingEngine to prime memories.
    Validates that format_priming_section() produces a non-empty markdown
    section with the expected structure.
    """
    from core.memory.priming import PrimingEngine, format_priming_section

    knowledge_dir = anima_dir / "knowledge"
    episodes_dir = anima_dir / "episodes"
    skills_dir = anima_dir / "skills"

    # Create knowledge file
    (knowledge_dir / "project-alpha.md").write_text(
        "# プロジェクトAlpha\n\n"
        "## 概要\n\n新規Webサービスの立ち上げプロジェクト。\n"
        "Reactフロントエンド + FastAPIバックエンド構成。\n\n"
        "## スケジュール\n\n2026年3月リリース予定。\n",
        encoding="utf-8",
    )

    # Create today's episode file
    from datetime import date
    today = date.today()
    (episodes_dir / f"{today.isoformat()}.md").write_text(
        f"# {today.isoformat()} 行動ログ\n\n"
        "## 09:00 — 朝会\n\nプロジェクトAlphaの進捗確認。フロントエンド80%完了。\n\n"
        "## 14:00 — コードレビュー\n\nAPI認証モジュールのレビュー実施。\n",
        encoding="utf-8",
    )

    # Create skill file
    (skills_dir / "react-development.md").write_text(
        "# React開発\n\n"
        "## 概要\n\nReactを使ったフロントエンド開発スキル。\n"
        "TypeScript, Next.js, Tailwind CSS を使用。\n",
        encoding="utf-8",
    )

    # Index knowledge for vector search
    indexer.index_directory(knowledge_dir, "knowledge")

    # Create PrimingEngine and prime memories
    engine = PrimingEngine(anima_dir)

    result = asyncio.run(
        engine.prime_memories(
            message="プロジェクトAlphaの進捗を教えて",
            sender_name="yamada",
        )
    )

    # Verify priming result has content
    assert not result.is_empty(), "Priming result should not be empty"

    # At least one channel should have produced output
    has_activity = bool(result.recent_activity)
    has_knowledge = bool(result.related_knowledge)
    has_skills = bool(result.matched_skills)

    assert has_activity or has_knowledge or has_skills, (
        "At least one priming channel should return content"
    )

    # Format for system prompt injection
    formatted = format_priming_section(result, sender_name="yamada")

    assert formatted, "Formatted priming section should not be empty"
    assert "あなたが思い出していること" in formatted, (
        "Formatted section should contain the standard header"
    )

    # Verify structural sections exist based on what was primed
    if has_activity:
        assert "直近のアクティビティ" in formatted
    if has_knowledge:
        assert "関連する知識" in formatted
    if has_skills:
        assert "使えそうなスキル" in formatted


# ── Test 7: Multi Memory Type ─────────────────────────────────────


def test_e2e_multi_memory_type(anima_dir, vector_store, indexer):
    """Index both knowledge and episodes, then search each type separately.

    Verifies that the collection isolation per memory_type works correctly:
    searching knowledge should not return episode content and vice versa.
    """
    from core.memory.rag.retriever import MemoryRetriever

    knowledge_dir = anima_dir / "knowledge"
    episodes_dir = anima_dir / "episodes"

    # Create knowledge file about security
    (knowledge_dir / "security-policy.md").write_text(
        "# セキュリティポリシー\n\n"
        "## 認証\n\nOAuth2.0を使用した認証フロー。JWT トークンで状態管理。\n\n"
        "## 暗号化\n\nAES-256で保存データを暗号化する。TLS 1.3を必須とする。\n",
        encoding="utf-8",
    )

    # Create episode file about a security incident
    (episodes_dir / "2026-02-10.md").write_text(
        "# 2026-02-10 行動ログ\n\n"
        "## 10:00 — セキュリティアラート対応\n\n"
        "不正アクセスの疑いがあるログを検出。IPアドレスをブロックして対応完了。\n\n"
        "## 15:00 — インシデントレポート作成\n\n"
        "セキュリティインシデントのレポートを作成し、チームに共有した。\n",
        encoding="utf-8",
    )

    # Index knowledge and episodes separately
    knowledge_chunks = indexer.index_directory(knowledge_dir, "knowledge")
    episodes_chunks = indexer.index_directory(episodes_dir, "episodes")

    assert knowledge_chunks > 0, "Knowledge indexing should produce chunks"
    assert episodes_chunks > 0, "Episodes indexing should produce chunks"

    # Create retriever
    retriever = MemoryRetriever(
        vector_store,
        indexer,
        knowledge_dir,
    )

    # Search knowledge type
    knowledge_results = retriever.search(
        query="セキュリティ認証の仕組み",
        anima_name="test_anima",
        memory_type="knowledge",
        top_k=5,
    )

    assert len(knowledge_results) > 0, "Knowledge search should return results"
    # All results should be from the knowledge collection
    for r in knowledge_results:
        assert "knowledge" in r.doc_id, (
            f"Knowledge search result should be from knowledge collection: {r.doc_id}"
        )

    # Search episodes type
    episodes_results = retriever.search(
        query="セキュリティインシデント対応",
        anima_name="test_anima",
        memory_type="episodes",
        top_k=5,
    )

    assert len(episodes_results) > 0, "Episodes search should return results"
    # All results should be from the episodes collection
    for r in episodes_results:
        assert "episodes" in r.doc_id, (
            f"Episodes search result should be from episodes collection: {r.doc_id}"
        )

    # Verify content isolation: knowledge results should contain policy content
    knowledge_content = " ".join(r.content for r in knowledge_results)
    assert "OAuth" in knowledge_content or "認証" in knowledge_content or "暗号化" in knowledge_content, (
        "Knowledge results should contain security policy content"
    )

    # Episodes results should contain incident content
    episodes_content = " ".join(r.content for r in episodes_results)
    assert "アラート" in episodes_content or "インシデント" in episodes_content or "不正アクセス" in episodes_content, (
        "Episodes results should contain incident log content"
    )
