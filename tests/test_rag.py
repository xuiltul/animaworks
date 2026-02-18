from __future__ import annotations
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for RAG (Retrieval-Augmented Generation) subsystem.

These tests require ChromaDB and sentence-transformers.
Install with: pip install 'animaworks[rag]'
"""

import pytest
from pathlib import Path
import tempfile
import shutil

# Skip all tests in this module if ChromaDB is not available
chromadb = pytest.importorskip("chromadb", reason="ChromaDB not installed. Install with: pip install 'animaworks[rag]'")

# ── Fixtures ────────────────────────────────────────────────────────


@pytest.fixture
def temp_anima_dir():
    """Create a temporary anima directory with sample memory files."""
    tmpdir = Path(tempfile.mkdtemp())
    anima_dir = tmpdir / "test_anima"
    anima_dir.mkdir()

    # Create memory directories
    knowledge_dir = anima_dir / "knowledge"
    episodes_dir = anima_dir / "episodes"
    procedures_dir = anima_dir / "procedures"
    skills_dir = anima_dir / "skills"

    for d in [knowledge_dir, episodes_dir, procedures_dir, skills_dir]:
        d.mkdir()

    # Sample knowledge file
    (knowledge_dir / "chatwork-policy.md").write_text(
        """# Chatwork対応方針

## 概要
Chatwork経由の依頼には以下のように対応する。

## 緊急対応
山田さんからの [IMPORTANT] マークがある依頼は優先対応。

## 通常対応
その他の依頼は通常業務として扱う。
""",
        encoding="utf-8",
    )

    (knowledge_dir / "slack-integration.md").write_text(
        """# Slack連携

## 通知設定
重要なアラートはSlackに通知する。

## チャンネル
- #general: 一般連絡
- #alerts: 緊急アラート
""",
        encoding="utf-8",
    )

    # Sample episode file
    (episodes_dir / "2026-02-14.md").write_text(
        """# 2026-02-14 行動ログ

## 09:30 — 山田さんから依頼受信

Chatworkで新規プロジェクトの見積もり依頼を受けた。

## 14:00 — 見積もり作成完了

見積書を作成し、Chatworkで送信した。
""",
        encoding="utf-8",
    )

    # Sample procedure
    (procedures_dir / "deploy-workflow.md").write_text(
        """# デプロイ手順

1. テスト実行
2. ビルド
3. ステージング環境へデプロイ
4. 動作確認
5. 本番環境へデプロイ
""",
        encoding="utf-8",
    )

    # Sample skill
    (skills_dir / "python-coding.md").write_text(
        """# Python coding

## 概要
Python でのコーディング支援。

## 得意分野
- FastAPI
- データ処理
- 自動化スクリプト
""",
        encoding="utf-8",
    )

    yield anima_dir

    # Cleanup
    shutil.rmtree(tmpdir)


@pytest.fixture
def temp_vector_store(temp_anima_dir):
    """Create a temporary ChromaDB vector store."""
    pytest.importorskip("chromadb")
    pytest.importorskip("sentence_transformers")

    from core.memory.rag.store import ChromaVectorStore

    vectordb_dir = temp_anima_dir / "vectordb"
    vectordb_dir.mkdir()

    store = ChromaVectorStore(persist_dir=vectordb_dir)
    yield store

    # Cleanup is handled by temp_anima_dir fixture


# ── ChromaDB Vector Store Tests ────────────────────────────────────


def test_chromadb_create_collection(temp_vector_store):
    """Test creating a ChromaDB collection."""
    from core.memory.rag.store import ChromaVectorStore

    store = temp_vector_store
    store.create_collection("test_knowledge", dimension=384)

    collections = store.list_collections()
    assert "test_knowledge" in collections


def test_chromadb_upsert_and_query(temp_vector_store):
    """Test upserting and querying documents."""
    from core.memory.rag.store import Document

    store = temp_vector_store
    store.create_collection("test_collection", dimension=3)

    # Create test documents with simple embeddings
    docs = [
        Document(
            id="doc1",
            content="This is about Python programming",
            embedding=[0.9, 0.1, 0.0],
            metadata={"topic": "programming", "language": "en"},
        ),
        Document(
            id="doc2",
            content="This is about Chatwork messaging",
            embedding=[0.1, 0.9, 0.0],
            metadata={"topic": "communication", "language": "en"},
        ),
        Document(
            id="doc3",
            content="This is about Slack integration",
            embedding=[0.1, 0.8, 0.1],
            metadata={"topic": "communication", "language": "en"},
        ),
    ]

    store.upsert("test_collection", docs)

    # Query with embedding similar to doc1
    results = store.query(
        collection="test_collection",
        embedding=[1.0, 0.0, 0.0],
        top_k=2,
    )

    assert len(results) == 2
    assert results[0].document.id == "doc1"  # Most similar


def test_chromadb_metadata_filter(temp_vector_store):
    """Test metadata filtering in queries."""
    from core.memory.rag.store import Document

    store = temp_vector_store
    store.create_collection("test_filter", dimension=3)

    docs = [
        Document(
            id="doc1",
            content="Python content",
            embedding=[0.5, 0.5, 0.0],
            metadata={"topic": "programming"},
        ),
        Document(
            id="doc2",
            content="Chatwork content",
            embedding=[0.5, 0.5, 0.0],
            metadata={"topic": "communication"},
        ),
    ]

    store.upsert("test_filter", docs)

    # Query with metadata filter
    results = store.query(
        collection="test_filter",
        embedding=[0.5, 0.5, 0.0],
        top_k=10,
        filter_metadata={"topic": "programming"},
    )

    assert len(results) == 1
    assert results[0].document.id == "doc1"


# ── MemoryIndexer Tests ─────────────────────────────────────────────


def test_indexer_chunk_by_markdown_headings(temp_anima_dir, temp_vector_store):
    """Test chunking knowledge files by Markdown headings."""
    pytest.importorskip("sentence_transformers")

    from core.memory.rag.indexer import MemoryIndexer

    indexer = MemoryIndexer(
        temp_vector_store, "test_anima", temp_anima_dir
    )

    knowledge_file = temp_anima_dir / "knowledge" / "chatwork-policy.md"
    chunks_indexed = indexer.index_file(knowledge_file, "knowledge")

    assert chunks_indexed > 0  # Should create multiple chunks


def test_indexer_chunk_by_time_headings(temp_anima_dir, temp_vector_store):
    """Test chunking episode files by time headings."""
    pytest.importorskip("sentence_transformers")

    from core.memory.rag.indexer import MemoryIndexer

    indexer = MemoryIndexer(
        temp_vector_store, "test_anima", temp_anima_dir
    )

    episode_file = temp_anima_dir / "episodes" / "2026-02-14.md"
    chunks_indexed = indexer.index_file(episode_file, "episodes")

    assert chunks_indexed >= 2  # Should have 2 time-based sections


def test_indexer_incremental_indexing(temp_anima_dir, temp_vector_store):
    """Test that unchanged files are not re-indexed."""
    pytest.importorskip("sentence_transformers")

    from core.memory.rag.indexer import MemoryIndexer

    indexer = MemoryIndexer(
        temp_vector_store, "test_anima", temp_anima_dir
    )

    knowledge_file = temp_anima_dir / "knowledge" / "chatwork-policy.md"

    # First index
    chunks1 = indexer.index_file(knowledge_file, "knowledge")
    assert chunks1 > 0

    # Second index (no changes)
    chunks2 = indexer.index_file(knowledge_file, "knowledge")
    assert chunks2 == 0  # Should skip unchanged file

    # Force re-index
    chunks3 = indexer.index_file(knowledge_file, "knowledge", force=True)
    assert chunks3 > 0


def test_indexer_metadata_extraction(temp_anima_dir, temp_vector_store):
    """Test metadata extraction (tags, importance, timestamps)."""
    pytest.importorskip("sentence_transformers")

    from core.memory.rag.indexer import MemoryIndexer

    # Create file with importance marker
    knowledge_file = temp_anima_dir / "knowledge" / "important.md"
    knowledge_file.write_text(
        """# Important Policy

[IMPORTANT] This is critical information.
""",
        encoding="utf-8",
    )

    indexer = MemoryIndexer(
        temp_vector_store, "test_anima", temp_anima_dir
    )

    chunks_indexed = indexer.index_file(knowledge_file, "knowledge")
    assert chunks_indexed > 0

    # Verify metadata in vector store
    collection_name = "test_anima_knowledge"
    results = indexer.vector_store.query(
        collection=collection_name,
        embedding=[0.0] * 384,  # Dummy query
        top_k=10,
    )

    # Find the important.md chunk
    important_chunks = [
        r for r in results
        if "important.md" in r.document.metadata.get("source_file", "")
    ]

    assert len(important_chunks) > 0
    assert important_chunks[0].document.metadata["importance"] == "important"


# ── MemoryRetriever Tests ──────────────────────────────────────────


def test_memory_retriever_vector_search(temp_anima_dir, temp_vector_store):
    """Test vector search component of MemoryRetriever."""
    pytest.importorskip("sentence_transformers")

    from core.memory.rag.indexer import MemoryIndexer
    from core.memory.rag.retriever import MemoryRetriever

    # Index knowledge files
    indexer = MemoryIndexer(
        temp_vector_store, "test_anima", temp_anima_dir
    )

    knowledge_dir = temp_anima_dir / "knowledge"
    indexer.index_directory(knowledge_dir, "knowledge")

    # Create retriever
    retriever = MemoryRetriever(
        temp_vector_store,
        indexer,
        knowledge_dir,
    )

    # Search for Chatwork-related content
    results = retriever._vector_search(
        query="Chatwork対応方針",
        anima_name="test_anima",
        memory_type="knowledge",
        top_k=3,
    )

    assert len(results) > 0
    # First result should be from chatwork-policy.md
    assert any("chatwork" in r[0].lower() for r in results)


def test_temporal_decay(temp_anima_dir, temp_vector_store):
    """Test temporal decay scoring."""
    from datetime import datetime, timedelta
    from core.memory.rag.retriever import MemoryRetriever, RetrievalResult

    indexer = None
    retriever = MemoryRetriever(
        temp_vector_store,
        indexer,
        temp_anima_dir / "knowledge",
    )

    # Create results with different ages
    now = datetime.now()
    recent_date = now - timedelta(days=1)
    old_date = now - timedelta(days=60)

    results = [
        RetrievalResult(
            doc_id="recent",
            content="Recent document",
            score=0.5,
            metadata={"updated_at": recent_date.isoformat()},
            source_scores={},
        ),
        RetrievalResult(
            doc_id="old",
            content="Old document",
            score=0.5,
            metadata={"updated_at": old_date.isoformat()},
            source_scores={},
        ),
    ]

    decayed = retriever._apply_score_adjustments(results)

    # Recent document should have higher final score
    recent_result = next(r for r in decayed if r.doc_id == "recent")
    old_result = next(r for r in decayed if r.doc_id == "old")

    assert recent_result.score > old_result.score


def test_memory_retriever_search_with_temporal_decay(temp_anima_dir, temp_vector_store):
    """Test full search pipeline: Vector + temporal decay integration."""
    pytest.importorskip("sentence_transformers")

    from core.memory.rag.indexer import MemoryIndexer
    from core.memory.rag.retriever import MemoryRetriever

    # Index knowledge files
    indexer = MemoryIndexer(
        temp_vector_store, "test_anima", temp_anima_dir
    )

    knowledge_dir = temp_anima_dir / "knowledge"
    indexer.index_directory(knowledge_dir, "knowledge")

    # Create retriever
    retriever = MemoryRetriever(
        temp_vector_store,
        indexer,
        knowledge_dir,
    )

    # Perform search
    results = retriever.search(
        query="Chatwork での山田さんからの依頼対応",
        anima_name="test_anima",
        memory_type="knowledge",
        top_k=3,
    )

    assert len(results) > 0
    # Results should have both vector and recency scores
    assert results[0].score > 0
    assert "vector" in results[0].source_scores
    assert "recency" in results[0].source_scores


# ── Integration Tests ───────────────────────────────────────────────


@pytest.mark.asyncio
async def test_priming_with_vector_search(temp_anima_dir, temp_vector_store):
    """Test priming layer with vector search integration."""
    pytest.importorskip("sentence_transformers")

    from core.memory.priming import PrimingEngine
    from core.memory.rag.indexer import MemoryIndexer

    # Index knowledge
    indexer = MemoryIndexer(
        temp_vector_store, "test_anima", temp_anima_dir
    )

    knowledge_dir = temp_anima_dir / "knowledge"
    indexer.index_directory(knowledge_dir, "knowledge")

    # Create priming engine and inject retriever using the same vector store
    engine = PrimingEngine(temp_anima_dir)

    from core.memory.rag.retriever import MemoryRetriever

    engine._retriever = MemoryRetriever(
        temp_vector_store, indexer, temp_anima_dir / "knowledge"
    )

    # Prime memories
    result = await engine.prime_memories(
        message="山田さんからChatworkで依頼が来ました",
        sender_name="yamada",
    )

    # Should find chatwork-policy.md in related knowledge
    assert result.related_knowledge
    assert "chatwork" in result.related_knowledge.lower() or "山田" in result.related_knowledge
