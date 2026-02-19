from __future__ import annotations
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""E2E test for memory access frequency and active forgetting pipeline.

Requires ChromaDB and sentence-transformers.
"""

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
    """Create an isolated anima directory with sample knowledge files.

    Redirects ANIMAWORKS_DATA_DIR to tmp_path so core.paths resolves
    to the temporary directory without touching ~/.animaworks/.
    """
    data_dir = tmp_path / ".animaworks"
    data_dir.mkdir()
    (data_dir / "models").mkdir()
    (data_dir / "shared" / "users").mkdir(parents=True)
    (data_dir / "common_skills").mkdir()

    monkeypatch.setenv("ANIMAWORKS_DATA_DIR", str(data_dir))

    # Invalidate cached paths so the monkeypatch takes effect
    from core.paths import _prompt_cache
    _prompt_cache.clear()

    anima_dir = data_dir / "animas" / "test_anima"
    anima_dir.mkdir(parents=True)

    # Create standard subdirectories
    for sub in ("knowledge", "episodes", "skills", "procedures", "state", "vectordb"):
        (anima_dir / sub).mkdir()

    # Sample knowledge files
    (anima_dir / "knowledge" / "chatwork-policy.md").write_text(
        "# Chatwork対応方針\n\n"
        "## 概要\n\nChatwork経由の依頼には以下のように対応する。\n\n"
        "## 緊急対応\n\n山田さんからの依頼は優先対応。\n\n"
        "## 通常対応\n\nその他の依頼は通常業務として扱う。\n",
        encoding="utf-8",
    )

    (anima_dir / "knowledge" / "slack-integration.md").write_text(
        "# Slack連携\n\n"
        "## 通知設定\n\n重要なアラートはSlackに通知する。\n\n"
        "## チャンネル\n\n- #general: 一般連絡\n- #alerts: 緊急アラート\n",
        encoding="utf-8",
    )

    (anima_dir / "knowledge" / "important-rule.md").write_text(
        "# 重要ルール\n\n"
        "## ルール\n\n[IMPORTANT] この情報は絶対に忘れてはいけない。\n",
        encoding="utf-8",
    )

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


# ── Test 1: Access Frequency Full Pipeline ─────────────────────────


def test_access_frequency_full_pipeline(anima_dir, vector_store, indexer, retriever):
    """Verify the full access frequency pipeline with real ChromaDB.

    Steps:
    1. Index knowledge files and verify initial metadata (access_count=0)
    2. Search and record_access, verify access_count incremented
    3. Search and record_access again, verify access_count=2
    4. Verify frequency boost affects score ordering
    """
    knowledge_dir = anima_dir / "knowledge"

    # Index all knowledge files
    total = indexer.index_directory(knowledge_dir, "knowledge")
    assert total > 0, "Indexing should produce at least one chunk"

    # -- Step 1: Verify initial metadata fields --
    collection_name = "test_anima_knowledge"
    coll = vector_store.client.get_collection(name=collection_name)
    all_data = coll.get(include=["metadatas"])

    assert len(all_data["ids"]) > 0, "Should have indexed chunks"

    for meta in all_data["metadatas"]:
        assert meta["access_count"] == 0, (
            f"New chunks should have access_count=0, got {meta['access_count']}"
        )
        assert meta["activation_level"] == "normal", (
            f"New chunks should have activation_level='normal', got {meta['activation_level']}"
        )

    # -- Step 2: Search and record_access --
    results = retriever.search(
        query="Chatwork対応方針",
        anima_name="test_anima",
        memory_type="knowledge",
        top_k=3,
    )
    assert len(results) > 0, "Search should return results"

    # Record access for the returned results
    retriever.record_access(results, "test_anima")

    # Query ChromaDB directly to verify access_count and last_accessed_at
    accessed_ids = [r.doc_id for r in results]
    coll = vector_store.client.get_collection(name=collection_name)
    verified_data = coll.get(ids=accessed_ids, include=["metadatas"])

    for meta in verified_data["metadatas"]:
        assert meta["access_count"] == 1, (
            f"After first access, access_count should be 1, got {meta['access_count']}"
        )
        assert meta["last_accessed_at"] != "", (
            "last_accessed_at should be set after record_access"
        )
        # Verify it's a valid ISO timestamp
        datetime.fromisoformat(str(meta["last_accessed_at"]))

    # -- Step 3: Search and record_access again --
    results2 = retriever.search(
        query="Chatwork対応方針",
        anima_name="test_anima",
        memory_type="knowledge",
        top_k=3,
    )
    retriever.record_access(results2, "test_anima")

    verified_data2 = coll.get(ids=accessed_ids, include=["metadatas"])
    for meta in verified_data2["metadatas"]:
        assert meta["access_count"] == 2, (
            f"After second access, access_count should be 2, got {meta['access_count']}"
        )

    # -- Step 4: Verify frequency boost affects score --
    # Get all chunk IDs
    all_data_after = coll.get(include=["metadatas"])
    accessed_set = set(accessed_ids)
    non_accessed_ids = [
        doc_id for doc_id in all_data_after["ids"]
        if doc_id not in accessed_set
    ]

    if non_accessed_ids:
        # Search with a broad query that should hit both accessed and non-accessed
        results_broad = retriever.search(
            query="対応方針 連携 通知",
            anima_name="test_anima",
            memory_type="knowledge",
            top_k=10,
        )

        accessed_results = [r for r in results_broad if r.doc_id in accessed_set]
        non_accessed_results = [r for r in results_broad if r.doc_id not in accessed_set]

        if accessed_results and non_accessed_results:
            # Accessed chunks should have a frequency boost > 0
            for r in accessed_results:
                assert r.source_scores.get("frequency", 0) > 0, (
                    "Accessed chunks should have a positive frequency boost"
                )

            # Non-accessed chunks should have frequency boost = 0
            for r in non_accessed_results:
                assert r.source_scores.get("frequency", 0) == 0.0, (
                    "Non-accessed chunks should have zero frequency boost"
                )


# ── Test 2: Synaptic Downscaling E2E ───────────────────────────────


def test_synaptic_downscaling_e2e(anima_dir, vector_store, indexer):
    """Verify synaptic downscaling marks old unaccessed chunks as low-activation.

    Steps:
    1. Index knowledge files
    2. Manually set metadata to simulate old, unaccessed chunks
    3. Run synaptic_downscaling()
    4. Verify chunks are marked activation_level="low"
    5. Verify protected chunks (importance="important") are NOT marked
    """
    from core.memory.forgetting import ForgettingEngine

    knowledge_dir = anima_dir / "knowledge"

    # Index all knowledge files
    total = indexer.index_directory(knowledge_dir, "knowledge")
    assert total > 0

    collection_name = "test_anima_knowledge"
    coll = vector_store.client.get_collection(name=collection_name)
    all_data = coll.get(include=["metadatas"])

    # Identify important vs normal chunks
    important_ids = []
    normal_ids = []
    for i, doc_id in enumerate(all_data["ids"]):
        meta = all_data["metadatas"][i]
        if meta.get("importance") == "important":
            important_ids.append(doc_id)
        else:
            normal_ids.append(doc_id)

    assert len(important_ids) > 0, "Should have at least one important chunk"
    assert len(normal_ids) > 0, "Should have at least one normal chunk"

    # Simulate old, unaccessed chunks (100 days ago)
    old_date = (datetime.now() - timedelta(days=100)).isoformat()
    all_ids = all_data["ids"]
    old_metas = [
        {"last_accessed_at": old_date, "access_count": 0}
        for _ in all_ids
    ]
    # Set updated_at to old date as well (for fallback logic)
    for meta in old_metas:
        meta["updated_at"] = old_date

    vector_store.update_metadata(collection_name, all_ids, old_metas)

    # Create ForgettingEngine with injected vector store
    engine = ForgettingEngine(anima_dir, "test_anima")
    # Monkey-patch to use our test vector store instead of singleton
    engine._get_vector_store = lambda: vector_store

    # Run synaptic downscaling
    result = engine.synaptic_downscaling()

    assert result["scanned"] > 0, "Should have scanned chunks"
    assert result["marked_low"] > 0, "Should have marked some chunks as low"

    # Verify normal chunks are marked as low-activation
    coll = vector_store.client.get_collection(name=collection_name)
    for doc_id in normal_ids:
        data = coll.get(ids=[doc_id], include=["metadatas"])
        meta = data["metadatas"][0]
        assert meta["activation_level"] == "low", (
            f"Normal chunk {doc_id} should be marked low, got {meta['activation_level']}"
        )
        assert meta["low_activation_since"] != "", (
            f"low_activation_since should be set for {doc_id}"
        )

    # Verify important chunks are NOT marked
    for doc_id in important_ids:
        data = coll.get(ids=[doc_id], include=["metadatas"])
        meta = data["metadatas"][0]
        assert meta["activation_level"] == "normal", (
            f"Important chunk {doc_id} should remain normal, got {meta['activation_level']}"
        )


# ── Test 3: Complete Forgetting E2E ────────────────────────────────


def test_complete_forgetting_e2e(anima_dir, vector_store, indexer):
    """Verify complete forgetting archives and deletes low-activation chunks.

    Steps:
    1. Index knowledge files
    2. Manually set metadata: activation_level="low", low_activation_since=90 days ago
    3. Run complete_forgetting()
    4. Verify chunks are deleted from ChromaDB
    5. Verify source files are moved to archive/forgotten/
    """
    from core.memory.forgetting import ForgettingEngine

    knowledge_dir = anima_dir / "knowledge"

    # Index only the non-important files for this test
    chatwork_file = knowledge_dir / "chatwork-policy.md"
    slack_file = knowledge_dir / "slack-integration.md"

    indexer.index_file(chatwork_file, "knowledge")
    indexer.index_file(slack_file, "knowledge")

    collection_name = "test_anima_knowledge"
    coll = vector_store.client.get_collection(name=collection_name)
    all_data = coll.get(include=["metadatas"])

    all_ids = all_data["ids"]
    initial_count = len(all_ids)
    assert initial_count > 0, "Should have indexed chunks"

    # Verify source files exist before forgetting
    assert chatwork_file.exists(), "Source file should exist before forgetting"
    assert slack_file.exists(), "Source file should exist before forgetting"

    # Simulate low activation for 90 days with zero access
    low_since = (datetime.now() - timedelta(days=90)).isoformat()
    low_metas = [
        {
            "activation_level": "low",
            "low_activation_since": low_since,
            "access_count": 0,
        }
        for _ in all_ids
    ]
    vector_store.update_metadata(collection_name, all_ids, low_metas)

    # Create ForgettingEngine with injected vector store
    engine = ForgettingEngine(anima_dir, "test_anima")
    engine._get_vector_store = lambda: vector_store

    # Run complete forgetting
    result = engine.complete_forgetting()

    assert result["forgotten_chunks"] > 0, "Should have forgotten some chunks"
    assert len(result["archived_files"]) > 0, "Should have archived source files"

    # Verify chunks are deleted from ChromaDB
    coll = vector_store.client.get_collection(name=collection_name)
    remaining_data = coll.get(include=["metadatas"])
    remaining_count = len(remaining_data["ids"])
    assert remaining_count < initial_count, (
        f"Chunks should be deleted: had {initial_count}, now {remaining_count}"
    )

    # Verify source files are moved to archive/forgotten/
    archive_dir = anima_dir / "archive" / "forgotten"
    assert archive_dir.exists(), "Archive directory should be created"

    archived_files = list(archive_dir.iterdir())
    assert len(archived_files) > 0, "Should have archived files in the directory"

    # At least one of the original source files should be gone
    source_files_gone = (
        not chatwork_file.exists() or not slack_file.exists()
    )
    assert source_files_gone, (
        "At least one source file should have been moved to archive"
    )
