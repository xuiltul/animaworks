from __future__ import annotations
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""E2E tests for common_knowledge RAG infrastructure.

Verifies the full flow of shared knowledge indexing, retrieval,
priming integration, and tool-handler file access — without requiring
actual LLM API calls.

Install dependencies with: pip install 'animaworks[rag]'
"""

import shutil
import tempfile
from pathlib import Path

import pytest

# Skip the entire module if ChromaDB or sentence-transformers are missing
chromadb = pytest.importorskip(
    "chromadb", reason="ChromaDB not installed. Install with: pip install 'animaworks[rag]'"
)
pytest.importorskip(
    "sentence_transformers",
    reason="sentence-transformers not installed. Install with: pip install 'animaworks[rag]'",
)


# ── Fixtures ────────────────────────────────────────────────────────


@pytest.fixture
def temp_dirs():
    """Create temporary anima_dir and common_knowledge dir with sample files.

    Yields a (anima_dir, common_knowledge_dir, data_dir) tuple.
    All directories are cleaned up after the test.
    """
    tmpdir = Path(tempfile.mkdtemp())
    data_dir = tmpdir / "data"
    data_dir.mkdir()

    # ── Anima directory ──────────────────────────
    anima_dir = data_dir / "animas" / "test_anima"
    anima_dir.mkdir(parents=True)

    for sub in ("knowledge", "episodes", "procedures", "skills", "state"):
        (anima_dir / sub).mkdir()

    (anima_dir / "knowledge" / "internal-policy.md").write_text(
        """# 社内ポリシー

## 概要
社内の情報セキュリティに関する基本方針。

## 機密情報の取り扱い
社外秘の情報は暗号化して保管する。
""",
        encoding="utf-8",
    )

    (anima_dir / "knowledge" / "project-alpha.md").write_text(
        """# プロジェクトアルファ

## 概要
2026年度の主要プロジェクト。

## スケジュール
Q1: 設計フェーズ
Q2: 実装フェーズ
""",
        encoding="utf-8",
    )

    # ── Common knowledge directory ────────────────
    common_knowledge_dir = data_dir / "common_knowledge"
    common_knowledge_dir.mkdir()

    (common_knowledge_dir / "company-handbook.md").write_text(
        """# 会社ハンドブック

## 概要
全社員が参照すべき共通ルールをまとめたドキュメント。

## 勤怠ルール
出社時間は9:00、退社時間は18:00。
フレックスタイム制度あり。

## 経費精算
経費は月末締め翌月15日支払い。
""",
        encoding="utf-8",
    )

    (common_knowledge_dir / "coding-standards.md").write_text(
        """# コーディング規約

## 概要
全プロジェクト共通のコーディングルール。

## Python
型ヒント必須、Black + Ruff でフォーマット。

## テスト
カバレッジ80%以上を目標とする。
""",
        encoding="utf-8",
    )

    (common_knowledge_dir / "test-file.md").write_text(
        """# Test File

This is a test file for read_memory_file E2E testing.
Contains sample content for verification.
""",
        encoding="utf-8",
    )

    yield anima_dir, common_knowledge_dir, data_dir

    shutil.rmtree(tmpdir)


@pytest.fixture
def vector_store(temp_dirs):
    """Create a temporary ChromaDB vector store."""
    anima_dir, _, _ = temp_dirs

    from core.memory.rag.store import ChromaVectorStore

    vectordb_dir = anima_dir.parent.parent / "vectordb"
    vectordb_dir.mkdir(parents=True, exist_ok=True)

    return ChromaVectorStore(persist_dir=vectordb_dir)


# ── Test 1: Shared Knowledge Indexing and Retrieval ─────────────────


def test_shared_knowledge_indexing_and_retrieval(temp_dirs, vector_store):
    """Index common_knowledge with collection_prefix='shared' and retrieve results.

    Verifies:
    - MemoryIndexer with collection_prefix='shared' creates shared_common_knowledge collection
    - Indexed documents are retrievable via MemoryRetriever.search(include_shared=True)
    - Shared results appear in merged results alongside personal results
    """
    anima_dir, common_knowledge_dir, data_dir = temp_dirs

    from core.memory.rag.indexer import MemoryIndexer
    from core.memory.rag.retriever import MemoryRetriever

    # 1. Index personal knowledge
    personal_indexer = MemoryIndexer(
        vector_store, "test_anima", anima_dir,
    )
    personal_chunks = personal_indexer.index_directory(
        anima_dir / "knowledge", "knowledge",
    )
    assert personal_chunks > 0, "Personal knowledge should produce chunks"

    # 2. Index shared (common_knowledge) — uses data_dir as anima_dir
    #    so that file_path.relative_to(anima_dir) resolves correctly.
    shared_indexer = MemoryIndexer(
        vector_store,
        "shared",
        data_dir,
        collection_prefix="shared",
    )
    shared_chunks = shared_indexer.index_directory(
        common_knowledge_dir, "common_knowledge",
    )
    assert shared_chunks > 0, "Common knowledge should produce chunks"

    # 3. Verify shared_common_knowledge collection was created
    collections = vector_store.list_collections()
    assert "shared_common_knowledge" in collections, (
        f"Expected 'shared_common_knowledge' in {collections}"
    )
    assert "test_anima_knowledge" in collections, (
        f"Expected 'test_anima_knowledge' in {collections}"
    )

    # 4. Create retriever and search with include_shared=True
    retriever = MemoryRetriever(
        vector_store,
        personal_indexer,
        anima_dir / "knowledge",
    )

    results = retriever.search(
        query="コーディング規約",
        anima_name="test_anima",
        memory_type="knowledge",
        top_k=5,
        include_shared=True,
    )

    assert len(results) > 0, "Search should return results"

    # 5. Verify shared results appear in results (check metadata anima=shared)
    shared_results = [
        r for r in results
        if r.metadata.get("anima") == "shared"
    ]
    assert len(shared_results) > 0, (
        "Shared common_knowledge results should appear in merged results"
    )


# ── Test 2: Hybrid Search (keyword + vector) ───────────────────────


def test_hybrid_search_common_knowledge(temp_dirs, vector_store, monkeypatch):
    """Search common_knowledge via MemoryManager.search_memory_text with scope='common_knowledge'.

    Verifies:
    - Keyword search finds matches in common_knowledge directory
    - Vector search augments keyword results when RAG is available
    - scope='common_knowledge' correctly targets common_knowledge dir
    """
    anima_dir, common_knowledge_dir, data_dir = temp_dirs

    from core.memory.rag.indexer import MemoryIndexer

    # Index personal knowledge
    personal_indexer = MemoryIndexer(
        vector_store, "test_anima", anima_dir,
    )
    personal_indexer.index_directory(anima_dir / "knowledge", "knowledge")

    # Index shared common_knowledge
    shared_indexer = MemoryIndexer(
        vector_store,
        "shared",
        data_dir,
        collection_prefix="shared",
    )
    shared_indexer.index_directory(common_knowledge_dir, "common_knowledge")

    # Patch get_common_knowledge_dir to point to our temp dir
    monkeypatch.setattr(
        "core.memory.manager.get_common_knowledge_dir",
        lambda: common_knowledge_dir,
    )
    # Patch get_shared_dir / get_company_dir / get_common_skills_dir for MemoryManager init
    monkeypatch.setattr(
        "core.memory.manager.get_company_dir",
        lambda: data_dir / "company",
    )
    monkeypatch.setattr(
        "core.memory.manager.get_common_skills_dir",
        lambda: data_dir / "common_skills",
    )
    monkeypatch.setattr(
        "core.memory.manager.get_shared_dir",
        lambda: data_dir / "shared",
    )

    # Ensure dirs exist for MemoryManager init
    (data_dir / "company").mkdir(exist_ok=True)
    (data_dir / "common_skills").mkdir(exist_ok=True)
    (data_dir / "shared").mkdir(exist_ok=True)

    from core.memory.manager import MemoryManager

    mm = MemoryManager(anima_dir, base_dir=data_dir)

    # Inject our pre-built indexer so MemoryManager uses the temp vector store
    mm._indexer = personal_indexer

    # Keyword search with scope="common_knowledge"
    results = mm.search_memory_text("勤怠ルール", scope="common_knowledge")
    assert len(results) > 0, (
        "Keyword search for '勤怠ルール' should find matches in common_knowledge"
    )

    # Verify results come from common_knowledge files
    filenames = [fname for fname, _ in results]
    assert any(
        "company-handbook" in fname for fname in filenames
    ), f"Expected company-handbook.md in results, got: {filenames}"


def test_hybrid_search_scope_all_includes_common(temp_dirs, vector_store, monkeypatch):
    """Scope 'all' also includes common_knowledge results.

    Verifies that searching with scope='all' returns results from
    both personal knowledge and common_knowledge directories.
    """
    anima_dir, common_knowledge_dir, data_dir = temp_dirs

    # Patch path helpers
    monkeypatch.setattr(
        "core.memory.manager.get_common_knowledge_dir",
        lambda: common_knowledge_dir,
    )
    monkeypatch.setattr(
        "core.memory.manager.get_company_dir",
        lambda: data_dir / "company",
    )
    monkeypatch.setattr(
        "core.memory.manager.get_common_skills_dir",
        lambda: data_dir / "common_skills",
    )
    monkeypatch.setattr(
        "core.memory.manager.get_shared_dir",
        lambda: data_dir / "shared",
    )

    (data_dir / "company").mkdir(exist_ok=True)
    (data_dir / "common_skills").mkdir(exist_ok=True)
    (data_dir / "shared").mkdir(exist_ok=True)

    from core.memory.manager import MemoryManager

    mm = MemoryManager(anima_dir, base_dir=data_dir)
    # Disable RAG indexer to test pure keyword search
    mm._indexer = None

    # Search with scope="all" — should include common_knowledge
    results = mm.search_memory_text("経費精算", scope="all")
    assert len(results) > 0, (
        "Scope 'all' should find '経費精算' in common_knowledge"
    )

    filenames = [fname for fname, _ in results]
    assert any("company-handbook" in fname for fname in filenames)


# ── Test 3: Priming with Shared Knowledge ──────────────────────────


@pytest.mark.asyncio
async def test_priming_with_shared_knowledge(temp_dirs, vector_store, monkeypatch):
    """PrimingEngine.prime_memories includes shared common_knowledge results.

    Verifies:
    - Channel C searches both personal and shared knowledge (include_shared=True)
    - related_knowledge in PrimingResult contains shared results
    - [shared] label appears in the formatted output
    """
    anima_dir, common_knowledge_dir, data_dir = temp_dirs

    from core.memory.rag.indexer import MemoryIndexer
    from core.memory.rag.retriever import MemoryRetriever
    from core.memory.priming import PrimingEngine, format_priming_section

    # Index personal knowledge
    personal_indexer = MemoryIndexer(
        vector_store, "test_anima", anima_dir,
    )
    personal_indexer.index_directory(anima_dir / "knowledge", "knowledge")

    # Index shared common_knowledge
    shared_indexer = MemoryIndexer(
        vector_store,
        "shared",
        data_dir,
        collection_prefix="shared",
    )
    shared_indexer.index_directory(common_knowledge_dir, "common_knowledge")

    # Create priming engine
    engine = PrimingEngine(anima_dir)

    # Inject retriever that uses our temp vector store
    engine._retriever = MemoryRetriever(
        vector_store,
        personal_indexer,
        anima_dir / "knowledge",
    )

    # Patch path helpers — priming.py uses local imports from core.paths
    monkeypatch.setattr(
        "core.paths.get_shared_dir",
        lambda: data_dir / "shared",
    )
    monkeypatch.setattr(
        "core.paths.get_common_skills_dir",
        lambda: data_dir / "common_skills",
    )
    (data_dir / "shared" / "users").mkdir(parents=True, exist_ok=True)
    (data_dir / "common_skills").mkdir(exist_ok=True)

    # Prime with a query that should match shared knowledge
    result = await engine.prime_memories(
        message="Pythonのコーディング規約について教えてください",
        sender_name="tester",
    )

    # related_knowledge should contain shared results
    assert result.related_knowledge, (
        "Priming should return related_knowledge from shared collection"
    )

    # Check that [shared] label appears in the output
    assert "[shared]" in result.related_knowledge, (
        f"Expected [shared] label in related_knowledge, got:\n{result.related_knowledge}"
    )

    # Format the priming section and verify structure
    formatted = format_priming_section(result, sender_name="tester")
    assert "関連する知識" in formatted, "Formatted output should contain knowledge section"


@pytest.mark.asyncio
async def test_priming_personal_and_shared_merged(temp_dirs, vector_store, monkeypatch):
    """Priming merges both personal and shared results by score.

    Verifies that results from both collections appear when the query
    matches content in both personal knowledge and common_knowledge.
    """
    anima_dir, common_knowledge_dir, data_dir = temp_dirs

    from core.memory.rag.indexer import MemoryIndexer
    from core.memory.rag.retriever import MemoryRetriever
    from core.memory.priming import PrimingEngine

    # Index personal knowledge
    personal_indexer = MemoryIndexer(
        vector_store, "test_anima", anima_dir,
    )
    personal_indexer.index_directory(anima_dir / "knowledge", "knowledge")

    # Index shared common_knowledge
    shared_indexer = MemoryIndexer(
        vector_store,
        "shared",
        data_dir,
        collection_prefix="shared",
    )
    shared_indexer.index_directory(common_knowledge_dir, "common_knowledge")

    # Create priming engine with injected retriever
    engine = PrimingEngine(anima_dir)
    engine._retriever = MemoryRetriever(
        vector_store,
        personal_indexer,
        anima_dir / "knowledge",
    )

    monkeypatch.setattr(
        "core.paths.get_shared_dir",
        lambda: data_dir / "shared",
    )
    monkeypatch.setattr(
        "core.paths.get_common_skills_dir",
        lambda: data_dir / "common_skills",
    )
    (data_dir / "shared" / "users").mkdir(parents=True, exist_ok=True)
    (data_dir / "common_skills").mkdir(exist_ok=True)

    # Query that should match both personal ("プロジェクト") and shared ("コーディング")
    result = await engine.prime_memories(
        message="プロジェクトアルファのコーディングルールについて",
        sender_name="tester",
    )

    assert result.related_knowledge, "Priming should return related knowledge"

    # Verify both [personal] and [shared] labels appear
    has_personal = "[personal]" in result.related_knowledge
    has_shared = "[shared]" in result.related_knowledge
    assert has_personal or has_shared, (
        "Priming output should contain at least one labeled result. "
        f"Got:\n{result.related_knowledge}"
    )


# ── Test 4: read_memory_file with common_knowledge prefix ──────────


def test_read_memory_file_common_knowledge_prefix(temp_dirs, monkeypatch):
    """ToolHandler.read_memory_file resolves common_knowledge/ prefix to shared dir.

    Verifies:
    - path="common_knowledge/test-file.md" reads from common_knowledge dir
    - File content is returned correctly
    - Non-existent files return 'File not found' message
    """
    anima_dir, common_knowledge_dir, data_dir = temp_dirs

    # Patch get_common_knowledge_dir — used by both ToolHandler (local import
    # from core.paths) and MemoryManager (module-level import).
    monkeypatch.setattr(
        "core.paths.get_common_knowledge_dir",
        lambda: common_knowledge_dir,
    )
    monkeypatch.setattr(
        "core.memory.manager.get_common_knowledge_dir",
        lambda: common_knowledge_dir,
    )
    monkeypatch.setattr(
        "core.memory.manager.get_company_dir",
        lambda: data_dir / "company",
    )
    monkeypatch.setattr(
        "core.memory.manager.get_common_skills_dir",
        lambda: data_dir / "common_skills",
    )
    monkeypatch.setattr(
        "core.memory.manager.get_shared_dir",
        lambda: data_dir / "shared",
    )

    (data_dir / "company").mkdir(exist_ok=True)
    (data_dir / "common_skills").mkdir(exist_ok=True)
    (data_dir / "shared").mkdir(exist_ok=True)

    from core.memory.manager import MemoryManager
    from core.tooling.handler import ToolHandler

    memory = MemoryManager(anima_dir, base_dir=data_dir)
    handler = ToolHandler(anima_dir, memory)

    # Read a file via common_knowledge/ prefix
    result = handler.handle(
        "read_memory_file",
        {"path": "common_knowledge/test-file.md"},
    )

    assert "Test File" in result, (
        f"Expected 'Test File' in result, got: {result}"
    )
    assert "sample content for verification" in result

    # Read a file via common_knowledge/ prefix — company handbook
    result2 = handler.handle(
        "read_memory_file",
        {"path": "common_knowledge/company-handbook.md"},
    )
    assert "会社ハンドブック" in result2

    # Non-existent file
    result3 = handler.handle(
        "read_memory_file",
        {"path": "common_knowledge/nonexistent.md"},
    )
    assert "not found" in result3.lower(), (
        f"Expected 'not found' for missing file, got: {result3}"
    )


def test_read_memory_file_personal_path_still_works(temp_dirs, monkeypatch):
    """ToolHandler.read_memory_file still works for personal (non-prefixed) paths.

    Verifies that the common_knowledge/ prefix logic does not break
    the default behavior of reading relative to anima_dir.
    """
    anima_dir, common_knowledge_dir, data_dir = temp_dirs

    monkeypatch.setattr(
        "core.memory.manager.get_common_knowledge_dir",
        lambda: common_knowledge_dir,
    )
    monkeypatch.setattr(
        "core.memory.manager.get_company_dir",
        lambda: data_dir / "company",
    )
    monkeypatch.setattr(
        "core.memory.manager.get_common_skills_dir",
        lambda: data_dir / "common_skills",
    )
    monkeypatch.setattr(
        "core.memory.manager.get_shared_dir",
        lambda: data_dir / "shared",
    )

    (data_dir / "company").mkdir(exist_ok=True)
    (data_dir / "common_skills").mkdir(exist_ok=True)
    (data_dir / "shared").mkdir(exist_ok=True)

    from core.memory.manager import MemoryManager
    from core.tooling.handler import ToolHandler

    memory = MemoryManager(anima_dir, base_dir=data_dir)
    handler = ToolHandler(anima_dir, memory)

    # Read personal knowledge file (no common_knowledge/ prefix)
    result = handler.handle(
        "read_memory_file",
        {"path": "knowledge/internal-policy.md"},
    )

    assert "社内ポリシー" in result, (
        f"Expected personal knowledge content, got: {result}"
    )
