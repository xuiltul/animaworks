from __future__ import annotations
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""E2E tests for memory system symmetry improvements.

Tests the full integration of:
1. Knowledge failure tracking (success_count/failure_count lifecycle)
2. Channel D vector search (semantic skill matching)
3. Contradiction history persistence + failure_count auto-increment

Requires ChromaDB and sentence-transformers.
"""

import json
from datetime import datetime
from pathlib import Path

import pytest
import yaml

chromadb = pytest.importorskip(
    "chromadb",
    reason="ChromaDB not installed. Install with: pip install 'animaworks[rag]'",
)
sentence_transformers = pytest.importorskip(
    "sentence_transformers",
    reason="sentence-transformers not installed. Install with: pip install 'animaworks[rag]'",
)


# ── Fixtures ────────────────────────────────────────────────────


@pytest.fixture
def anima_dir(tmp_path, monkeypatch):
    """Isolated anima directory with all required subdirectories."""
    data_dir = tmp_path / ".animaworks"
    data_dir.mkdir()
    (data_dir / "models").mkdir()
    (data_dir / "shared" / "users").mkdir(parents=True)
    (data_dir / "common_skills").mkdir()
    (data_dir / "common_knowledge").mkdir()

    monkeypatch.setenv("ANIMAWORKS_DATA_DIR", str(data_dir))

    from core.paths import _prompt_cache
    _prompt_cache.clear()

    anima = data_dir / "animas" / "test_anima"
    anima.mkdir(parents=True)

    for sub in ("knowledge", "episodes", "skills", "procedures", "state",
                "vectordb", "shortterm", "activity_log", "archive"):
        (anima / sub).mkdir()

    yield anima


def _write_knowledge(anima_dir: Path, name: str, body: str, meta: dict | None = None) -> Path:
    """Write a knowledge file with YAML frontmatter."""
    path = anima_dir / "knowledge" / name
    if meta:
        fm = yaml.dump(meta, default_flow_style=False, allow_unicode=True)
        path.write_text(f"---\n{fm}---\n\n{body}", encoding="utf-8")
    else:
        path.write_text(body, encoding="utf-8")
    return path


def _write_skill(base_dir: Path, name: str, description: str, body: str) -> Path:
    """Write a skill file with YAML frontmatter."""
    path = base_dir / name
    path.write_text(
        f"---\ndescription: {description}\ntags: [test]\n---\n\n{body}",
        encoding="utf-8",
    )
    return path


# ── Test 1: Knowledge File Lifecycle ────────────────────────────


def test_knowledge_lifecycle_report_and_protection(anima_dir):
    """Full lifecycle: create → report success → verify protection from forgetting."""
    from core.memory.activity import ActivityLogger
    from core.memory.forgetting import ForgettingEngine
    from core.memory.manager import MemoryManager
    from core.memory.rag.indexer import MemoryIndexer
    from core.memory.rag.store import ChromaVectorStore
    from core.tooling.handler import ToolHandler

    # Step 1: Create knowledge with tracking metadata
    _write_knowledge(anima_dir, "policy.md", "# Policy\n\nImportant rules for deployment and operations.\n\n## Details\n\nThis policy document covers the comprehensive guidelines for system administration, security protocols, and operational procedures. All team members must follow these rules. The policy includes multiple sections for adequate indexing length.\n\n## Security\n\nAll access must be authenticated and authorized.\n", {
        "confidence": 0.7,
        "success_count": 0,
        "failure_count": 0,
        "version": 1,
        "last_used": "",
    })

    # Step 2: Report success twice via handler
    mm = MemoryManager(anima_dir)
    handler = ToolHandler.__new__(ToolHandler)
    handler._anima_dir = anima_dir
    handler._memory = mm
    from unittest.mock import MagicMock
    handler._model_config = MagicMock()
    handler._activity = ActivityLogger(anima_dir)

    for _ in range(2):
        result = handler._handle_report_knowledge_outcome({
            "path": "knowledge/policy.md",
            "success": True,
        })
        assert "成功" in result

    # Step 3: Verify metadata
    meta = mm.read_knowledge_metadata(anima_dir / "knowledge" / "policy.md")
    assert meta["success_count"] == 2
    assert meta["failure_count"] == 0
    assert meta["confidence"] == 1.0
    assert meta["last_used"] != ""

    # Step 4: Index and verify ChromaDB metadata
    store = ChromaVectorStore(persist_dir=anima_dir / "vectordb")
    indexer = MemoryIndexer(store, "test_anima", anima_dir)
    total = indexer.index_directory(anima_dir / "knowledge", "knowledge")
    assert total > 0

    coll = store.client.get_collection(name="test_anima_knowledge")
    all_data = coll.get(include=["metadatas"])
    for m in all_data["metadatas"]:
        if "policy" in m.get("source_file", ""):
            assert m.get("success_count") == 2

    # Step 5: Verify forgetting protection
    engine = ForgettingEngine(anima_dir, "test_anima")
    assert engine._is_protected_knowledge({"success_count": 2}) is True
    assert engine._is_protected({"memory_type": "knowledge", "success_count": 2}) is True


# ── Test 2: Knowledge Reconsolidation Target Detection ─────────


@pytest.mark.asyncio
async def test_reconsolidation_targets_e2e(anima_dir):
    """Files with failure_count >= 2 and confidence < 0.6 become targets."""
    from core.memory.activity import ActivityLogger
    from core.memory.manager import MemoryManager
    from core.memory.reconsolidation import ReconsolidationEngine

    # Create files with varying quality signals
    _write_knowledge(anima_dir, "failing.md", "Bad info.", {
        "confidence": 0.3,
        "success_count": 0,
        "failure_count": 3,
        "version": 1,
    })
    _write_knowledge(anima_dir, "healthy.md", "Good info.", {
        "confidence": 0.9,
        "success_count": 5,
        "failure_count": 0,
        "version": 1,
    })
    _write_knowledge(anima_dir, "borderline.md", "Okay info.", {
        "confidence": 0.55,
        "success_count": 1,
        "failure_count": 1,
        "version": 1,
    })

    mm = MemoryManager(anima_dir)
    al = ActivityLogger(anima_dir)
    engine = ReconsolidationEngine(
        anima_dir, "test_anima",
        memory_manager=mm, activity_logger=al,
    )

    targets = await engine.find_knowledge_reconsolidation_targets()
    names = [t.name for t in targets]

    assert "failing.md" in names
    assert "healthy.md" not in names
    assert "borderline.md" not in names


# ── Test 3: Contradiction History Full Pipeline ────────────────


def test_contradiction_history_and_failure_increment(anima_dir, monkeypatch):
    """Contradiction resolution persists history and increments failure_count."""
    from core.memory.contradiction import ContradictionDetector, ContradictionPair
    from core.memory.manager import MemoryManager

    # Set shared_dir to our test data_dir
    shared_dir = anima_dir.parent.parent / "shared"
    shared_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(
        "core.paths.get_shared_dir",
        lambda: shared_dir,
    )

    # Create two knowledge files
    _write_knowledge(anima_dir, "newer.md", "Updated policy.", {
        "confidence": 0.8,
        "success_count": 2,
        "failure_count": 0,
        "version": 1,
        "created_at": "2026-02-19",
    })
    _write_knowledge(anima_dir, "older.md", "Outdated policy.", {
        "confidence": 0.7,
        "success_count": 1,
        "failure_count": 0,
        "version": 1,
        "created_at": "2026-01-01",
    })

    detector = ContradictionDetector(anima_dir, "test_anima")
    pair = ContradictionPair(
        file_a=anima_dir / "knowledge" / "newer.md",
        file_b=anima_dir / "knowledge" / "older.md",
        text_a="Updated policy.",
        text_b="Outdated policy.",
        confidence=0.85,
        resolution="supersede",
        reason="Newer policy supersedes older",
    )

    # Simulate the resolution flow manually:
    # 1. Increment failure_count before resolution
    detector._increment_failure_count(pair.file_b)
    # 2. Persist history
    detector._persist_contradiction_history(pair, "supersede")

    # Verify failure_count was incremented on older file
    mm = MemoryManager(anima_dir)
    meta_old = mm.read_knowledge_metadata(anima_dir / "knowledge" / "older.md")
    assert meta_old["failure_count"] == 1
    assert meta_old["confidence"] == 0.5  # 1/(1+1) = 0.5

    # Verify newer file is unchanged
    meta_new = mm.read_knowledge_metadata(anima_dir / "knowledge" / "newer.md")
    assert meta_new["failure_count"] == 0
    assert meta_new["success_count"] == 2

    # Verify JSONL history
    history_path = shared_dir / "contradiction_history.jsonl"
    assert history_path.exists()

    entries = [json.loads(line) for line in history_path.read_text().strip().split("\n")]
    assert len(entries) == 1
    entry = entries[0]
    assert entry["anima"] == "test_anima"
    assert entry["file_a"] == "newer.md"
    assert entry["file_b"] == "older.md"
    assert entry["confidence"] == 0.85
    assert entry["resolution"] == "supersede"
    assert entry["reason"] == "Newer policy supersedes older"
    datetime.fromisoformat(entry["ts"])


# ── Test 4: Channel D Semantic Skill Matching ──────────────────


def test_channel_d_vector_search_integration(anima_dir, monkeypatch):
    """Channel D should find semantically related skills via vector search."""
    from core.memory.manager import MemoryManager
    from core.memory.priming import PrimingEngine
    from core.memory.rag.indexer import MemoryIndexer
    from core.memory.rag.retriever import MemoryRetriever
    from core.memory.rag.store import ChromaVectorStore

    # Create skills with descriptions
    skills_dir = anima_dir / "skills"
    _write_skill(skills_dir, "deploy_procedure.md",
                 "「デプロイ」「deploy」AWSへのサーバーデプロイ手順",
                 "# Deploy\n\n1. Build 2. Push 3. Deploy")
    _write_skill(skills_dir, "database_migration.md",
                 "「DB」「マイグレーション」データベースのスキーマ変更手順",
                 "# DB Migration\n\n1. Backup 2. Migrate 3. Verify")
    _write_skill(skills_dir, "meeting_notes.md",
                 "「会議」「議事録」会議メモの書き方",
                 "# Meeting Notes\n\n議事録テンプレート")

    # Index skills
    store = ChromaVectorStore(persist_dir=anima_dir / "vectordb")
    indexer = MemoryIndexer(store, "test_anima", anima_dir)
    indexer.index_directory(skills_dir, "skills")

    # Create retriever and priming engine
    retriever = MemoryRetriever(store, indexer, anima_dir / "knowledge")
    mm = MemoryManager(anima_dir)

    engine = PrimingEngine(anima_dir)
    engine._retriever = retriever
    engine._retriever_initialized = True

    # Test: keyword match should find deploy skill
    import asyncio
    result = asyncio.get_event_loop().run_until_complete(
        engine._channel_d_skill_match("サーバーをデプロイしたい", ["デプロイ"]),
    )
    assert "deploy_procedure" in result


# ── Test 5: Backward Compatibility ─────────────────────────────


def test_backward_compatibility_legacy_knowledge(anima_dir):
    """Knowledge files without new metadata fields should work without errors."""
    from core.memory.activity import ActivityLogger
    from core.memory.forgetting import ForgettingEngine
    from core.memory.manager import MemoryManager
    from core.tooling.handler import ToolHandler
    from unittest.mock import MagicMock

    # Legacy file: no tracking fields at all
    _write_knowledge(anima_dir, "legacy_no_meta.md", "Old content without frontmatter")

    # Legacy file: only confidence
    _write_knowledge(anima_dir, "legacy_partial.md", "Content with partial meta", {
        "confidence": 0.7,
        "created_at": "2026-01-01",
    })

    mm = MemoryManager(anima_dir)

    # Reading metadata should not crash
    meta1 = mm.read_knowledge_metadata(anima_dir / "knowledge" / "legacy_no_meta.md")
    assert meta1.get("success_count", 0) == 0
    assert meta1.get("failure_count", 0) == 0

    meta2 = mm.read_knowledge_metadata(anima_dir / "knowledge" / "legacy_partial.md")
    assert meta2["confidence"] == 0.7
    assert meta2.get("success_count", 0) == 0

    # report_knowledge_outcome should work on legacy files
    handler = ToolHandler.__new__(ToolHandler)
    handler._anima_dir = anima_dir
    handler._memory = mm
    handler._model_config = MagicMock()
    handler._activity = ActivityLogger(anima_dir)

    result = handler._handle_report_knowledge_outcome({
        "path": "knowledge/legacy_partial.md",
        "success": True,
    })
    assert "成功" in result

    meta_after = mm.read_knowledge_metadata(anima_dir / "knowledge" / "legacy_partial.md")
    assert meta_after["success_count"] == 1
    assert meta_after["confidence"] == 1.0

    # Forgetting protection with defaults
    engine = ForgettingEngine(anima_dir, "test_anima")
    assert engine._is_protected_knowledge({}) is False
    assert engine._is_protected_knowledge({"success_count": 0}) is False
