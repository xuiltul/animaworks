from __future__ import annotations
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for knowledge failure tracking (success_count, failure_count,
version, last_used) — Issue: memory-system-symmetry-and-skill-vector-search.

Tests cover:
- Knowledge metadata creation via consolidation
- update_knowledge_metadata helper
- report_knowledge_outcome handler
- Knowledge protection in forgetting
- Knowledge reconsolidation targets
- Contradiction-driven failure_count increment
- Contradiction history JSONL persistence
- Backward compatibility with legacy knowledge files
"""

import json
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ── Fixtures ────────────────────────────────────────────────────


@pytest.fixture
def anima_dir(tmp_path: Path) -> Path:
    """Minimal anima directory with knowledge subdirectory."""
    d = tmp_path / "animas" / "test_anima"
    for sub in ("knowledge", "episodes", "skills", "procedures", "state",
                "shortterm", "activity_log", "archive"):
        (d / sub).mkdir(parents=True)
    return d


@pytest.fixture
def shared_dir(tmp_path: Path) -> Path:
    """Shared directory for contradiction history."""
    d = tmp_path / "shared"
    d.mkdir(parents=True)
    return d


def _write_knowledge(
    anima_dir: Path,
    name: str,
    content: str,
    meta: dict | None = None,
) -> Path:
    """Helper: write a knowledge file with optional YAML frontmatter."""
    import yaml

    path = anima_dir / "knowledge" / name
    if meta:
        fm = yaml.dump(meta, default_flow_style=False, allow_unicode=True)
        path.write_text(f"---\n{fm}---\n\n{content}", encoding="utf-8")
    else:
        path.write_text(content, encoding="utf-8")
    return path


# ── 1. MemoryManager helpers ───────────────────────────────────


class TestMemoryManagerKnowledgeHelpers:

    def test_read_knowledge_metadata_with_tracking_fields(self, anima_dir: Path) -> None:
        from core.memory.manager import MemoryManager

        _write_knowledge(anima_dir, "topic.md", "body", {
            "confidence": 0.7,
            "success_count": 3,
            "failure_count": 1,
            "version": 2,
            "last_used": "2026-02-19T10:00:00",
        })

        mm = MemoryManager(anima_dir)
        meta = mm.read_knowledge_metadata(anima_dir / "knowledge" / "topic.md")

        assert meta["success_count"] == 3
        assert meta["failure_count"] == 1
        assert meta["version"] == 2
        assert meta["last_used"] == "2026-02-19T10:00:00"

    def test_read_knowledge_metadata_defaults_for_legacy_files(self, anima_dir: Path) -> None:
        """Legacy files without tracking fields should return empty dict (no crash)."""
        from core.memory.manager import MemoryManager

        _write_knowledge(anima_dir, "legacy.md", "old content")
        mm = MemoryManager(anima_dir)
        meta = mm.read_knowledge_metadata(anima_dir / "knowledge" / "legacy.md")
        assert meta.get("success_count", 0) == 0
        assert meta.get("failure_count", 0) == 0

    def test_update_knowledge_metadata_merges(self, anima_dir: Path) -> None:
        from core.memory.manager import MemoryManager

        _write_knowledge(anima_dir, "topic.md", "body text", {
            "confidence": 0.7,
            "success_count": 0,
            "failure_count": 0,
            "version": 1,
        })
        mm = MemoryManager(anima_dir)
        target = anima_dir / "knowledge" / "topic.md"
        mm.update_knowledge_metadata(target, {"success_count": 5, "last_used": "now"})

        meta = mm.read_knowledge_metadata(target)
        assert meta["success_count"] == 5
        assert meta["last_used"] == "now"
        assert meta["confidence"] == 0.7  # unchanged

    def test_update_knowledge_metadata_relative_path(self, anima_dir: Path) -> None:
        from core.memory.manager import MemoryManager

        _write_knowledge(anima_dir, "rel.md", "body", {"confidence": 0.5})
        mm = MemoryManager(anima_dir)
        mm.update_knowledge_metadata(Path("rel.md"), {"failure_count": 3})

        meta = mm.read_knowledge_metadata(anima_dir / "knowledge" / "rel.md")
        assert meta["failure_count"] == 3


# ── 2. report_knowledge_outcome handler ────────────────────────


class TestReportKnowledgeOutcome:

    def _make_handler(self, anima_dir: Path):
        """Create a minimal ToolHandler for testing."""
        from core.memory.activity import ActivityLogger
        from core.memory.manager import MemoryManager
        from core.tooling.handler import ToolHandler

        mm = MemoryManager(anima_dir)
        handler = ToolHandler.__new__(ToolHandler)
        handler._anima_dir = anima_dir
        handler._memory = mm
        handler._model_config = MagicMock()
        handler._activity = ActivityLogger(anima_dir)
        return handler

    def test_success_increments_count(self, anima_dir: Path) -> None:
        _write_knowledge(anima_dir, "topic.md", "body", {
            "confidence": 0.7,
            "success_count": 0,
            "failure_count": 0,
            "version": 1,
            "last_used": "",
        })
        handler = self._make_handler(anima_dir)
        result = handler._handle_report_knowledge_outcome({
            "path": "knowledge/topic.md",
            "success": True,
        })
        assert "成功" in result

        from core.memory.manager import MemoryManager
        meta = MemoryManager(anima_dir).read_knowledge_metadata(
            anima_dir / "knowledge" / "topic.md",
        )
        assert meta["success_count"] == 1
        assert meta["failure_count"] == 0
        assert meta["confidence"] == 1.0
        assert meta["last_used"] != ""

    def test_failure_increments_and_recalculates(self, anima_dir: Path) -> None:
        _write_knowledge(anima_dir, "topic.md", "body", {
            "confidence": 0.7,
            "success_count": 1,
            "failure_count": 0,
            "version": 1,
            "last_used": "",
        })
        handler = self._make_handler(anima_dir)
        result = handler._handle_report_knowledge_outcome({
            "path": "knowledge/topic.md",
            "success": False,
        })
        assert "失敗" in result

        from core.memory.manager import MemoryManager
        meta = MemoryManager(anima_dir).read_knowledge_metadata(
            anima_dir / "knowledge" / "topic.md",
        )
        assert meta["success_count"] == 1
        assert meta["failure_count"] == 1
        assert meta["confidence"] == 0.5

    def test_missing_path_returns_error(self, anima_dir: Path) -> None:
        handler = self._make_handler(anima_dir)
        result = handler._handle_report_knowledge_outcome({"path": "", "success": True})
        assert "Error" in result or "error" in result.lower()

    def test_nonexistent_file_returns_error(self, anima_dir: Path) -> None:
        handler = self._make_handler(anima_dir)
        result = handler._handle_report_knowledge_outcome({
            "path": "knowledge/nonexistent.md",
            "success": True,
        })
        assert "FileNotFound" in result or "not found" in result.lower()


# ── 3. Forgetting protection ──────────────────────────────────


class TestKnowledgeForgettingProtection:

    def test_success_count_2_is_protected(self, anima_dir: Path) -> None:
        from core.memory.forgetting import ForgettingEngine

        engine = ForgettingEngine(anima_dir, "test_anima")
        assert engine._is_protected_knowledge({"success_count": 2}) is True

    def test_success_count_1_is_not_protected(self, anima_dir: Path) -> None:
        from core.memory.forgetting import ForgettingEngine

        engine = ForgettingEngine(anima_dir, "test_anima")
        assert engine._is_protected_knowledge({"success_count": 1}) is False

    def test_important_tag_is_protected(self, anima_dir: Path) -> None:
        from core.memory.forgetting import ForgettingEngine

        engine = ForgettingEngine(anima_dir, "test_anima")
        assert engine._is_protected_knowledge({"importance": "important"}) is True

    def test_knowledge_type_routes_to_protection(self, anima_dir: Path) -> None:
        from core.memory.forgetting import ForgettingEngine

        engine = ForgettingEngine(anima_dir, "test_anima")
        assert engine._is_protected({
            "memory_type": "knowledge",
            "success_count": 5,
        }) is True

    def test_knowledge_no_protection_by_default(self, anima_dir: Path) -> None:
        from core.memory.forgetting import ForgettingEngine

        engine = ForgettingEngine(anima_dir, "test_anima")
        assert engine._is_protected({
            "memory_type": "knowledge",
            "success_count": 0,
        }) is False

    def test_success_count_string_coercion(self, anima_dir: Path) -> None:
        """ChromaDB may return metadata values as strings."""
        from core.memory.forgetting import ForgettingEngine

        engine = ForgettingEngine(anima_dir, "test_anima")
        assert engine._is_protected_knowledge({"success_count": "3"}) is True


# ── 4. Reconsolidation targets ─────────────────────────────────


class TestKnowledgeReconsolidation:

    @pytest.fixture
    def engine(self, anima_dir: Path):
        from core.memory.activity import ActivityLogger
        from core.memory.manager import MemoryManager
        from core.memory.reconsolidation import ReconsolidationEngine

        mm = MemoryManager(anima_dir)
        al = ActivityLogger(anima_dir)
        return ReconsolidationEngine(
            anima_dir, "test_anima",
            memory_manager=mm, activity_logger=al,
        )

    @pytest.mark.asyncio
    async def test_finds_targets_with_high_failure_low_confidence(
        self, anima_dir: Path, engine,
    ) -> None:
        _write_knowledge(anima_dir, "bad.md", "bad content", {
            "failure_count": 3,
            "confidence": 0.3,
            "success_count": 0,
            "version": 1,
        })
        _write_knowledge(anima_dir, "good.md", "good content", {
            "failure_count": 0,
            "confidence": 0.9,
            "success_count": 5,
            "version": 1,
        })

        targets = await engine.find_knowledge_reconsolidation_targets()
        names = [t.name for t in targets]
        assert "bad.md" in names
        assert "good.md" not in names

    @pytest.mark.asyncio
    async def test_ignores_below_threshold(self, anima_dir: Path, engine) -> None:
        _write_knowledge(anima_dir, "borderline.md", "content", {
            "failure_count": 1,
            "confidence": 0.5,
            "success_count": 1,
            "version": 1,
        })

        targets = await engine.find_knowledge_reconsolidation_targets()
        assert len(targets) == 0

    @pytest.mark.asyncio
    async def test_defaults_when_no_metadata(self, anima_dir: Path, engine) -> None:
        """Legacy files without frontmatter should not be targets."""
        _write_knowledge(anima_dir, "legacy.md", "no frontmatter")

        targets = await engine.find_knowledge_reconsolidation_targets()
        assert len(targets) == 0


# ── 5. Contradiction failure_count increment ───────────────────


class TestContradictionFailureIncrement:

    def test_increment_failure_count(self, anima_dir: Path) -> None:
        from core.memory.contradiction import ContradictionDetector

        _write_knowledge(anima_dir, "old.md", "old info", {
            "confidence": 0.7,
            "success_count": 1,
            "failure_count": 0,
            "version": 1,
        })

        detector = ContradictionDetector(anima_dir, "test_anima")
        detector._increment_failure_count(anima_dir / "knowledge" / "old.md")

        from core.memory.manager import MemoryManager
        meta = MemoryManager(anima_dir).read_knowledge_metadata(
            anima_dir / "knowledge" / "old.md",
        )
        assert meta["failure_count"] == 1
        assert meta["confidence"] == 0.5  # 1/(1+1) = 0.5

    def test_increment_missing_file_no_crash(self, anima_dir: Path) -> None:
        from core.memory.contradiction import ContradictionDetector

        detector = ContradictionDetector(anima_dir, "test_anima")
        # Should not raise — gracefully skips
        detector._increment_failure_count(anima_dir / "knowledge" / "gone.md")

    def test_increment_no_frontmatter(self, anima_dir: Path) -> None:
        from core.memory.contradiction import ContradictionDetector

        _write_knowledge(anima_dir, "plain.md", "no meta")
        detector = ContradictionDetector(anima_dir, "test_anima")
        detector._increment_failure_count(anima_dir / "knowledge" / "plain.md")

        from core.memory.manager import MemoryManager
        meta = MemoryManager(anima_dir).read_knowledge_metadata(
            anima_dir / "knowledge" / "plain.md",
        )
        assert meta["failure_count"] == 1
        assert meta["confidence"] == 0.0  # 0/(0+1) = 0


# ── 6. Contradiction history JSONL persistence ─────────────────


class TestContradictionHistoryPersistence:

    def test_persist_creates_jsonl_entry(
        self, anima_dir: Path, shared_dir: Path, monkeypatch,
    ) -> None:
        from core.memory.contradiction import ContradictionDetector, ContradictionPair

        monkeypatch.setattr(
            "core.paths.get_shared_dir",
            lambda: shared_dir,
        )

        detector = ContradictionDetector(anima_dir, "test_anima")
        pair = ContradictionPair(
            file_a=Path("newer.md"),
            file_b=Path("older.md"),
            text_a="new text",
            text_b="old text",
            confidence=0.85,
            resolution="supersede",
            reason="newer info",
            merged_content=None,
        )
        detector._persist_contradiction_history(pair, "supersede")

        history = shared_dir / "contradiction_history.jsonl"
        assert history.exists()

        entries = [json.loads(line) for line in history.read_text().strip().split("\n")]
        assert len(entries) == 1
        entry = entries[0]
        assert entry["anima"] == "test_anima"
        assert entry["file_a"] == "newer.md"
        assert entry["file_b"] == "older.md"
        assert entry["confidence"] == 0.85
        assert entry["resolution"] == "supersede"
        assert entry["reason"] == "newer info"
        assert entry["merged_content"] is None
        assert "ts" in entry
        # Verify ts is valid ISO8601
        datetime.fromisoformat(entry["ts"])

    def test_persist_appends_multiple_entries(
        self, anima_dir: Path, shared_dir: Path, monkeypatch,
    ) -> None:
        from core.memory.contradiction import ContradictionDetector, ContradictionPair

        monkeypatch.setattr(
            "core.paths.get_shared_dir",
            lambda: shared_dir,
        )

        detector = ContradictionDetector(anima_dir, "test_anima")
        for i in range(3):
            pair = ContradictionPair(
                file_a=Path(f"a{i}.md"),
                file_b=Path(f"b{i}.md"),
                text_a="",
                text_b="",
                confidence=0.9,
                resolution="coexist",
                reason="context-dependent",
            )
            detector._persist_contradiction_history(pair, "coexist")

        history = shared_dir / "contradiction_history.jsonl"
        entries = [json.loads(line) for line in history.read_text().strip().split("\n")]
        assert len(entries) == 3


# ── 7. ChromaDB metadata extraction ───────────────────────────


class TestIndexerMetadataExtraction:

    def test_extracts_failure_tracking_fields(self, tmp_path: Path) -> None:
        """Verify that indexing picks up the new tracking fields in ChromaDB."""
        pytest.importorskip("chromadb")
        pytest.importorskip("sentence_transformers")

        # Set up a full data dir structure for model loading
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        (data_dir / "models").mkdir()
        import os
        old = os.environ.get("ANIMAWORKS_DATA_DIR")
        os.environ["ANIMAWORKS_DATA_DIR"] = str(data_dir)
        try:
            from core.paths import _prompt_cache
            _prompt_cache.clear()

            a_dir = data_dir / "animas" / "idx_test"
            (a_dir / "knowledge").mkdir(parents=True)
            (a_dir / "vectordb").mkdir()

            _write_knowledge(a_dir, "tracked.md", "# Topic\n\nContent body text here for indexing.\n\n## Details\n\nThis is a longer knowledge file with enough content to be indexed by the chunking algorithm. It contains multiple paragraphs and sections to ensure the text passes minimum length thresholds for embedding generation. The topic covers important policy information that should be searchable via vector similarity.\n\n## Additional Context\n\nMore information about the policy including background context and related references that provide depth to the knowledge entry.\n", {
                "confidence": 0.8,
                "success_count": 5,
                "failure_count": 2,
                "version": 3,
                "last_used": "2026-02-19T12:00:00",
            })

            from core.memory.rag.indexer import MemoryIndexer
            from core.memory.rag.store import ChromaVectorStore

            store = ChromaVectorStore(persist_dir=a_dir / "vectordb")
            indexer = MemoryIndexer(store, "idx_test", a_dir)
            total = indexer.index_directory(a_dir / "knowledge", "knowledge")
            assert total > 0

            coll = store.client.get_collection(name="idx_test_knowledge")
            all_data = coll.get(include=["metadatas"])
            found = False
            for meta in all_data["metadatas"]:
                if "tracked" in meta.get("source_file", ""):
                    assert meta.get("success_count") == 5
                    assert meta.get("failure_count") == 2
                    assert meta.get("version") == 3
                    assert meta.get("confidence") == 0.8
                    assert meta.get("last_used") == "2026-02-19T12:00:00"
                    found = True
                    break
            assert found, "tracked.md metadata not found in ChromaDB"
        finally:
            if old is not None:
                os.environ["ANIMAWORKS_DATA_DIR"] = old
            else:
                os.environ.pop("ANIMAWORKS_DATA_DIR", None)
            _prompt_cache.clear()
