"""Unit tests for shared-index change detection in core/memory/rag_search.py."""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from core.memory.rag_search import (
    RAGMemorySearch,
    _compute_dir_hash,
    _read_shared_hash,
    _write_shared_hash,
)

# ── _compute_dir_hash ─────────────────────────────────────


class TestComputeDirHash:
    def test_empty_dir(self, tmp_path: Path) -> None:
        """Empty directory produces a deterministic hash."""
        h = _compute_dir_hash(tmp_path, "*.md")
        assert isinstance(h, str) and len(h) == 64

    def test_stable_for_same_content(self, tmp_path: Path) -> None:
        """Same files produce the same hash."""
        (tmp_path / "a.md").write_text("hello")
        h1 = _compute_dir_hash(tmp_path, "*.md")
        h2 = _compute_dir_hash(tmp_path, "*.md")
        assert h1 == h2

    def test_changes_when_file_added(self, tmp_path: Path) -> None:
        """Adding a file changes the hash."""
        (tmp_path / "a.md").write_text("hello")
        h1 = _compute_dir_hash(tmp_path, "*.md")
        (tmp_path / "b.md").write_text("world")
        h2 = _compute_dir_hash(tmp_path, "*.md")
        assert h1 != h2

    def test_changes_when_file_modified(self, tmp_path: Path) -> None:
        """Modifying a file changes the hash (mtime differs)."""
        f = tmp_path / "a.md"
        f.write_text("v1")
        h1 = _compute_dir_hash(tmp_path, "*.md")
        time.sleep(0.05)
        f.write_text("v2")
        h2 = _compute_dir_hash(tmp_path, "*.md")
        assert h1 != h2

    def test_changes_when_file_removed(self, tmp_path: Path) -> None:
        """Removing a file changes the hash."""
        (tmp_path / "a.md").write_text("x")
        (tmp_path / "b.md").write_text("y")
        h1 = _compute_dir_hash(tmp_path, "*.md")
        (tmp_path / "b.md").unlink()
        h2 = _compute_dir_hash(tmp_path, "*.md")
        assert h1 != h2

    def test_ignores_non_matching_files(self, tmp_path: Path) -> None:
        """Non-matching files don't affect the hash."""
        (tmp_path / "a.md").write_text("hello")
        h1 = _compute_dir_hash(tmp_path, "*.md")
        (tmp_path / "ignored.txt").write_text("no effect")
        h2 = _compute_dir_hash(tmp_path, "*.md")
        assert h1 == h2

    def test_includes_subdirectory_files(self, tmp_path: Path) -> None:
        """rglob picks up nested .md files."""
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "nested.md").write_text("deep")
        h1 = _compute_dir_hash(tmp_path, "*.md")
        assert isinstance(h1, str) and len(h1) == 64
        (sub / "another.md").write_text("more")
        h2 = _compute_dir_hash(tmp_path, "*.md")
        assert h1 != h2


# ── _read_shared_hash / _write_shared_hash ────────────────


class TestReadWriteSharedHash:
    def test_read_missing_file(self, tmp_path: Path) -> None:
        """Returns None when file doesn't exist."""
        assert _read_shared_hash(tmp_path / "nope.json", "key") is None

    def test_read_corrupted_json(self, tmp_path: Path) -> None:
        """Returns None on malformed JSON."""
        p = tmp_path / "meta.json"
        p.write_text("{invalid", encoding="utf-8")
        assert _read_shared_hash(p, "key") is None

    def test_read_missing_key(self, tmp_path: Path) -> None:
        """Returns None when key is absent."""
        p = tmp_path / "meta.json"
        p.write_text('{"other": "val"}', encoding="utf-8")
        assert _read_shared_hash(p, "key") is None

    def test_write_creates_file(self, tmp_path: Path) -> None:
        """Creates index_meta.json when it doesn't exist."""
        p = tmp_path / "meta.json"
        _write_shared_hash(p, "my_key", "abc123")
        data = json.loads(p.read_text(encoding="utf-8"))
        assert data["my_key"] == "abc123"

    def test_write_preserves_existing_keys(self, tmp_path: Path) -> None:
        """Writing a key preserves other keys."""
        p = tmp_path / "meta.json"
        p.write_text('{"existing": "keep"}', encoding="utf-8")
        _write_shared_hash(p, "new_key", "val")
        data = json.loads(p.read_text(encoding="utf-8"))
        assert data["existing"] == "keep"
        assert data["new_key"] == "val"

    def test_write_overwrites_existing_key(self, tmp_path: Path) -> None:
        """Overwriting a key updates its value."""
        p = tmp_path / "meta.json"
        _write_shared_hash(p, "k", "old")
        _write_shared_hash(p, "k", "new")
        assert _read_shared_hash(p, "k") == "new"

    def test_roundtrip(self, tmp_path: Path) -> None:
        """Write then read returns same value."""
        p = tmp_path / "meta.json"
        _write_shared_hash(p, "shared_common_knowledge_hash", "deadbeef")
        assert _read_shared_hash(p, "shared_common_knowledge_hash") == "deadbeef"


# ── _ensure_shared_knowledge_indexed (change detection) ───


@pytest.fixture
def anima_dir(tmp_path: Path) -> Path:
    d = tmp_path / "animas" / "alice"
    d.mkdir(parents=True)
    return d


@pytest.fixture
def common_knowledge_dir(tmp_path: Path) -> Path:
    d = tmp_path / "common_knowledge"
    d.mkdir()
    return d


@pytest.fixture
def common_skills_dir(tmp_path: Path) -> Path:
    d = tmp_path / "common_skills"
    d.mkdir()
    return d


@pytest.fixture
def rag(
    anima_dir: Path,
    common_knowledge_dir: Path,
    common_skills_dir: Path,
) -> RAGMemorySearch:
    return RAGMemorySearch(anima_dir, common_knowledge_dir, common_skills_dir)


class TestEnsureSharedKnowledgeChangeDetection:
    """Verify that _ensure_shared_knowledge_indexed uses hash-based skip."""

    def test_skips_when_no_md_files(self, rag: RAGMemorySearch) -> None:
        """Empty common_knowledge dir → no indexing attempted."""
        mock_vs = MagicMock()
        rag._indexer = MagicMock()
        rag._ensure_shared_knowledge_indexed(mock_vs)
        meta = rag._anima_dir / "index_meta.json"
        assert not meta.exists()

    def test_indexes_on_first_call(
        self,
        rag: RAGMemorySearch,
        common_knowledge_dir: Path,
    ) -> None:
        """First call with files → indexes and writes hash."""
        (common_knowledge_dir / "guide.md").write_text("# Guide")
        mock_vs = MagicMock()
        rag._indexer = MagicMock()

        with patch("core.memory.rag.MemoryIndexer") as MockIdx:
            mock_shared = MagicMock()
            mock_shared.index_directory.return_value = 5
            MockIdx.return_value = mock_shared

            rag._ensure_shared_knowledge_indexed(mock_vs)

        meta_path = rag._anima_dir / "index_meta.json"
        assert meta_path.exists()
        stored = _read_shared_hash(meta_path, "shared_common_knowledge_hash")
        assert stored is not None

    def test_skips_on_second_call_unchanged(
        self,
        rag: RAGMemorySearch,
        common_knowledge_dir: Path,
    ) -> None:
        """Second call with same files → skips (hash match + collection exists)."""
        (common_knowledge_dir / "guide.md").write_text("# Guide")
        mock_vs = MagicMock()
        # Simulate the shared collection existing in the vector store
        # so the existence check after hash match also passes.
        mock_vs.list_collections.return_value = ["shared_common_knowledge"]
        rag._indexer = MagicMock()

        with patch("core.memory.rag.MemoryIndexer") as MockIdx:
            mock_shared = MagicMock()
            mock_shared.index_directory.return_value = 5
            MockIdx.return_value = mock_shared

            rag._ensure_shared_knowledge_indexed(mock_vs)
            MockIdx.reset_mock()
            mock_shared.reset_mock()

            rag._ensure_shared_knowledge_indexed(mock_vs)
            MockIdx.assert_not_called()

    def test_re_indexes_after_file_change(
        self,
        rag: RAGMemorySearch,
        common_knowledge_dir: Path,
    ) -> None:
        """After a file is modified, re-indexing occurs."""
        f = common_knowledge_dir / "guide.md"
        f.write_text("# Guide v1")
        mock_vs = MagicMock()
        rag._indexer = MagicMock()

        with patch("core.memory.rag.MemoryIndexer") as MockIdx:
            mock_shared = MagicMock()
            mock_shared.index_directory.return_value = 3
            MockIdx.return_value = mock_shared

            rag._ensure_shared_knowledge_indexed(mock_vs)
            MockIdx.reset_mock()

            time.sleep(0.05)
            f.write_text("# Guide v2")

            MockIdx.return_value = mock_shared
            rag._ensure_shared_knowledge_indexed(mock_vs)
            MockIdx.assert_called_once()

    def test_re_indexes_when_collection_missing(
        self,
        rag: RAGMemorySearch,
        common_knowledge_dir: Path,
    ) -> None:
        """Hash unchanged but collection missing in vectordb → force re-index.

        Recovery scenario: vectordb was wiped/recreated since the last
        index, leaving the per-anima index_meta.json hash stale.  The
        framework must detect the missing collection and force re-index
        with ``force=True`` so it gets recreated.
        """
        (common_knowledge_dir / "guide.md").write_text("# Guide")
        mock_vs = MagicMock()
        # First call: no collection yet, should index normally
        mock_vs.list_collections.return_value = []
        rag._indexer = MagicMock()

        with patch("core.memory.rag.MemoryIndexer") as MockIdx:
            mock_shared = MagicMock()
            mock_shared.index_directory.return_value = 5
            MockIdx.return_value = mock_shared

            rag._ensure_shared_knowledge_indexed(mock_vs)
            assert MockIdx.call_count == 1
            # Verify hash was stored
            meta_path = rag._anima_dir / "index_meta.json"
            stored = _read_shared_hash(meta_path, "shared_common_knowledge_hash")
            assert stored is not None

            MockIdx.reset_mock()
            mock_shared.reset_mock()

            # Second call: hash matches but collection STILL missing
            # (simulating a vectordb wipe between calls).
            # → must force re-index so collection is recreated.
            rag._ensure_shared_knowledge_indexed(mock_vs)
            assert MockIdx.call_count == 1
            mock_shared.index_directory.assert_called_once_with(
                common_knowledge_dir,
                "common_knowledge",
                force=True,
            )


class TestEnsureSharedSkillsChangeDetection:
    """Verify that _ensure_shared_skills_indexed uses hash-based skip."""

    def test_skips_when_no_skill_files(self, rag: RAGMemorySearch) -> None:
        """Empty common_skills dir → no indexing."""
        mock_vs = MagicMock()
        rag._indexer = MagicMock()
        rag._ensure_shared_skills_indexed(mock_vs)
        meta = rag._anima_dir / "index_meta.json"
        assert not meta.exists()

    def test_indexes_on_first_call(
        self,
        rag: RAGMemorySearch,
        common_skills_dir: Path,
    ) -> None:
        """First call with SKILL.md files → indexes and writes hash."""
        skill_dir = common_skills_dir / "deploy"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("---\ndescription: deploy\n---\n# Deploy")
        mock_vs = MagicMock()
        rag._indexer = MagicMock()

        with patch("core.memory.rag.MemoryIndexer") as MockIdx:
            mock_shared = MagicMock()
            mock_shared.index_directory.return_value = 2
            MockIdx.return_value = mock_shared

            rag._ensure_shared_skills_indexed(mock_vs)

        meta_path = rag._anima_dir / "index_meta.json"
        stored = _read_shared_hash(meta_path, "shared_common_skills_hash")
        assert stored is not None

    def test_skips_on_second_call_unchanged(
        self,
        rag: RAGMemorySearch,
        common_skills_dir: Path,
    ) -> None:
        """Second call with same files → skips (hash match + collection exists)."""
        skill_dir = common_skills_dir / "deploy"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("# Deploy")
        mock_vs = MagicMock()
        mock_vs.list_collections.return_value = ["shared_common_skills"]
        rag._indexer = MagicMock()

        with patch("core.memory.rag.MemoryIndexer") as MockIdx:
            mock_shared = MagicMock()
            mock_shared.index_directory.return_value = 2
            MockIdx.return_value = mock_shared

            rag._ensure_shared_skills_indexed(mock_vs)
            MockIdx.reset_mock()

            rag._ensure_shared_skills_indexed(mock_vs)
            MockIdx.assert_not_called()

    def test_re_indexes_when_collection_missing(
        self,
        rag: RAGMemorySearch,
        common_skills_dir: Path,
    ) -> None:
        """Hash unchanged but shared_common_skills missing → force re-index.

        Recovery scenario for skills mirror of the knowledge case.
        """
        skill_dir = common_skills_dir / "deploy"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("# Deploy")
        mock_vs = MagicMock()
        mock_vs.list_collections.return_value = []
        rag._indexer = MagicMock()

        with patch("core.memory.rag.MemoryIndexer") as MockIdx:
            mock_shared = MagicMock()
            mock_shared.index_directory.return_value = 2
            MockIdx.return_value = mock_shared

            rag._ensure_shared_skills_indexed(mock_vs)
            MockIdx.reset_mock()
            mock_shared.reset_mock()

            rag._ensure_shared_skills_indexed(mock_vs)
            mock_shared.index_directory.assert_called_once_with(
                common_skills_dir,
                "common_skills",
                force=True,
            )


# ── _get_indexer calls _check_shared_collections ──────────


class TestGetIndexerSharedCheck:
    """_get_indexer() calls _check_shared_collections() on every access."""

    def test_check_shared_called_every_time(self, rag: RAGMemorySearch) -> None:
        """_check_shared_collections is called on each _get_indexer() call."""
        rag._indexer_initialized = True
        rag._indexer = MagicMock()

        with patch.object(rag, "_check_shared_collections") as mock_check:
            rag._get_indexer()
            rag._get_indexer()
            assert mock_check.call_count == 2

    def test_check_shared_skipped_when_no_indexer(self, rag: RAGMemorySearch) -> None:
        """_check_shared_collections is a no-op when indexer is None."""
        rag._indexer_initialized = True
        rag._indexer = None

        with patch.object(rag, "_ensure_shared_knowledge_indexed") as mock_ck:
            rag._check_shared_collections()
            mock_ck.assert_not_called()
