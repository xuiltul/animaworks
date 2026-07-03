# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for .ragignore support in MemoryIndexer.

Verifies:
- is_ragignored() when .ragignore doesn't exist
- is_ragignored() with exact filename and glob patterns
- _load_ragignore() ignores comments and empty lines, caches by mtime
- index_file() skips ragignored files
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from core.memory.rag.indexer import MemoryIndexer

# ── Cache cleanup ───────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def clear_ragignore_cache():
    """Clear class-level ragignore cache after each test."""
    yield
    MemoryIndexer._ragignore_cache = None


# ── is_ragignored when .ragignore doesn't exist ─────────────────────


class TestIsRagignoredNoFile:
    def test_returns_false_when_ragignore_missing(
        self,
        tmp_path: Path,
    ) -> None:
        """is_ragignored() returns False when .ragignore doesn't exist."""
        with patch("core.paths.get_data_dir", return_value=tmp_path):
            result = MemoryIndexer.is_ragignored(tmp_path / "any_file.md")
        assert result is False


# ── is_ragignored with patterns ──────────────────────────────────────


class TestIsRagignoredExactMatch:
    def test_returns_true_for_exact_filename_match(
        self,
        tmp_path: Path,
    ) -> None:
        """is_ragignored() returns True for exact filename match (e.g. 00_index.md)."""
        ragignore = tmp_path / ".ragignore"
        ragignore.write_text("00_index.md\n", encoding="utf-8")

        with patch("core.paths.get_data_dir", return_value=tmp_path):
            result = MemoryIndexer.is_ragignored(tmp_path / "00_index.md")
        assert result is True

    def test_returns_true_for_exact_match_in_subdir(
        self,
        tmp_path: Path,
    ) -> None:
        """is_ragignored() matches by path when glob pattern matches (e.g. *knowledge/00_index.md)."""
        ragignore = tmp_path / ".ragignore"
        ragignore.write_text("*knowledge/00_index.md\n", encoding="utf-8")

        with patch("core.paths.get_data_dir", return_value=tmp_path):
            result = MemoryIndexer.is_ragignored(tmp_path / "knowledge" / "00_index.md")
        assert result is True


class TestIsRagignoredGlobPattern:
    def test_returns_true_for_glob_pattern(
        self,
        tmp_path: Path,
    ) -> None:
        """is_ragignored() returns True for glob patterns (e.g. *.tmp)."""
        ragignore = tmp_path / ".ragignore"
        ragignore.write_text("*.tmp\n", encoding="utf-8")

        with patch("core.paths.get_data_dir", return_value=tmp_path):
            assert MemoryIndexer.is_ragignored(tmp_path / "foo.tmp") is True
            assert MemoryIndexer.is_ragignored(tmp_path / "bar.tmp") is True

    def test_returns_false_for_non_matching_file(
        self,
        tmp_path: Path,
    ) -> None:
        """is_ragignored() returns False for non-matching files."""
        ragignore = tmp_path / ".ragignore"
        ragignore.write_text("00_index.md\n*.tmp\n", encoding="utf-8")

        with patch("core.paths.get_data_dir", return_value=tmp_path):
            result = MemoryIndexer.is_ragignored(tmp_path / "included.md")
        assert result is False


# ── _load_ragignore comments and empty lines ──────────────────────────


class TestLoadRagignoreFormat:
    def test_ignores_comment_lines_and_empty(
        self,
        tmp_path: Path,
    ) -> None:
        """_load_ragignore() ignores lines starting with # and empty lines."""
        ragignore = tmp_path / ".ragignore"
        ragignore.write_text(
            "# comment line\n\n00_index.md\n  # inline comment (stripped starts with #, so skipped)\n*.tmp\n\n",
            encoding="utf-8",
        )

        with patch("core.paths.get_data_dir", return_value=tmp_path):
            patterns = MemoryIndexer._load_ragignore()

        assert patterns == ["00_index.md", "*.tmp"]
        assert "# comment" not in patterns
        assert "" not in patterns


# ── _load_ragignore caching ──────────────────────────────────────────


class TestLoadRagignoreCache:
    def test_caches_by_mtime(
        self,
        tmp_path: Path,
    ) -> None:
        """_load_ragignore() caches results by mtime; second call returns cached."""
        ragignore = tmp_path / ".ragignore"
        ragignore.write_text("00_index.md\n", encoding="utf-8")

        read_count = 0
        original_read_text = Path.read_text

        def counting_read_text(self, encoding="utf-8"):
            nonlocal read_count
            if self == ragignore:
                read_count += 1
            return original_read_text(self, encoding=encoding)

        with (
            patch("core.paths.get_data_dir", return_value=tmp_path),
            patch.object(Path, "read_text", counting_read_text),
        ):
            p1 = MemoryIndexer._load_ragignore()
            p2 = MemoryIndexer._load_ragignore()

        assert p1 == p2 == ["00_index.md"]
        assert read_count == 1, "Second call should use cache, not re-read file"


# ── index_file skips ragignored ──────────────────────────────────────


class TestIndexFileSkipsRagignored:
    def test_index_file_skips_ragignored_files(
        self,
        tmp_path: Path,
    ) -> None:
        """index_file() skips files that match .ragignore (mock ragignore check)."""
        from unittest.mock import MagicMock

        anima_dir = tmp_path / "anima"
        anima_dir.mkdir()
        (anima_dir / "knowledge").mkdir()
        ragignore = tmp_path / ".ragignore"
        ragignore.write_text("excluded.md\n", encoding="utf-8")

        excluded = anima_dir / "knowledge" / "excluded.md"
        excluded.write_text("# Excluded content\n\nShould not be indexed.", encoding="utf-8")

        mock_store = MagicMock()
        with (
            patch("core.paths.get_data_dir", return_value=tmp_path),
            patch.object(MemoryIndexer, "_init_embedding_model"),
        ):
            indexer = MemoryIndexer(
                mock_store,
                anima_name="test",
                anima_dir=anima_dir,
            )
            result = indexer.index_file(excluded, "knowledge")

        assert result == 0
        mock_store.upsert.assert_not_called()

    def test_index_file_purges_existing_chunks_when_newly_ragignored(
        self,
        tmp_path: Path,
    ) -> None:
        """F6: a file that matches .ragignore only after being indexed has its
        stale chunks removed instead of lingering in the collection."""
        from unittest.mock import MagicMock

        anima_dir = tmp_path / "anima"
        anima_dir.mkdir()
        (anima_dir / "knowledge").mkdir()
        ragignore = tmp_path / ".ragignore"
        ragignore.write_text("excluded.md\n", encoding="utf-8")

        excluded = anima_dir / "knowledge" / "excluded.md"
        excluded.write_text("# Excluded content\n\nBody.", encoding="utf-8")

        mock_store = MagicMock()
        with (
            patch("core.paths.get_data_dir", return_value=tmp_path),
            patch.object(MemoryIndexer, "_init_embedding_model"),
        ):
            indexer = MemoryIndexer(mock_store, anima_name="test", anima_dir=anima_dir)
            indexer.delete_indexed_file = MagicMock(return_value=2)
            result = indexer.index_file(excluded, "knowledge")

        assert result == 0
        mock_store.upsert.assert_not_called()
        indexer.delete_indexed_file.assert_called_once_with(excluded, "knowledge")
