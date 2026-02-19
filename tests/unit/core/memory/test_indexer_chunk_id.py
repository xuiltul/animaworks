# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for MemoryIndexer chunk ID generation and memory_type propagation.

Verifies:
- _make_chunk_id produces IDs without path duplication
- _chunk_by_markdown_headings passes memory_type correctly
- _chunk_by_time_headings passes memory_type correctly
- _chunk_file dispatches memory_type to all chunking strategies
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


class TestChunkIdFormat:
    """Verify chunk IDs do not contain doubled directory paths."""

    @pytest.fixture
    def anima_dir(self, tmp_path: Path) -> Path:
        d = tmp_path / "test_anima"
        d.mkdir()
        for sub in ("knowledge", "episodes", "procedures", "skills"):
            (d / sub).mkdir()
        return d

    def _make_indexer(self, anima_dir: Path, prefix: str | None = None):
        from core.memory.rag.indexer import MemoryIndexer

        with patch.object(MemoryIndexer, "_init_embedding_model"):
            return MemoryIndexer(
                MagicMock(),
                anima_name=anima_dir.name,
                anima_dir=anima_dir,
                collection_prefix=prefix,
            )

    def test_knowledge_chunk_id(self, anima_dir: Path):
        indexer = self._make_indexer(anima_dir)
        f = anima_dir / "knowledge" / "topic.md"
        f.write_text("x", encoding="utf-8")
        cid = indexer._make_chunk_id(f, "knowledge", 0)
        assert cid == f"{anima_dir.name}/knowledge/topic.md#0"

    def test_episodes_chunk_id(self, anima_dir: Path):
        indexer = self._make_indexer(anima_dir)
        f = anima_dir / "episodes" / "2026-02-16.md"
        f.write_text("x", encoding="utf-8")
        cid = indexer._make_chunk_id(f, "episodes", 2)
        assert cid == f"{anima_dir.name}/episodes/2026-02-16.md#2"

    def test_shared_common_knowledge_chunk_id(self, anima_dir: Path):
        indexer = self._make_indexer(anima_dir, prefix="shared")
        ck = anima_dir / "common_knowledge"
        ck.mkdir(exist_ok=True)
        f = ck / "guide.md"
        f.write_text("x", encoding="utf-8")
        cid = indexer._make_chunk_id(f, "common_knowledge", 0)
        assert cid == "shared/common_knowledge/guide.md#0"
        assert "common_knowledge/common_knowledge/" not in cid


class TestChunkByMarkdownHeadingsMemoryType:
    """Verify _chunk_by_markdown_headings uses the memory_type parameter."""

    @pytest.fixture
    def anima_dir(self, tmp_path: Path) -> Path:
        d = tmp_path / "anima"
        d.mkdir()
        (d / "knowledge").mkdir()
        (d / "common_knowledge").mkdir()
        return d

    def _make_indexer(self, anima_dir: Path, prefix: str | None = None):
        from core.memory.rag.indexer import MemoryIndexer

        with patch.object(MemoryIndexer, "_init_embedding_model"):
            return MemoryIndexer(
                MagicMock(),
                anima_name=anima_dir.name,
                anima_dir=anima_dir,
                collection_prefix=prefix,
            )

    def test_memory_type_propagated_to_chunk_ids(self, anima_dir: Path):
        """When called with memory_type='common_knowledge', chunk IDs reflect that."""
        indexer = self._make_indexer(anima_dir, prefix="shared")
        f = anima_dir / "common_knowledge" / "guide.md"
        f.write_text("preamble\n\n## Section A\n\nBody A", encoding="utf-8")

        chunks = indexer._chunk_by_markdown_headings(f, f.read_text(), "common_knowledge")
        assert len(chunks) >= 1
        for chunk in chunks:
            assert "common_knowledge" in chunk.id
            assert chunk.metadata["memory_type"] == "common_knowledge"
            # Must NOT contain 'knowledge/' (that would mean the old hardcoded value leaked)
            assert "/knowledge/" not in chunk.id or "/common_knowledge/" in chunk.id

    def test_memory_type_knowledge(self, anima_dir: Path):
        """Standard knowledge type still works correctly."""
        indexer = self._make_indexer(anima_dir)
        f = anima_dir / "knowledge" / "topic.md"
        f.write_text("intro\n\n## Heading\n\nContent here", encoding="utf-8")

        chunks = indexer._chunk_by_markdown_headings(f, f.read_text(), "knowledge")
        assert len(chunks) >= 1
        for chunk in chunks:
            assert chunk.metadata["memory_type"] == "knowledge"


class TestChunkByMarkdownHeadingsPreambleCollision:
    """Verify preamble and heading sections get unique sequential IDs."""

    @pytest.fixture
    def anima_dir(self, tmp_path: Path) -> Path:
        d = tmp_path / "anima"
        d.mkdir()
        (d / "knowledge").mkdir()
        return d

    def _make_indexer(self, anima_dir: Path):
        from core.memory.rag.indexer import MemoryIndexer

        with patch.object(MemoryIndexer, "_init_embedding_model"):
            return MemoryIndexer(
                MagicMock(),
                anima_name=anima_dir.name,
                anima_dir=anima_dir,
            )

    def test_preamble_and_first_heading_have_different_ids(self, anima_dir: Path):
        """Preamble (#0) and first heading section (#1) must not collide."""
        indexer = self._make_indexer(anima_dir)
        f = anima_dir / "knowledge" / "topic.md"
        # Preamble is >50 chars to trigger inclusion
        f.write_text(
            "# Topic Title\n\n[AUTO-CONSOLIDATED: 2026-02-16 02:00]\n\n"
            "## Section A\n\nBody A\n\n## Section B\n\nBody B",
            encoding="utf-8",
        )

        chunks = indexer._chunk_by_markdown_headings(f, f.read_text(), "knowledge")

        assert len(chunks) == 3  # preamble + 2 heading sections
        ids = [c.id for c in chunks]
        # All IDs must be unique
        assert len(ids) == len(set(ids)), f"Duplicate IDs found: {ids}"
        # Sequential numbering: #0, #1, #2
        assert ids[0].endswith("#0")
        assert ids[1].endswith("#1")
        assert ids[2].endswith("#2")

    def test_no_preamble_starts_at_zero(self, anima_dir: Path):
        """Without preamble, first heading section gets #0."""
        indexer = self._make_indexer(anima_dir)
        f = anima_dir / "knowledge" / "topic.md"
        f.write_text(
            "short\n\n## Section A\n\nBody A\n\n## Section B\n\nBody B",
            encoding="utf-8",
        )

        chunks = indexer._chunk_by_markdown_headings(f, f.read_text(), "knowledge")

        assert len(chunks) == 2  # no preamble (too short), 2 headings
        assert chunks[0].id.endswith("#0")
        assert chunks[1].id.endswith("#1")

    def test_auto_consolidated_header_triggers_preamble(self, anima_dir: Path):
        """Real-world pattern: [AUTO-CONSOLIDATED: ...] creates >50 char preamble."""
        indexer = self._make_indexer(anima_dir)
        f = anima_dir / "knowledge" / "system-integration.md"
        content = (
            "# system-integration\n\n"
            "[AUTO-CONSOLIDATED: 2026-02-16 02:00]\n\n"
            "```markdown\n"
            "## Basic Policy\n\nDo things step by step\n\n"
            "## Lessons\n\nImportant stuff\n\n"
            "## Examples\n\nReal examples\n\n"
            "## Guidelines\n\nKeep it simple\n"
            "```"
        )
        f.write_text(content, encoding="utf-8")

        chunks = indexer._chunk_by_markdown_headings(f, f.read_text(), "knowledge")

        ids = [c.id for c in chunks]
        assert len(ids) == len(set(ids)), f"Duplicate IDs: {ids}"
        # Preamble + 4 headings = 5 chunks
        assert len(chunks) == 5
        for i, chunk in enumerate(chunks):
            assert chunk.id.endswith(f"#{i}")

    def test_single_heading_with_preamble(self, anima_dir: Path):
        """File with preamble and exactly one heading section."""
        indexer = self._make_indexer(anima_dir)
        f = anima_dir / "knowledge" / "topic.md"
        f.write_text(
            "# Title\n\nSome long preamble that is definitely over fifty characters.\n\n"
            "## The Only Section\n\nContent here",
            encoding="utf-8",
        )

        chunks = indexer._chunk_by_markdown_headings(f, f.read_text(), "knowledge")

        assert len(chunks) == 2
        assert chunks[0].id.endswith("#0")  # preamble
        assert chunks[1].id.endswith("#1")  # heading
        assert chunks[0].id != chunks[1].id

    def test_chunk_indices_in_metadata(self, anima_dir: Path):
        """chunk_index in metadata matches the sequential ID."""
        indexer = self._make_indexer(anima_dir)
        f = anima_dir / "knowledge" / "topic.md"
        f.write_text(
            "# Title\n\n[AUTO-CONSOLIDATED: 2026-02-16]\nExtra context line here.\n\n"
            "## A\n\nBody A\n\n## B\n\nBody B",
            encoding="utf-8",
        )

        chunks = indexer._chunk_by_markdown_headings(f, f.read_text(), "knowledge")

        for i, chunk in enumerate(chunks):
            assert chunk.metadata["chunk_index"] == i


class TestChunkByTimeHeadingsMemoryType:
    """Verify _chunk_by_time_headings uses the memory_type parameter."""

    @pytest.fixture
    def anima_dir(self, tmp_path: Path) -> Path:
        d = tmp_path / "anima"
        d.mkdir()
        (d / "episodes").mkdir()
        return d

    def _make_indexer(self, anima_dir: Path):
        from core.memory.rag.indexer import MemoryIndexer

        with patch.object(MemoryIndexer, "_init_embedding_model"):
            return MemoryIndexer(
                MagicMock(),
                anima_name=anima_dir.name,
                anima_dir=anima_dir,
            )

    def test_memory_type_propagated_to_chunk_ids(self, anima_dir: Path):
        indexer = self._make_indexer(anima_dir)
        f = anima_dir / "episodes" / "2026-02-16.md"
        f.write_text(
            "# 2026-02-16\n\n## 09:30 — Morning\n\nDid stuff\n\n## 14:00 — Afternoon\n\nMore stuff",
            encoding="utf-8",
        )

        chunks = indexer._chunk_by_time_headings(f, f.read_text(), "episodes")
        assert len(chunks) == 2
        for chunk in chunks:
            assert chunk.metadata["memory_type"] == "episodes"
            assert "episodes/episodes/" not in chunk.id


class TestChunkFileDispatches:
    """Verify _chunk_file dispatches memory_type to all strategies."""

    @pytest.fixture
    def anima_dir(self, tmp_path: Path) -> Path:
        d = tmp_path / "anima"
        d.mkdir()
        for sub in ("knowledge", "episodes", "procedures"):
            (d / sub).mkdir()
        return d

    def _make_indexer(self, anima_dir: Path):
        from core.memory.rag.indexer import MemoryIndexer

        with patch.object(MemoryIndexer, "_init_embedding_model"):
            return MemoryIndexer(
                MagicMock(),
                anima_name=anima_dir.name,
                anima_dir=anima_dir,
            )

    def test_common_knowledge_dispatches_with_correct_type(self, anima_dir: Path):
        indexer = self._make_indexer(anima_dir)
        ck_dir = anima_dir / "common_knowledge"
        ck_dir.mkdir(exist_ok=True)
        f = ck_dir / "doc.md"
        content = "preamble\n\n## Section\n\nBody"
        f.write_text(content, encoding="utf-8")

        chunks = indexer._chunk_file(f, content, "common_knowledge")
        for chunk in chunks:
            assert chunk.metadata["memory_type"] == "common_knowledge"

    def test_episodes_dispatches_with_correct_type(self, anima_dir: Path):
        indexer = self._make_indexer(anima_dir)
        f = anima_dir / "episodes" / "2026-02-16.md"
        content = "# Day\n\n## 10:00 — Test\n\nContent"
        f.write_text(content, encoding="utf-8")

        chunks = indexer._chunk_file(f, content, "episodes")
        for chunk in chunks:
            assert chunk.metadata["memory_type"] == "episodes"

    def test_procedures_dispatches_with_correct_type(self, anima_dir: Path):
        indexer = self._make_indexer(anima_dir)
        f = anima_dir / "procedures" / "deploy.md"
        content = "# Deploy procedure\n\nStep 1: do the thing"
        f.write_text(content, encoding="utf-8")

        chunks = indexer._chunk_file(f, content, "procedures")
        assert len(chunks) == 1
        assert chunks[0].metadata["memory_type"] == "procedures"
