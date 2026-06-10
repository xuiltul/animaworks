from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.


"""Tests for the active forgetting mechanism.

Tests cover:
- ForgettingEngine._is_protected() classification
- Synaptic downscaling (Stage 1)
- Complete forgetting with archival (Stage 3)
- Integration with ConsolidationEngine (daily + weekly hooks)
"""

import logging
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.memory.forgetting import (
    ForgettingEngine,
)
from core.time_utils import now_jst

# ── Fixtures ────────────────────────────────────────────────────────


@pytest.fixture
def anima_dir(tmp_path: Path) -> Path:
    """Create a temporary anima directory structure."""
    anima_dir = tmp_path / "test_anima"
    anima_dir.mkdir()
    (anima_dir / "knowledge").mkdir()
    (anima_dir / "episodes").mkdir()
    (anima_dir / "procedures").mkdir()
    (anima_dir / "skills").mkdir()
    return anima_dir


@pytest.fixture
def forgetting_engine(anima_dir: Path) -> ForgettingEngine:
    """Create a ForgettingEngine instance."""
    return ForgettingEngine(anima_dir=anima_dir, anima_name="test_anima")


def _make_chunk(
    doc_id: str = "chunk1",
    content: str = "test content",
    memory_type: str = "knowledge",
    importance: str = "normal",
    access_count: int = 0,
    last_accessed_at: str = "",
    updated_at: str = "",
    activation_level: str = "normal",
    low_activation_since: str = "",
    source_file: str = "knowledge/test.md",
) -> dict[str, Any]:
    """Helper to create a chunk dict matching _get_all_chunks format."""
    return {
        "id": doc_id,
        "content": content,
        "metadata": {
            "memory_type": memory_type,
            "importance": importance,
            "access_count": access_count,
            "last_accessed_at": last_accessed_at,
            "updated_at": updated_at,
            "activation_level": activation_level,
            "low_activation_since": low_activation_since,
            "source_file": source_file,
        },
    }


def _make_indexed_chunk(
    engine: ForgettingEngine,
    path: Path,
    *,
    doc_id: str,
    chunk_index: int,
    content: str,
    memory_type: str = "knowledge",
) -> dict[str, Any]:
    chunk = _make_chunk(
        doc_id=doc_id,
        content=content,
        memory_type=memory_type,
        source_file=str(path.relative_to(engine.anima_dir)),
        activation_level="low",
    )
    chunk["metadata"].update(
        {
            "chunk_index": chunk_index,
            "source_hash": engine._compute_source_hash(path),
            "source_mtime_ns": path.stat().st_mtime_ns,
        }
    )
    return chunk


# ── _is_protected Tests ─────────────────────────────────────────────


class TestIsProtected:
    """Test _is_protected() classification of chunks."""

    def test_is_protected_procedures_not_blanket(self, forgetting_engine):
        """Verify that memory_type='procedures' is NOT blanket-protected.

        Procedures are no longer in PROTECTED_MEMORY_TYPES. They use
        utility-based protection via _is_protected_procedure instead.
        A basic procedure with importance='normal' and version=1 is not protected.
        """
        meta = {"memory_type": "procedures", "importance": "normal", "version": 1}
        assert forgetting_engine._is_protected(meta) is False

    def test_is_protected_skills(self, forgetting_engine):
        """Verify that memory_type='skills' is protected from forgetting."""
        meta = {"memory_type": "skills", "importance": "normal"}
        assert forgetting_engine._is_protected(meta) is True

    def test_is_protected_shared_users(self, forgetting_engine):
        """Verify that memory_type='shared_users' is protected from forgetting."""
        meta = {"memory_type": "shared_users", "importance": "normal"}
        assert forgetting_engine._is_protected(meta) is True

    def test_is_protected_important(self, forgetting_engine):
        """Verify that importance='important' is protected regardless of type."""
        meta = {"memory_type": "knowledge", "importance": "important"}
        assert forgetting_engine._is_protected(meta) is True

    def test_is_protected_normal_knowledge(self, forgetting_engine):
        """Verify that memory_type='knowledge', importance='normal' is NOT protected."""
        meta = {"memory_type": "knowledge", "importance": "normal"}
        assert forgetting_engine._is_protected(meta) is False

    def test_is_protected_normal_episodes(self, forgetting_engine):
        """Verify that memory_type='episodes', importance='normal' is NOT protected."""
        meta = {"memory_type": "episodes", "importance": "normal"}
        assert forgetting_engine._is_protected(meta) is False


# ── Synaptic Downscaling Tests ──────────────────────────────────────


class TestSynapticDownscaling:
    """Test synaptic_downscaling() (Stage 1: daily mark low-activation)."""

    def test_synaptic_downscaling_marks_old_chunks(self, forgetting_engine):
        """Test that old chunks with no access are marked as low activation.

        Chunks that are >90 days old with access_count=0 should be marked
        with activation_level='low'.
        """
        old_date = (now_jst() - timedelta(days=120)).isoformat()
        knowledge_chunks = [
            _make_chunk(
                doc_id="old_chunk",
                access_count=0,
                last_accessed_at="",
                updated_at=old_date,
                activation_level="normal",
            ),
        ]

        def get_chunks(collection_name):
            if "knowledge" in collection_name:
                return knowledge_chunks
            return []  # No episode chunks

        mock_store = MagicMock()
        mock_store.update_metadata = MagicMock()

        with patch.object(forgetting_engine, "_get_vector_store", return_value=mock_store):
            with patch.object(forgetting_engine, "_get_all_chunks", side_effect=get_chunks):
                result = forgetting_engine.synaptic_downscaling()

        assert result["scanned"] == 1
        assert result["marked_low"] == 1

        # Verify update_metadata was called with correct args
        mock_store.update_metadata.assert_called_once()
        call_args = mock_store.update_metadata.call_args[0]
        assert call_args[0] == "test_anima_knowledge"
        assert call_args[1] == ["old_chunk"]
        assert call_args[2][0]["activation_level"] == "low"
        assert call_args[2][0]["low_activation_since"] != ""

    def test_synaptic_downscaling_skips_protected(self, forgetting_engine):
        """Test that chunks with importance='important' are NOT marked.

        Protected chunks should be skipped even if they are old and unaccessed.
        """
        old_date = (now_jst() - timedelta(days=120)).isoformat()
        knowledge_chunks = [
            _make_chunk(
                doc_id="important_chunk",
                importance="important",
                access_count=0,
                last_accessed_at="",
                updated_at=old_date,
                activation_level="normal",
            ),
        ]

        def get_chunks(collection_name):
            if "knowledge" in collection_name:
                return knowledge_chunks
            return []

        mock_store = MagicMock()
        mock_store.update_metadata = MagicMock()

        with patch.object(forgetting_engine, "_get_vector_store", return_value=mock_store):
            with patch.object(forgetting_engine, "_get_all_chunks", side_effect=get_chunks):
                result = forgetting_engine.synaptic_downscaling()

        assert result["scanned"] == 1
        assert result["marked_low"] == 0
        mock_store.update_metadata.assert_not_called()

    def test_synaptic_downscaling_skips_frequently_accessed(self, forgetting_engine):
        """Test that chunks with access_count >= threshold are NOT marked.

        Frequently accessed chunks (access_count >= DOWNSCALING_ACCESS_THRESHOLD)
        should be skipped even if they are old.
        """
        old_date = (now_jst() - timedelta(days=120)).isoformat()
        chunks = [
            _make_chunk(
                doc_id="accessed_chunk",
                access_count=5,  # Above threshold of 3
                last_accessed_at="",
                updated_at=old_date,
                activation_level="normal",
            ),
        ]

        mock_store = MagicMock()
        mock_store.update_metadata = MagicMock()

        with patch.object(forgetting_engine, "_get_vector_store", return_value=mock_store):
            with patch.object(forgetting_engine, "_get_all_chunks", return_value=chunks):
                result = forgetting_engine.synaptic_downscaling()

        assert result["marked_low"] == 0
        mock_store.update_metadata.assert_not_called()

    def test_synaptic_downscaling_skips_recent(self, forgetting_engine):
        """Test that recently accessed chunks are NOT marked.

        Chunks accessed within the last 90 days should not be marked
        regardless of access_count.
        """
        recent_date = (now_jst() - timedelta(days=30)).isoformat()
        chunks = [
            _make_chunk(
                doc_id="recent_chunk",
                access_count=0,
                last_accessed_at=recent_date,
                updated_at=recent_date,
                activation_level="normal",
            ),
        ]

        mock_store = MagicMock()
        mock_store.update_metadata = MagicMock()

        with patch.object(forgetting_engine, "_get_vector_store", return_value=mock_store):
            with patch.object(forgetting_engine, "_get_all_chunks", return_value=chunks):
                result = forgetting_engine.synaptic_downscaling()

        assert result["marked_low"] == 0
        mock_store.update_metadata.assert_not_called()

    def test_synaptic_downscaling_skips_already_low(self, forgetting_engine):
        """Test that chunks already at low activation are skipped."""
        old_date = (now_jst() - timedelta(days=120)).isoformat()
        chunks = [
            _make_chunk(
                doc_id="already_low",
                access_count=0,
                last_accessed_at="",
                updated_at=old_date,
                activation_level="low",
                low_activation_since=old_date,
            ),
        ]

        mock_store = MagicMock()
        mock_store.update_metadata = MagicMock()

        with patch.object(forgetting_engine, "_get_vector_store", return_value=mock_store):
            with patch.object(forgetting_engine, "_get_all_chunks", return_value=chunks):
                result = forgetting_engine.synaptic_downscaling()

        assert result["marked_low"] == 0
        mock_store.update_metadata.assert_not_called()

    def test_synaptic_downscaling_scans_knowledge_episodes_procedures(self, forgetting_engine):
        """Test that downscaling scans knowledge, episodes, and procedures."""
        chunks_knowledge = [
            _make_chunk(doc_id="k1", memory_type="knowledge"),
        ]
        chunks_episodes = [
            _make_chunk(doc_id="e1", memory_type="episodes"),
        ]
        chunks_procedures = [
            _make_chunk(doc_id="p1", memory_type="procedures"),
        ]

        call_count = {"n": 0}

        def side_effect_chunks(collection_name):
            call_count["n"] += 1
            if "knowledge" in collection_name:
                return chunks_knowledge
            if "episodes" in collection_name:
                return chunks_episodes
            return chunks_procedures

        mock_store = MagicMock()
        mock_store.update_metadata = MagicMock()

        with patch.object(forgetting_engine, "_get_vector_store", return_value=mock_store):
            with patch.object(
                forgetting_engine,
                "_get_all_chunks",
                side_effect=side_effect_chunks,
            ):
                result = forgetting_engine.synaptic_downscaling()

        # Should have been called for all three collections
        assert call_count["n"] == 3
        assert result["scanned"] == 3


# ── Neurogenesis Source Sync Tests ─────────────────────────────────


class TestNeurogenesisSourceSync:
    """Test chunk-level source syncing for neurogenesis merges."""

    def test_find_similar_pairs_batches_embeddings(self, forgetting_engine):
        chunks = [
            _make_chunk(doc_id="a", content="alpha"),
            _make_chunk(doc_id="b", content="alpha duplicate"),
            _make_chunk(doc_id="c", content="gamma"),
        ]
        store = MagicMock()

        def fake_query(*, collection, embedding, top_k):
            assert collection == "test_anima_knowledge"
            assert top_k == 5
            if embedding == [1.0]:
                return [
                    SimpleNamespace(document=SimpleNamespace(id="a"), score=1.0),
                    SimpleNamespace(document=SimpleNamespace(id="b"), score=0.86),
                ]
            return []

        store.query.side_effect = fake_query

        with patch(
            "core.memory.rag.singleton.generate_embeddings",
            return_value=[[1.0], [2.0], [3.0]],
        ) as mock_embeddings:
            pairs = forgetting_engine._find_similar_pairs(chunks, "test_anima_knowledge", store)

        mock_embeddings.assert_called_once_with(["alpha", "alpha duplicate", "gamma"])
        assert [(a["id"], b["id"], score) for a, b, score in pairs] == [("a", "b", 0.86)]

    def test_sync_replaces_only_target_chunk(self, forgetting_engine, anima_dir):
        primary = anima_dir / "knowledge" / "primary.md"
        secondary = anima_dir / "knowledge" / "secondary.md"
        primary.write_text(
            "## Keep One\n\nAlpha stays.\n\n## Merge Me\n\nOld duplicate detail.\n\n## Keep Two\n\nGamma stays.",
            encoding="utf-8",
        )
        secondary.write_text("## Merge Source\n\nNew duplicate detail.", encoding="utf-8")

        chunk_a = _make_indexed_chunk(
            forgetting_engine,
            primary,
            doc_id="a",
            chunk_index=1,
            content="## Merge Me\n\nOld duplicate detail.",
        )
        chunk_b = _make_indexed_chunk(
            forgetting_engine,
            secondary,
            doc_id="b",
            chunk_index=0,
            content="## Merge Source\n\nNew duplicate detail.",
        )

        assert forgetting_engine._sync_merged_source_files(chunk_a, chunk_b, "## Merged\n\nCombined detail.")

        updated = primary.read_text(encoding="utf-8")
        assert "## Keep One\n\nAlpha stays." in updated
        assert "## Keep Two\n\nGamma stays." in updated
        assert "## Merged\n\nCombined detail." in updated
        assert "Old duplicate detail." not in updated
        assert not secondary.exists()

    def test_sync_preserves_frontmatter(self, forgetting_engine, anima_dir):
        primary = anima_dir / "knowledge" / "frontmatter.md"
        secondary = anima_dir / "knowledge" / "dup.md"
        frontmatter = "---\ntitle: Deployment Notes\nconfidence: 0.8\n---"
        primary.write_text(
            f"{frontmatter}\n\n## Merge\n\nOld deploy detail.\n\n## Keep\n\nKeep this.",
            encoding="utf-8",
        )
        secondary.write_text("## Duplicate\n\nNew deploy detail.", encoding="utf-8")

        chunk_a = _make_indexed_chunk(
            forgetting_engine,
            primary,
            doc_id="a",
            chunk_index=0,
            content="## Merge\n\nOld deploy detail.",
        )
        chunk_b = _make_indexed_chunk(
            forgetting_engine,
            secondary,
            doc_id="b",
            chunk_index=0,
            content="## Duplicate\n\nNew deploy detail.",
        )

        assert forgetting_engine._sync_merged_source_files(chunk_a, chunk_b, "## Merge\n\nMerged deploy detail.")

        updated = primary.read_text(encoding="utf-8")
        assert updated.startswith(frontmatter)
        assert "confidence: 0.8" in updated
        assert "Merged deploy detail." in updated
        assert "## Keep\n\nKeep this." in updated

    def test_sync_skips_when_source_changed_since_index(
        self,
        forgetting_engine,
        anima_dir,
        caplog,
    ):
        primary = anima_dir / "knowledge" / "changed.md"
        secondary = anima_dir / "knowledge" / "secondary.md"
        primary.write_text("## Merge\n\nOld indexed detail.", encoding="utf-8")
        secondary.write_text("## Duplicate\n\nSecondary detail.", encoding="utf-8")

        chunk_a = _make_indexed_chunk(
            forgetting_engine,
            primary,
            doc_id="a",
            chunk_index=0,
            content="## Merge\n\nOld indexed detail.",
        )
        chunk_b = _make_indexed_chunk(
            forgetting_engine,
            secondary,
            doc_id="b",
            chunk_index=0,
            content="## Duplicate\n\nSecondary detail.",
        )
        primary.write_text("## Merge\n\nUser edited after indexing.", encoding="utf-8")

        with caplog.at_level(logging.WARNING, logger="animaworks.forgetting"):
            result = forgetting_engine._sync_merged_source_files(chunk_a, chunk_b, "## Merged\n\nShould skip.")

        assert result is False
        assert primary.read_text(encoding="utf-8") == "## Merge\n\nUser edited after indexing."
        assert "source file content changed since indexing" in caplog.text
        assert not (anima_dir / "archive" / "merged").exists()

    @pytest.mark.asyncio
    async def test_merge_reject_skips_llm_merge(self, forgetting_engine):
        with patch(
            "core.memory._llm_utils.one_shot_completion",
            new=AsyncMock(return_value="MERGE_REJECT\nDifferent topics."),
        ):
            merged = await forgetting_engine._merge_chunks_llm(
                {"id": "a", "content": "A"},
                {"id": "b", "content": "B"},
                0.91,
                "test-model",
            )

        assert merged is None

    def test_secondary_partial_absorption_keeps_file_unarchived(self, forgetting_engine, anima_dir):
        primary = anima_dir / "knowledge" / "primary.md"
        secondary = anima_dir / "knowledge" / "secondary.md"
        primary.write_text("## Primary\n\nPrimary duplicate.", encoding="utf-8")
        secondary.write_text(
            "## Keep One\n\nFirst survives.\n\n## Remove Me\n\nDuplicate absorbed.\n\n## Keep Two\n\nSecond survives.",
            encoding="utf-8",
        )

        chunk_a = _make_indexed_chunk(
            forgetting_engine,
            primary,
            doc_id="a",
            chunk_index=0,
            content="## Primary\n\nPrimary duplicate.",
        )
        chunk_b = _make_indexed_chunk(
            forgetting_engine,
            secondary,
            doc_id="b",
            chunk_index=1,
            content="## Remove Me\n\nDuplicate absorbed.",
        )

        assert forgetting_engine._sync_merged_source_files(chunk_a, chunk_b, "## Merged\n\nMerged duplicate.")

        secondary_text = secondary.read_text(encoding="utf-8")
        assert "## Keep One\n\nFirst survives." in secondary_text
        assert "## Keep Two\n\nSecond survives." in secondary_text
        assert "Duplicate absorbed." not in secondary_text

        archive_names = [path.name for path in (anima_dir / "archive" / "merged").iterdir()]
        assert any(name.startswith("primary_") for name in archive_names)
        assert not any(name.startswith("secondary_") for name in archive_names)


# ── Complete Forgetting Tests ───────────────────────────────────────


class TestCompleteForgetting:
    """Test complete_forgetting() (Stage 3: monthly archive and delete)."""

    def test_complete_forgetting_archives_and_deletes(self, forgetting_engine, anima_dir):
        """Test that low-activation chunks are archived and deleted.

        Chunks with low_activation_since > 60 days ago and access_count=0
        should have their source files moved to archive/forgotten/ and
        be deleted from the vector store.
        """
        old_low_since = (now_jst() - timedelta(days=90)).isoformat()

        # Create source file in the anima dir
        source_file = anima_dir / "knowledge" / "forgotten-topic.md"
        source_file.write_text("# Old topic\n\nThis will be forgotten.", encoding="utf-8")

        knowledge_chunks = [
            _make_chunk(
                doc_id="forget_me",
                access_count=0,
                activation_level="low",
                low_activation_since=old_low_since,
                source_file="knowledge/forgotten-topic.md",
            ),
        ]

        def get_chunks(collection_name):
            if "knowledge" in collection_name:
                return knowledge_chunks
            return []

        mock_store = MagicMock()
        mock_store.delete_documents = MagicMock()

        with patch.object(forgetting_engine, "_get_vector_store", return_value=mock_store):
            with patch.object(forgetting_engine, "_get_all_chunks", side_effect=get_chunks):
                result = forgetting_engine.complete_forgetting()

        assert result["forgotten_chunks"] == 1
        assert len(result["archived_files"]) == 1
        assert "knowledge/forgotten-topic.md" in result["archived_files"]

        # Verify source file was moved to archive
        archive_dir = anima_dir / "archive" / "forgotten"
        assert archive_dir.exists()
        archived_files = list(archive_dir.iterdir())
        assert len(archived_files) == 1
        assert "forgotten-topic" in archived_files[0].name

        # Verify original file is gone
        assert not source_file.exists()

        # Verify delete_documents was called once (for the knowledge collection)
        mock_store.delete_documents.assert_called_once()
        call_args = mock_store.delete_documents.call_args[0]
        assert call_args[0] == "test_anima_knowledge"
        assert call_args[1] == ["forget_me"]

    def test_complete_forgetting_skips_frequently_accessed_chunks(self, forgetting_engine, anima_dir):
        """Test that low-activation chunks with access_count > 2 are NOT deleted.

        Even if low_activation_since is old, access_count above the threshold
        (FORGETTING_MAX_ACCESS_COUNT=2) means the memory should survive.
        """
        old_low_since = (now_jst() - timedelta(days=120)).isoformat()
        chunks = [
            _make_chunk(
                doc_id="accessed_low",
                access_count=3,
                activation_level="low",
                low_activation_since=old_low_since,
                source_file="knowledge/accessed.md",
            ),
        ]

        mock_store = MagicMock()
        mock_store.delete_documents = MagicMock()

        with patch.object(forgetting_engine, "_get_vector_store", return_value=mock_store):
            with patch.object(forgetting_engine, "_get_all_chunks", return_value=chunks):
                result = forgetting_engine.complete_forgetting()

        assert result["forgotten_chunks"] == 0
        assert len(result["archived_files"]) == 0
        mock_store.delete_documents.assert_not_called()

    def test_complete_forgetting_skips_protected(self, forgetting_engine, anima_dir):
        """Test that protected chunks are not forgotten even if low-activation."""
        old_low_since = (now_jst() - timedelta(days=90)).isoformat()
        chunks = [
            _make_chunk(
                doc_id="protected_chunk",
                access_count=0,
                activation_level="low",
                low_activation_since=old_low_since,
                importance="important",
            ),
        ]

        mock_store = MagicMock()
        mock_store.delete_documents = MagicMock()

        with patch.object(forgetting_engine, "_get_vector_store", return_value=mock_store):
            with patch.object(forgetting_engine, "_get_all_chunks", return_value=chunks):
                result = forgetting_engine.complete_forgetting()

        assert result["forgotten_chunks"] == 0
        mock_store.delete_documents.assert_not_called()

    def test_complete_forgetting_skips_normal_activation(self, forgetting_engine, anima_dir):
        """Test that chunks with activation_level='normal' are not forgotten."""
        chunks = [
            _make_chunk(
                doc_id="normal_chunk",
                access_count=0,
                activation_level="normal",
                low_activation_since="",
            ),
        ]

        mock_store = MagicMock()
        mock_store.delete_documents = MagicMock()

        with patch.object(forgetting_engine, "_get_vector_store", return_value=mock_store):
            with patch.object(forgetting_engine, "_get_all_chunks", return_value=chunks):
                result = forgetting_engine.complete_forgetting()

        assert result["forgotten_chunks"] == 0
        mock_store.delete_documents.assert_not_called()

    def test_complete_forgetting_skips_recent_low_activation(self, forgetting_engine, anima_dir):
        """Test that chunks recently marked as low are NOT yet forgotten.

        Chunks that have been low for less than FORGETTING_LOW_ACTIVATION_DAYS
        should not be deleted yet.
        """
        recent_low = (now_jst() - timedelta(days=10)).isoformat()
        chunks = [
            _make_chunk(
                doc_id="recent_low",
                access_count=0,
                activation_level="low",
                low_activation_since=recent_low,
            ),
        ]

        mock_store = MagicMock()
        mock_store.delete_documents = MagicMock()

        with patch.object(forgetting_engine, "_get_vector_store", return_value=mock_store):
            with patch.object(forgetting_engine, "_get_all_chunks", return_value=chunks):
                result = forgetting_engine.complete_forgetting()

        assert result["forgotten_chunks"] == 0
        mock_store.delete_documents.assert_not_called()


# ── Consolidation Integration Tests ─────────────────────────────────


class TestConsolidationForgettingHooks:
    """Test that lifecycle hooks into ForgettingEngine correctly.

    After the Anima-driven consolidation refactoring, forgetting hooks
    are called from lifecycle.py as post-processing steps:
    - Daily: lifecycle._handle_daily_consolidation calls synaptic_downscaling
    - Weekly: lifecycle._handle_weekly_integration calls neurogenesis_reorganize
    """

    @pytest.mark.asyncio
    async def test_consolidation_daily_calls_downscaling(self, tmp_path: Path):
        """Test that _handle_daily_consolidation calls synaptic_downscaling().

        After the Anima-driven consolidation, the lifecycle handler
        invokes ForgettingEngine.synaptic_downscaling as post-processing.
        """
        from core.lifecycle import LifecycleManager

        manager = LifecycleManager()

        anima_dir = tmp_path / "test_anima"
        anima_dir.mkdir(parents=True)

        # Mock anima
        mock_anima = MagicMock()
        mock_anima.name = "test_anima"
        mock_anima.memory = MagicMock()
        mock_anima.memory.anima_dir = anima_dir
        mock_anima.count_recent_episodes.return_value = 3

        mock_result = MagicMock()
        mock_result.duration_ms = 30_000
        mock_result.summary = "consolidated"
        mock_anima.run_consolidation = AsyncMock(return_value=mock_result)

        manager.animas["test_anima"] = mock_anima
        manager._schedule_consolidation_retry = MagicMock()

        mock_config = MagicMock()
        mock_consolidation_cfg = MagicMock()
        mock_consolidation_cfg.daily_enabled = True
        mock_consolidation_cfg.min_episodes_threshold = 1
        mock_consolidation_cfg.max_turns = 30
        mock_config.consolidation = mock_consolidation_cfg

        mock_downscaling_result = {"scanned": 10, "marked_low": 2}

        gate = SimpleNamespace(
            should_run=True,
            activity_count=3,
            episode_count=0,
            carryover_count=0,
            threshold=1,
        )

        with (
            patch("core.config.load_config", return_value=mock_config),
            patch("core.lifecycle.system_consolidation.evaluate_daily_consolidation_gate", return_value=gate),
            patch("core.lifecycle.system_consolidation.run_knowledge_self_correction_if_enabled", AsyncMock()),
            patch("core.lifecycle.system_consolidation.detect_communities_if_neo4j", AsyncMock()),
            patch("core.memory.forgetting.ForgettingEngine") as MockForgettingEngine,
            patch("core.memory.consolidation.ConsolidationEngine"),
        ):
            mock_forgetter = MagicMock()
            mock_forgetter.synaptic_downscaling.return_value = mock_downscaling_result
            MockForgettingEngine.return_value = mock_forgetter

            await manager._handle_daily_consolidation()

        # Verify downscaling was called
        mock_forgetter.synaptic_downscaling.assert_called_once()

    @pytest.mark.asyncio
    async def test_consolidation_weekly_calls_reorganization(self, tmp_path: Path):
        """Test that _handle_weekly_integration calls neurogenesis_reorganize().

        After the Anima-driven consolidation, the lifecycle handler
        invokes ForgettingEngine.neurogenesis_reorganize as post-processing.
        """
        from core.lifecycle import LifecycleManager

        manager = LifecycleManager()

        anima_dir = tmp_path / "test_anima"
        anima_dir.mkdir(parents=True)

        # Mock anima
        mock_anima = MagicMock()
        mock_anima.name = "test_anima"
        mock_anima.memory = MagicMock()
        mock_anima.memory.anima_dir = anima_dir

        mock_result = MagicMock()
        mock_result.duration_ms = 200
        mock_result.summary = "weekly integration done"
        mock_anima.run_consolidation = AsyncMock(return_value=mock_result)

        manager.animas["test_anima"] = mock_anima

        mock_config = MagicMock()
        mock_consolidation_cfg = MagicMock()
        mock_consolidation_cfg.weekly_enabled = True
        mock_consolidation_cfg.llm_model = "anthropic/claude-sonnet-4-6"
        mock_consolidation_cfg.max_turns = 30
        mock_config.consolidation = mock_consolidation_cfg

        mock_reorg_result = {"merged_count": 3, "merged_pairs": ["a+b", "c+d", "e+f"]}

        with (
            patch("core.config.load_config", return_value=mock_config),
            patch("core.lifecycle.system_consolidation.detect_communities_if_neo4j", AsyncMock()),
            patch("core.memory.forgetting.ForgettingEngine") as MockForgettingEngine,
            patch("core.memory.consolidation.ConsolidationEngine"),
        ):
            mock_forgetter = MagicMock()
            mock_forgetter.neurogenesis_reorganize = AsyncMock(
                return_value=mock_reorg_result,
            )
            MockForgettingEngine.return_value = mock_forgetter

            await manager._handle_weekly_integration()

        # Verify neurogenesis_reorganize was called
        mock_forgetter.neurogenesis_reorganize.assert_called_once()


# ── Monthly Forgetting Hook Test ────────────────────────────────────


class TestMonthlyForgettingHook:
    """Test ConsolidationEngine.monthly_forget() integration."""

    @pytest.fixture
    def consolidation_engine(self, tmp_path: Path):
        """Create a ConsolidationEngine instance."""
        from core.memory.consolidation import ConsolidationEngine

        anima_dir = tmp_path / "test_anima"
        (anima_dir / "episodes").mkdir(parents=True)
        (anima_dir / "knowledge").mkdir(parents=True)
        return ConsolidationEngine(
            anima_dir=anima_dir,
            anima_name="test_anima",
        )

    @pytest.mark.asyncio
    async def test_monthly_forget_calls_complete_forgetting(self, consolidation_engine):
        """Test that monthly_forget() calls ForgettingEngine.complete_forgetting()."""
        mock_result = {"forgotten_chunks": 5, "archived_files": ["a.md", "b.md"]}

        with patch("core.memory.forgetting.ForgettingEngine") as MockForgettingEngine:
            mock_forgetter = MagicMock()
            mock_forgetter.complete_forgetting.return_value = mock_result
            MockForgettingEngine.return_value = mock_forgetter

            with patch.object(consolidation_engine, "_rebuild_rag_index"):
                result = await consolidation_engine.monthly_forget()

        mock_forgetter.complete_forgetting.assert_called_once()
        assert result["forgotten_chunks"] == 5
        assert len(result["archived_files"]) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
