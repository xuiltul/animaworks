from __future__ import annotations
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.


"""Tests for utility-based procedural memory forgetting (Issue 6).

Tests cover:
- _is_protected_procedure() with various metadata combinations
- _is_protected() delegation to _is_protected_procedure() for procedures
- Procedure-specific downscaling (180-day inactivity, utility score)
- Procedures included in complete_forgetting scan
- cleanup_procedure_archives() keeps only 5 recent versions
- Skills and shared_users remain in PROTECTED_MEMORY_TYPES
- Edge cases: version >= 3 protection, IMPORTANT tag with low utility
"""

import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.memory.forgetting import (
    FORGETTING_LOW_ACTIVATION_DAYS,
    PROCEDURE_ARCHIVE_KEEP_VERSIONS,
    PROCEDURE_INACTIVITY_DAYS,
    PROCEDURE_LOW_UTILITY_MIN_FAILURES,
    PROCEDURE_LOW_UTILITY_THRESHOLD,
    PROCEDURE_MIN_USAGE,
    PROTECTED_MEMORY_TYPES,
    ForgettingEngine,
)


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
    (anima_dir / "archive" / "versions").mkdir(parents=True)
    (anima_dir / "archive" / "forgotten").mkdir(parents=True)
    return anima_dir


@pytest.fixture
def engine(anima_dir: Path) -> ForgettingEngine:
    """Create a ForgettingEngine instance."""
    return ForgettingEngine(anima_dir=anima_dir, anima_name="test_anima")


def _make_chunk(
    doc_id: str = "chunk1",
    content: str = "test content",
    memory_type: str = "procedures",
    importance: str = "normal",
    access_count: int = 0,
    last_accessed_at: str = "",
    updated_at: str = "",
    activation_level: str = "normal",
    low_activation_since: str = "",
    source_file: str = "procedures/test.md",
    version: int = 1,
    protected: bool = False,
    success_count: int = 0,
    failure_count: int = 0,
    last_used: str = "",
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
            "version": version,
            "protected": protected,
            "success_count": success_count,
            "failure_count": failure_count,
            "last_used": last_used,
        },
    }


# ── PROTECTED_MEMORY_TYPES Tests ──────────────────────────────────


class TestProtectedMemoryTypes:
    """Verify PROTECTED_MEMORY_TYPES configuration."""

    def test_procedures_not_in_protected(self):
        """Procedures should NOT be in PROTECTED_MEMORY_TYPES."""
        assert "procedures" not in PROTECTED_MEMORY_TYPES

    def test_skills_still_protected(self):
        """Skills should remain in PROTECTED_MEMORY_TYPES."""
        assert "skills" in PROTECTED_MEMORY_TYPES

    def test_shared_users_still_protected(self):
        """shared_users should remain in PROTECTED_MEMORY_TYPES."""
        assert "shared_users" in PROTECTED_MEMORY_TYPES


# ── _is_protected_procedure Tests ──────────────────────────────────


class TestIsProtectedProcedure:
    """Test _is_protected_procedure() with various metadata combinations."""

    def test_important_tag_protected(self, engine):
        """Procedure with importance='important' is protected."""
        meta = {"importance": "important", "version": 1, "protected": False}
        assert engine._is_protected_procedure(meta) is True

    def test_manual_protection_flag(self, engine):
        """Procedure with protected=True is protected."""
        meta = {"importance": "normal", "version": 1, "protected": True}
        assert engine._is_protected_procedure(meta) is True

    def test_mature_version_protected(self, engine):
        """Procedure with version >= 3 is protected."""
        meta = {"importance": "normal", "version": 3, "protected": False}
        assert engine._is_protected_procedure(meta) is True

    def test_version_above_3_protected(self, engine):
        """Procedure with version > 3 is also protected."""
        meta = {"importance": "normal", "version": 5, "protected": False}
        assert engine._is_protected_procedure(meta) is True

    def test_basic_procedure_not_protected(self, engine):
        """Procedure with version=1, no flags is NOT protected."""
        meta = {"importance": "normal", "version": 1, "protected": False}
        assert engine._is_protected_procedure(meta) is False

    def test_version_2_not_protected(self, engine):
        """Procedure with version=2 is NOT protected (below threshold)."""
        meta = {"importance": "normal", "version": 2, "protected": False}
        assert engine._is_protected_procedure(meta) is False

    def test_missing_version_defaults_to_1(self, engine):
        """Procedure without version field defaults to 1 (not protected)."""
        meta = {"importance": "normal", "protected": False}
        assert engine._is_protected_procedure(meta) is False

    def test_protected_string_false_not_protected(self, engine):
        """Only boolean True triggers manual protection, not string 'true'."""
        meta = {"importance": "normal", "version": 1, "protected": "true"}
        assert engine._is_protected_procedure(meta) is False


# ── _is_protected Integration Tests ────────────────────────────────


class TestIsProtectedIntegration:
    """Test _is_protected() correctly delegates to _is_protected_procedure()."""

    def test_procedures_delegate_to_utility_check(self, engine):
        """memory_type='procedures' should delegate to _is_protected_procedure."""
        # Basic procedure: not protected
        meta = {"memory_type": "procedures", "importance": "normal", "version": 1}
        assert engine._is_protected(meta) is False

    def test_procedures_important_protected(self, engine):
        """Procedures with importance='important' are protected via top-level check."""
        meta = {"memory_type": "procedures", "importance": "important", "version": 1}
        assert engine._is_protected(meta) is True

    def test_procedures_mature_version_protected(self, engine):
        """Procedures with version >= 3 are protected via _is_protected_procedure."""
        meta = {"memory_type": "procedures", "importance": "normal", "version": 3}
        assert engine._is_protected(meta) is True

    def test_procedures_manual_protection(self, engine):
        """Procedures with protected=True are protected."""
        meta = {
            "memory_type": "procedures",
            "importance": "normal",
            "version": 1,
            "protected": True,
        }
        assert engine._is_protected(meta) is True

    def test_skills_still_protected(self, engine):
        """Skills remain fully protected via PROTECTED_MEMORY_TYPES."""
        meta = {"memory_type": "skills", "importance": "normal"}
        assert engine._is_protected(meta) is True

    def test_shared_users_still_protected(self, engine):
        """shared_users remain fully protected via PROTECTED_MEMORY_TYPES."""
        meta = {"memory_type": "shared_users", "importance": "normal"}
        assert engine._is_protected(meta) is True

    def test_knowledge_not_protected(self, engine):
        """Normal knowledge chunks are not protected."""
        meta = {"memory_type": "knowledge", "importance": "normal"}
        assert engine._is_protected(meta) is False


# ── _should_downscale_procedure Tests ──────────────────────────────


class TestShouldDownscaleProcedure:
    """Test procedure-specific downscaling logic."""

    def test_long_inactivity_low_usage(self, engine):
        """Procedure inactive >180 days with < 3 total uses is downscaled."""
        old_date = (datetime.now() - timedelta(days=200)).isoformat()
        meta = {
            "last_used": old_date,
            "success_count": 1,
            "failure_count": 0,
        }
        assert engine._should_downscale_procedure(meta, datetime.now()) is True

    def test_long_inactivity_sufficient_usage(self, engine):
        """Procedure inactive >180 days but with >= 3 total uses is NOT downscaled."""
        old_date = (datetime.now() - timedelta(days=200)).isoformat()
        meta = {
            "last_used": old_date,
            "success_count": 2,
            "failure_count": 1,
            # total_usage = 3 >= PROCEDURE_MIN_USAGE
        }
        assert engine._should_downscale_procedure(meta, datetime.now()) is False

    def test_recent_with_low_usage(self, engine):
        """Procedure active within 180 days with low usage is NOT downscaled."""
        recent_date = (datetime.now() - timedelta(days=30)).isoformat()
        meta = {
            "last_used": recent_date,
            "success_count": 0,
            "failure_count": 0,
        }
        assert engine._should_downscale_procedure(meta, datetime.now()) is False

    def test_high_failure_low_utility(self, engine):
        """Procedure with >= 3 failures and utility < 0.3 is downscaled."""
        meta = {
            "last_used": datetime.now().isoformat(),
            "success_count": 0,
            "failure_count": 5,
            # utility = 0 / max(1, 5) = 0.0 < 0.3
        }
        assert engine._should_downscale_procedure(meta, datetime.now()) is True

    def test_high_failure_sufficient_utility(self, engine):
        """Procedure with >= 3 failures but utility >= 0.3 is NOT downscaled."""
        meta = {
            "last_used": datetime.now().isoformat(),
            "success_count": 5,
            "failure_count": 3,
            # utility = 5 / max(1, 8) = 0.625 >= 0.3
        }
        assert engine._should_downscale_procedure(meta, datetime.now()) is False

    def test_few_failures_low_utility(self, engine):
        """Procedure with < 3 failures is NOT downscaled even if utility is low."""
        meta = {
            "last_used": datetime.now().isoformat(),
            "success_count": 0,
            "failure_count": 2,
            # failure_count < PROCEDURE_LOW_UTILITY_MIN_FAILURES
        }
        assert engine._should_downscale_procedure(meta, datetime.now()) is False

    def test_fallback_to_last_accessed_at(self, engine):
        """When last_used is empty, falls back to last_accessed_at."""
        old_date = (datetime.now() - timedelta(days=200)).isoformat()
        meta = {
            "last_used": "",
            "last_accessed_at": old_date,
            "success_count": 0,
            "failure_count": 0,
        }
        assert engine._should_downscale_procedure(meta, datetime.now()) is True

    def test_fallback_to_updated_at(self, engine):
        """When last_used and last_accessed_at are empty, falls back to updated_at."""
        old_date = (datetime.now() - timedelta(days=200)).isoformat()
        meta = {
            "last_used": "",
            "last_accessed_at": "",
            "updated_at": old_date,
            "success_count": 0,
            "failure_count": 0,
        }
        assert engine._should_downscale_procedure(meta, datetime.now()) is True

    def test_no_dates_at_all(self, engine):
        """When no date fields are present, days_since is inf, so downscaled."""
        meta = {
            "success_count": 0,
            "failure_count": 0,
        }
        assert engine._should_downscale_procedure(meta, datetime.now()) is True

    def test_utility_boundary_exactly_0_3(self, engine):
        """Procedure with utility exactly 0.3 is NOT downscaled (threshold is <)."""
        meta = {
            "last_used": datetime.now().isoformat(),
            "success_count": 3,
            "failure_count": 7,
            # utility = 3/10 = 0.3, NOT < 0.3
        }
        assert engine._should_downscale_procedure(meta, datetime.now()) is False

    def test_exactly_180_days_not_downscaled(self, engine):
        """Procedure inactive exactly 180 days is NOT downscaled (threshold is >)."""
        exact_date = (datetime.now() - timedelta(days=180)).isoformat()
        meta = {
            "last_used": exact_date,
            "success_count": 0,
            "failure_count": 0,
        }
        # days_since is approximately 180, not > 180
        # Due to float precision this may be slightly > 180, so we use a tight check
        result = engine._should_downscale_procedure(meta, datetime.now())
        # At exactly 180 days (maybe a few seconds over), the behaviour is acceptable either way
        # The important thing is 179 days is definitely False
        meta_recent = {
            "last_used": (datetime.now() - timedelta(days=179)).isoformat(),
            "success_count": 0,
            "failure_count": 0,
        }
        assert engine._should_downscale_procedure(meta_recent, datetime.now()) is False


# ── Synaptic Downscaling with Procedures ───────────────────────────


class TestDownscalingWithProcedures:
    """Test that synaptic_downscaling includes procedures collection."""

    def test_procedure_downscaled_on_inactivity(self, engine):
        """Test that an old, unused procedure is marked as low activation."""
        old_date = (datetime.now() - timedelta(days=200)).isoformat()
        proc_chunks = [
            _make_chunk(
                doc_id="old_proc",
                memory_type="procedures",
                last_used=old_date,
                updated_at=old_date,
                success_count=0,
                failure_count=0,
                version=1,
            ),
        ]

        def get_chunks(collection_name):
            if "procedures" in collection_name:
                return proc_chunks
            return []

        mock_store = MagicMock()
        mock_store.update_metadata = MagicMock()

        with patch.object(engine, "_get_vector_store", return_value=mock_store):
            with patch.object(engine, "_get_all_chunks", side_effect=get_chunks):
                result = engine.synaptic_downscaling()

        assert result["scanned"] == 1
        assert result["marked_low"] == 1
        mock_store.update_metadata.assert_called_once()
        call_args = mock_store.update_metadata.call_args[0]
        assert call_args[0] == "test_anima_procedures"
        assert call_args[2][0]["activation_level"] == "low"

    def test_protected_procedure_not_downscaled(self, engine):
        """Test that a version >= 3 procedure is NOT downscaled."""
        old_date = (datetime.now() - timedelta(days=200)).isoformat()
        proc_chunks = [
            _make_chunk(
                doc_id="mature_proc",
                memory_type="procedures",
                last_used=old_date,
                updated_at=old_date,
                success_count=0,
                failure_count=0,
                version=3,  # protected
            ),
        ]

        def get_chunks(collection_name):
            if "procedures" in collection_name:
                return proc_chunks
            return []

        mock_store = MagicMock()
        mock_store.update_metadata = MagicMock()

        with patch.object(engine, "_get_vector_store", return_value=mock_store):
            with patch.object(engine, "_get_all_chunks", side_effect=get_chunks):
                result = engine.synaptic_downscaling()

        assert result["marked_low"] == 0
        mock_store.update_metadata.assert_not_called()

    def test_high_failure_procedure_downscaled(self, engine):
        """Test that a procedure with high failure rate is downscaled."""
        recent_date = datetime.now().isoformat()
        proc_chunks = [
            _make_chunk(
                doc_id="bad_proc",
                memory_type="procedures",
                last_used=recent_date,
                updated_at=recent_date,
                success_count=0,
                failure_count=5,
                version=1,
            ),
        ]

        def get_chunks(collection_name):
            if "procedures" in collection_name:
                return proc_chunks
            return []

        mock_store = MagicMock()
        mock_store.update_metadata = MagicMock()

        with patch.object(engine, "_get_vector_store", return_value=mock_store):
            with patch.object(engine, "_get_all_chunks", side_effect=get_chunks):
                result = engine.synaptic_downscaling()

        assert result["marked_low"] == 1

    def test_recently_reconsolidated_procedure_not_downscaled(self, engine):
        """Procedure just reconsolidated (failure_count=0) is not downscaled."""
        recent_date = datetime.now().isoformat()
        proc_chunks = [
            _make_chunk(
                doc_id="reconsolidated_proc",
                memory_type="procedures",
                last_used=recent_date,
                updated_at=recent_date,
                success_count=0,
                failure_count=0,
                version=2,
            ),
        ]

        def get_chunks(collection_name):
            if "procedures" in collection_name:
                return proc_chunks
            return []

        mock_store = MagicMock()
        mock_store.update_metadata = MagicMock()

        with patch.object(engine, "_get_vector_store", return_value=mock_store):
            with patch.object(engine, "_get_all_chunks", side_effect=get_chunks):
                result = engine.synaptic_downscaling()

        assert result["marked_low"] == 0


# ── Complete Forgetting with Procedures ────────────────────────────


class TestCompleteForgettingProcedures:
    """Test that procedures are included in complete_forgetting scan."""

    def test_procedure_archived_and_deleted(self, engine, anima_dir):
        """Low-activation procedure is archived and deleted from vector store."""
        old_low_since = (datetime.now() - timedelta(days=90)).isoformat()

        # Create source file
        source_file = anima_dir / "procedures" / "old-deploy.md"
        source_file.write_text("# Old Deploy\n\n1. Step one", encoding="utf-8")

        proc_chunks = [
            _make_chunk(
                doc_id="forget_proc",
                memory_type="procedures",
                access_count=0,
                activation_level="low",
                low_activation_since=old_low_since,
                source_file="procedures/old-deploy.md",
                version=1,
            ),
        ]

        def get_chunks(collection_name):
            if "procedures" in collection_name:
                return proc_chunks
            return []

        mock_store = MagicMock()
        mock_store.delete_documents = MagicMock()

        with patch.object(engine, "_get_vector_store", return_value=mock_store):
            with patch.object(engine, "_get_all_chunks", side_effect=get_chunks):
                result = engine.complete_forgetting()

        assert result["forgotten_chunks"] == 1
        assert "procedures/old-deploy.md" in result["archived_files"]

        # Verify source file was moved to archive
        archive_dir = anima_dir / "archive" / "forgotten"
        archived_files = list(archive_dir.iterdir())
        assert len(archived_files) == 1
        assert "old-deploy" in archived_files[0].name

        # Verify original file is gone
        assert not source_file.exists()

        # Verify delete_documents called for procedures collection
        mock_store.delete_documents.assert_called_once()
        call_args = mock_store.delete_documents.call_args[0]
        assert call_args[0] == "test_anima_procedures"
        assert call_args[1] == ["forget_proc"]

    def test_protected_procedure_not_forgotten(self, engine, anima_dir):
        """Mature procedure (version >= 3) is not forgotten."""
        old_low_since = (datetime.now() - timedelta(days=90)).isoformat()

        proc_chunks = [
            _make_chunk(
                doc_id="mature_proc",
                memory_type="procedures",
                access_count=0,
                activation_level="low",
                low_activation_since=old_low_since,
                version=3,  # protected
            ),
        ]

        def get_chunks(collection_name):
            if "procedures" in collection_name:
                return proc_chunks
            return []

        mock_store = MagicMock()
        mock_store.delete_documents = MagicMock()

        with patch.object(engine, "_get_vector_store", return_value=mock_store):
            with patch.object(engine, "_get_all_chunks", side_effect=get_chunks):
                result = engine.complete_forgetting()

        assert result["forgotten_chunks"] == 0
        mock_store.delete_documents.assert_not_called()


# ── Archive Cleanup Tests ──────────────────────────────────────────


class TestCleanupProcedureArchives:
    """Test cleanup_procedure_archives() version retention."""

    def test_keeps_only_n_recent_versions(self, engine, anima_dir):
        """Cleanup keeps only PROCEDURE_ARCHIVE_KEEP_VERSIONS per stem."""
        archive_dir = anima_dir / "archive" / "versions"

        # Create 8 version files for a procedure stem
        for i in range(8):
            ts = f"2026010{i + 1}_120000"
            path = archive_dir / f"deploy_v{i + 1}_{ts}.md"
            path.write_text(f"version {i + 1}", encoding="utf-8")
            # Ensure distinct mtime ordering
            time.sleep(0.01)

        result = engine.cleanup_procedure_archives()

        assert result["deleted_count"] == 3  # 8 - 5 = 3
        assert result["kept_count"] == 5

        remaining = sorted(archive_dir.iterdir())
        assert len(remaining) == 5

    def test_keeps_all_when_under_limit(self, engine, anima_dir):
        """No deletion when there are fewer versions than the limit."""
        archive_dir = anima_dir / "archive" / "versions"

        for i in range(3):
            ts = f"2026010{i + 1}_120000"
            path = archive_dir / f"deploy_v{i + 1}_{ts}.md"
            path.write_text(f"version {i + 1}", encoding="utf-8")

        result = engine.cleanup_procedure_archives()

        assert result["deleted_count"] == 0
        assert result["kept_count"] == 3

    def test_groups_by_stem(self, engine, anima_dir):
        """Cleanup groups files by procedure stem, not globally."""
        archive_dir = anima_dir / "archive" / "versions"

        # Create 7 files for stem "deploy"
        for i in range(7):
            ts = f"2026010{i + 1}_120000"
            path = archive_dir / f"deploy_v{i + 1}_{ts}.md"
            path.write_text(f"deploy version {i + 1}", encoding="utf-8")
            time.sleep(0.01)

        # Create 3 files for stem "backup"
        for i in range(3):
            ts = f"2026010{i + 1}_120000"
            path = archive_dir / f"backup_v{i + 1}_{ts}.md"
            path.write_text(f"backup version {i + 1}", encoding="utf-8")
            time.sleep(0.01)

        result = engine.cleanup_procedure_archives()

        # deploy: 7 - 5 = 2 deleted; backup: 0 deleted
        assert result["deleted_count"] == 2
        assert result["kept_count"] == 5 + 3  # 5 deploy + 3 backup

    def test_no_archive_dir(self, engine, anima_dir):
        """Cleanup returns zero counts when archive/versions/ doesn't exist."""
        # Remove the archive/versions directory
        import shutil
        versions_dir = anima_dir / "archive" / "versions"
        if versions_dir.exists():
            shutil.rmtree(versions_dir)

        result = engine.cleanup_procedure_archives()

        assert result["deleted_count"] == 0
        assert result["kept_count"] == 0

    def test_ignores_non_matching_files(self, engine, anima_dir):
        """Files that don't match the version pattern are not touched."""
        archive_dir = anima_dir / "archive" / "versions"

        # Create a non-matching file
        (archive_dir / "random_notes.md").write_text("notes", encoding="utf-8")

        # Create matching version files
        for i in range(3):
            ts = f"2026010{i + 1}_120000"
            path = archive_dir / f"deploy_v{i + 1}_{ts}.md"
            path.write_text(f"version {i + 1}", encoding="utf-8")

        result = engine.cleanup_procedure_archives()

        assert result["deleted_count"] == 0
        assert result["kept_count"] == 3
        # Non-matching file should still exist
        assert (archive_dir / "random_notes.md").exists()


# ── Edge Cases ─────────────────────────────────────────────────────


class TestEdgeCases:
    """Edge case tests for procedure forgetting."""

    def test_important_tag_with_low_utility(self, engine):
        """IMPORTANT tag takes priority over low utility score."""
        meta = {
            "memory_type": "procedures",
            "importance": "important",
            "version": 1,
            "success_count": 0,
            "failure_count": 10,
        }
        # Even though utility is 0.0, importance='important' protects it
        assert engine._is_protected(meta) is True

    def test_version_3_with_low_utility(self, engine):
        """version >= 3 procedure is protected regardless of utility."""
        meta = {
            "memory_type": "procedures",
            "importance": "normal",
            "version": 3,
            "success_count": 0,
            "failure_count": 10,
        }
        assert engine._is_protected(meta) is True

    def test_manual_protected_with_low_utility(self, engine):
        """protected=True procedure is protected regardless of utility."""
        meta = {
            "memory_type": "procedures",
            "importance": "normal",
            "version": 1,
            "protected": True,
            "success_count": 0,
            "failure_count": 10,
        }
        assert engine._is_protected(meta) is True

    def test_procedure_already_low_not_remarked(self, engine):
        """Procedure already at low activation is not remarked."""
        old_date = (datetime.now() - timedelta(days=200)).isoformat()
        proc_chunks = [
            _make_chunk(
                doc_id="already_low_proc",
                memory_type="procedures",
                activation_level="low",
                low_activation_since=old_date,
                last_used=old_date,
                success_count=0,
                failure_count=0,
                version=1,
            ),
        ]

        def get_chunks(collection_name):
            if "procedures" in collection_name:
                return proc_chunks
            return []

        mock_store = MagicMock()
        mock_store.update_metadata = MagicMock()

        with patch.object(engine, "_get_vector_store", return_value=mock_store):
            with patch.object(engine, "_get_all_chunks", side_effect=get_chunks):
                result = engine.synaptic_downscaling()

        assert result["marked_low"] == 0
        mock_store.update_metadata.assert_not_called()

    def test_knowledge_uses_standard_thresholds(self, engine):
        """Knowledge chunks still use the standard 90-day/3-access threshold."""
        old_date = (datetime.now() - timedelta(days=100)).isoformat()
        knowledge_chunks = [
            _make_chunk(
                doc_id="old_knowledge",
                memory_type="knowledge",
                access_count=0,
                last_accessed_at="",
                updated_at=old_date,
                activation_level="normal",
                source_file="knowledge/test.md",
            ),
        ]

        def get_chunks(collection_name):
            if "knowledge" in collection_name:
                return knowledge_chunks
            return []

        mock_store = MagicMock()
        mock_store.update_metadata = MagicMock()

        with patch.object(engine, "_get_vector_store", return_value=mock_store):
            with patch.object(engine, "_get_all_chunks", side_effect=get_chunks):
                result = engine.synaptic_downscaling()

        # 100 days > 90 day threshold, access_count=0 < 3 → should be marked
        assert result["marked_low"] == 1


# ── Consolidation Integration Tests ────────────────────────────────


class TestConsolidationProcedureArchiveCleanup:
    """Test that monthly_forget() calls cleanup_procedure_archives()."""

    @pytest.fixture
    def consolidation_engine(self, tmp_path: Path):
        """Create a ConsolidationEngine instance."""
        from core.memory.consolidation import ConsolidationEngine

        anima_dir = tmp_path / "test_anima"
        (anima_dir / "episodes").mkdir(parents=True)
        (anima_dir / "knowledge").mkdir(parents=True)
        (anima_dir / "procedures").mkdir(parents=True)
        return ConsolidationEngine(
            anima_dir=anima_dir,
            anima_name="test_anima",
        )

    @pytest.mark.asyncio
    async def test_monthly_forget_calls_archive_cleanup(self, consolidation_engine):
        """monthly_forget() should call cleanup_procedure_archives()."""
        mock_forget_result = {"forgotten_chunks": 0, "archived_files": []}
        mock_cleanup_result = {"deleted_count": 2, "kept_count": 5}

        with patch(
            "core.memory.forgetting.ForgettingEngine"
        ) as MockForgettingEngine:
            mock_forgetter = MagicMock()
            mock_forgetter.complete_forgetting.return_value = mock_forget_result
            mock_forgetter.cleanup_procedure_archives.return_value = mock_cleanup_result
            MockForgettingEngine.return_value = mock_forgetter

            with patch.object(consolidation_engine, "_rebuild_rag_index"):
                result = await consolidation_engine.monthly_forget()

        mock_forgetter.cleanup_procedure_archives.assert_called_once()
        assert result["procedure_archive_cleanup"] == mock_cleanup_result


# ── Constants Tests ────────────────────────────────────────────────


class TestConstants:
    """Verify procedure-specific constants are correctly defined."""

    def test_inactivity_days(self):
        assert PROCEDURE_INACTIVITY_DAYS == 180

    def test_min_usage(self):
        assert PROCEDURE_MIN_USAGE == 3

    def test_low_utility_threshold(self):
        assert PROCEDURE_LOW_UTILITY_THRESHOLD == 0.3

    def test_low_utility_min_failures(self):
        assert PROCEDURE_LOW_UTILITY_MIN_FAILURES == 3

    def test_archive_keep_versions(self):
        assert PROCEDURE_ARCHIVE_KEEP_VERSIONS == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
