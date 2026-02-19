from __future__ import annotations
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""E2E tests for forgetting pipeline fixes.

Tests three aspects of the forgetting pipeline that were fixed:
1. Stage 2: _sync_merged_source_files() archives originals and writes merged content
2. Stage 3: complete_forgetting() deletes vectors before archiving files
3. Monthly forgetting schedule: ProcessSupervisor registers the monthly job
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ── Test 1: Stage 2 — _sync_merged_source_files ──────────────────


class TestSyncMergedSourceFiles:
    """Verify _sync_merged_source_files archives originals and writes merged content."""

    def test_sync_merged_source_files(self, tmp_path: Path) -> None:
        """After neurogenesis merge, source files are updated on disk."""
        from core.memory.forgetting import ForgettingEngine

        anima_dir = tmp_path / "test_anima"
        (anima_dir / "knowledge").mkdir(parents=True)

        # Create two source files
        file_a = anima_dir / "knowledge" / "topic-a.md"
        file_b = anima_dir / "knowledge" / "topic-b.md"
        file_a.write_text("Original content A", encoding="utf-8")
        file_b.write_text("Original content B", encoding="utf-8")

        engine = ForgettingEngine(anima_dir, "test_anima")

        chunk_a = {"id": "chunk_a", "metadata": {"source_file": "knowledge/topic-a.md"}}
        chunk_b = {"id": "chunk_b", "metadata": {"source_file": "knowledge/topic-b.md"}}
        merged = "Merged content from A and B"

        engine._sync_merged_source_files(chunk_a, chunk_b, merged)

        # Primary file should have merged content
        assert file_a.read_text(encoding="utf-8") == merged
        # Secondary file should be deleted
        assert not file_b.exists()
        # Archive should have both originals
        archive_dir = anima_dir / "archive" / "merged"
        assert archive_dir.exists()
        archived = list(archive_dir.iterdir())
        assert len(archived) == 2

    def test_sync_merged_source_files_skips_empty_source(self, tmp_path: Path) -> None:
        """When source_file is empty or 'merged', no files are touched."""
        from core.memory.forgetting import ForgettingEngine

        anima_dir = tmp_path / "test_anima"
        anima_dir.mkdir(parents=True)

        engine = ForgettingEngine(anima_dir, "test_anima")

        chunk_a = {"id": "a", "metadata": {"source_file": ""}}
        chunk_b = {"id": "b", "metadata": {"source_file": "merged"}}
        # Should not raise
        engine._sync_merged_source_files(chunk_a, chunk_b, "content")

        # No archive directory should be created
        assert not (anima_dir / "archive" / "merged").exists()

    def test_sync_merged_source_files_handles_missing_files(self, tmp_path: Path) -> None:
        """Gracefully handles source files that no longer exist on disk."""
        from core.memory.forgetting import ForgettingEngine

        anima_dir = tmp_path / "test_anima"
        (anima_dir / "knowledge").mkdir(parents=True)

        engine = ForgettingEngine(anima_dir, "test_anima")

        chunk_a = {"id": "a", "metadata": {"source_file": "knowledge/nonexistent-a.md"}}
        chunk_b = {"id": "b", "metadata": {"source_file": "knowledge/nonexistent-b.md"}}

        # Should not raise even though files don't exist
        engine._sync_merged_source_files(chunk_a, chunk_b, "merged content")

        # Primary file should be created with merged content
        primary = anima_dir / "knowledge" / "nonexistent-a.md"
        assert primary.exists()
        assert primary.read_text(encoding="utf-8") == "merged content"

    def test_sync_merged_source_files_same_source(self, tmp_path: Path) -> None:
        """When both chunks come from the same file, only one archive copy is made."""
        from core.memory.forgetting import ForgettingEngine

        anima_dir = tmp_path / "test_anima"
        (anima_dir / "knowledge").mkdir(parents=True)

        source_file = anima_dir / "knowledge" / "shared.md"
        source_file.write_text("Original content", encoding="utf-8")

        engine = ForgettingEngine(anima_dir, "test_anima")

        chunk_a = {"id": "a", "metadata": {"source_file": "knowledge/shared.md"}}
        chunk_b = {"id": "b", "metadata": {"source_file": "knowledge/shared.md"}}

        engine._sync_merged_source_files(chunk_a, chunk_b, "merged")

        # Primary file should have merged content
        assert source_file.read_text(encoding="utf-8") == "merged"
        # Archive should have exactly one copy (secondary == primary, so no removal)
        archive_dir = anima_dir / "archive" / "merged"
        assert archive_dir.exists()
        archived = list(archive_dir.iterdir())
        assert len(archived) == 1


# ── Test 2: Stage 3 — complete_forgetting() archive order ────────


class TestCompleteForgettingOrder:
    """Verify complete_forgetting() deletes vectors FIRST, then archives files."""

    def test_complete_forgetting_skips_archive_on_vector_failure(
        self, tmp_path: Path,
    ) -> None:
        """If vector deletion fails, source files should NOT be archived."""
        from core.memory.forgetting import ForgettingEngine

        anima_dir = tmp_path / "test_anima"
        (anima_dir / "knowledge").mkdir(parents=True)
        source = anima_dir / "knowledge" / "test.md"
        source.write_text("test content", encoding="utf-8")

        engine = ForgettingEngine(anima_dir, "test_anima")

        # Build a mock vector store whose delete_documents always fails.
        # _get_all_chunks is called internally via store.client.get_collection(),
        # so we mock at the _get_all_chunks level AND _get_vector_store level.
        mock_store = MagicMock()
        mock_store.delete_documents.side_effect = Exception("ChromaDB error")

        engine._get_vector_store = lambda: mock_store

        # Mock _get_all_chunks to return a forgettable chunk
        original_get_all_chunks = engine._get_all_chunks

        def fake_get_all_chunks(collection_name: str):
            if "knowledge" in collection_name:
                return [{
                    "id": "chunk1",
                    "metadata": {
                        "memory_type": "knowledge",
                        "activation_level": "low",
                        "low_activation_since": "2025-01-01T00:00:00",
                        "access_count": 0,
                        "source_file": "knowledge/test.md",
                        "importance": "",
                    },
                    "content": "test content",
                }]
            return []

        engine._get_all_chunks = fake_get_all_chunks

        result = engine.complete_forgetting()

        # Source file should still exist (not archived) because vector delete failed
        assert source.exists(), (
            "Source file should NOT be archived when vector deletion fails"
        )
        assert result["forgotten_chunks"] == 0

    def test_complete_forgetting_archives_after_successful_delete(
        self, tmp_path: Path,
    ) -> None:
        """When vector deletion succeeds, source files are archived."""
        from core.memory.forgetting import ForgettingEngine

        anima_dir = tmp_path / "test_anima"
        (anima_dir / "knowledge").mkdir(parents=True)
        source = anima_dir / "knowledge" / "test.md"
        source.write_text("test content", encoding="utf-8")

        engine = ForgettingEngine(anima_dir, "test_anima")

        mock_store = MagicMock()
        mock_store.delete_documents.return_value = None  # Success

        engine._get_vector_store = lambda: mock_store

        def fake_get_all_chunks(collection_name: str):
            if "knowledge" in collection_name:
                return [{
                    "id": "chunk1",
                    "metadata": {
                        "memory_type": "knowledge",
                        "activation_level": "low",
                        "low_activation_since": "2025-01-01T00:00:00",
                        "access_count": 0,
                        "source_file": "knowledge/test.md",
                        "importance": "",
                    },
                    "content": "test content",
                }]
            return []

        engine._get_all_chunks = fake_get_all_chunks

        result = engine.complete_forgetting()

        # Source file should be archived (moved)
        assert not source.exists(), (
            "Source file should be archived after successful vector deletion"
        )
        assert result["forgotten_chunks"] == 1
        assert "knowledge/test.md" in result["archived_files"]

        # Verify archive directory has the file
        archive_dir = anima_dir / "archive" / "forgotten"
        assert archive_dir.exists()
        archived = list(archive_dir.iterdir())
        assert len(archived) == 1

    def test_complete_forgetting_delete_then_archive_ordering(
        self, tmp_path: Path,
    ) -> None:
        """Verify that delete_documents is called before _archive_source_file."""
        from core.memory.forgetting import ForgettingEngine

        anima_dir = tmp_path / "test_anima"
        (anima_dir / "knowledge").mkdir(parents=True)
        source = anima_dir / "knowledge" / "ordered.md"
        source.write_text("content", encoding="utf-8")

        engine = ForgettingEngine(anima_dir, "test_anima")

        call_order: list[str] = []

        mock_store = MagicMock()

        def track_delete(*args, **kwargs):
            call_order.append("vector_delete")

        mock_store.delete_documents.side_effect = track_delete
        engine._get_vector_store = lambda: mock_store

        original_archive = engine._archive_source_file

        def track_archive(rel_path: str):
            call_order.append("file_archive")
            original_archive(rel_path)

        engine._archive_source_file = track_archive

        def fake_get_all_chunks(collection_name: str):
            if "knowledge" in collection_name:
                return [{
                    "id": "chunk1",
                    "metadata": {
                        "memory_type": "knowledge",
                        "activation_level": "low",
                        "low_activation_since": "2025-01-01T00:00:00",
                        "access_count": 0,
                        "source_file": "knowledge/ordered.md",
                        "importance": "",
                    },
                    "content": "content",
                }]
            return []

        engine._get_all_chunks = fake_get_all_chunks

        engine.complete_forgetting()

        assert call_order == ["vector_delete", "file_archive"], (
            f"Expected vector_delete before file_archive, got: {call_order}"
        )


# ── Test 3: Monthly forgetting schedule ───────────────────────────


class TestMonthlyForgettingSchedule:
    """Verify ProcessSupervisor._setup_system_crons registers monthly forgetting."""

    def test_setup_system_crons_registers_monthly(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """ProcessSupervisor._setup_system_crons registers monthly forgetting job."""
        from core.supervisor.manager import ProcessSupervisor

        monkeypatch.setenv("ANIMAWORKS_DATA_DIR", str(tmp_path))

        supervisor = ProcessSupervisor.__new__(ProcessSupervisor)
        supervisor.animas_dir = tmp_path / "animas"
        supervisor.animas_dir.mkdir(parents=True)
        supervisor.processes = {}

        mock_scheduler = MagicMock()
        supervisor.scheduler = mock_scheduler

        with patch("core.config.load_config") as mock_config:
            mock_cfg = MagicMock()
            mock_cfg.consolidation = None
            mock_config.return_value = mock_cfg
            supervisor._setup_system_crons()

        # Extract job IDs from add_job calls
        calls = mock_scheduler.add_job.call_args_list
        job_ids = [call.kwargs.get("id") for call in calls]

        monthly_found = "system_monthly_forgetting" in job_ids
        assert monthly_found, (
            f"Monthly forgetting job should be registered. "
            f"Registered jobs: {job_ids}"
        )

    def test_setup_system_crons_monthly_disabled(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """When monthly_enabled=False, the monthly forgetting job is NOT registered."""
        from core.supervisor.manager import ProcessSupervisor

        monkeypatch.setenv("ANIMAWORKS_DATA_DIR", str(tmp_path))

        supervisor = ProcessSupervisor.__new__(ProcessSupervisor)
        supervisor.animas_dir = tmp_path / "animas"
        supervisor.animas_dir.mkdir(parents=True)
        supervisor.processes = {}

        mock_scheduler = MagicMock()
        supervisor.scheduler = mock_scheduler

        with patch("core.config.load_config") as mock_config:
            mock_cfg = MagicMock()
            mock_consolidation = MagicMock()
            mock_consolidation.monthly_enabled = False
            # Keep defaults for other settings
            mock_consolidation.daily_enabled = True
            mock_consolidation.daily_time = "02:00"
            mock_consolidation.weekly_enabled = True
            mock_consolidation.weekly_time = "sun:03:00"
            mock_consolidation.monthly_time = "1:04:00"
            mock_cfg.consolidation = mock_consolidation
            mock_config.return_value = mock_cfg
            supervisor._setup_system_crons()

        calls = mock_scheduler.add_job.call_args_list
        job_ids = [call.kwargs.get("id") for call in calls]

        assert "system_monthly_forgetting" not in job_ids, (
            f"Monthly forgetting job should NOT be registered when disabled. "
            f"Registered jobs: {job_ids}"
        )

    def test_setup_system_crons_registers_all_three(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """All three system crons (daily, weekly, monthly) are registered by default."""
        from core.supervisor.manager import ProcessSupervisor

        monkeypatch.setenv("ANIMAWORKS_DATA_DIR", str(tmp_path))

        supervisor = ProcessSupervisor.__new__(ProcessSupervisor)
        supervisor.animas_dir = tmp_path / "animas"
        supervisor.animas_dir.mkdir(parents=True)
        supervisor.processes = {}

        mock_scheduler = MagicMock()
        supervisor.scheduler = mock_scheduler

        with patch("core.config.load_config") as mock_config:
            mock_cfg = MagicMock()
            mock_cfg.consolidation = None
            mock_config.return_value = mock_cfg
            supervisor._setup_system_crons()

        calls = mock_scheduler.add_job.call_args_list
        job_ids = [call.kwargs.get("id") for call in calls]

        expected_ids = [
            "system_daily_consolidation",
            "system_weekly_integration",
            "system_monthly_forgetting",
        ]
        for expected_id in expected_ids:
            assert expected_id in job_ids, (
                f"{expected_id} should be registered. Registered: {job_ids}"
            )
