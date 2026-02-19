from __future__ import annotations
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""E2E tests for episode compression and stale task cleanup fixes.

Validates:
  1. _compress_old_episodes() handles suffixed filenames (YYYY-MM-DD_xxx.md)
  2. Original episode is backed up before LLM compression overwrites it
  3. Stale RUNNING tasks (>48h) are cleaned up as crash orphans
  4. Recent RUNNING tasks (<48h) are preserved during cleanup
"""

import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

pytestmark = pytest.mark.e2e


# ── Helpers ──────────────────────────────────────────────────


def _make_compress_mock_response(summary_text: str) -> MagicMock:
    """Build a mock litellm.acompletion return value for compression."""
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = summary_text
    return mock_response


# ── 1. Suffixed episode compression ─────────────────────────


class TestSuffixedEpisodeCompression:
    """_compress_old_episodes() handles YYYY-MM-DD_xxx.md filenames."""

    @pytest.mark.asyncio
    async def test_compress_suffixed_episodes(self, tmp_path: Path) -> None:
        """Suffixed episode files older than retention are compressed."""
        from core.memory.consolidation import ConsolidationEngine

        anima_dir = tmp_path / "test_anima"
        episodes_dir = anima_dir / "episodes"
        knowledge_dir = anima_dir / "knowledge"
        episodes_dir.mkdir(parents=True)
        knowledge_dir.mkdir(parents=True)

        engine = ConsolidationEngine(
            anima_dir=anima_dir, anima_name="test_anima",
        )

        # Create a suffixed episode file older than retention period
        old_date = (datetime.now() - timedelta(days=60)).strftime("%Y-%m-%d")
        suffixed = episodes_dir / f"{old_date}_heartbeat.md"
        suffixed.write_text(
            "## Heartbeat check\nAll systems normal.",
            encoding="utf-8",
        )

        mock_response = _make_compress_mock_response(
            f"## {old_date} \u8981\u7d04\n- \u30b7\u30b9\u30c6\u30e0\u6b63\u5e38\u7a3c\u50cd",
        )

        with patch(
            "litellm.acompletion",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            result = await engine._compress_old_episodes(retention_days=30)

        assert result > 0, "Should have compressed at least one episode"

        # Verify the file was compressed
        content = suffixed.read_text(encoding="utf-8")
        assert "[COMPRESSED:" in content

    @pytest.mark.asyncio
    async def test_compress_skips_recent_suffixed_episodes(
        self, tmp_path: Path,
    ) -> None:
        """Suffixed episodes within retention period are not compressed."""
        from core.memory.consolidation import ConsolidationEngine

        anima_dir = tmp_path / "test_anima"
        episodes_dir = anima_dir / "episodes"
        knowledge_dir = anima_dir / "knowledge"
        episodes_dir.mkdir(parents=True)
        knowledge_dir.mkdir(parents=True)

        engine = ConsolidationEngine(
            anima_dir=anima_dir, anima_name="test_anima",
        )

        # Create a suffixed episode file within retention period
        recent_date = (datetime.now() - timedelta(days=5)).strftime("%Y-%m-%d")
        suffixed = episodes_dir / f"{recent_date}_cron.md"
        original_content = "## Cron run\nBatch processed 100 items."
        suffixed.write_text(original_content, encoding="utf-8")

        with patch(
            "litellm.acompletion",
            new_callable=AsyncMock,
        ) as mock_llm:
            result = await engine._compress_old_episodes(retention_days=30)

        assert result == 0, "Should not compress recent episodes"
        mock_llm.assert_not_awaited()

        # Content should be unchanged
        assert suffixed.read_text(encoding="utf-8") == original_content


# ── 2. Episode compression backup ───────────────────────────


class TestEpisodeCompressionBackup:
    """Original episode is backed up to archive/episodes/ before compression."""

    @pytest.mark.asyncio
    async def test_compress_backup_before_overwrite(
        self, tmp_path: Path,
    ) -> None:
        """Backup is created in archive/episodes/ with original content."""
        from core.memory.consolidation import ConsolidationEngine

        anima_dir = tmp_path / "test_anima"
        episodes_dir = anima_dir / "episodes"
        knowledge_dir = anima_dir / "knowledge"
        episodes_dir.mkdir(parents=True)
        knowledge_dir.mkdir(parents=True)

        engine = ConsolidationEngine(
            anima_dir=anima_dir, anima_name="test_anima",
        )

        old_date = (datetime.now() - timedelta(days=60)).strftime("%Y-%m-%d")
        episode = episodes_dir / f"{old_date}.md"
        original_content = (
            "## Original episode content\nImportant details here."
        )
        episode.write_text(original_content, encoding="utf-8")

        mock_response = _make_compress_mock_response(
            f"## {old_date} \u8981\u7d04\n- \u8981\u70b9\u306e\u307f",
        )

        with patch(
            "litellm.acompletion",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            result = await engine._compress_old_episodes(retention_days=30)

        assert result > 0

        # Verify backup exists
        backup_dir = anima_dir / "archive" / "episodes"
        assert backup_dir.exists(), "Backup directory should be created"
        backups = list(backup_dir.iterdir())
        assert len(backups) == 1, (
            f"Should have exactly one backup, got {len(backups)}"
        )

        # Verify backup contains original content
        backup_content = backups[0].read_text(encoding="utf-8")
        assert backup_content == original_content

        # Verify original was compressed
        compressed = episode.read_text(encoding="utf-8")
        assert "[COMPRESSED:" in compressed

    @pytest.mark.asyncio
    async def test_compress_backup_suffixed_episode(
        self, tmp_path: Path,
    ) -> None:
        """Backup also works for suffixed episode filenames."""
        from core.memory.consolidation import ConsolidationEngine

        anima_dir = tmp_path / "test_anima"
        episodes_dir = anima_dir / "episodes"
        knowledge_dir = anima_dir / "knowledge"
        episodes_dir.mkdir(parents=True)
        knowledge_dir.mkdir(parents=True)

        engine = ConsolidationEngine(
            anima_dir=anima_dir, anima_name="test_anima",
        )

        old_date = (datetime.now() - timedelta(days=60)).strftime("%Y-%m-%d")
        suffixed = episodes_dir / f"{old_date}_heartbeat.md"
        original_content = "## Heartbeat\nCheck completed successfully."
        suffixed.write_text(original_content, encoding="utf-8")

        mock_response = _make_compress_mock_response(
            f"## {old_date} \u8981\u7d04\n- \u30c1\u30a7\u30c3\u30af\u5b8c\u4e86",
        )

        with patch(
            "litellm.acompletion",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            result = await engine._compress_old_episodes(retention_days=30)

        assert result > 0

        # Verify backup was created with original filename
        backup_dir = anima_dir / "archive" / "episodes"
        backup_file = backup_dir / f"{old_date}_heartbeat.md"
        assert backup_file.exists(), (
            f"Backup should preserve original filename: {old_date}_heartbeat.md"
        )
        assert backup_file.read_text(encoding="utf-8") == original_content


# ── 3. Stale RUNNING task cleanup ───────────────────────────


class TestStaleRunningTaskCleanup:
    """Tasks running >48h are treated as crash orphans and cleaned up."""

    def test_cleanup_stale_running_tasks(self, tmp_path: Path) -> None:
        """Stale running tasks (>48h) are removed; recent ones preserved."""
        from core.background import BackgroundTaskManager

        anima_dir = tmp_path / "animas" / "cleanup-stale"
        anima_dir.mkdir(parents=True)
        (anima_dir / "state").mkdir()

        mgr = BackgroundTaskManager(
            anima_dir, anima_name="cleanup-stale",
            eligible_tools={"some_tool": 5},
        )

        bg_dir = anima_dir / "state" / "background_tasks"

        # Create a stale running task (created 72 hours ago)
        stale_task = {
            "task_id": "stale-123",
            "anima_name": "cleanup-stale",
            "tool_name": "some_tool",
            "tool_args": {},
            "status": "running",
            "created_at": time.time() - (72 * 3600),
        }
        (bg_dir / "stale-123.json").write_text(
            json.dumps(stale_task), encoding="utf-8",
        )

        # Create a recent running task (created 1 hour ago)
        recent_task = {
            "task_id": "recent-456",
            "anima_name": "cleanup-stale",
            "tool_name": "some_tool",
            "tool_args": {},
            "status": "running",
            "created_at": time.time() - (1 * 3600),
        }
        (bg_dir / "recent-456.json").write_text(
            json.dumps(recent_task), encoding="utf-8",
        )

        # Create a completed task older than max_age
        completed_task = {
            "task_id": "done-789",
            "anima_name": "cleanup-stale",
            "tool_name": "some_tool",
            "tool_args": {},
            "status": "completed",
            "created_at": time.time() - (72 * 3600),
            "completed_at": time.time() - (72 * 3600),
        }
        (bg_dir / "done-789.json").write_text(
            json.dumps(completed_task), encoding="utf-8",
        )

        removed = mgr.cleanup_old_tasks(max_age_hours=24)

        # Stale running and old completed should be removed
        assert removed >= 2, (
            f"Should remove stale running + old completed, removed={removed}"
        )
        # Recent running should still exist
        assert (bg_dir / "recent-456.json").exists(), (
            "Recent running task should NOT be removed"
        )
        # Stale running should be gone
        assert not (bg_dir / "stale-123.json").exists(), (
            "Stale running task should be removed"
        )
        # Old completed should be gone
        assert not (bg_dir / "done-789.json").exists(), (
            "Old completed task should be removed"
        )


# ── 4. Recent RUNNING task preservation ──────────────────────


class TestRecentRunningTaskPreservation:
    """Tasks running <48h should NOT be cleaned up."""

    def test_cleanup_preserves_recent_running_tasks(
        self, tmp_path: Path,
    ) -> None:
        """A running task from 12 hours ago is preserved by cleanup."""
        from core.background import BackgroundTaskManager

        anima_dir = tmp_path / "animas" / "cleanup-preserve"
        anima_dir.mkdir(parents=True)
        (anima_dir / "state").mkdir()

        mgr = BackgroundTaskManager(
            anima_dir, anima_name="cleanup-preserve",
            eligible_tools={"some_tool": 5},
        )

        bg_dir = anima_dir / "state" / "background_tasks"

        # Create a running task from 12 hours ago
        task = {
            "task_id": "active-001",
            "anima_name": "cleanup-preserve",
            "tool_name": "some_tool",
            "tool_args": {},
            "status": "running",
            "created_at": time.time() - (12 * 3600),
        }
        (bg_dir / "active-001.json").write_text(
            json.dumps(task), encoding="utf-8",
        )

        removed = mgr.cleanup_old_tasks(max_age_hours=24)

        assert removed == 0
        assert (bg_dir / "active-001.json").exists()
