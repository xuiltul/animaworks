from __future__ import annotations
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""E2E tests for stale task cleanup fixes.

Validates:
  1. Stale RUNNING tasks (>48h) are cleaned up as crash orphans
  2. Recent RUNNING tasks (<48h) are preserved during cleanup

NOTE: TestSuffixedEpisodeCompression and TestEpisodeCompressionBackup were
removed because _compress_old_episodes is no longer a method on
ConsolidationEngine. Episode compression is now handled by the Anima's
tool-call loop via run_consolidation(). See core/memory/consolidation.py.
"""

import json
import time
from pathlib import Path

import pytest

pytestmark = pytest.mark.e2e


# ── 1. Stale RUNNING task cleanup ───────────────────────────


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
