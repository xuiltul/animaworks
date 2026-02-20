"""Unit tests for PendingTaskExecutor."""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.supervisor.pending_executor import PendingTaskExecutor


def _make_executor(tmp_path: Path) -> PendingTaskExecutor:
    """Create a PendingTaskExecutor with minimal dependencies."""
    anima_dir = tmp_path / "animas" / "test-anima"
    anima_dir.mkdir(parents=True, exist_ok=True)
    mock_anima = MagicMock()
    mock_anima.agent.background_manager = MagicMock()

    return PendingTaskExecutor(
        anima=mock_anima,
        anima_name="test-anima",
        anima_dir=anima_dir,
        shutdown_event=asyncio.Event(),
    )


class TestPendingTaskExecutorInit:
    """Test PendingTaskExecutor initialization."""

    def test_creates_instance(self, tmp_path):
        executor = _make_executor(tmp_path)
        assert executor._anima_name == "test-anima"

    def test_independent_instantiation(self, tmp_path):
        """PendingTaskExecutor can be instantiated without AnimaRunner."""
        anima_dir = tmp_path / "animas" / "standalone"
        anima_dir.mkdir(parents=True)
        executor = PendingTaskExecutor(
            anima=MagicMock(),
            anima_name="standalone",
            anima_dir=anima_dir,
            shutdown_event=asyncio.Event(),
        )
        assert executor._anima_name == "standalone"


class TestExecutePendingTask:
    """Test pending task execution."""

    @pytest.mark.asyncio
    async def test_submits_to_background_manager(self, tmp_path):
        """Task should be submitted to BackgroundTaskManager."""
        executor = _make_executor(tmp_path)

        task_desc = {
            "task_id": "test-123",
            "tool_name": "web_search",
            "subcommand": "search",
            "raw_args": ["query"],
            "anima_dir": str(tmp_path / "animas" / "test-anima"),
        }

        await executor.execute_pending_task(task_desc)

        executor._anima.agent.background_manager.submit.assert_called_once()
        call_args = executor._anima.agent.background_manager.submit.call_args
        assert call_args[0][0] == "web_search:search"

    @pytest.mark.asyncio
    async def test_skips_when_no_anima(self, tmp_path):
        """Should skip when anima is None."""
        executor = _make_executor(tmp_path)
        executor._anima = None

        # Should not raise
        await executor.execute_pending_task({"tool_name": "test"})

    @pytest.mark.asyncio
    async def test_skips_when_no_background_manager(self, tmp_path):
        """Should skip when background_manager is None."""
        executor = _make_executor(tmp_path)
        executor._anima.agent.background_manager = None

        # Should not raise
        await executor.execute_pending_task({"tool_name": "test"})


class TestWatcherLoop:
    """Test pending task watcher loop."""

    @pytest.mark.asyncio
    async def test_picks_up_pending_files(self, tmp_path):
        """Watcher should pick up and process .json files in pending dir."""
        executor = _make_executor(tmp_path)
        pending_dir = executor._anima_dir / "state" / "background_tasks" / "pending"
        pending_dir.mkdir(parents=True, exist_ok=True)

        # Write a pending task file
        task = {"task_id": "test-1", "tool_name": "test_tool", "subcommand": "", "raw_args": []}
        (pending_dir / "task1.json").write_text(json.dumps(task))

        # Stop after first iteration
        iteration_count = 0
        original_sleep = asyncio.sleep

        async def mock_sleep(duration):
            nonlocal iteration_count
            iteration_count += 1
            if iteration_count >= 1:
                executor._shutdown_event.set()
            await original_sleep(0)

        with patch("asyncio.sleep", side_effect=mock_sleep):
            await executor.watcher_loop()

        # File should have been removed
        assert not (pending_dir / "task1.json").exists()
        # Background manager should have been called
        executor._anima.agent.background_manager.submit.assert_called_once()

    @pytest.mark.asyncio
    async def test_handles_invalid_json(self, tmp_path):
        """Watcher should handle invalid JSON files gracefully."""
        executor = _make_executor(tmp_path)
        pending_dir = executor._anima_dir / "state" / "background_tasks" / "pending"
        pending_dir.mkdir(parents=True, exist_ok=True)

        # Write invalid JSON
        (pending_dir / "bad.json").write_text("not json")

        iteration_count = 0
        original_sleep = asyncio.sleep

        async def mock_sleep(duration):
            nonlocal iteration_count
            iteration_count += 1
            if iteration_count >= 1:
                executor._shutdown_event.set()
            await original_sleep(0)

        with patch("asyncio.sleep", side_effect=mock_sleep):
            await executor.watcher_loop()

        # Bad file should be removed
        assert not (pending_dir / "bad.json").exists()
