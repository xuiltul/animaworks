"""Tests for Channel E task_results reading in priming.py.

Verifies that _channel_e_pending_tasks() includes completed background
task results from state/task_results/ in the Priming output.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from core.memory.priming import PrimingEngine


@pytest.fixture
def anima_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        d = Path(tmpdir) / "animas" / "test"
        for sub in ["episodes", "knowledge", "skills", "state"]:
            (d / sub).mkdir(parents=True)
        yield d


class TestChannelETaskResults:
    @pytest.mark.asyncio
    async def test_no_task_results_dir(self, anima_dir: Path):
        """No task_results dir → no error, empty or just queue content."""
        engine = PrimingEngine(anima_dir)
        result = await engine._channel_e_pending_tasks()
        assert "完了済みバックグラウンドタスク" not in result

    @pytest.mark.asyncio
    async def test_empty_task_results_dir(self, anima_dir: Path):
        """Empty task_results dir → no completed tasks section."""
        (anima_dir / "state" / "task_results").mkdir(parents=True)
        engine = PrimingEngine(anima_dir)
        result = await engine._channel_e_pending_tasks()
        assert "完了済みバックグラウンドタスク" not in result

    @pytest.mark.asyncio
    async def test_single_task_result(self, anima_dir: Path):
        """One task result file → appears in output."""
        results_dir = anima_dir / "state" / "task_results"
        results_dir.mkdir(parents=True)

        (results_dir / "abc123.md").write_text(
            "# Task Result\nSuccessfully completed the research task.",
            encoding="utf-8",
        )

        engine = PrimingEngine(anima_dir)
        result = await engine._channel_e_pending_tasks()
        assert "完了済みバックグラウンドタスク" in result
        assert "abc123" in result
        assert "Successfully completed" in result

    @pytest.mark.asyncio
    async def test_multiple_task_results_limited_to_5(self, anima_dir: Path):
        """Only the 5 most recent task results should appear."""
        import time

        results_dir = anima_dir / "state" / "task_results"
        results_dir.mkdir(parents=True)

        for i in range(8):
            f = results_dir / f"task_{i:03d}.md"
            f.write_text(f"Result for task {i}", encoding="utf-8")
            time.sleep(0.01)

        engine = PrimingEngine(anima_dir)
        result = await engine._channel_e_pending_tasks()

        assert "完了済みバックグラウンドタスク" in result
        # The 5 most recent (003-007) should be present
        assert "task_007" in result
        assert "task_006" in result
        assert "task_005" in result
        assert "task_004" in result
        assert "task_003" in result
        # The oldest 3 (000-002) should NOT be present
        assert "task_000" not in result
        assert "task_001" not in result
        assert "task_002" not in result

    @pytest.mark.asyncio
    async def test_task_result_preview_truncated(self, anima_dir: Path):
        """Long task result content is truncated to 150 chars."""
        results_dir = anima_dir / "state" / "task_results"
        results_dir.mkdir(parents=True)

        long_content = "x" * 300
        (results_dir / "longresult.md").write_text(long_content, encoding="utf-8")

        engine = PrimingEngine(anima_dir)
        result = await engine._channel_e_pending_tasks()

        # The preview should be truncated — "x" * 300 becomes "x" * 150
        assert "完了済みバックグラウンドタスク" in result
        assert "longresult" in result

    @pytest.mark.asyncio
    async def test_task_results_coexist_with_queue(self, anima_dir: Path):
        """Task results should appear alongside pending queue entries."""
        from core.memory.task_queue import TaskQueueManager

        manager = TaskQueueManager(anima_dir)
        manager.add_task(
            source="human",
            original_instruction="Pending task",
            assignee="test",
            summary="Pending summary",
        )

        results_dir = anima_dir / "state" / "task_results"
        results_dir.mkdir(parents=True)
        (results_dir / "done_task.md").write_text(
            "Task completed successfully",
            encoding="utf-8",
        )

        engine = PrimingEngine(anima_dir)
        result = await engine._channel_e_pending_tasks()

        assert "Pending summary" in result
        assert "完了済みバックグラウンドタスク" in result
        assert "done_task" in result

    @pytest.mark.asyncio
    async def test_non_md_files_ignored(self, anima_dir: Path):
        """Only .md files should be read from task_results/."""
        results_dir = anima_dir / "state" / "task_results"
        results_dir.mkdir(parents=True)

        (results_dir / "result.md").write_text("Good result", encoding="utf-8")
        (results_dir / "result.json").write_text('{"status": "ok"}', encoding="utf-8")
        (results_dir / "result.txt").write_text("Text file", encoding="utf-8")

        engine = PrimingEngine(anima_dir)
        result = await engine._channel_e_pending_tasks()

        assert "Good result" in result
        # JSON and TXT files should not appear
        assert "status" not in result
        assert "Text file" not in result
