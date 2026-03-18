from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from core.memory.priming import PrimingEngine, PrimingResult, _BUDGET_PENDING_TASKS


@pytest.fixture
def temp_anima_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        anima_dir = Path(tmpdir) / "animas" / "test"
        for d in ["episodes", "knowledge", "skills", "state"]:
            (anima_dir / d).mkdir(parents=True)
        yield anima_dir


class TestPrimingResultWithPendingTasks:
    def test_pending_tasks_field(self):
        result = PrimingResult(pending_tasks="test tasks")
        assert result.pending_tasks == "test tasks"

    def test_is_empty_with_pending_tasks(self):
        result = PrimingResult(pending_tasks="task")
        assert not result.is_empty()

    def test_is_empty_without_pending_tasks(self):
        result = PrimingResult()
        assert result.is_empty()

    def test_total_chars_includes_pending(self):
        result = PrimingResult(pending_tasks="12345")
        assert result.total_chars() >= 5


class TestChannelE:
    @pytest.mark.asyncio
    async def test_channel_e_no_tasks(self, temp_anima_dir):
        engine = PrimingEngine(temp_anima_dir)
        result = await engine._channel_e_pending_tasks()
        assert result == ""

    @pytest.mark.asyncio
    async def test_channel_e_with_tasks(self, temp_anima_dir):
        from core.memory.task_queue import TaskQueueManager
        manager = TaskQueueManager(temp_anima_dir)
        manager.add_task(
            source="human",
            original_instruction="Test task",
            assignee="rin",
            summary="Test summary",
            deadline="1h",
        )
        engine = PrimingEngine(temp_anima_dir)
        result = await engine._channel_e_pending_tasks()
        assert "🔴 HIGH" in result
        assert "Test summary" in result

    def test_budget_constant(self):
        assert _BUDGET_PENDING_TASKS == 500


class TestChannelEOverflowInbox:
    @pytest.mark.asyncio
    async def test_no_overflow_dir(self, temp_anima_dir):
        engine = PrimingEngine(temp_anima_dir)
        result = await engine._channel_e_pending_tasks()
        assert "overflow_inbox" not in result

    @pytest.mark.asyncio
    async def test_empty_overflow_dir(self, temp_anima_dir):
        (temp_anima_dir / "state" / "overflow_inbox").mkdir(parents=True)
        engine = PrimingEngine(temp_anima_dir)
        result = await engine._channel_e_pending_tasks()
        assert "overflow_inbox" not in result

    @pytest.mark.asyncio
    async def test_overflow_files_shown(self, temp_anima_dir):
        overflow_dir = temp_anima_dir / "state" / "overflow_inbox"
        overflow_dir.mkdir(parents=True)
        for i in range(3):
            (overflow_dir / f"20260318_1200_sender{i}.md").write_text(
                f"---\nfrom: sender{i}\n---\ntest message {i}",
                encoding="utf-8",
            )
        engine = PrimingEngine(temp_anima_dir)
        result = await engine._channel_e_pending_tasks()
        assert "3" in result
        assert "overflow_inbox" in result
        assert "read_memory_file" in result
        assert "archive_memory_file" in result

    @pytest.mark.asyncio
    async def test_overflow_more_than_5_shows_count(self, temp_anima_dir):
        overflow_dir = temp_anima_dir / "state" / "overflow_inbox"
        overflow_dir.mkdir(parents=True)
        for i in range(8):
            (overflow_dir / f"20260318_1200_sender{i}.md").write_text(
                f"---\nfrom: sender{i}\n---\nmsg",
                encoding="utf-8",
            )
        engine = PrimingEngine(temp_anima_dir)
        result = await engine._channel_e_pending_tasks()
        assert "8" in result
        assert "(+3)" in result

    @pytest.mark.asyncio
    async def test_non_md_files_ignored(self, temp_anima_dir):
        overflow_dir = temp_anima_dir / "state" / "overflow_inbox"
        overflow_dir.mkdir(parents=True)
        (overflow_dir / "not_a_message.txt").write_text("ignored")
        (overflow_dir / "actual.md").write_text("---\nfrom: a\n---\nmsg")
        engine = PrimingEngine(temp_anima_dir)
        result = await engine._channel_e_pending_tasks()
        assert "1 " in result or "1件" in result


class TestPrimeMemoriesIncludesChannelE:
    @pytest.mark.asyncio
    async def test_prime_memories_returns_pending_tasks(self, temp_anima_dir):
        from core.memory.task_queue import TaskQueueManager
        manager = TaskQueueManager(temp_anima_dir)
        manager.add_task(
            source="human",
            original_instruction="Important task",
            assignee="test",
            summary="Important task summary",
            deadline="1h",
        )

        engine = PrimingEngine(temp_anima_dir)
        result = await engine.prime_memories("hello", sender_name="test")
        assert result.pending_tasks != ""
        assert "Important task" in result.pending_tasks
