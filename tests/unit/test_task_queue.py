from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

from core.memory.task_queue import (
    TaskQueueManager,
    _elapsed_seconds,
    _format_deadline_display,
    _format_elapsed_from_sec,
    _parse_deadline,
    _STALE_TASK_THRESHOLD_SEC,
)
from core.schemas import TaskEntry
from core.time_utils import now_jst

JST = timezone(timedelta(hours=9))


@pytest.fixture
def task_queue(tmp_path):
    """Create a TaskQueueManager with a temp anima dir."""
    anima_dir = tmp_path / "animas" / "test"
    (anima_dir / "state").mkdir(parents=True)
    return TaskQueueManager(anima_dir)


# ── Existing tests (updated: deadline is now mandatory) ──────


class TestAddTask:
    def test_add_task_creates_entry(self, task_queue):
        entry = task_queue.add_task(
            source="human",
            original_instruction="Issue全取得してPR作成",
            assignee="rin",
            summary="Issue取得とPR作成",
            deadline="1h",
        )
        assert isinstance(entry, TaskEntry)
        assert entry.source == "human"
        assert entry.assignee == "rin"
        assert entry.status == "pending"
        assert len(entry.task_id) == 12

    def test_add_task_persists_to_jsonl(self, task_queue):
        task_queue.add_task(
            source="human",
            original_instruction="test",
            assignee="rin",
            summary="test",
            deadline="1h",
        )
        assert task_queue.queue_path.exists()
        lines = task_queue.queue_path.read_text().strip().splitlines()
        assert len(lines) == 1
        data = json.loads(lines[0])
        assert data["source"] == "human"

    def test_add_multiple_tasks(self, task_queue):
        task_queue.add_task(
            source="human", original_instruction="t1", assignee="a",
            summary="s1", deadline="1h",
        )
        task_queue.add_task(
            source="anima", original_instruction="t2", assignee="b",
            summary="s2", deadline="2h",
        )
        tasks = task_queue.list_tasks()
        assert len(tasks) == 2

    def test_add_task_with_relay_chain(self, task_queue):
        entry = task_queue.add_task(
            source="human",
            original_instruction="test",
            assignee="rin",
            summary="test",
            deadline="1h",
            relay_chain=["taka", "sakura", "rin"],
        )
        assert entry.relay_chain == ["taka", "sakura", "rin"]

    def test_add_task_with_deadline(self, task_queue):
        entry = task_queue.add_task(
            source="human",
            original_instruction="test",
            assignee="rin",
            summary="test",
            deadline="2026-03-01T00:00:00",
        )
        assert entry.deadline == "2026-03-01T00:00:00"


class TestUpdateStatus:
    def test_update_status(self, task_queue):
        entry = task_queue.add_task(
            source="human", original_instruction="t", assignee="a",
            summary="s", deadline="1h",
        )
        updated = task_queue.update_status(entry.task_id, "in_progress")
        assert updated is not None
        assert updated.status == "in_progress"

    def test_update_status_persists(self, task_queue):
        entry = task_queue.add_task(
            source="human", original_instruction="t", assignee="a",
            summary="s", deadline="1h",
        )
        task_queue.update_status(entry.task_id, "done")
        # Reload from file
        tasks = task_queue.list_tasks()
        done_tasks = [t for t in tasks if t.status == "done"]
        assert len(done_tasks) == 1

    def test_update_nonexistent_task(self, task_queue):
        result = task_queue.update_status("nonexistent", "done")
        assert result is None

    def test_update_invalid_status(self, task_queue):
        entry = task_queue.add_task(
            source="human", original_instruction="t", assignee="a",
            summary="s", deadline="1h",
        )
        result = task_queue.update_status(entry.task_id, "invalid_status")
        assert result is None

    def test_update_summary(self, task_queue):
        entry = task_queue.add_task(
            source="human", original_instruction="t", assignee="a",
            summary="original", deadline="1h",
        )
        updated = task_queue.update_status(entry.task_id, "in_progress", summary="updated")
        assert updated.summary == "updated"


class TestGetPending:
    def test_get_pending_empty(self, task_queue):
        assert task_queue.get_pending() == []

    def test_get_pending_filters_done(self, task_queue):
        e1 = task_queue.add_task(
            source="human", original_instruction="t1", assignee="a",
            summary="s1", deadline="1h",
        )
        e2 = task_queue.add_task(
            source="human", original_instruction="t2", assignee="b",
            summary="s2", deadline="1h",
        )
        task_queue.update_status(e1.task_id, "done")
        pending = task_queue.get_pending()
        assert len(pending) == 1
        assert pending[0].task_id == e2.task_id

    def test_get_pending_includes_in_progress(self, task_queue):
        e1 = task_queue.add_task(
            source="human", original_instruction="t1", assignee="a",
            summary="s1", deadline="1h",
        )
        task_queue.update_status(e1.task_id, "in_progress")
        pending = task_queue.get_pending()
        assert len(pending) == 1


class TestGetHumanTasks:
    def test_get_human_tasks_filters_source(self, task_queue):
        task_queue.add_task(
            source="human", original_instruction="t1", assignee="a",
            summary="s1", deadline="1h",
        )
        task_queue.add_task(
            source="anima", original_instruction="t2", assignee="b",
            summary="s2", deadline="1h",
        )
        human = task_queue.get_human_tasks()
        assert len(human) == 1
        assert human[0].source == "human"


class TestFormatForPriming:
    def test_format_empty(self, task_queue):
        assert task_queue.format_for_priming() == ""

    def test_format_human_high_priority(self, task_queue):
        task_queue.add_task(
            source="human", original_instruction="t", assignee="a",
            summary="Important task", deadline="1h",
        )
        output = task_queue.format_for_priming()
        assert "\U0001f534 HIGH" in output
        assert "Important task" in output

    def test_format_anima_normal_priority(self, task_queue):
        task_queue.add_task(
            source="anima", original_instruction="t", assignee="a",
            summary="Normal task", deadline="1h",
        )
        output = task_queue.format_for_priming()
        assert "\u26aa" in output
        assert "Normal task" in output

    def test_format_respects_budget(self, task_queue):
        # Add many tasks
        for i in range(50):
            task_queue.add_task(
                source="human", original_instruction=f"task {i}", assignee="a",
                summary=f"Very long task description number {i} with lots of detail",
                deadline="1h",
            )
        output = task_queue.format_for_priming(budget_tokens=100)
        # Should be limited by budget (100 tokens * 4 chars = 400 chars)
        assert len(output) <= 500  # Some margin

    def test_format_shows_relay_chain(self, task_queue):
        task_queue.add_task(
            source="human", original_instruction="t", assignee="rin",
            summary="Delegated task", deadline="1h",
            relay_chain=["taka", "sakura", "rin"],
        )
        output = task_queue.format_for_priming()
        assert "chain:" in output
        assert "taka" in output


class TestCompact:
    def test_compact_removes_done_tasks(self, task_queue):
        e1 = task_queue.add_task(
            source="human", original_instruction="t1", assignee="a",
            summary="s1", deadline="1h",
        )
        e2 = task_queue.add_task(
            source="human", original_instruction="t2", assignee="b",
            summary="s2", deadline="1h",
        )
        task_queue.update_status(e1.task_id, "done")
        removed = task_queue.compact()
        assert removed == 1
        tasks = task_queue.list_tasks()
        assert len(tasks) == 1
        assert tasks[0].task_id == e2.task_id

    def test_compact_removes_cancelled_tasks(self, task_queue):
        e1 = task_queue.add_task(
            source="anima", original_instruction="t1", assignee="a",
            summary="s1", deadline="1h",
        )
        task_queue.update_status(e1.task_id, "cancelled")
        removed = task_queue.compact()
        assert removed == 1
        assert task_queue.list_tasks() == []

    def test_compact_no_terminal_tasks(self, task_queue):
        task_queue.add_task(
            source="human", original_instruction="t1", assignee="a",
            summary="s1", deadline="1h",
        )
        removed = task_queue.compact()
        assert removed == 0

    def test_compact_empty_queue(self, task_queue):
        removed = task_queue.compact()
        assert removed == 0


class TestSourceValidation:
    def test_invalid_source_raises(self, task_queue):
        with pytest.raises(ValueError, match="Invalid source"):
            task_queue.add_task(
                source="invalid",
                original_instruction="t",
                assignee="a",
                summary="s",
                deadline="1h",
            )

    def test_valid_sources(self, task_queue):
        e1 = task_queue.add_task(
            source="human", original_instruction="t1", assignee="a",
            summary="s1", deadline="1h",
        )
        e2 = task_queue.add_task(
            source="anima", original_instruction="t2", assignee="b",
            summary="s2", deadline="1h",
        )
        assert e1.source == "human"
        assert e2.source == "anima"


class TestInstructionSizeCap:
    def test_long_instruction_truncated(self, task_queue):
        long_text = "x" * 20_000
        entry = task_queue.add_task(
            source="human",
            original_instruction=long_text,
            assignee="a",
            summary="s",
            deadline="1h",
        )
        assert len(entry.original_instruction) == 10_000


class TestCorruptedFile:
    def test_corrupted_line_skipped(self, task_queue):
        task_queue.queue_path.parent.mkdir(parents=True, exist_ok=True)
        # Write a corrupted line + valid line
        task_queue.add_task(
            source="human", original_instruction="valid", assignee="a",
            summary="valid task", deadline="1h",
        )
        with task_queue.queue_path.open("a") as f:
            f.write("THIS IS NOT VALID JSON\n")
        tasks = task_queue.list_tasks()
        assert len(tasks) == 1
        assert tasks[0].summary == "valid task"


# ── New tests: _parse_deadline ───────────────────────────────


class TestParseDeadline:
    """Tests for the module-level _parse_deadline() function."""

    def test_parse_relative_minutes(self):
        """'30m' should produce an ISO8601 timestamp ~30 minutes from now."""
        before = now_jst()
        result = _parse_deadline("30m")
        after = now_jst()
        parsed = datetime.fromisoformat(result)
        # parsed should be ~30 minutes after 'before'
        assert parsed >= before + timedelta(minutes=29, seconds=59)
        assert parsed <= after + timedelta(minutes=30, seconds=1)

    def test_parse_relative_hours(self):
        """'2h' should produce an ISO8601 timestamp ~2 hours from now."""
        before = now_jst()
        result = _parse_deadline("2h")
        after = now_jst()
        parsed = datetime.fromisoformat(result)
        assert parsed >= before + timedelta(hours=1, minutes=59, seconds=59)
        assert parsed <= after + timedelta(hours=2, seconds=1)

    def test_parse_relative_days(self):
        """'1d' should produce an ISO8601 timestamp ~1 day from now."""
        before = now_jst()
        result = _parse_deadline("1d")
        after = now_jst()
        parsed = datetime.fromisoformat(result)
        assert parsed >= before + timedelta(days=1) - timedelta(seconds=1)
        assert parsed <= after + timedelta(days=1) + timedelta(seconds=1)

    def test_parse_iso8601_passthrough(self):
        """An ISO8601 string should pass through unchanged."""
        iso_str = "2026-03-01T14:00:00"
        result = _parse_deadline(iso_str)
        assert result == iso_str

    def test_parse_invalid_raises(self):
        """An unrecognised format should raise ValueError."""
        with pytest.raises(ValueError, match="Invalid deadline format"):
            _parse_deadline("abc")

    def test_parse_empty_string_raises_in_add_task(self, task_queue):
        """An empty deadline string should raise ValueError in add_task."""
        with pytest.raises(ValueError, match="deadline is required"):
            task_queue.add_task(
                source="human",
                original_instruction="test",
                assignee="rin",
                summary="test",
                deadline="",
            )


# ── New tests: deadline mandatory in add_task ────────────────


class TestDeadlineMandatory:
    """Tests that deadline is a required parameter for add_task()."""

    def test_add_task_without_deadline_raises(self, task_queue):
        """Calling add_task without a deadline keyword raises TypeError (missing arg)."""
        with pytest.raises(TypeError):
            task_queue.add_task(
                source="human",
                original_instruction="test",
                assignee="rin",
                summary="test",
            )

    def test_add_task_with_relative_deadline_converts(self, task_queue):
        """A relative deadline ('1h') should be converted to ISO8601 in the stored entry."""
        before = now_jst()
        entry = task_queue.add_task(
            source="human",
            original_instruction="test",
            assignee="rin",
            summary="test",
            deadline="1h",
        )
        after = now_jst()
        # The stored deadline should be a valid ISO8601 timestamp
        parsed = datetime.fromisoformat(entry.deadline)
        assert parsed >= before + timedelta(minutes=59, seconds=59)
        assert parsed <= after + timedelta(hours=1, seconds=1)

    def test_add_task_with_iso8601_deadline(self, task_queue):
        """An ISO8601 deadline should be stored as-is."""
        entry = task_queue.add_task(
            source="human",
            original_instruction="test",
            assignee="rin",
            summary="test",
            deadline="2026-03-01T14:00:00",
        )
        assert entry.deadline == "2026-03-01T14:00:00"


# ── New tests: format_for_priming with staleness/deadline ────


class TestFormatForPrimingWithStaleness:
    """Tests for staleness and deadline markers in format_for_priming().

    Uses unittest.mock.patch to control datetime.now() in the task_queue module.
    Tasks are written directly to JSONL with specific timestamps.
    """

    def _write_task_entry(self, task_queue, *, updated_at, deadline=None):
        """Write a task entry directly to JSONL with controlled timestamps."""
        import uuid

        task_id = uuid.uuid4().hex[:12]
        entry = {
            "task_id": task_id,
            "ts": updated_at,
            "source": "human",
            "original_instruction": "test instruction",
            "assignee": "rin",
            "status": "pending",
            "summary": "Test task",
            "deadline": deadline,
            "relay_chain": [],
            "updated_at": updated_at,
        }
        task_queue.queue_path.parent.mkdir(parents=True, exist_ok=True)
        with task_queue.queue_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        return task_id

    def test_format_shows_elapsed_time(self, task_queue):
        """Output should contain elapsed time indicator."""
        now = datetime(2026, 3, 1, 12, 0, 0, tzinfo=JST)
        # Task updated 15 minutes ago
        updated_at = (now - timedelta(minutes=15)).isoformat()
        self._write_task_entry(task_queue, updated_at=updated_at, deadline="2026-03-01T14:00:00+09:00")

        with patch("core.memory.task_queue.now_jst", return_value=now):
            output = task_queue.format_for_priming()

        assert "\u23f1\ufe0f 15\u5206\u7d4c\u904e" in output

    def test_format_shows_stale_marker(self, task_queue):
        """Task updated 45 minutes ago should show STALE marker."""
        now = datetime(2026, 3, 1, 12, 0, 0, tzinfo=JST)
        updated_at = (now - timedelta(minutes=45)).isoformat()
        self._write_task_entry(task_queue, updated_at=updated_at, deadline="2026-03-01T14:00:00+09:00")

        with patch("core.memory.task_queue.now_jst", return_value=now):
            output = task_queue.format_for_priming()

        assert "\u26a0\ufe0f STALE" in output

    def test_format_no_stale_for_recent_task(self, task_queue):
        """Task updated 5 minutes ago should NOT show STALE marker."""
        now = datetime(2026, 3, 1, 12, 0, 0, tzinfo=JST)
        updated_at = (now - timedelta(minutes=5)).isoformat()
        self._write_task_entry(task_queue, updated_at=updated_at, deadline="2026-03-01T14:00:00+09:00")

        with patch("core.memory.task_queue.now_jst", return_value=now):
            output = task_queue.format_for_priming()

        assert "\u26a0\ufe0f STALE" not in output

    def test_format_shows_deadline(self, task_queue):
        """Task with a future deadline should show deadline display."""
        now = datetime(2026, 3, 1, 12, 0, 0, tzinfo=JST)
        updated_at = (now - timedelta(minutes=5)).isoformat()
        self._write_task_entry(
            task_queue, updated_at=updated_at,
            deadline="2026-03-01T14:30:00+09:00",
        )

        with patch("core.memory.task_queue.now_jst", return_value=now):
            output = task_queue.format_for_priming()

        assert "\U0001f4c5 14:30\u307e\u3067" in output

    def test_format_shows_overdue(self, task_queue):
        """Task with a past deadline should show OVERDUE marker."""
        now = datetime(2026, 3, 1, 15, 0, 0, tzinfo=JST)
        updated_at = (now - timedelta(minutes=10)).isoformat()
        self._write_task_entry(
            task_queue, updated_at=updated_at,
            deadline="2026-03-01T14:00:00+09:00",
        )

        with patch("core.memory.task_queue.now_jst", return_value=now):
            output = task_queue.format_for_priming()

        assert "\U0001f534 OVERDUE" in output

    def test_format_handles_null_deadline(self, task_queue):
        """Existing task with deadline=None should not crash format_for_priming."""
        now = datetime(2026, 3, 1, 12, 0, 0, tzinfo=JST)
        updated_at = (now - timedelta(minutes=5)).isoformat()
        self._write_task_entry(task_queue, updated_at=updated_at, deadline=None)

        with patch("core.memory.task_queue.now_jst", return_value=now):
            output = task_queue.format_for_priming()

        # Should produce output without crashing; no deadline markers
        assert "Test task" in output
        assert "\U0001f4c5" not in output
        assert "\U0001f534 OVERDUE" not in output

    def test_format_handles_invalid_updated_at(self, task_queue):
        """Task with non-ISO updated_at should not crash format_for_priming."""
        # Write a task entry with an invalid updated_at directly
        entry = {
            "task_id": "abc123def456",
            "ts": "not-a-date",
            "source": "human",
            "original_instruction": "test",
            "assignee": "rin",
            "status": "pending",
            "summary": "Bad timestamp task",
            "deadline": None,
            "relay_chain": [],
            "updated_at": "not-a-valid-iso-timestamp",
        }
        task_queue.queue_path.parent.mkdir(parents=True, exist_ok=True)
        with task_queue.queue_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        now = datetime(2026, 3, 1, 12, 0, 0, tzinfo=JST)
        with patch("core.memory.task_queue.now_jst", return_value=now):
            # Should not raise
            output = task_queue.format_for_priming()

        assert "Bad timestamp task" in output


# ── New tests: get_stale_tasks ───────────────────────────────


class TestGetStaleTasks:
    """Tests for the get_stale_tasks() method."""

    def _write_task_entry(self, task_queue, *, updated_at, status="pending"):
        """Write a task entry directly to JSONL with controlled timestamps."""
        import uuid

        task_id = uuid.uuid4().hex[:12]
        entry = {
            "task_id": task_id,
            "ts": updated_at,
            "source": "human",
            "original_instruction": "test",
            "assignee": "rin",
            "status": status,
            "summary": "Stale test task",
            "deadline": "2026-03-01T23:59:59",
            "relay_chain": [],
            "updated_at": updated_at,
        }
        task_queue.queue_path.parent.mkdir(parents=True, exist_ok=True)
        with task_queue.queue_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        return task_id

    def test_returns_stale_tasks(self, task_queue):
        """Tasks with updated_at older than 30 minutes should be returned."""
        now = datetime(2026, 3, 1, 12, 0, 0, tzinfo=JST)
        old_time = (now - timedelta(minutes=45)).isoformat()
        task_id = self._write_task_entry(task_queue, updated_at=old_time)

        with patch("core.memory.task_queue.now_jst", return_value=now):
            stale = task_queue.get_stale_tasks()

        assert len(stale) == 1
        assert stale[0].task_id == task_id

    def test_excludes_recent_tasks(self, task_queue):
        """Recently updated tasks should not be returned as stale."""
        now = datetime(2026, 3, 1, 12, 0, 0, tzinfo=JST)
        recent_time = (now - timedelta(minutes=10)).isoformat()
        self._write_task_entry(task_queue, updated_at=recent_time)

        with patch("core.memory.task_queue.now_jst", return_value=now):
            stale = task_queue.get_stale_tasks()

        assert len(stale) == 0

    def test_empty_queue_returns_empty(self, task_queue):
        """An empty task queue should return an empty list."""
        stale = task_queue.get_stale_tasks()
        assert stale == []

    def test_excludes_done_tasks(self, task_queue):
        """Done tasks should not appear in stale results (get_pending filters them)."""
        now = datetime(2026, 3, 1, 12, 0, 0, tzinfo=JST)
        old_time = (now - timedelta(minutes=45)).isoformat()
        self._write_task_entry(task_queue, updated_at=old_time, status="done")

        with patch("core.memory.task_queue.now_jst", return_value=now):
            stale = task_queue.get_stale_tasks()

        assert len(stale) == 0

    def test_stale_threshold_is_30_minutes(self):
        """Verify the stale threshold constant is 1800 seconds (30 minutes)."""
        assert _STALE_TASK_THRESHOLD_SEC == 1800


# ── New tests: helper function unit tests ────────────────────


class TestElapsedSeconds:
    """Tests for the _elapsed_seconds() helper."""

    def test_valid_timestamps(self):
        now = datetime(2026, 3, 1, 12, 0, 0, tzinfo=JST)
        updated_at = "2026-03-01T11:30:00+09:00"
        result = _elapsed_seconds(updated_at, now)
        assert result == 1800.0

    def test_invalid_timestamp_returns_none(self):
        now = datetime(2026, 3, 1, 12, 0, 0, tzinfo=JST)
        result = _elapsed_seconds("not-valid", now)
        assert result is None

    def test_none_timestamp_returns_none(self):
        now = datetime(2026, 3, 1, 12, 0, 0, tzinfo=JST)
        result = _elapsed_seconds(None, now)
        assert result is None


class TestFormatElapsedFromSec:
    """Tests for the _format_elapsed_from_sec() helper."""

    def test_minutes_only(self):
        result = _format_elapsed_from_sec(13 * 60)  # 13 minutes
        assert result == "\u23f1\ufe0f 13\u5206\u7d4c\u904e"

    def test_hours_and_minutes(self):
        result = _format_elapsed_from_sec(2 * 3600 + 15 * 60)  # 2h15m
        assert result == "\u23f1\ufe0f 2\u6642\u959315\u5206\u7d4c\u904e"

    def test_exact_hours(self):
        result = _format_elapsed_from_sec(2 * 3600)  # exactly 2h
        assert result == "\u23f1\ufe0f 2\u6642\u9593\u7d4c\u904e"

    def test_none_returns_empty(self):
        result = _format_elapsed_from_sec(None)
        assert result == ""

    def test_negative_returns_empty(self):
        result = _format_elapsed_from_sec(-300)
        assert result == ""


class TestFormatDeadlineDisplay:
    """Tests for the _format_deadline_display() helper."""

    def test_future_deadline(self):
        now = datetime(2026, 3, 1, 12, 0, 0, tzinfo=JST)
        result = _format_deadline_display("2026-03-01T14:30:00+09:00", now)
        assert result == "\U0001f4c5 14:30\u307e\u3067"

    def test_past_deadline(self):
        now = datetime(2026, 3, 1, 15, 0, 0, tzinfo=JST)
        result = _format_deadline_display("2026-03-01T14:00:00+09:00", now)
        assert result == "\U0001f534 OVERDUE(14:00\u671f\u9650)"

    def test_invalid_deadline_returns_empty(self):
        now = datetime(2026, 3, 1, 12, 0, 0, tzinfo=JST)
        result = _format_deadline_display("not-valid", now)
        assert result == ""

    def test_none_deadline_returns_empty(self):
        now = datetime(2026, 3, 1, 12, 0, 0, tzinfo=JST)
        result = _format_deadline_display(None, now)
        assert result == ""
