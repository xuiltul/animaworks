"""E2E tests for task staleness detection and delegation workflow.

Verifies the full lifecycle:
1. Task creation with relative deadline → ISO8601 conversion
2. format_for_priming output with elapsed / STALE / OVERDUE markers
3. get_stale_tasks returns correct results
4. Heartbeat delegation prompt injection for animas with subordinates
"""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from core.time_utils import now_jst
from core.memory.task_queue import (
    TaskQueueManager,
    _STALE_TASK_THRESHOLD_SEC,
    _parse_deadline,
)
from core.paths import load_prompt, _prompt_cache
from core.schemas import TaskEntry


# ── Fixtures ──────────────────────────────────────────────────


@pytest.fixture
def anima_dir(tmp_path: Path) -> Path:
    """Create an isolated anima directory with required state subdirectory."""
    d = tmp_path / "animas" / "test-anima"
    (d / "state").mkdir(parents=True)
    return d


@pytest.fixture
def task_queue(anima_dir: Path) -> TaskQueueManager:
    """Create a TaskQueueManager backed by the temp anima directory."""
    return TaskQueueManager(anima_dir)


def _write_task_entry_to_jsonl(
    queue_path: Path,
    *,
    task_id: str,
    source: str = "human",
    original_instruction: str = "test instruction",
    assignee: str = "test-anima",
    status: str = "pending",
    summary: str = "test task",
    deadline: str | None = None,
    relay_chain: list[str] | None = None,
    ts: str | None = None,
    updated_at: str | None = None,
) -> None:
    """Write a raw task entry directly to the JSONL file.

    This allows tests to inject tasks with specific timestamps that would
    be impossible to achieve through the normal add_task() API.
    """
    now = datetime.now().isoformat()
    entry = {
        "task_id": task_id,
        "ts": ts or now,
        "source": source,
        "original_instruction": original_instruction,
        "assignee": assignee,
        "status": status,
        "summary": summary,
        "deadline": deadline,
        "relay_chain": relay_chain or [],
        "updated_at": updated_at or now,
    }
    queue_path.parent.mkdir(parents=True, exist_ok=True)
    with queue_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


# ── Test 1: Full task lifecycle with staleness ─────────────────


class TestFullTaskLifecycleWithStaleness:
    """Verify the complete flow: create task, detect staleness, format for priming."""

    def test_relative_deadline_converted_to_iso8601(self, task_queue: TaskQueueManager):
        """add_task with deadline='1h' produces a valid ISO8601 deadline."""
        before = now_jst()
        entry = task_queue.add_task(
            source="human",
            original_instruction="Deploy to staging",
            assignee="rin",
            summary="Deploy staging",
            deadline="1h",
        )
        after = now_jst()

        # Deadline must be valid ISO8601
        deadline_dt = datetime.fromisoformat(entry.deadline)
        assert deadline_dt >= before + timedelta(hours=1) - timedelta(seconds=2)
        assert deadline_dt <= after + timedelta(hours=1) + timedelta(seconds=2)

    def test_relative_deadline_minutes(self, task_queue: TaskQueueManager):
        """add_task with deadline='30m' resolves correctly."""
        before = now_jst()
        entry = task_queue.add_task(
            source="anima",
            original_instruction="Quick check",
            assignee="sakura",
            summary="Quick check",
            deadline="30m",
        )
        deadline_dt = datetime.fromisoformat(entry.deadline)
        expected_min = before + timedelta(minutes=30) - timedelta(seconds=2)
        expected_max = before + timedelta(minutes=30) + timedelta(seconds=5)
        assert expected_min <= deadline_dt <= expected_max

    def test_relative_deadline_days(self, task_queue: TaskQueueManager):
        """add_task with deadline='1d' resolves correctly."""
        before = now_jst()
        entry = task_queue.add_task(
            source="human",
            original_instruction="Weekly report",
            assignee="taka",
            summary="Weekly report",
            deadline="1d",
        )
        deadline_dt = datetime.fromisoformat(entry.deadline)
        expected_min = before + timedelta(days=1) - timedelta(seconds=2)
        expected_max = before + timedelta(days=1) + timedelta(seconds=5)
        assert expected_min <= deadline_dt <= expected_max

    def test_absolute_iso8601_deadline_accepted(self, task_queue: TaskQueueManager):
        """add_task with an absolute ISO8601 deadline passes through unchanged."""
        iso_deadline = "2026-06-01T12:00:00"
        entry = task_queue.add_task(
            source="human",
            original_instruction="Future task",
            assignee="rin",
            summary="Future task",
            deadline=iso_deadline,
        )
        assert entry.deadline == iso_deadline

    def test_stale_marker_in_format_for_priming(
        self, task_queue: TaskQueueManager, anima_dir: Path,
    ):
        """Old task gets STALE marker; fresh task does not."""
        # Add a fresh task via API
        fresh = task_queue.add_task(
            source="human",
            original_instruction="Fresh task",
            assignee="test-anima",
            summary="Fresh task",
            deadline="2h",
        )

        # Write an old task directly with updated_at 45 minutes ago
        old_updated = (datetime.now() - timedelta(minutes=45)).isoformat()
        old_deadline = (datetime.now() + timedelta(hours=1)).isoformat()
        _write_task_entry_to_jsonl(
            task_queue.queue_path,
            task_id="old_task_001",
            source="human",
            summary="Old stale task",
            assignee="test-anima",
            deadline=old_deadline,
            updated_at=old_updated,
            ts=old_updated,
        )

        output = task_queue.format_for_priming()

        # Old task should have STALE marker
        assert "STALE" in output
        assert "Old stale task" in output

        # Fresh task should NOT have STALE marker
        # Split into lines and check each one
        lines = output.strip().split("\n")
        for line in lines:
            if "Fresh task" in line:
                assert "STALE" not in line, (
                    "Fresh task should not be marked as STALE"
                )

    def test_get_stale_tasks_returns_old_task_only(
        self, task_queue: TaskQueueManager, anima_dir: Path,
    ):
        """get_stale_tasks returns only tasks updated 30+ minutes ago."""
        # Fresh task via API
        task_queue.add_task(
            source="human",
            original_instruction="Fresh",
            assignee="test-anima",
            summary="Fresh task",
            deadline="2h",
        )

        # Old task: updated 45 minutes ago
        old_updated = (datetime.now() - timedelta(minutes=45)).isoformat()
        _write_task_entry_to_jsonl(
            task_queue.queue_path,
            task_id="stale_task_01",
            source="anima",
            summary="Stale task",
            assignee="test-anima",
            deadline=(datetime.now() + timedelta(hours=2)).isoformat(),
            updated_at=old_updated,
            ts=old_updated,
        )

        stale = task_queue.get_stale_tasks()
        assert len(stale) == 1
        assert stale[0].task_id == "stale_task_01"

    def test_get_stale_tasks_empty_when_all_fresh(
        self, task_queue: TaskQueueManager,
    ):
        """No stale tasks when all tasks were just created."""
        task_queue.add_task(
            source="human",
            original_instruction="Just created",
            assignee="test-anima",
            summary="Brand new",
            deadline="1h",
        )
        assert task_queue.get_stale_tasks() == []

    def test_stale_threshold_boundary(
        self, task_queue: TaskQueueManager, anima_dir: Path,
    ):
        """Task updated exactly at the threshold boundary is stale."""
        # Exactly 30 minutes ago (the threshold)
        boundary_updated = (
            datetime.now() - timedelta(seconds=_STALE_TASK_THRESHOLD_SEC)
        ).isoformat()
        _write_task_entry_to_jsonl(
            task_queue.queue_path,
            task_id="boundary_task",
            source="human",
            summary="Boundary task",
            assignee="test-anima",
            deadline=(datetime.now() + timedelta(hours=1)).isoformat(),
            updated_at=boundary_updated,
        )

        stale = task_queue.get_stale_tasks()
        assert len(stale) == 1
        assert stale[0].task_id == "boundary_task"

    def test_done_tasks_not_in_stale_results(
        self, task_queue: TaskQueueManager, anima_dir: Path,
    ):
        """Completed tasks should not appear in stale results even if old."""
        old_updated = (datetime.now() - timedelta(hours=2)).isoformat()
        _write_task_entry_to_jsonl(
            task_queue.queue_path,
            task_id="done_old_task",
            source="human",
            summary="Old done task",
            assignee="test-anima",
            status="done",
            deadline=(datetime.now() - timedelta(hours=1)).isoformat(),
            updated_at=old_updated,
        )

        stale = task_queue.get_stale_tasks()
        assert len(stale) == 0


# ── Test 2: OVERDUE detection ────────────────────────────────


class TestOverdueDetection:
    """Verify that tasks past their deadline are marked OVERDUE in priming output."""

    def test_overdue_marker_in_format_for_priming(
        self, task_queue: TaskQueueManager, anima_dir: Path,
    ):
        """Task with a deadline in the past gets the OVERDUE marker."""
        past_deadline = (datetime.now() - timedelta(hours=1)).isoformat()
        _write_task_entry_to_jsonl(
            task_queue.queue_path,
            task_id="overdue_001",
            source="human",
            summary="Overdue deployment",
            assignee="test-anima",
            deadline=past_deadline,
            updated_at=datetime.now().isoformat(),
        )

        output = task_queue.format_for_priming()
        assert "OVERDUE" in output
        assert "Overdue deployment" in output

    def test_future_deadline_not_overdue(
        self, task_queue: TaskQueueManager,
    ):
        """Task with a future deadline should NOT have OVERDUE marker."""
        entry = task_queue.add_task(
            source="human",
            original_instruction="Future work",
            assignee="test-anima",
            summary="Future deadline task",
            deadline="2h",
        )

        output = task_queue.format_for_priming()
        assert "OVERDUE" not in output
        assert "Future deadline task" in output

    def test_overdue_and_stale_both_appear(
        self, task_queue: TaskQueueManager, anima_dir: Path,
    ):
        """A task that is both stale and overdue should show both markers."""
        old_time = (datetime.now() - timedelta(hours=2)).isoformat()
        past_deadline = (datetime.now() - timedelta(hours=1)).isoformat()
        _write_task_entry_to_jsonl(
            task_queue.queue_path,
            task_id="stale_overdue",
            source="human",
            summary="Stale and overdue",
            assignee="test-anima",
            deadline=past_deadline,
            updated_at=old_time,
            ts=old_time,
        )

        output = task_queue.format_for_priming()
        # Find the line for this task
        lines = output.strip().split("\n")
        task_line = [l for l in lines if "Stale and overdue" in l]
        assert len(task_line) == 1
        assert "STALE" in task_line[0]
        assert "OVERDUE" in task_line[0]

    def test_elapsed_time_displayed(
        self, task_queue: TaskQueueManager, anima_dir: Path,
    ):
        """Priming output includes elapsed time indicator."""
        updated = (datetime.now() - timedelta(minutes=15)).isoformat()
        _write_task_entry_to_jsonl(
            task_queue.queue_path,
            task_id="elapsed_task",
            source="anima",
            summary="Task with elapsed time",
            assignee="test-anima",
            deadline=(datetime.now() + timedelta(hours=1)).isoformat(),
            updated_at=updated,
        )

        output = task_queue.format_for_priming()
        # Should contain elapsed time marker
        assert "15分経過" in output or "14分経過" in output or "16分経過" in output


# ── Test 3: Deadline mandatory enforcement via ToolHandler ────


class TestDeadlineMandatoryEnforcement:
    """Verify that add_task without deadline is rejected at the ToolHandler level."""

    @pytest.fixture
    def handler(self, tmp_path: Path):
        """Create a ToolHandler with minimal setup."""
        from core.memory import MemoryManager
        from core.tooling.handler import ToolHandler

        anima_dir = tmp_path / "animas" / "test-anima"
        for d in ["state", "episodes", "knowledge", "procedures", "skills"]:
            (anima_dir / d).mkdir(parents=True)
        memory = MemoryManager(anima_dir)
        return ToolHandler(anima_dir, memory)

    def test_add_task_without_deadline_returns_error(self, handler):
        """ToolHandler rejects add_task when deadline is missing."""
        result = handler.handle("add_task", {
            "source": "human",
            "original_instruction": "Do something",
            "assignee": "rin",
            "summary": "Something",
            # deadline intentionally omitted
        })
        data = json.loads(result)
        assert data["status"] == "error"
        assert data["error_type"] == "InvalidArguments"
        assert "deadline" in data["message"].lower()

    def test_add_task_with_empty_deadline_returns_error(self, handler):
        """ToolHandler rejects add_task when deadline is empty string."""
        result = handler.handle("add_task", {
            "source": "human",
            "original_instruction": "Do something",
            "assignee": "rin",
            "summary": "Something",
            "deadline": "",
        })
        data = json.loads(result)
        assert data["status"] == "error"
        assert data["error_type"] == "InvalidArguments"
        assert "deadline" in data["message"].lower()

    def test_add_task_with_invalid_deadline_returns_error(self, handler):
        """ToolHandler rejects add_task when deadline format is invalid."""
        result = handler.handle("add_task", {
            "source": "human",
            "original_instruction": "Do something",
            "assignee": "rin",
            "summary": "Something",
            "deadline": "next tuesday",
        })
        data = json.loads(result)
        assert data["status"] == "error"
        assert data["error_type"] == "InvalidArguments"

    def test_add_task_with_valid_deadline_succeeds(self, handler):
        """ToolHandler accepts add_task when deadline is provided."""
        result = handler.handle("add_task", {
            "source": "human",
            "original_instruction": "Deploy to production",
            "assignee": "rin",
            "summary": "Deploy prod",
            "deadline": "2h",
        })
        data = json.loads(result)
        assert "task_id" in data
        assert data["status"] == "pending"
        assert data["deadline"] is not None
        # Deadline should be valid ISO8601
        datetime.fromisoformat(data["deadline"])

    def test_add_task_mandatory_at_task_queue_level(self, task_queue: TaskQueueManager):
        """TaskQueueManager.add_task raises ValueError when deadline is empty."""
        with pytest.raises(ValueError, match="deadline is required"):
            task_queue.add_task(
                source="human",
                original_instruction="No deadline",
                assignee="rin",
                summary="Missing deadline",
                deadline="",
            )


# ── Test 4: Heartbeat delegation prompt injection ────────────


class TestHeartbeatDelegationInjection:
    """Verify load_prompt('heartbeat_subordinate_check', ...) works correctly."""

    @pytest.fixture(autouse=True)
    def _clear_prompt_cache(self):
        """Clear the prompt template cache before and after each test."""
        _prompt_cache.clear()
        yield
        _prompt_cache.clear()

    def test_load_prompt_renders_subordinates(self):
        """load_prompt substitutes {subordinates} into the template."""
        subordinates = "rin, sakura, taka"
        result = load_prompt(
            "heartbeat_subordinate_check",
            subordinates=subordinates,
            animas_dir="/home/test/.animaworks/animas",
        )

        assert "rin, sakura, taka" in result
        assert "STALE" in result
        assert "OVERDUE" in result
        assert "委任" in result
        assert "報告検証" in result
        assert "activity_log" in result

    def test_template_contains_delegation_checklist(self):
        """The delegation template includes key decision criteria."""
        result = load_prompt(
            "heartbeat_subordinate_check",
            subordinates="alice, bob",
            animas_dir="/home/test/.animaworks/animas",
        )

        # Verify the template contains the expected decision sections
        assert "判断・承認系" in result
        assert "実行・調査系" in result
        assert "deadline" in result
        assert "部下" in result

    def test_template_mentions_idle_check(self):
        """The delegation template instructs checking subordinate availability."""
        result = load_prompt(
            "heartbeat_subordinate_check",
            subordinates="rin",
            animas_dir="/home/test/.animaworks/animas",
        )

        assert "idle" in result or "稼働" in result

    def test_load_prompt_with_single_subordinate(self):
        """Template works with a single subordinate name."""
        result = load_prompt(
            "heartbeat_subordinate_check",
            subordinates="rin",
            animas_dir="/home/test/.animaworks/animas",
        )
        assert "rin" in result
        assert "部下" in result


# ── Test 5: _parse_deadline unit validation ───────────────────


class TestParseDeadlineFormats:
    """Validate all accepted deadline formats via _parse_deadline."""

    def test_minutes_format(self):
        before = now_jst()
        result = _parse_deadline("30m")
        result_dt = datetime.fromisoformat(result)
        assert result_dt >= before + timedelta(minutes=30) - timedelta(seconds=2)

    def test_hours_format(self):
        before = now_jst()
        result = _parse_deadline("2h")
        result_dt = datetime.fromisoformat(result)
        assert result_dt >= before + timedelta(hours=2) - timedelta(seconds=2)

    def test_days_format(self):
        before = now_jst()
        result = _parse_deadline("1d")
        result_dt = datetime.fromisoformat(result)
        assert result_dt >= before + timedelta(days=1) - timedelta(seconds=2)

    def test_iso8601_passthrough(self):
        iso = "2026-12-31T23:59:59"
        assert _parse_deadline(iso) == iso

    def test_invalid_format_raises(self):
        with pytest.raises(ValueError, match="Invalid deadline format"):
            _parse_deadline("next week")

    def test_whitespace_stripped(self):
        """Leading/trailing whitespace is stripped before parsing."""
        before = now_jst()
        result = _parse_deadline("  1h  ")
        result_dt = datetime.fromisoformat(result)
        assert result_dt >= before + timedelta(hours=1) - timedelta(seconds=2)


# ── Test 6: Integration — priming output structure ───────────


class TestPrimingOutputStructure:
    """Verify format_for_priming produces correctly structured output."""

    def test_human_tasks_sorted_first(
        self, task_queue: TaskQueueManager, anima_dir: Path,
    ):
        """Human-origin tasks appear before anima-origin tasks."""
        # Add anima task first (chronologically earlier)
        early_ts = (datetime.now() - timedelta(minutes=10)).isoformat()
        _write_task_entry_to_jsonl(
            task_queue.queue_path,
            task_id="anima_first",
            source="anima",
            summary="Anima task first",
            assignee="test-anima",
            deadline=(datetime.now() + timedelta(hours=1)).isoformat(),
            updated_at=early_ts,
            ts=early_ts,
        )

        # Add human task second (chronologically later)
        _write_task_entry_to_jsonl(
            task_queue.queue_path,
            task_id="human_second",
            source="human",
            summary="Human task second",
            assignee="test-anima",
            deadline=(datetime.now() + timedelta(hours=1)).isoformat(),
            updated_at=datetime.now().isoformat(),
        )

        output = task_queue.format_for_priming()
        lines = output.strip().split("\n")
        assert len(lines) == 2

        # Human task should come first despite being added second
        assert "Human task second" in lines[0]
        assert "Anima task first" in lines[1]

    def test_priority_markers_correct(
        self, task_queue: TaskQueueManager,
    ):
        """Human tasks get HIGH priority; anima tasks get normal priority."""
        task_queue.add_task(
            source="human",
            original_instruction="Human request",
            assignee="test-anima",
            summary="Human priority task",
            deadline="1h",
        )
        task_queue.add_task(
            source="anima",
            original_instruction="Anima request",
            assignee="test-anima",
            summary="Anima normal task",
            deadline="1h",
        )

        output = task_queue.format_for_priming()
        lines = output.strip().split("\n")

        human_line = [l for l in lines if "Human priority task" in l][0]
        anima_line = [l for l in lines if "Anima normal task" in l][0]

        assert "HIGH" in human_line
        assert "HIGH" not in anima_line

    def test_deadline_display_shows_time(
        self, task_queue: TaskQueueManager, anima_dir: Path,
    ):
        """Future deadline shows target time in HH:MM format."""
        future = datetime.now() + timedelta(hours=3)
        _write_task_entry_to_jsonl(
            task_queue.queue_path,
            task_id="future_dl",
            source="human",
            summary="Task with future deadline",
            assignee="test-anima",
            deadline=future.isoformat(),
            updated_at=datetime.now().isoformat(),
        )

        output = task_queue.format_for_priming()
        expected_time = future.strftime("%H:%M")
        assert expected_time in output
        assert "まで" in output

    def test_budget_respected(
        self, task_queue: TaskQueueManager, anima_dir: Path,
    ):
        """Output should not exceed the token budget."""
        # Write many tasks
        for i in range(50):
            _write_task_entry_to_jsonl(
                task_queue.queue_path,
                task_id=f"budget_{i:03d}",
                source="human",
                summary=f"Very long task description number {i} with lots of detail here",
                assignee="test-anima",
                deadline=(datetime.now() + timedelta(hours=1)).isoformat(),
                updated_at=datetime.now().isoformat(),
            )

        output = task_queue.format_for_priming(budget_tokens=100)
        # 100 tokens * 4 chars = 400 chars max, with some margin for the last line
        assert len(output) <= 600
