"""E2E tests for task staleness detection and delegation workflow.

Verifies the full lifecycle:
1. Task creation without the retired deadline field
2. format_for_priming output with elapsed / STALE markers
3. get_stale_tasks returns correct results
4. Heartbeat delegation prompt injection for animas with subordinates
"""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path

import pytest

from core.memory.task_queue import (
    _STALE_TASK_THRESHOLD_SEC,
    TaskQueueManager,
)
from core.paths import _prompt_cache, load_prompt
from core.time_utils import now_jst

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
    now = now_jst().isoformat()
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

    def test_stale_marker_in_format_for_priming(
        self,
        task_queue: TaskQueueManager,
        anima_dir: Path,
    ):
        """Old task gets STALE marker; fresh task does not."""
        # Add a fresh task via API
        task_queue.add_task(
            source="human",
            original_instruction="Fresh task",
            assignee="test-anima",
            summary="Fresh task",
        )

        # Write an old task directly with updated_at 45 minutes ago
        old_updated = (now_jst() - timedelta(minutes=45)).isoformat()
        _write_task_entry_to_jsonl(
            task_queue.queue_path,
            task_id="old_task_001",
            source="human",
            summary="Old stale task",
            assignee="test-anima",
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
                assert "STALE" not in line, "Fresh task should not be marked as STALE"

    def test_get_stale_tasks_returns_old_task_only(
        self,
        task_queue: TaskQueueManager,
        anima_dir: Path,
    ):
        """get_stale_tasks returns only tasks updated 30+ minutes ago."""
        # Fresh task via API
        task_queue.add_task(
            source="human",
            original_instruction="Fresh",
            assignee="test-anima",
            summary="Fresh task",
        )

        # Old task: updated 45 minutes ago
        old_updated = (now_jst() - timedelta(minutes=45)).isoformat()
        _write_task_entry_to_jsonl(
            task_queue.queue_path,
            task_id="stale_task_01",
            source="anima",
            summary="Stale task",
            assignee="test-anima",
            updated_at=old_updated,
            ts=old_updated,
        )

        stale = task_queue.get_stale_tasks()
        assert len(stale) == 1
        assert stale[0].task_id == "stale_task_01"

    def test_get_stale_tasks_empty_when_all_fresh(
        self,
        task_queue: TaskQueueManager,
    ):
        """No stale tasks when all tasks were just created."""
        task_queue.add_task(
            source="human",
            original_instruction="Just created",
            assignee="test-anima",
            summary="Brand new",
        )
        assert task_queue.get_stale_tasks() == []

    def test_stale_threshold_boundary(
        self,
        task_queue: TaskQueueManager,
        anima_dir: Path,
    ):
        """Task updated exactly at the threshold boundary is stale."""
        # Exactly 30 minutes ago (the threshold)
        boundary_updated = (now_jst() - timedelta(seconds=_STALE_TASK_THRESHOLD_SEC)).isoformat()
        _write_task_entry_to_jsonl(
            task_queue.queue_path,
            task_id="boundary_task",
            source="human",
            summary="Boundary task",
            assignee="test-anima",
            updated_at=boundary_updated,
        )

        stale = task_queue.get_stale_tasks()
        assert len(stale) == 1
        assert stale[0].task_id == "boundary_task"

    def test_done_tasks_not_in_stale_results(
        self,
        task_queue: TaskQueueManager,
        anima_dir: Path,
    ):
        """Completed tasks should not appear in stale results even if old."""
        old_updated = (now_jst() - timedelta(hours=2)).isoformat()
        _write_task_entry_to_jsonl(
            task_queue.queue_path,
            task_id="done_old_task",
            source="human",
            summary="Old done task",
            assignee="test-anima",
            status="done",
            updated_at=old_updated,
        )

        stale = task_queue.get_stale_tasks()
        assert len(stale) == 0

    # ── Test 4: Heartbeat delegation prompt injection ────────────

    def test_elapsed_time_displayed(
        self,
        task_queue: TaskQueueManager,
        anima_dir: Path,
    ):
        """Priming output includes elapsed time indicator."""
        updated = (now_jst() - timedelta(minutes=15)).isoformat()
        _write_task_entry_to_jsonl(
            task_queue.queue_path,
            task_id="elapsed_task",
            source="anima",
            summary="Task with elapsed time",
            assignee="test-anima",
            updated_at=updated,
        )

        output = task_queue.format_for_priming()
        # Should contain elapsed time marker
        assert "15分経過" in output or "14分経過" in output or "16分経過" in output


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
        subordinates = "rin, sakura, owner"
        result = load_prompt(
            "heartbeat_subordinate_check",
            subordinates=subordinates,
            animas_dir="/home/test/.animaworks/animas",
        )

        assert "rin, sakura, owner" in result
        assert "STALE" in result
        assert "OVERDUE" not in result
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
        assert "deadline" not in result
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


# ── Test 6: Integration — priming output structure ───────────


class TestPrimingOutputStructure:
    """Verify format_for_priming produces correctly structured output."""

    def test_human_tasks_sorted_first(
        self,
        task_queue: TaskQueueManager,
        anima_dir: Path,
    ):
        """Human-origin tasks appear before anima-origin tasks."""
        # Add anima task first (chronologically earlier)
        early_ts = (now_jst() - timedelta(minutes=10)).isoformat()
        _write_task_entry_to_jsonl(
            task_queue.queue_path,
            task_id="anima_first",
            source="anima",
            summary="Anima task first",
            assignee="test-anima",
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
            updated_at=now_jst().isoformat(),
        )

        output = task_queue.format_for_priming()
        lines = output.strip().split("\n")
        assert len(lines) == 2

        # Human task should come first despite being added second
        assert "Human task second" in lines[0]
        assert "Anima task first" in lines[1]

    def test_priority_markers_correct(
        self,
        task_queue: TaskQueueManager,
    ):
        """Human tasks get HIGH priority; anima tasks get normal priority."""
        task_queue.add_task(
            source="human",
            original_instruction="Human request",
            assignee="test-anima",
            summary="Human priority task",
        )
        task_queue.add_task(
            source="anima",
            original_instruction="Anima request",
            assignee="test-anima",
            summary="Anima normal task",
        )

        output = task_queue.format_for_priming()
        lines = output.strip().split("\n")

        human_line = [line for line in lines if "Human priority task" in line][0]
        anima_line = [line for line in lines if "Anima normal task" in line][0]

        assert "HIGH" in human_line
        assert "HIGH" not in anima_line

    def test_budget_respected(
        self,
        task_queue: TaskQueueManager,
        anima_dir: Path,
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
                updated_at=now_jst().isoformat(),
            )

        output = task_queue.format_for_priming(budget_tokens=100)
        # 100 tokens * 4 chars = 400 chars max, with some margin for the last line
        assert len(output) <= 600


def test_legacy_deadline_is_ignored_when_loading_and_priming(task_queue: TaskQueueManager):
    """Old queue rows remain readable without reviving deadline enforcement."""
    _write_task_entry_to_jsonl(
        task_queue.queue_path,
        task_id="legacy-deadline",
        summary="Legacy task",
        deadline=(now_jst() - timedelta(days=1)).isoformat(),
    )
    entry = task_queue.get_task_by_id("legacy-deadline")
    assert entry is not None
    assert "deadline" not in entry.model_dump()
    output = task_queue.format_for_priming()
    assert "Legacy task" in output
    assert "OVERDUE" not in output
