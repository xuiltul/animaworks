from __future__ import annotations

import json
from pathlib import Path

from core.goals import GoalJudgment, GoalManager
from core.memory.task_queue import TaskQueueManager
from core.taskboard.models import AttentionVisibility
from core.taskboard.store import TaskBoardStore
from core.tooling.handler import ToolHandler
from core.tooling.schemas import _goal_tools, build_tool_list


def _anima_dir(tmp_path: Path, monkeypatch) -> Path:
    data_dir = tmp_path / "data"
    monkeypatch.setenv("ANIMAWORKS_DATA_DIR", str(data_dir))
    anima_dir = data_dir / "animas" / "alice"
    for dirname in ("state", "episodes", "knowledge", "procedures", "skills"):
        (anima_dir / dirname).mkdir(parents=True, exist_ok=True)
    (data_dir / "common_skills").mkdir(parents=True, exist_ok=True)
    return anima_dir


def test_goal_manager_replays_transitions_and_max_iteration_block(tmp_path: Path, monkeypatch) -> None:
    anima_dir = _anima_dir(tmp_path, monkeypatch)
    manager = GoalManager(anima_dir)

    state = manager.set_goal(
        title="Ship report",
        objective="Create a weekly report",
        success_criteria=["Report exists", "Summary is sent"],
        max_iterations=1,
    )
    manager.pause(state.goal_id, reason="wait")
    manager.resume(state.goal_id, reason="continue")
    updated = manager.record_judgment(
        state.goal_id,
        GoalJudgment(
            goal_id=state.goal_id,
            task_id="t1",
            verdict="continue",
            reason="not enough evidence",
            iteration=1,
        ),
        result_summary="draft only",
    )

    assert updated is not None
    assert updated.status == "blocked"
    assert updated.iteration_count == 1
    assert "max_iterations_reached:1/1" in updated.blocked_reason

    replayed = GoalManager(anima_dir).get_goal(state.goal_id)
    assert replayed is not None
    assert replayed.status == "blocked"
    assert replayed.last_task_id == "t1"


def test_enqueue_continuation_creates_normal_task_and_avoids_duplicates(tmp_path: Path, monkeypatch) -> None:
    anima_dir = _anima_dir(tmp_path, monkeypatch)
    manager = GoalManager(anima_dir)
    state = manager.set_goal(
        objective="Finish the deployment",
        success_criteria=["Deployment verified"],
        max_iterations=3,
    )
    manager.record_judgment(
        state.goal_id,
        GoalJudgment(
            goal_id=state.goal_id,
            task_id="seed",
            verdict="continue",
            reason="needs verification",
            continuation_prompt="Run verification and fix failures.",
            iteration=1,
        ),
        result_summary="build prepared",
    )
    judgment = manager.get_goal(state.goal_id).last_judgment  # type: ignore[union-attr]

    entry = manager.enqueue_continuation(
        state.goal_id,
        judgment,
        source_task_desc={"working_directory": "/tmp/work", "reply_to": "alice"},
        result_summary="build prepared",
    )
    duplicate = manager.enqueue_continuation(
        state.goal_id,
        judgment,
        source_task_desc={},
        result_summary="build prepared",
    )

    assert entry is not None
    assert duplicate is not None
    assert duplicate.task_id == entry.task_id
    assert entry.meta["goal_id"] == state.goal_id
    assert entry.meta["executor"] == "taskexec"
    pending_json = anima_dir / "state" / "pending" / f"{entry.task_id}.json"
    assert pending_json.exists()
    assert json.loads(pending_json.read_text(encoding="utf-8"))["task_type"] == "llm"


def test_human_task_defers_continuation(tmp_path: Path, monkeypatch) -> None:
    anima_dir = _anima_dir(tmp_path, monkeypatch)
    manager = GoalManager(anima_dir)
    state = manager.set_goal(objective="Do background work", max_iterations=3)
    TaskQueueManager(anima_dir).add_task(
        source="human",
        original_instruction="urgent user task",
        assignee="alice",
        summary="urgent",
        deadline="1h",
    )

    result = manager.enqueue_continuation(
        state.goal_id,
        GoalJudgment(goal_id=state.goal_id, verdict="continue", iteration=1),
        source_task_desc={},
        result_summary="needs more",
    )

    assert result is None


def test_get_active_goal_task_ignores_archived_taskboard_rows(tmp_path: Path, monkeypatch) -> None:
    anima_dir = _anima_dir(tmp_path, monkeypatch)
    queue = TaskQueueManager(anima_dir)
    entry = queue.add_task(
        source="anima",
        original_instruction="continue",
        assignee="alice",
        summary="continue",
        status="in_progress",
        meta={"goal_id": "goal-1"},
    )
    assert queue.get_active_goal_task("goal-1").task_id == entry.task_id

    TaskBoardStore().upsert_metadata(
        anima_name="alice",
        task_id=entry.task_id,
        actor="test",
        visibility=AttentionVisibility.ARCHIVED,
    )

    assert queue.get_active_goal_task("goal-1") is None


def test_goal_tool_schema_and_handler(tmp_path: Path, monkeypatch) -> None:
    anima_dir = _anima_dir(tmp_path, monkeypatch)
    tools = build_tool_list(include_goal_tools=True)
    assert _goal_tools()[0]["name"] == "goal"
    assert "goal" in {tool["name"] for tool in tools}

    handler = ToolHandler(anima_dir, memory=object())  # type: ignore[arg-type]
    result = json.loads(
        handler.handle(
            "goal",
            {
                "action": "set",
                "objective": "Prepare a release",
                "success_criteria": ["tag exists"],
                "max_iterations": 2,
            },
        )
    )

    assert result["status"] == "active"
    assert result["objective"] == "Prepare a release"
