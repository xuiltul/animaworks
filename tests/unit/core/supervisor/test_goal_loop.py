from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

from core.goals import GoalManager
from core.memory.task_queue import TaskQueueManager
from core.supervisor.pending_executor import PendingTaskExecutor


def _setup(tmp_path: Path, monkeypatch) -> Path:
    data_dir = tmp_path / "data"
    monkeypatch.setenv("ANIMAWORKS_DATA_DIR", str(data_dir))
    anima_dir = data_dir / "animas" / "alice"
    for dirname in ("state", "episodes", "knowledge", "procedures", "skills"):
        (anima_dir / dirname).mkdir(parents=True, exist_ok=True)
    return anima_dir


def _executor(anima_dir: Path) -> PendingTaskExecutor:
    anima = SimpleNamespace(agent=SimpleNamespace(human_notifier=None))
    return PendingTaskExecutor(
        anima=anima,  # type: ignore[arg-type]
        anima_name="alice",
        anima_dir=anima_dir,
        shutdown_event=asyncio.Event(),
    )


async def test_taskexec_completion_continues_goal_without_duplicates(tmp_path: Path, monkeypatch) -> None:
    anima_dir = _setup(tmp_path, monkeypatch)
    manager = GoalManager(anima_dir)
    goal = manager.set_goal(objective="Complete rollout", success_criteria=["rollout verified"], max_iterations=3)
    queue = TaskQueueManager(anima_dir)
    queue.add_task(
        source="anima",
        original_instruction="seed",
        assignee="alice",
        summary="seed",
        task_id="seed",
        status="in_progress",
        meta={"executor": "taskexec", "goal_id": goal.goal_id},
    )
    queue.update_status("seed", "done", summary="seed complete")

    executor = _executor(anima_dir)
    executor._goal_judge_fn = lambda _prompt, _payload: {  # type: ignore[attr-defined]
        "verdict": "continue",
        "reason": "missing verification",
        "continuation_prompt": "Verify rollout.",
    }

    await executor._handle_goal_completion({"task_id": "seed"}, "seed complete")
    await executor._handle_goal_completion({"task_id": "seed"}, "seed complete")

    active = TaskQueueManager(anima_dir).get_active_goal_task(goal.goal_id)
    assert active is not None
    assert active.meta["goal_id"] == goal.goal_id
    assert len(list((anima_dir / "state" / "pending").glob("*.json"))) == 1


async def test_paused_goal_does_not_auto_continue(tmp_path: Path, monkeypatch) -> None:
    anima_dir = _setup(tmp_path, monkeypatch)
    manager = GoalManager(anima_dir)
    goal = manager.set_goal(objective="Paused work")
    manager.pause(goal.goal_id, reason="manual")
    TaskQueueManager(anima_dir).add_task(
        source="anima",
        original_instruction="seed",
        assignee="alice",
        summary="seed",
        task_id="seed",
        status="in_progress",
        meta={"executor": "taskexec", "goal_id": goal.goal_id},
    )

    executor = _executor(anima_dir)
    executor._goal_judge_fn = lambda _prompt, _payload: {"verdict": "continue"}  # type: ignore[attr-defined]
    await executor._handle_goal_completion({"task_id": "seed"}, "result")

    assert not (anima_dir / "state" / "pending").exists()


async def test_max_iterations_blocks_and_emits_call_human_activity(tmp_path: Path, monkeypatch) -> None:
    anima_dir = _setup(tmp_path, monkeypatch)
    manager = GoalManager(anima_dir)
    goal = manager.set_goal(objective="Impossible goal", max_iterations=1)
    TaskQueueManager(anima_dir).add_task(
        source="anima",
        original_instruction="seed",
        assignee="alice",
        summary="seed",
        task_id="seed",
        status="in_progress",
        meta={"executor": "taskexec", "goal_id": goal.goal_id},
    )

    executor = _executor(anima_dir)
    executor._goal_judge_fn = lambda _prompt, _payload: {"verdict": "continue", "reason": "still missing"}  # type: ignore[attr-defined]
    await executor._handle_goal_completion({"task_id": "seed"}, "not done")

    updated = GoalManager(anima_dir).get_goal(goal.goal_id)
    assert updated is not None
    assert updated.status == "blocked"
    activity_files = list((anima_dir / "activity_log").glob("*.jsonl"))
    assert activity_files
    activity_text = "\n".join(path.read_text(encoding="utf-8") for path in activity_files)
    assert '"tool": "call_human"' in activity_text
    assert "max_iterations_reached" in activity_text
