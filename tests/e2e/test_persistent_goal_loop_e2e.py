from __future__ import annotations

import asyncio
from types import SimpleNamespace

from core.goals import GoalManager
from core.memory.task_queue import TaskQueueManager
from core.supervisor.pending_executor import PendingTaskExecutor


async def test_persistent_goal_loop_continue_then_done(data_dir) -> None:
    anima_dir = data_dir / "animas" / "alice"
    for dirname in ("state", "episodes", "knowledge", "procedures", "skills"):
        (anima_dir / dirname).mkdir(parents=True, exist_ok=True)

    manager = GoalManager(anima_dir)
    goal = manager.set_goal(
        objective="Publish the release checklist",
        success_criteria=["Checklist is published", "Verification is complete"],
        max_iterations=3,
    )
    queue = TaskQueueManager(anima_dir)
    queue.add_task(
        source="anima",
        original_instruction="Create checklist",
        assignee="alice",
        summary="Create checklist",
        task_id="seed",
        status="in_progress",
        meta={"executor": "taskexec", "goal_id": goal.goal_id},
    )
    queue.update_status("seed", "done", summary="Checklist drafted")

    executor = PendingTaskExecutor(
        anima=SimpleNamespace(agent=SimpleNamespace(human_notifier=None)),  # type: ignore[arg-type]
        anima_name="alice",
        anima_dir=anima_dir,
        shutdown_event=asyncio.Event(),
    )
    verdicts = iter(
        [
            {"verdict": "continue", "reason": "verification missing", "continuation_prompt": "Verify it."},
            {"verdict": "done", "reason": "all success criteria are satisfied"},
        ]
    )
    executor._goal_judge_fn = lambda _prompt, _payload: next(verdicts)  # type: ignore[attr-defined]

    await executor._handle_goal_completion({"task_id": "seed"}, "Checklist drafted")
    continuation = TaskQueueManager(anima_dir).get_active_goal_task(goal.goal_id)
    assert continuation is not None
    assert (anima_dir / "state" / "pending" / f"{continuation.task_id}.json").exists()

    queue.update_status(continuation.task_id, "done", summary="Checklist verified and published")
    await executor._handle_goal_completion(
        {"task_id": continuation.task_id},
        "Checklist verified and published",
    )

    final = GoalManager(anima_dir).get_goal(goal.goal_id)
    assert final is not None
    assert final.status == "done"
    assert final.iteration_count == 2
    assert list((anima_dir / "episodes").glob("*_goal_loop.md"))
