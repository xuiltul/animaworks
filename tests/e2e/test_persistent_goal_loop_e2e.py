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
    queue.submit(
        {"task_id": "seed", "task_type": "llm", "title": "Create checklist", "description": "Create checklist"},
        meta={"executor": "taskexec", "goal_id": goal.goal_id},
    )
    seed_attempt = queue.store.claim("alice", "seed", {})
    assert seed_attempt is not None
    assert queue.store.finish(
        seed_attempt["_attempt_token"], status="done", stop_kind="completed", summary="Checklist drafted"
    )

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
    descriptor = queue.store.get_input("alice", continuation.task_id)
    assert descriptor is not None
    assert descriptor["task_type"] == "llm"
    assert descriptor["acceptance_criteria"] == goal.success_criteria
    assert "Verify it." in descriptor["description"]
    assert continuation.task_id in {item["task_id"] for item in queue.store.pending("alice")}
    assert not (anima_dir / "state" / "pending" / f"{continuation.task_id}.json").exists()

    continuation_attempt = queue.store.claim("alice", continuation.task_id, {})
    assert continuation_attempt is not None
    assert queue.store.finish(
        continuation_attempt["_attempt_token"],
        status="done",
        stop_kind="completed",
        summary="Checklist verified and published",
    )
    await executor._handle_goal_completion(
        {"task_id": continuation.task_id},
        "Checklist verified and published",
    )

    final = GoalManager(anima_dir).get_goal(goal.goal_id)
    assert final is not None
    assert final.status == "done"
    assert final.iteration_count == 2
    assert queue.get_active_goal_task(goal.goal_id) is None
    assert queue.store.pending("alice") == []
    assert list((anima_dir / "episodes").glob("*_goal_loop.md"))
