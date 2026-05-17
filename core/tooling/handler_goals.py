from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Goal tool handler."""

import asyncio
import json as _json
import logging
from pathlib import Path
from typing import Any

from core.goals import GoalJudge, GoalJudgment, GoalManager
from core.tooling.handler_base import _error_result

logger = logging.getLogger("animaworks.tool_handler")


class GoalsToolsMixin:
    """Persistent goal loop tool handler."""

    _anima_dir: Path
    _activity: Any

    def _handle_goal(self, args: dict[str, Any]) -> str:
        action = str(args.get("action", "")).strip().lower()
        manager = GoalManager(self._anima_dir)

        try:
            if action == "set":
                state = manager.set_goal(
                    title=str(args.get("title") or ""),
                    objective=str(args.get("objective") or ""),
                    success_criteria=args.get("success_criteria") or [],
                    max_iterations=int(args.get("max_iterations") or 5),
                    judge_model=args.get("judge_model") or None,
                    goal_id=args.get("goal_id") or None,
                    skill_refs=_string_list(args.get("skills")),
                    actor="goal_tool",
                )
                self._activity.log(
                    "goal_set",
                    summary=f"Goal set: {state.title}",
                    meta={"goal_id": state.goal_id},
                    safe=True,
                )
                return _json.dumps(state.model_dump(mode="json"), ensure_ascii=False, indent=2)

            if action == "status":
                goal_id = args.get("goal_id") or ""
                if goal_id:
                    state = manager.get_goal(str(goal_id))
                    if state is None:
                        return _error_result("GoalNotFound", f"Goal not found: {goal_id}")
                    return _json.dumps(self._goal_status_payload(manager, state), ensure_ascii=False, indent=2)
                return _json.dumps(
                    [self._goal_status_payload(manager, goal) for goal in manager.list_goals()],
                    ensure_ascii=False,
                    indent=2,
                )

            if action in {"pause", "resume", "clear"}:
                goal_id = self._resolve_goal_id(manager, args)
                if not goal_id:
                    return _error_result("GoalNotFound", "goal_id is required when no active goal exists")
                reason = str(args.get("reason") or "")
                if action == "pause":
                    state = manager.pause(goal_id, reason=reason)
                elif action == "resume":
                    state = manager.resume(goal_id, reason=reason)
                else:
                    state = manager.clear(goal_id, reason=reason)
                if state is None:
                    return _error_result("GoalNotFound", f"Goal not found: {goal_id}")
                self._activity.log(
                    f"goal_{action}",
                    summary=f"Goal {action}: {state.title}",
                    meta={"goal_id": state.goal_id, "reason": reason},
                    safe=True,
                )
                return _json.dumps(state.model_dump(mode="json"), ensure_ascii=False, indent=2)

            if action == "judge":
                return self._handle_goal_judge(args, manager)

            return _error_result("InvalidArguments", "action must be one of set, pause, resume, clear, status, judge")
        except ValueError as exc:
            return _error_result("InvalidArguments", str(exc))
        except Exception as exc:
            logger.exception("goal tool failed")
            return _error_result("GoalError", f"Goal tool failed: {exc}")

    def _handle_goal_judge(self, args: dict[str, Any], manager: GoalManager) -> str:
        goal_id = self._resolve_goal_id(manager, args)
        if not goal_id:
            return _error_result("GoalNotFound", "goal_id is required when no active goal exists")
        state = manager.get_goal(goal_id)
        if state is None:
            return _error_result("GoalNotFound", f"Goal not found: {goal_id}")
        verdict = str(args.get("verdict") or "").strip().lower()
        task_id = str(args.get("task_id") or "")
        result_summary = str(args.get("result_summary") or "")
        verification_output = str(args.get("verification_output") or "")
        if verdict:
            judgment = GoalJudgment(
                goal_id=goal_id,
                task_id=task_id,
                verdict=verdict,  # type: ignore[arg-type]
                reason=str(args.get("reason") or ""),
                continuation_prompt=str(args.get("continuation_prompt") or ""),
                verification_output=verification_output,
                iteration=state.iteration_count + 1,
            )
        else:
            judgment = _run_sync(
                GoalJudge(self._anima_dir).judge(
                    state,
                    task_id=task_id,
                    result_summaries=[result_summary] if result_summary else [],
                    verification_output=verification_output,
                )
            )

        updated = manager.record_judgment(goal_id, judgment, result_summary=result_summary, actor="goal_tool")
        if updated is None:
            return _error_result("GoalNotFound", f"Goal not found: {goal_id}")
        return _json.dumps(updated.model_dump(mode="json"), ensure_ascii=False, indent=2)

    @staticmethod
    def _resolve_goal_id(manager: GoalManager, args: dict[str, Any]) -> str:
        goal_id = str(args.get("goal_id") or "").strip()
        if goal_id:
            return goal_id
        current = manager.current_goal()
        return current.goal_id if current else ""

    def _goal_status_payload(self, manager: GoalManager, state) -> dict[str, Any]:
        payload = state.model_dump(mode="json")
        payload["related_tasks"] = _related_task_payloads(manager.anima_dir, state)
        return payload


def _string_list(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]
    return [str(value).strip()] if str(value).strip() else []


def _related_task_payloads(anima_dir: Path, state) -> list[dict[str, Any]]:
    try:
        from core.memory.task_queue import TaskQueueManager

        queue = TaskQueueManager(anima_dir)
        tasks = {task.task_id: task for task in queue.list_goal_tasks(state.goal_id)}
        for task_id in state.related_task_ids:
            if task_id not in tasks:
                task = queue.get_task_by_id(task_id)
                if task is not None:
                    tasks[task_id] = task
        ordered = sorted(tasks.values(), key=lambda task: task.updated_at, reverse=True)
        return [
            {
                "task_id": task.task_id,
                "status": task.status,
                "summary": task.summary,
                "source": task.source,
                "updated_at": task.updated_at,
                "goal_iteration": task.meta.get("goal_iteration"),
                "executor": task.meta.get("executor"),
            }
            for task in ordered
        ]
    except Exception:
        logger.debug("Failed to build goal related task payloads", exc_info=True)
        return []


def _run_sync(coro):
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    if not loop.is_running():
        return loop.run_until_complete(coro)
    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result(timeout=90)
