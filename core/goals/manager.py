from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Append-only persistent goal state manager."""

import json
import logging
import os
import threading
import uuid
from pathlib import Path
from typing import Any

from core.goals.models import GoalEvent, GoalJudgment, GoalState
from core.memory.activity import ActivityLogger
from core.time_utils import now_iso, today_local

logger = logging.getLogger("animaworks.goals")

_LOCKS: dict[str, threading.Lock] = {}
_LOCKS_GUARD = threading.Lock()
_TERMINAL_GOAL_STATUSES = {"done", "blocked", "cleared"}


class GoalManager:
    """Manage ``state/goal_state.jsonl`` as the goal loop SSoT."""

    def __init__(self, anima_dir: Path) -> None:
        self.anima_dir = anima_dir
        self.state_path = anima_dir / "state" / "goal_state.jsonl"

    # ── Public state API ─────────────────────────────────────

    def set_goal(
        self,
        *,
        objective: str,
        success_criteria: list[str] | str | None = None,
        title: str = "",
        max_iterations: int = 5,
        judge_model: str | None = None,
        goal_id: str | None = None,
        skill_refs: list[str] | None = None,
        actor: str = "goal_tool",
    ) -> GoalState:
        objective = objective.strip()
        if not objective:
            raise ValueError("objective is required")
        usable_skills, rejected_skills = self._filter_skill_refs(skill_refs or [])
        resolved_goal_id = goal_id or uuid.uuid4().hex[:12]
        payload = {
            "goal_id": resolved_goal_id,
            "title": title.strip() or objective[:80],
            "objective": objective,
            "success_criteria": success_criteria or [],
            "status": "active",
            "max_iterations": max(1, int(max_iterations or 1)),
            "judge_model": judge_model or None,
            "skill_refs": usable_skills,
            "skill_rejections": rejected_skills,
            "created_at": now_iso(),
            "updated_at": now_iso(),
        }
        self._append_event("set", resolved_goal_id, payload, actor=actor)
        return self.get_goal(resolved_goal_id)  # type: ignore[return-value]

    def pause(self, goal_id: str, *, reason: str = "", actor: str = "goal_tool") -> GoalState | None:
        if self.get_goal(goal_id) is None:
            return None
        self._append_event("pause", goal_id, {"reason": reason, "paused_at": now_iso()}, actor=actor)
        return self.get_goal(goal_id)

    def resume(self, goal_id: str, *, reason: str = "", actor: str = "goal_tool") -> GoalState | None:
        state = self.get_goal(goal_id)
        if state is None or state.status in _TERMINAL_GOAL_STATUSES:
            return state
        self._append_event("resume", goal_id, {"reason": reason}, actor=actor)
        return self.get_goal(goal_id)

    def clear(self, goal_id: str, *, reason: str = "", actor: str = "goal_tool") -> GoalState | None:
        if self.get_goal(goal_id) is None:
            return None
        self._append_event("clear", goal_id, {"reason": reason, "cleared_at": now_iso()}, actor=actor)
        return self.get_goal(goal_id)

    def get_goal(self, goal_id: str) -> GoalState | None:
        return self.replay().get(goal_id)

    def list_goals(self, *, include_terminal: bool = True) -> list[GoalState]:
        goals = list(self.replay().values())
        if not include_terminal:
            goals = [goal for goal in goals if goal.status not in _TERMINAL_GOAL_STATUSES]
        return sorted(goals, key=lambda goal: goal.updated_at, reverse=True)

    def current_goal(self) -> GoalState | None:
        for goal in self.list_goals(include_terminal=False):
            if goal.status == "active":
                return goal
        return None

    def record_judgment(
        self,
        goal_id: str,
        judgment: GoalJudgment,
        *,
        result_summary: str = "",
        actor: str = "goal_judge",
    ) -> GoalState | None:
        state = self.get_goal(goal_id)
        if state is None:
            return None

        iteration = judgment.iteration or state.iteration_count + 1
        if judgment.verdict == "continue" and iteration >= state.max_iterations:
            judgment = judgment.model_copy(
                update={
                    "verdict": "blocked",
                    "reason": (
                        judgment.reason
                        + ("; " if judgment.reason else "")
                        + f"max_iterations_reached:{iteration}/{state.max_iterations}"
                    ),
                    "iteration": iteration,
                }
            )
        elif judgment.iteration != iteration:
            judgment = judgment.model_copy(update={"iteration": iteration})

        self._append_event(
            "judgment",
            goal_id,
            {
                "judgment": judgment.model_dump(mode="json"),
                "result_summary": result_summary[:2000],
            },
            actor=actor,
        )
        if judgment.failed_open:
            self._record_fail_open_activity(state, judgment)
        return self.get_goal(goal_id)

    def enqueue_continuation(
        self,
        goal_id: str,
        judgment: GoalJudgment,
        *,
        source_task_desc: dict[str, Any],
        result_summary: str,
        respect_human_priority: bool = True,
    ):
        """Create the next normal TaskExec task when no active continuation exists."""
        lock = self._lock_for_goal(goal_id)
        with lock:
            state = self.get_goal(goal_id)
            if state is None or state.status != "active":
                return None

            from core.memory.task_queue import TaskQueueManager

            queue = TaskQueueManager(self.anima_dir)
            existing = queue.get_active_goal_task(goal_id)
            if existing is not None:
                return existing
            if respect_human_priority and queue.get_human_tasks():
                logger.info("Goal continuation deferred behind human task: goal=%s", goal_id)
                return None

            task_id = f"goal_{goal_id[:8]}_{uuid.uuid4().hex[:6]}"
            next_iteration = state.iteration_count + 1
            title = f"Goal continuation {next_iteration}: {state.title or state.objective[:40]}"
            description = _continuation_description(state, judgment, result_summary)
            task_desc = {
                "task_type": "llm",
                "task_id": task_id,
                "batch_id": f"goal:{goal_id}",
                "title": title,
                "description": description,
                "parallel": False,
                "depends_on": [],
                "context": _continuation_context(state, judgment, result_summary),
                "acceptance_criteria": state.success_criteria,
                "constraints": ["Do not mutate system prompts or toolsets; use normal TaskExec tools only."],
                "file_paths": [],
                "submitted_by": "goal_loop",
                "submitted_at": now_iso(),
                "reply_to": source_task_desc.get("reply_to", self.anima_dir.name),
                "working_directory": source_task_desc.get("working_directory", ""),
            }
            pending_dir = self.anima_dir / "state" / "pending"
            pending_dir.mkdir(parents=True, exist_ok=True)
            (pending_dir / f"{task_id}.json").write_text(
                json.dumps(task_desc, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )

            entry = queue.add_task(
                source="anima",
                original_instruction=description[:5000],
                assignee=self.anima_dir.name,
                summary=title,
                task_id=task_id,
                status="in_progress",
                meta={
                    "executor": "taskexec",
                    "batch_id": f"goal:{goal_id}",
                    "goal_id": goal_id,
                    "goal_iteration": next_iteration,
                    "skills": list(state.skill_refs),
                    "task_desc": task_desc,
                },
            )
            self._record_taskboard_link(task_id, goal_id)
            self._append_event(
                "continuation_created",
                goal_id,
                {"task_id": task_id, "iteration": next_iteration},
                actor="goal_loop",
            )
            ActivityLogger(self.anima_dir).log(
                "task_created",
                summary=f"Goal continuation queued: {title}",
                meta={"goal_id": goal_id, "task_id": task_id},
                safe=True,
            )
            return entry

    def mark_done_activity(self, state: GoalState) -> None:
        """Record user-visible goal completion in activity log and episodes."""
        summary = f"Goal done: {state.title or state.objective[:80]}"
        ActivityLogger(self.anima_dir).log(
            "goal_done",
            content=state.last_judgment.reason if state.last_judgment else "",
            summary=summary,
            meta={"goal_id": state.goal_id},
            safe=True,
        )
        self._append_episode(summary, state.last_judgment.reason if state.last_judgment else "")

    def mark_blocked_activity(self, state: GoalState) -> None:
        summary = f"Goal blocked: {state.title or state.objective[:80]}"
        reason = state.blocked_reason or (state.last_judgment.reason if state.last_judgment else "")
        activity = ActivityLogger(self.anima_dir)
        activity.log(
            "goal_blocked",
            content=reason,
            summary=summary,
            meta={"goal_id": state.goal_id},
            safe=True,
        )
        activity.log(
            "tool_use",
            content=reason,
            summary=summary,
            tool="call_human",
            meta={"goal_id": state.goal_id, "auto_emitted": True},
            safe=True,
        )

    def _record_fail_open_activity(self, state: GoalState, judgment: GoalJudgment) -> None:
        summary = f"Goal judge failed open: {state.title or state.objective[:80]}"
        ActivityLogger(self.anima_dir).log(
            "goal_judge_failed_open",
            content=judgment.reason,
            summary=summary,
            meta={
                "goal_id": state.goal_id,
                "task_id": judgment.task_id,
                "iteration": judgment.iteration,
            },
            safe=True,
        )

    # ── Replay ───────────────────────────────────────────────

    def replay(self) -> dict[str, GoalState]:
        goals: dict[str, GoalState] = {}
        if not self.state_path.exists():
            return goals
        try:
            lines = self.state_path.read_text(encoding="utf-8").splitlines()
        except OSError:
            logger.warning("Failed to read goal state: %s", self.state_path, exc_info=True)
            return goals

        for line in lines:
            if not line.strip():
                continue
            try:
                event = GoalEvent(**json.loads(line))
            except Exception:
                logger.warning("Skipping invalid goal event line: %s", line[:120])
                continue
            _apply_event(goals, event)
        return goals

    # ── Internals ────────────────────────────────────────────

    def _append_event(self, event_type: str, goal_id: str, payload: dict[str, Any], *, actor: str) -> None:
        event = GoalEvent(
            event_id=uuid.uuid4().hex[:16],
            event_type=event_type,  # type: ignore[arg-type]
            goal_id=goal_id,
            actor=actor,
            payload=payload,
        )
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(event.model_dump(mode="json"), ensure_ascii=False)
        with self.state_path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
            f.flush()
            os.fsync(f.fileno())

    def _lock_for_goal(self, goal_id: str) -> threading.Lock:
        key = f"{self.state_path.resolve(strict=False)}:{goal_id}"
        with _LOCKS_GUARD:
            lock = _LOCKS.get(key)
            if lock is None:
                lock = threading.Lock()
                _LOCKS[key] = lock
            return lock

    def _record_taskboard_link(self, task_id: str, goal_id: str) -> None:
        try:
            from core.taskboard.store import TaskBoardStore

            TaskBoardStore().upsert_metadata(
                anima_name=self.anima_dir.name,
                task_id=task_id,
                actor="goal_loop",
                source_ref=f"aw://goal/{goal_id}",
            )
        except Exception:
            logger.debug("Failed to link goal task in TaskBoard metadata", exc_info=True)

    def _append_episode(self, summary: str, body: str) -> None:
        try:
            episodes_dir = self.anima_dir / "episodes"
            episodes_dir.mkdir(parents=True, exist_ok=True)
            path = episodes_dir / f"{today_local().isoformat()}_goal_loop.md"
            text = f"\n## {now_iso()} - {summary}\n\n{body.strip()}\n"
            with path.open("a", encoding="utf-8") as f:
                f.write(text)
        except OSError:
            logger.debug("Failed to append goal episode", exc_info=True)

    def _filter_skill_refs(self, refs: list[str]) -> tuple[list[str], list[dict[str, str]]]:
        usable: list[str] = []
        rejected: list[dict[str, str]] = []
        if not refs:
            return usable, rejected
        try:
            from core.paths import get_common_skills_dir
            from core.skills.index import SkillIndex
            from core.skills.loader import skill_access_decision

            index = SkillIndex(
                self.anima_dir / "skills",
                get_common_skills_dir(),
                self.anima_dir / "procedures",
                anima_dir=self.anima_dir,
            )
            for ref in refs:
                value = str(ref).strip()
                if not value:
                    continue
                meta = index.resolve_skill_reference(value)
                if meta is None:
                    rejected.append({"skill": value, "reason": "not_found"})
                    continue
                allowed, reason = skill_access_decision(meta, anima_dir=self.anima_dir)
                if not allowed:
                    rejected.append({"skill": value, "reason": reason})
                    continue
                if value not in usable:
                    usable.append(value)
        except Exception:
            logger.debug("Skill reference filtering failed for goal", exc_info=True)
            for ref in refs:
                value = str(ref).strip()
                if value and value not in usable:
                    usable.append(value)
        return usable, rejected


def _apply_event(goals: dict[str, GoalState], event: GoalEvent) -> None:
    payload = dict(event.payload or {})
    goal_id = event.goal_id
    if event.event_type == "set":
        state_payload = dict(payload)
        state_payload["goal_id"] = goal_id
        goals[goal_id] = GoalState(**state_payload)
        return

    state = goals.get(goal_id)
    if state is None:
        return

    if event.event_type == "pause":
        goals[goal_id] = state.model_copy(
            update={"status": "paused", "paused_at": payload.get("paused_at") or event.ts, "updated_at": event.ts}
        )
    elif event.event_type == "resume":
        goals[goal_id] = state.model_copy(update={"status": "active", "paused_at": None, "updated_at": event.ts})
    elif event.event_type == "clear":
        goals[goal_id] = state.model_copy(
            update={"status": "cleared", "cleared_at": payload.get("cleared_at") or event.ts, "updated_at": event.ts}
        )
    elif event.event_type == "judgment":
        judgment = GoalJudgment(**payload.get("judgment", {}))
        related = list(state.related_task_ids)
        if judgment.task_id and judgment.task_id not in related:
            related.append(judgment.task_id)
        update: dict[str, Any] = {
            "iteration_count": max(state.iteration_count, judgment.iteration),
            "last_task_id": judgment.task_id,
            "last_result_summary": str(payload.get("result_summary") or ""),
            "last_judgment": judgment,
            "related_task_ids": related,
            "updated_at": event.ts,
        }
        if judgment.verdict == "done":
            update["status"] = "done"
            update["completed_at"] = event.ts
        elif judgment.verdict == "blocked":
            update["status"] = "blocked"
            update["blocked_reason"] = judgment.reason
        else:
            update["status"] = "active"
        goals[goal_id] = state.model_copy(update=update)
    elif event.event_type == "continuation_created":
        related = list(state.related_task_ids)
        task_id = str(payload.get("task_id") or "")
        if task_id and task_id not in related:
            related.append(task_id)
        goals[goal_id] = state.model_copy(update={"related_task_ids": related, "updated_at": event.ts})


def _continuation_context(state: GoalState, judgment: GoalJudgment, result_summary: str) -> str:
    return (
        f"Persistent goal: {state.objective}\n"
        f"Success criteria:\n{_bullet_list(state.success_criteria)}\n\n"
        f"Previous result summary:\n{result_summary[:2000]}\n\n"
        f"Judge reason:\n{judgment.reason[:1000]}"
    )


def _continuation_description(state: GoalState, judgment: GoalJudgment, result_summary: str) -> str:
    next_step = judgment.continuation_prompt.strip()
    if not next_step:
        next_step = "Continue the smallest useful next step toward the persistent goal."
    return (
        f"{next_step}\n\n"
        f"Goal objective: {state.objective}\n"
        f"Success criteria:\n{_bullet_list(state.success_criteria)}\n\n"
        f"Previous task result:\n{result_summary[:2000]}\n"
    )


def _bullet_list(items: list[str]) -> str:
    return "\n".join(f"- {item}" for item in items) if items else "- (none provided)"
