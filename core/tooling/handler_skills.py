from __future__ import annotations
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""SkillsToolsMixin — tool management, procedure/knowledge outcomes, skills, task queue."""

import json as _json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from core.i18n import t
from core.time_utils import now_iso

from core.tooling.handler_base import _error_result

if TYPE_CHECKING:
    from core.memory import MemoryManager
    from core.memory.activity import ActivityLogger
    from core.tooling.dispatch import ExternalToolDispatcher

logger = logging.getLogger("animaworks.tool_handler")


class SkillsToolsMixin:
    """Tool management, procedure/knowledge outcome tracking, skills, and task queue."""

    # Declared for type-checker visibility
    _anima_dir: Path
    _anima_name: str
    _memory: MemoryManager
    _activity: ActivityLogger
    _external: ExternalToolDispatcher
    _session_id: str

    # ── Tool management ───────────────────────────────────────

    def _handle_refresh_tools(self, args: dict[str, Any]) -> str:
        """Re-discover personal and common tools, update dispatcher."""
        from core.tools import discover_common_tools, discover_personal_tools

        personal = discover_personal_tools(self._anima_dir)
        common = discover_common_tools()
        merged = {**common, **personal}
        self._external.update_personal_tools(merged)

        if not merged:
            return "No personal or common tools found."

        names = ", ".join(sorted(merged.keys()))
        logger.info("refresh_tools: discovered %d tools: %s", len(merged), names)
        return (
            f"Refreshed tools ({len(merged)} discovered): {names}\n"
            "These tools are now available for use."
        )

    def _handle_share_tool(self, args: dict[str, Any]) -> str:
        """Copy a personal tool to common_tools/ for all animas."""
        import shutil

        from core.paths import get_data_dir

        tool_name = args["tool_name"]

        if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", tool_name):
            return _error_result(
                "InvalidArguments",
                f"Invalid tool name '{tool_name}'. Must be a valid Python identifier.",
                suggestion="Use only letters, digits, and underscores",
            )

        src = self._anima_dir / "tools" / f"{tool_name}.py"
        if not src.exists():
            return _error_result(
                "FileNotFound",
                f"Personal tool '{tool_name}' not found at {src}",
                suggestion="Check tool name with refresh_tools first",
            )

        if not self._check_tool_creation_permission("共有ツール"):
            return _error_result(
                "PermissionDenied",
                t("handler.shared_tool_denied"),
            )

        common_dir = get_data_dir() / "common_tools"
        common_dir.mkdir(parents=True, exist_ok=True)
        dst = common_dir / f"{tool_name}.py"
        if dst.exists():
            return _error_result(
                "FileExists",
                f"Common tool '{tool_name}' already exists at {dst}",
                suggestion="Choose a different name or remove the existing tool",
            )

        shutil.copy2(src, dst)
        logger.info("share_tool: copied %s → %s", src, dst)
        return f"Shared tool '{tool_name}' to common_tools/. All animas can now use it after refresh_tools."

    # ── Procedure outcome tracking ────────────────────────────

    def _handle_report_procedure_outcome(self, args: dict[str, Any]) -> str:
        """Report success/failure of a procedure and update its metadata."""
        rel = args.get("path", "")
        success = args.get("success", True)
        notes = args.get("notes", "")

        if not rel:
            return _error_result("InvalidArguments", "path is required")

        target = self._anima_dir / rel
        if not target.exists():
            return _error_result(
                "FileNotFound",
                f"File not found: {rel}",
                suggestion="Check the path (e.g. procedures/deploy.md)",
            )

        if not target.resolve().is_relative_to(self._anima_dir.resolve()):
            return _error_result("PermissionDenied", "Path resolves outside anima directory")

        meta = self._memory.read_procedure_metadata(target)

        if success:
            meta["success_count"] = meta.get("success_count", 0) + 1
        else:
            meta["failure_count"] = meta.get("failure_count", 0) + 1

        meta["last_used"] = now_iso()

        s = meta.get("success_count", 0)
        f = meta.get("failure_count", 0)
        meta["confidence"] = s / max(1, s + f)

        meta["_reported_session_id"] = self._session_id

        body = self._memory.read_procedure_content(target)
        self._memory.write_procedure_with_meta(target, body, meta)

        logger.info(
            "report_procedure_outcome path=%s success=%s confidence=%.2f",
            rel, success, meta["confidence"],
        )

        outcome_label = t("handler.outcome_success") if success else t("handler.outcome_failure")
        result = (
            f"Procedure outcome recorded: {rel} -> {outcome_label}\n"
            f"confidence: {meta['confidence']:.2f} "
            f"(success: {meta['success_count']}, failure: {meta['failure_count']})"
        )
        if notes:
            result += f"\nnotes: {notes}"

        return result

    # ── Knowledge outcome tracking ────────────────────────────

    def _handle_report_knowledge_outcome(self, args: dict[str, Any]) -> str:
        """Report success/failure of a knowledge file and update its metadata."""
        rel = args.get("path", "")
        success = args.get("success", True)
        notes = args.get("notes", "")

        if not rel:
            return _error_result("InvalidArguments", "path is required")

        target = self._anima_dir / rel
        if not target.exists():
            return _error_result(
                "FileNotFound",
                f"File not found: {rel}",
                suggestion="Check the path (e.g. knowledge/topic.md)",
            )

        if not target.resolve().is_relative_to(self._anima_dir.resolve()):
            return _error_result("PermissionDenied", "Path resolves outside anima directory")

        meta = self._memory.read_knowledge_metadata(target)

        if success:
            meta["success_count"] = meta.get("success_count", 0) + 1
        else:
            meta["failure_count"] = meta.get("failure_count", 0) + 1

        meta["last_used"] = datetime.now().isoformat()

        s = meta.get("success_count", 0)
        f = meta.get("failure_count", 0)
        meta["confidence"] = s / max(1, s + f)

        content = self._memory.read_knowledge_content(target)
        self._memory.write_knowledge_with_meta(target, content, meta)

        logger.info(
            "report_knowledge_outcome path=%s success=%s confidence=%.2f",
            rel, success, meta["confidence"],
        )

        self._activity.log(
            "knowledge_outcome",
            summary=f"{t('handler.outcome_success') if success else t('handler.outcome_failure')}: {rel}",
            meta={
                "path": rel,
                "success": success,
                "confidence": meta["confidence"],
                "notes": notes[:200] if notes else "",
            },
        )

        outcome_label = t("handler.outcome_success") if success else t("handler.outcome_failure")
        result = (
            f"Knowledge outcome recorded: {rel} -> {outcome_label}\n"
            f"confidence: {meta.get('confidence', 0):.2f} "
            f"(success: {meta.get('success_count', 0)}, failure: {meta.get('failure_count', 0)})"
        )
        if notes:
            result += f"\nnotes: {notes}"

        return result

    # ── Skill tool handler ────────────────────────────────────

    def _handle_skill(self, args: dict[str, Any]) -> str:
        """Handle skill tool invocation — load and return skill content."""
        from core.tooling.skill_tool import load_and_render_skill
        from core.paths import get_common_skills_dir

        skill_name = args.get("skill_name", "")
        context = args.get("context", "")

        if not skill_name:
            return t("handler.skill_name_required")

        return load_and_render_skill(
            skill_name=skill_name,
            anima_dir=self._anima_dir,
            skills_dir=self._anima_dir / "skills",
            common_skills_dir=get_common_skills_dir(),
            procedures_dir=self._anima_dir / "procedures",
            context=context,
        )

    def _handle_create_skill(self, args: dict[str, Any]) -> str:
        """Handle create_skill tool — create skill directory structure."""
        from core.paths import get_common_skills_dir
        from core.tooling.skill_creator import create_skill_directory

        skill_name = args.get("skill_name", "")
        description = args.get("description", "")
        body = args.get("body", "")
        location = args.get("location", "personal")
        references = args.get("references")
        templates = args.get("templates")
        allowed_tools = args.get("allowed_tools")

        if not skill_name:
            return "skill_name パラメータは必須です。"
        if not description:
            return "description パラメータは必須です。"
        if not body:
            return "body パラメータは必須です。"

        if location == "common":
            base_dir = get_common_skills_dir()
        else:
            base_dir = self._anima_dir / "skills"

        return create_skill_directory(
            skill_name=skill_name,
            description=description,
            body=body,
            base_dir=base_dir,
            references=references,
            templates=templates,
            allowed_tools=allowed_tools,
        )

    # ── Task queue handlers ───────────────────────────────────

    def _handle_add_task(self, args: dict[str, Any]) -> str:
        from core.memory.task_queue import TaskQueueManager

        manager = TaskQueueManager(self._anima_dir)
        source = args.get("source", "anima")
        instruction = args.get("original_instruction", "")
        assignee = args.get("assignee", "")
        summary = args.get("summary", "") or instruction[:100]
        deadline = args.get("deadline")
        relay_chain = args.get("relay_chain", [])

        if not instruction:
            return _error_result("InvalidArguments", "original_instruction is required")
        if not assignee:
            return _error_result("InvalidArguments", "assignee is required")
        if not deadline:
            return _error_result(
                "InvalidArguments",
                "deadline is required. Use relative format ('30m', '2h', '1d') or ISO8601.",
            )

        try:
            entry = manager.add_task(
                source=source,
                original_instruction=instruction,
                assignee=assignee,
                summary=summary,
                deadline=deadline,
                relay_chain=relay_chain,
            )
        except ValueError as e:
            return _error_result("InvalidArguments", str(e))

        self._activity.log(
            "task_created",
            summary=t("handler.task_add_log", summary=summary[:100]),
            meta={"task_id": entry.task_id, "source": source, "assignee": assignee},
        )

        return _json.dumps(entry.model_dump(), ensure_ascii=False, indent=2)

    def _handle_update_task(self, args: dict[str, Any]) -> str:
        from core.memory.task_queue import TaskQueueManager

        manager = TaskQueueManager(self._anima_dir)
        task_id = args.get("task_id", "")
        status = args.get("status", "")
        summary = args.get("summary")

        if not task_id:
            return _error_result("InvalidArguments", "task_id is required")
        if not status:
            return _error_result("InvalidArguments", "status is required")

        entry = manager.update_status(task_id, status, summary=summary)
        if entry is None:
            return _error_result(
                "TaskNotFound",
                f"Task not found or invalid status: {task_id}",
            )

        self._activity.log(
            "task_updated",
            summary=t("handler.task_update_log", summary=entry.summary[:100], status=status),
            meta={"task_id": task_id, "status": status},
        )

        return _json.dumps(entry.model_dump(), ensure_ascii=False, indent=2)

    def _handle_list_tasks(self, args: dict[str, Any]) -> str:
        from core.memory.task_queue import TaskQueueManager

        manager = TaskQueueManager(self._anima_dir)
        status_filter = args.get("status")
        tasks = manager.list_tasks(status=status_filter)
        result = [t.model_dump() for t in tasks]
        return _json.dumps(result, ensure_ascii=False, indent=2)

    # ── plan_tasks handler (DAG batch submission) ────────────

    def _handle_plan_tasks(self, args: dict[str, Any]) -> str:
        """Validate and write a DAG batch of tasks to state/pending/.

        Performs cycle detection, duplicate ID check, and dependency
        reference validation before writing task files.
        """
        from datetime import datetime as _dt, timezone as _tz

        batch_id = args.get("batch_id", "")
        tasks = args.get("tasks", [])

        if not batch_id:
            return _error_result("batch_id is required")
        if not tasks:
            return _error_result("tasks must contain at least one task")

        # Validate task IDs are unique
        task_ids = [t.get("task_id", "") for t in tasks]
        if len(task_ids) != len(set(task_ids)):
            return _error_result("Duplicate task_id found in batch")

        task_id_set = set(task_ids)

        # Validate depends_on references
        for t in tasks:
            for dep in t.get("depends_on", []):
                if dep not in task_id_set:
                    return _error_result(
                        f"Task '{t['task_id']}' depends on unknown task_id '{dep}'"
                    )

        # Validate required fields
        for t in tasks:
            if not t.get("task_id") or not t.get("title") or not t.get("description"):
                return _error_result(
                    f"Task missing required fields (task_id, title, description): "
                    f"{t.get('task_id', '?')}"
                )

        # Cycle detection via topological sort
        from core.supervisor.pending_executor import _topological_sort
        try:
            _topological_sort(tasks)
        except ValueError:
            return _error_result("Cycle detected in depends_on references")

        # Write task files to state/pending/
        pending_dir = self._anima_dir / "state" / "pending"
        pending_dir.mkdir(parents=True, exist_ok=True)
        now_iso = _dt.now(_tz.utc).isoformat()

        written: list[str] = []
        for t in tasks:
            task_desc = {
                "task_type": "llm",
                "task_id": t["task_id"],
                "batch_id": batch_id,
                "title": t["title"],
                "description": t["description"],
                "parallel": t.get("parallel", False),
                "depends_on": t.get("depends_on", []),
                "acceptance_criteria": t.get("acceptance_criteria", []),
                "constraints": t.get("constraints", []),
                "file_paths": t.get("file_paths", []),
                "submitted_by": self._anima_name,
                "submitted_at": now_iso,
            }
            path = pending_dir / f"{t['task_id']}.json"
            path.write_text(
                _json.dumps(task_desc, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            written.append(t["task_id"])

        # Wake the pending executor
        if hasattr(self, "_pending_executor_wake") and self._pending_executor_wake:
            self._pending_executor_wake()

        return _json.dumps({
            "status": "submitted",
            "batch_id": batch_id,
            "task_count": len(written),
            "task_ids": written,
            "message": (
                f"Batch '{batch_id}' submitted with {len(written)} tasks. "
                f"Parallel tasks will execute concurrently. "
                f"Tasks with depends_on will wait for dependencies."
            ),
        }, ensure_ascii=False)

    # ── Background task handlers ─────────────────────────────

    def _handle_check_background_task(self, args: dict[str, Any]) -> str:
        task_id = args.get("task_id", "")
        if not task_id:
            return _error_result("ValidationError", t("handler.bg_task_id_required"))

        mgr = self._background_manager
        if mgr is None:
            return _error_result("NotEnabled", t("handler.bg_not_enabled"))

        task = mgr.get_task(task_id)
        if task is None:
            return _error_result(
                "NotFound",
                t("handler.bg_task_not_found", task_id=task_id),
            )

        return _json.dumps(task.to_dict(), ensure_ascii=False, indent=2)

    def _handle_list_background_tasks(self, args: dict[str, Any]) -> str:
        mgr = self._background_manager
        if mgr is None:
            return _error_result("NotEnabled", t("handler.bg_not_enabled"))

        from core.background import TaskStatus

        status_filter: TaskStatus | None = None
        raw_status = args.get("status")
        if raw_status:
            try:
                status_filter = TaskStatus(raw_status)
            except ValueError:
                return _error_result(
                    "ValidationError",
                    t("handler.bg_invalid_status", status=raw_status),
                )

        tasks = mgr.list_tasks(status=status_filter)
        return _json.dumps(
            [t_item.to_dict() for t_item in tasks],
            ensure_ascii=False,
            indent=2,
        )
