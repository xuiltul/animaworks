from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""SkillsToolsMixin — tool management, procedure/knowledge outcomes, skills, task queue."""

import json as _json
import logging
import re
from datetime import UTC
from pathlib import Path
from typing import TYPE_CHECKING, Any

from core.i18n import t
from core.time_utils import now_iso, now_local
from core.tooling.handler_base import _error_result

if TYPE_CHECKING:
    from core.memory import MemoryManager
    from core.memory.activity import ActivityLogger
    from core.schemas import TaskEntry
    from core.tooling.dispatch import ExternalToolDispatcher

logger = logging.getLogger("animaworks.tool_handler")

_INSTRUCTION_TRUNCATE_LEN = 200


class TaskSuppressedError(RuntimeError):
    """Raised when TaskBoard attention policy blocks task regeneration."""

    def __init__(self, task_id: str, reason: str) -> None:
        self.task_id = task_id
        self.reason = reason
        super().__init__(f"Task {task_id} is suppressed by TaskBoard: {reason}")


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
        return f"Refreshed tools ({len(merged)} discovered): {names}\nThese tools are now available for use."

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

        if not self._check_tool_creation_permission(t("handler.shared_tool_keyword")):
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

    # ── Procedure/Skill outcome tracking ─────────────────────

    def _handle_report_procedure_outcome(self, args: dict[str, Any]) -> str:
        """Report success/failure of a procedure or skill and update its metadata."""
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
                suggestion="Check the path (e.g. procedures/deploy.md or skills/my-skill/SKILL.md)",
            )

        if not target.resolve().is_relative_to(self._anima_dir.resolve()):
            return _error_result("PermissionDenied", "Path resolves outside anima directory")

        is_skill = rel.startswith("skills/")

        # Record event in SkillUsageTracker
        from core.skills.models import SkillUsageEventType
        from core.skills.usage import SkillUsageTracker

        tracker = SkillUsageTracker(self._anima_dir)
        skill_name = Path(rel).parent.name if is_skill else Path(rel).stem
        event_type = SkillUsageEventType.success if success else SkillUsageEventType.failure
        tracker.record(
            skill_name,
            event_type,
            is_common=False,
            is_procedure=not is_skill,
            ref=rel,
            notes=notes or None,
        )

        if is_skill:
            # Skills use JSONL only — no frontmatter write
            outcome_label = t("handler.outcome_success") if success else t("handler.outcome_failure")
            stats = tracker.get_stats(skill_name)
            logger.info(
                "report_skill_outcome path=%s success=%s",
                rel,
                success,
            )
            result = (
                f"Skill outcome recorded: {rel} -> {outcome_label}\n"
                f"(success: {stats.success_count}, failure: {stats.failure_count})"
            )
            if notes:
                result += f"\nnotes: {notes}"
            return result

        # Procedures — maintain existing frontmatter behaviour
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
            rel,
            success,
            meta["confidence"],
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

        meta["last_used"] = now_local().isoformat()

        s = meta.get("success_count", 0)
        f = meta.get("failure_count", 0)
        meta["confidence"] = s / max(1, s + f)

        content = self._memory.read_knowledge_content(target)
        self._memory.write_knowledge_with_meta(target, content, meta)

        logger.info(
            "report_knowledge_outcome path=%s success=%s confidence=%.2f",
            rel,
            success,
            meta["confidence"],
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
        trust_level = args.get("trust_level")
        source_type = args.get("source_type")
        source_origin = args.get("source_origin")
        category = args.get("category")
        promotion_status = args.get("promotion_status")
        skill_policy = args.get("skill_policy")
        use_when = args.get("use_when")
        trigger_phrases = args.get("trigger_phrases")
        negative_phrases = args.get("negative_phrases")
        domains = args.get("domains")
        routing_examples = args.get("routing_examples")

        if not skill_name:
            return t("handler.skill_name_required")
        if not description:
            return t("handler.description_param_required")
        if not body:
            return t("handler.body_param_required")

        if location == "common":
            base_dir = get_common_skills_dir()
        else:
            base_dir = self._anima_dir / "skills"

        skill_dir = base_dir / skill_name
        result = create_skill_directory(
            skill_name=skill_name,
            description=description,
            body=body,
            base_dir=base_dir,
            references=references,
            templates=templates,
            allowed_tools=allowed_tools,
            trust_level=trust_level,
            source_type=source_type,
            source_origin=source_origin,
            source_owner_anima=self._anima_dir.name,
            category=category,
            promotion_status=promotion_status,
            skill_policy=skill_policy,
            use_when=use_when,
            trigger_phrases=trigger_phrases,
            negative_phrases=negative_phrases,
            domains=domains,
            routing_examples=routing_examples,
        )

        # Record create event in usage tracker
        if (skill_dir / "SKILL.md").exists():
            try:
                from core.skills.models import SkillUsageEventType
                from core.skills.usage import SkillUsageTracker

                tracker = SkillUsageTracker(self._anima_dir)
                tracker.record(
                    skill_name,
                    SkillUsageEventType.create,
                    is_common=(location == "common"),
                    ref=f"{'common_skills' if location == 'common' else 'skills'}/{skill_name}/SKILL.md",
                    source_origin=source_origin or "manual",
                )
            except Exception:
                logger.debug("Failed to record skill create event", exc_info=True)

        # Run security scan on the newly created skill
        scan_summary = self._scan_created_skill(skill_dir, trust_level)
        if scan_summary:
            result += f"\n\n{scan_summary}"

        return result

    def _handle_trust_skill(self, args: dict[str, Any]) -> str:
        """Promote a safe skill to trusted operating guidance."""
        from core.skills.trust import promote_skill_to_trusted
        from core.skills.trust_gate import trust_skill_enabled_for_context

        if not trust_skill_enabled_for_context(self._trigger, self._session_origin):
            return _error_result("PermissionDenied", "trust_skill requires an explicit human-origin session")

        ref = str(args.get("ref") or args.get("skill_name") or "").strip()
        if not ref:
            return _error_result("InvalidArguments", "ref is required")
        trusted_by = "user"
        trust_reason = str(args.get("trust_reason") or "human_instruction").strip() or "human_instruction"
        try:
            result = promote_skill_to_trusted(
                self._anima_dir,
                ref,
                trusted_by=trusted_by,
                trust_reason=trust_reason,
            )
        except Exception as exc:
            logger.exception("trust_skill failed")
            return _error_result("TrustSkillFailed", str(exc))
        return _json.dumps({"status": "trusted", **result.to_dict()}, ensure_ascii=False, indent=2)

    def _handle_promote_procedure_to_skill(self, args: dict[str, Any]) -> str:
        """Create or approve a reviewed skill generated from a procedure."""
        from core.tooling.skill_promotion_tool import handle_promote_procedure_to_skill

        return handle_promote_procedure_to_skill(self, args)

    def _curator(self):
        from core.paths import get_common_skills_dir
        from core.skills.curator import SkillCurator

        return SkillCurator(self._anima_dir, common_skills_dir=get_common_skills_dir())

    def _curator_index_entries(self):
        from core.paths import get_common_skills_dir
        from core.skills.index import SkillIndex

        index = SkillIndex(
            self._anima_dir / "skills",
            get_common_skills_dir(),
            self._anima_dir / "procedures",
            anima_dir=self._anima_dir,
        )
        index.build_index()
        return index.search("", include_blocked=True)

    def _handle_curate_skills(self, args: dict[str, Any]) -> str:
        """Return a deterministic curator report for the current skill catalog."""
        del args
        try:
            report = self._curator().generate_report(self._curator_index_entries())
        except Exception as exc:
            logger.exception("curate_skills failed")
            return _error_result("CuratorFailed", str(exc))
        return _json.dumps(report, ensure_ascii=False, indent=2, default=str)

    def _handle_archive_skill(self, args: dict[str, Any]) -> str:
        return self._handle_curator_state_change(args, "archived")

    def _handle_restore_skill(self, args: dict[str, Any]) -> str:
        return self._handle_curator_state_change(args, "active")

    def _handle_block_skill(self, args: dict[str, Any]) -> str:
        return self._handle_curator_state_change(args, "blocked")

    def _handle_unblock_skill(self, args: dict[str, Any]) -> str:
        return self._handle_curator_state_change(args, "active")

    def _handle_delete_skill(self, args: dict[str, Any]) -> str:
        return self._handle_curator_state_change(args, "deleted")

    def _handle_set_skill_lifecycle(self, args: dict[str, Any]) -> str:
        state = args.get("state", "")
        if not state:
            return _error_result("InvalidArguments", "state is required")
        return self._handle_curator_state_change(args, state)

    def _handle_curator_state_change(self, args: dict[str, Any], state: str) -> str:
        skill_name = str(args.get("skill_name") or "").strip()
        reason = str(args.get("reason") or "").strip()
        absorbed_into = args.get("absorbed_into")
        if not skill_name:
            return _error_result("InvalidArguments", "skill_name is required")
        if not reason:
            return _error_result("InvalidArguments", "reason is required")
        absorbed_target = str(absorbed_into).strip() if absorbed_into is not None else ""
        try:
            event = self._curator().change_state(
                skill_name,
                state,
                reason=reason,
                actor=self._anima_name,
                absorbed_into=absorbed_target or None,
            )
        except ValueError as exc:
            return _error_result("InvalidArguments", str(exc))
        except Exception as exc:
            logger.exception("skill lifecycle change failed")
            return _error_result("CuratorFailed", str(exc))
        return event.model_dump_json(indent=2)

    def _scan_created_skill(self, skill_dir: Path, trust_level: str | None) -> str:
        """Run security scan on a newly created skill and persist results."""
        from datetime import datetime

        import yaml

        from core.memory.frontmatter import parse_frontmatter
        from core.skills.guard import SCANNER_VERSION, SkillScanner
        from core.skills.models import SkillScanVerdict

        scanner = SkillScanner()
        scan_result = scanner.scan_skill(skill_dir)

        # Persist scan result into SKILL.md frontmatter
        skill_md_path = skill_dir / "SKILL.md"
        if skill_md_path.exists():
            text = skill_md_path.read_text(encoding="utf-8")
            meta, body = parse_frontmatter(text)
            meta["security"] = {
                "verdict": scan_result.verdict.value,
                "scan_status": "scanned",
                "findings": [f.model_dump() for f in scan_result.findings],
                "scanned_at": datetime.now(UTC).isoformat(),
                "scanner_version": SCANNER_VERSION,
            }
            frontmatter = yaml.dump(meta, allow_unicode=True, default_flow_style=False, sort_keys=False).strip()
            skill_md_path.write_text(f"---\n{frontmatter}\n---\n\n{body}\n", encoding="utf-8")

        # Build summary message
        verdict = scan_result.verdict
        if verdict == SkillScanVerdict.safe:
            return t("handler.skill_scan_safe")
        elif verdict == SkillScanVerdict.dangerous:
            categories = sorted({f.category for f in scan_result.findings})
            return t(
                "handler.skill_scan_dangerous",
                count=len(scan_result.findings),
                categories=", ".join(categories),
            )
        else:
            return t(
                "handler.skill_scan_warning",
                verdict=verdict.value,
                count=len(scan_result.findings),
            )

    # ── Task queue handlers ───────────────────────────────────

    def _handle_backlog_task(self, args: dict[str, Any]) -> str:
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
        except Exception as e:
            logger.error("Task persistence failed in backlog_task: %s", e)
            return _error_result("PersistenceFailed", f"Failed to persist task: {e}")

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

        if status == "pending":
            current = manager.get_task_by_id(task_id)
            if current is None:
                return _error_result(
                    "TaskNotFound",
                    f"Task not found or invalid status: {task_id}",
                )
            decision = self._retry_attention_decision(task_id, queue_status="pending")
            if not decision.executable:
                return _error_result(
                    "TaskSuppressed",
                    f"Task {task_id} is suppressed by TaskBoard: {decision.reason}",
                )

        try:
            entry = manager.update_status(task_id, status, summary=summary)
        except Exception as e:
            logger.error("Task persistence failed in update_task: %s", e)
            return _error_result("PersistenceFailed", f"Failed to update task: {e}")
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

        # Retry flow: if status changed to "pending" and task has task_desc in meta,
        # regenerate Layer 1 JSON and re-submit to PendingTaskExecutor
        if status == "pending" and entry and entry.meta.get("task_desc"):
            try:
                regenerated = self._regenerate_pending_json(entry)
                if regenerated:
                    # Immediately set to in_progress since TaskExec will pick it up
                    entry = manager.update_status(task_id, "in_progress") or entry
            except TaskSuppressedError as e:
                return _error_result("TaskSuppressed", str(e))
            except Exception:
                logger.warning(
                    "Failed to regenerate pending JSON for retry: %s",
                    task_id,
                    exc_info=True,
                )

        return _json.dumps(entry.model_dump(), ensure_ascii=False, indent=2)

    _MAX_TASK_RETRY = 3

    def _retry_attention_decision(self, task_id: str, *, queue_status: str | None = None):
        try:
            from core.taskboard.attention_resolver import resolver_for_anima_dir
            from core.taskboard.models import AttentionDecision

            return resolver_for_anima_dir(self._anima_dir).should_execute(
                self._anima_name,
                task_id,
                queue_status=queue_status,
            )
        except Exception:
            logger.warning(
                "TaskBoard retry gate unavailable for task %s; failing open",
                task_id,
                exc_info=True,
            )
            from core.taskboard.models import AttentionDecision

            return AttentionDecision(reason="active")

    def _regenerate_pending_json(self, entry: TaskEntry) -> bool:
        """Regenerate Layer 1 JSON from task_queue entry for retry execution."""
        pending_dir = self._anima_dir / "state" / "pending"
        processing_dir = pending_dir / "processing"
        task_file = f"{entry.task_id}.json"

        if (pending_dir / task_file).exists() or (processing_dir / task_file).exists():
            logger.warning("Task %s already in pipeline, skip regeneration", entry.task_id)
            return True

        decision = self._retry_attention_decision(entry.task_id, queue_status="pending")
        if not decision.executable:
            raise TaskSuppressedError(entry.task_id, decision.reason)

        retry_count = entry.meta.get("retry_count", 0)
        if retry_count >= self._MAX_TASK_RETRY:
            logger.warning(
                "Task %s exceeded max retries (%d), skip",
                entry.task_id,
                self._MAX_TASK_RETRY,
            )
            return False

        next_retry_count = retry_count + 1
        entry.meta["retry_count"] = next_retry_count
        try:
            from core.memory.task_queue import TaskQueueManager

            updated = TaskQueueManager(self._anima_dir).update_meta(
                entry.task_id,
                {"retry_count": next_retry_count},
            )
            if updated is not None:
                entry.meta = updated.meta
        except Exception:
            logger.warning("Failed to persist retry_count for task %s", entry.task_id, exc_info=True)

        task_desc_meta = entry.meta.get("task_desc", {})
        task_desc = {
            "task_type": "llm",
            "task_id": entry.task_id,
            "batch_id": entry.meta.get("batch_id", ""),
            "title": task_desc_meta.get("title", entry.summary),
            "description": task_desc_meta.get("description", entry.original_instruction),
            "parallel": False,  # retry は常に単体実行
            "depends_on": [],  # retry は依存なし
            "context": task_desc_meta.get("context", ""),
            "acceptance_criteria": task_desc_meta.get("acceptance_criteria", []),
            "constraints": task_desc_meta.get("constraints", []),
            "file_paths": task_desc_meta.get("file_paths", []),
            "submitted_by": self._anima_name,
            "submitted_at": now_iso(),
            "reply_to": task_desc_meta.get("reply_to", self._anima_name),
            "working_directory": task_desc_meta.get("working_directory", ""),
        }

        pending_dir = self._anima_dir / "state" / "pending"
        pending_dir.mkdir(parents=True, exist_ok=True)
        path = pending_dir / f"{entry.task_id}.json"
        path.write_text(
            _json.dumps(task_desc, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        # Wake the pending executor
        if hasattr(self, "_pending_executor_wake") and self._pending_executor_wake:
            self._pending_executor_wake()

        logger.info("Regenerated pending JSON for retry: %s", entry.task_id)
        return True

    def _handle_list_tasks(self, args: dict[str, Any]) -> str:
        from core.memory.task_queue import TaskQueueManager

        manager = TaskQueueManager(self._anima_dir)
        status_filter = args.get("status")
        detail = args.get("detail", False)
        tasks = manager.list_tasks(status=status_filter)
        result = [t.model_dump() for t in tasks]
        if not detail:
            for item in result:
                instr = item.get("original_instruction", "")
                if len(instr) > _INSTRUCTION_TRUNCATE_LEN:
                    item["original_instruction"] = instr[:_INSTRUCTION_TRUNCATE_LEN] + "..."
        return _json.dumps(result, ensure_ascii=False)

    # ── submit_tasks handler (DAG batch submission) ────────────

    def _handle_submit_tasks(self, args: dict[str, Any]) -> str:
        """Validate and write a DAG batch of tasks to state/pending/.

        Performs cycle detection, duplicate ID check, and dependency
        reference validation before writing task files.
        """
        batch_id = args.get("batch_id", "")
        tasks = args.get("tasks", [])

        if not batch_id:
            return _error_result("InvalidArguments", "batch_id is required")
        if not tasks:
            return _error_result("InvalidArguments", "tasks must contain at least one task")

        # Validate task IDs are unique
        task_ids = [t.get("task_id", "") for t in tasks]
        if len(task_ids) != len(set(task_ids)):
            return _error_result("InvalidArguments", "Duplicate task_id found in batch")

        task_id_set = set(task_ids)

        # Validate depends_on references
        for t in tasks:  # noqa: F402
            for dep in t.get("depends_on", []):
                if dep not in task_id_set:
                    return _error_result(
                        "InvalidArguments", f"Task '{t['task_id']}' depends on unknown task_id '{dep}'"
                    )

        # Validate required fields
        for t in tasks:
            if not t.get("task_id") or not t.get("title") or not t.get("description"):
                return _error_result(
                    "InvalidArguments",
                    f"Task missing required fields (task_id, title, description): {t.get('task_id', '?')}",
                )

        # Cycle detection via topological sort
        from core.supervisor.pending_executor import _topological_sort

        try:
            _topological_sort(tasks)
        except ValueError:
            return _error_result("InvalidArguments", "Cycle detected in depends_on references")

        # Write task files to state/pending/ AND register in task_queue.jsonl
        pending_dir = self._anima_dir / "state" / "pending"
        pending_dir.mkdir(parents=True, exist_ok=True)
        submitted_at = now_iso()

        from core.memory.task_queue import TaskQueueManager

        manager = TaskQueueManager(self._anima_dir)

        written: list[str] = []
        for t in tasks:
            workspace_raw = t.get("workspace", "")
            resolved_wd = ""
            if workspace_raw:
                try:
                    from core.workspace import resolve_workspace

                    resolved_wd = str(resolve_workspace(workspace_raw))
                except ValueError as e:
                    return _error_result("InvalidArguments", f"Workspace resolution failed: {e}")

            task_desc = {
                "task_type": "llm",
                "task_id": t["task_id"],
                "batch_id": batch_id,
                "title": t["title"],
                "description": t["description"],
                "parallel": t.get("parallel", False),
                "depends_on": t.get("depends_on", []),
                "context": t.get("context", ""),
                "acceptance_criteria": t.get("acceptance_criteria", []),
                "constraints": t.get("constraints", []),
                "file_paths": t.get("file_paths", []),
                "submitted_by": self._anima_name,
                "submitted_at": submitted_at,
                "reply_to": t.get("reply_to", self._anima_name),
                "working_directory": resolved_wd,
            }

            # Layer 1: Write JSON to state/pending/
            path = pending_dir / f"{t['task_id']}.json"
            path.write_text(
                _json.dumps(task_desc, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )

            # Layer 2: Register in task_queue.jsonl
            try:
                manager.add_task(
                    source="anima",
                    original_instruction=t["description"][:5000],
                    assignee=self._anima_name,
                    summary=t["title"],
                    task_id=t["task_id"],
                    status="pending" if t.get("depends_on") else "in_progress",
                    meta={
                        "executor": "taskexec",
                        "batch_id": batch_id,
                        "depends_on": t.get("depends_on", []),
                        "parallel": t.get("parallel", False),
                        "task_desc": {
                            "title": t["title"],
                            "description": t["description"],
                            "acceptance_criteria": t.get("acceptance_criteria", []),
                            "constraints": t.get("constraints", []),
                            "file_paths": t.get("file_paths", []),
                            "context": t.get("context", ""),
                            "reply_to": t.get("reply_to", self._anima_name),
                            "working_directory": resolved_wd,
                        },
                    },
                )
            except Exception:
                logger.warning(
                    "Failed to register submit_task in task_queue: %s",
                    t["task_id"],
                    exc_info=True,
                )

            written.append(t["task_id"])

        # Wake the pending executor
        if hasattr(self, "_pending_executor_wake") and self._pending_executor_wake:
            self._pending_executor_wake()

        return _json.dumps(
            {
                "status": "submitted",
                "batch_id": batch_id,
                "task_count": len(written),
                "task_ids": written,
                "message": (
                    f"Batch '{batch_id}' submitted with {len(written)} tasks. "
                    f"Parallel tasks will execute concurrently. "
                    f"Tasks with depends_on will wait for dependencies."
                ),
            },
            ensure_ascii=False,
        )

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
