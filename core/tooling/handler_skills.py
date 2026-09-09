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
    from core.tooling.dispatch import ExternalToolDispatcher

logger = logging.getLogger("animaworks.tool_handler")

_INSTRUCTION_TRUNCATE_LEN = 120


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
            recorder = getattr(self, "_record_memory_file_used", None)
            if callable(recorder):
                recorder(rel)
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

        recorder = getattr(self, "_record_memory_file_used", None)
        if callable(recorder):
            recorder(rel)
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

        recorder = getattr(self, "_record_memory_file_used", None)
        if callable(recorder):
            recorder(rel)
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
        self._mark_curator_reviewed()
        return _json.dumps(report, ensure_ascii=False, indent=2, default=str)

    def _mark_curator_reviewed(self) -> None:
        """Record the curator review time so heartbeat stops re-injecting reports."""
        try:
            marker_dir = self._anima_dir / "state" / "skill_curator"
            marker_dir.mkdir(parents=True, exist_ok=True)
            (marker_dir / "last_reviewed.json").write_text(
                _json.dumps({"reviewed_at": now_iso()}, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception:
            logger.debug("Failed to update curator review marker", exc_info=True)

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
            from core.config import load_config

            curator = self._curator()
            # Security quarantine remains immediate. Routine curation records
            # proposals; a host-side explicit action can accept them later.
            apply_change = state == "blocked" or load_config().consolidation.curator_auto_apply_enabled
            operation = curator.change_state if apply_change else curator.propose_state_change
            event = operation(
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
        relay_chain = args.get("relay_chain", [])

        if not instruction:
            return _error_result("InvalidArguments", "original_instruction is required")
        if not assignee:
            return _error_result("InvalidArguments", "assignee is required")

        try:
            entry = manager.add_task(
                source=source,
                original_instruction=instruction,
                assignee=assignee,
                summary=summary,
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
        from core.tasks_dispatch import update_task

        manager = TaskQueueManager(self._anima_dir)
        task_id = args.get("task_id", "")
        status = args.get("status", "")
        summary = args.get("summary")
        result = args.get("result")

        if not task_id:
            return _error_result("InvalidArguments", "task_id is required")
        if not status:
            return _error_result("InvalidArguments", "status is required")
        if status in ("blocked", "failed"):
            return _error_result(
                "InvalidArguments",
                f"status={status!r} was retired. Use status='cancelled' and message the requester with the reason.",
            )
        if status == "in_progress":
            return _error_result(
                "InvalidArguments",
                "status 'in_progress' is written only by the running TaskExec. "
                "To (re)start a task, submit it with the submit_tasks tool. "
                "To close it, use --status done or cancelled.",
            )
        if result is not None and not isinstance(result, str):
            return _error_result("InvalidArguments", "result must be a string")
        if result is not None:
            summary = result

        try:
            entry = update_task(manager, task_id, status, summary=summary, result=result)
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

        return _json.dumps(entry.model_dump(), ensure_ascii=False, indent=2)

    def _handle_list_tasks(self, args: dict[str, Any]) -> str:
        from core.memory.task_queue import TaskQueueManager, mark_executability

        manager = TaskQueueManager(self._anima_dir)
        status_filter = args.get("status")
        detail = args.get("detail", False)
        tasks = manager.list_tasks(status=status_filter)
        result = [t.model_dump() for t in tasks]
        if not detail:
            _META_SUMMARY_KEYS = {
                "last_run_stop_kind",
                "last_run_note",
                "depends_on",
                "batch_id",
            }
            projected: list[dict[str, Any]] = []
            for item in result:
                meta = item.get("meta") or {}
                meta_subset = {k: v for k, v in meta.items() if k in _META_SUMMARY_KEYS}

                def _truncate(value: str) -> str:
                    if len(value) > _INSTRUCTION_TRUNCATE_LEN:
                        return value[:_INSTRUCTION_TRUNCATE_LEN] + "..."
                    return value

                truncated = {
                    "task_id": item.get("task_id", ""),
                    "status": item.get("status", ""),
                    "summary": _truncate(item.get("summary", "")),
                    "original_instruction": _truncate(item.get("original_instruction", "")),
                    "assignee": item.get("assignee", ""),
                    "source": item.get("source", ""),
                    "updated_at": item.get("updated_at", ""),
                }
                if meta_subset:
                    truncated["meta"] = meta_subset
                projected.append(truncated)
            result = projected
        mark_executability(result, self._anima_dir)
        return _json.dumps(result, ensure_ascii=False)

    # ── submit_tasks handler (DAG batch submission) ────────────

    def _handle_submit_tasks(self, args: dict[str, Any]) -> str:
        """Validate a complete DAG batch, then publish it in one transaction."""
        from core.tasks_dispatch import publish_tasks

        batch_id = args.get("batch_id", "")
        tasks = args.get("tasks", [])
        if not isinstance(batch_id, str) or not batch_id:
            return _error_result("InvalidArguments", "batch_id is required")
        if not isinstance(tasks, list) or not tasks or not all(isinstance(task, dict) for task in tasks):
            return _error_result("InvalidArguments", "tasks must contain at least one task")
        submitted_at = now_iso()
        payloads = [
            dict(task)
            if task.get("resume") is True
            else {
                "task_type": "llm",
                "task_id": task.get("task_id"),
                "batch_id": batch_id,
                "title": task.get("title"),
                "description": task.get("description"),
                "parallel": task.get("parallel", False),
                "depends_on": task.get("depends_on", []),
                "context": task.get("context", ""),
                "acceptance_criteria": task.get("acceptance_criteria", []),
                "constraints": task.get("constraints", []),
                "file_paths": task.get("file_paths", []),
                "submitted_by": self._anima_name,
                "submitted_at": submitted_at,
                "reply_to": task.get("reply_to", self._anima_name),
                "workspace": task.get("workspace", ""),
                "model": task.get("model", ""),
            }
            for task in tasks
        ]
        try:
            entries = publish_tasks(self._anima_dir, payloads)
        except ValueError as exc:
            return _error_result("InvalidArguments", str(exc))
        except Exception as exc:
            logger.exception("Failed to submit task batch %s", batch_id)
            return _error_result("PersistenceFailed", str(exc))

        if getattr(self, "_pending_executor_wake", None):
            self._pending_executor_wake()
        return _json.dumps(
            {
                "status": "submitted",
                "batch_id": batch_id,
                "task_count": len(entries),
                "task_ids": [entry.task_id for entry in entries],
                "message": (
                    f"Batch '{batch_id}' submitted with {len(entries)} tasks. "
                    "Parallel tasks will execute concurrently. "
                    "Tasks with depends_on will wait for dependencies."
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
