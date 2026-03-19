from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""DelegationMixin — delegate_task and task_tracker."""

import json as _json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from core.i18n import t
from core.tooling.handler_base import _error_result, build_outgoing_origin_chain
from core.tooling.org_helpers import OrgHelpersMixin

if TYPE_CHECKING:
    from core.memory.activity import ActivityLogger
    from core.messenger import Messenger

logger = logging.getLogger("animaworks.tool_handler")


class DelegationMixin(OrgHelpersMixin):
    """Mixin for delegate_task and task_tracker tools."""

    # Declared for type-checker visibility
    _anima_dir: Path
    _anima_name: str
    _activity: ActivityLogger
    _messenger: Messenger | None
    _session_origin: str
    _session_origin_chain: list[str]

    def _handle_delegate_task(self, args: dict[str, Any]) -> str:
        """Delegate a task to a direct subordinate."""
        target_name = args.get("name", "")
        instruction = args.get("instruction", "")
        summary = args.get("summary", "") or instruction[:100]
        deadline = args.get("deadline", "")

        workspace_raw = args.get("workspace", "")
        resolved_wd = ""
        if workspace_raw:
            try:
                from core.workspace import resolve_workspace

                resolved_wd = str(resolve_workspace(workspace_raw))
            except ValueError as e:
                return _error_result("InvalidArguments", f"Workspace resolution failed: {e}")

        if not target_name:
            return _error_result("InvalidArguments", "name is required")
        if not instruction:
            return _error_result("InvalidArguments", "instruction is required")
        if not deadline:
            return _error_result(
                "InvalidArguments",
                "deadline is required. Use relative format ('30m', '2h', '1d') or ISO8601.",
            )

        err = self._check_subordinate(target_name)
        if err:
            return err

        from core.memory.task_queue import TaskQueueManager
        from core.paths import get_animas_dir

        target_dir = get_animas_dir() / target_name

        sub_tqm = TaskQueueManager(target_dir)
        try:
            sub_entry = sub_tqm.add_task(
                source="anima",
                original_instruction=instruction,
                assignee=target_name,
                summary=summary,
                deadline=deadline,
                relay_chain=[self._anima_name],
            )
        except ValueError as e:
            return _error_result("InvalidArguments", str(e))
        except Exception as e:
            logger.error("Task persistence failed in delegate_task (subordinate queue): %s", e)
            return _error_result("PersistenceFailed", f"Failed to persist task to subordinate queue: {e}")

        # Write pending task JSON so PendingTaskExecutor picks it up for immediate execution
        task_desc = {
            "task_type": "llm",
            "task_id": sub_entry.task_id,
            "title": summary,
            "description": instruction,
            "context": "",
            "acceptance_criteria": [],
            "constraints": [],
            "file_paths": [],
            "submitted_by": self._anima_name,
            "submitted_at": datetime.now(UTC).isoformat(),
            "reply_to": self._anima_name,
            "source": "delegation",
            "working_directory": resolved_wd,
        }
        pending_dir = target_dir / "state" / "pending"
        pending_dir.mkdir(parents=True, exist_ok=True)
        (pending_dir / f"{sub_entry.task_id}.json").write_text(
            _json.dumps(task_desc, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        # Build outgoing origin_chain (provenance Phase 3)
        outgoing_chain = build_outgoing_origin_chain(
            self._session_origin,
            self._session_origin_chain,
        )

        dm_result = ""
        if self._messenger:
            try:
                self._messenger.send(
                    to=target_name,
                    content=t(
                        "handler.delegation_dm_content",
                        instruction=instruction,
                        deadline=deadline,
                        task_id=sub_entry.task_id,
                    ),
                    intent="delegation",
                    origin_chain=outgoing_chain,
                    meta={"task_id": sub_entry.task_id},
                )
                dm_result = t("handler.dm_sent")
            except Exception as e:
                dm_result = t("handler.dm_send_failed", e=e)
                logger.warning("delegate_task DM failed: %s -> %s: %s", self._anima_name, target_name, e)
        else:
            dm_result = t("handler.messenger_not_set")

        process_warning = ""
        try:
            from core.paths import get_data_dir

            sock = get_data_dir() / "run" / "sockets" / f"{target_name}.sock"
            if not sock.exists():
                status_file = target_dir / "status.json"
                if status_file.exists():
                    sdata = _json.loads(status_file.read_text(encoding="utf-8"))
                    if not sdata.get("enabled", True):
                        process_warning = t("handler.subordinate_disabled_warning", target_name=target_name)
        except Exception:
            logger.debug("Failed to check subordinate process status for %s", target_name, exc_info=True)

        own_tqm = TaskQueueManager(self._anima_dir)
        try:
            own_entry = own_tqm.add_delegated_task(
                original_instruction=instruction,
                assignee=target_name,
                summary=t("handler.delegation_summary", summary=summary),
                deadline=deadline,
                relay_chain=[self._anima_name, target_name],
                meta={
                    "delegated_to": target_name,
                    "delegated_task_id": sub_entry.task_id,
                },
            )
        except Exception as e:
            logger.warning("Failed to persist tracking entry for delegate_task (DM already sent): %s", e)
            own_entry = None

        own_id = own_entry.task_id if own_entry else "persist_failed"
        self._activity.log(
            "tool_use",
            tool="delegate_task",
            summary=t("handler.delegate_log", target_name=target_name, summary=summary[:80]),
            meta={
                "target": target_name,
                "own_task_id": own_id,
                "sub_task_id": sub_entry.task_id,
            },
        )

        result = t(
            "handler.delegated_success",
            target_name=target_name,
            sub_id=sub_entry.task_id,
            own_id=own_id,
            dm_result=dm_result,
        )
        return result + process_warning

    def _handle_task_tracker(self, args: dict[str, Any]) -> str:
        """Track progress of delegated tasks."""
        status_filter = args.get("status", "active")

        from core.memory.task_queue import TaskQueueManager
        from core.paths import get_animas_dir

        own_tqm = TaskQueueManager(self._anima_dir)
        delegated = own_tqm.get_delegated_tasks()

        if not delegated:
            return t("handler.no_delegated_tasks")

        animas_dir = get_animas_dir()
        results: list[dict[str, Any]] = []

        for task in delegated:
            meta = task.meta or {}
            delegated_to = meta.get("delegated_to", "")
            delegated_task_id = meta.get("delegated_task_id", "")

            entry: dict[str, Any] = {
                "my_task_id": task.task_id,
                "delegated_to": delegated_to,
                "summary": task.summary,
                "delegated_at": task.ts,
                "deadline": task.deadline or "",
                "subordinate_status": "unknown",
                "last_updated": "",
            }

            if delegated_to and delegated_task_id:
                target_dir = animas_dir / delegated_to
                try:
                    sub_tqm = TaskQueueManager(target_dir)
                    sub_task = sub_tqm.get_task_by_id(delegated_task_id)
                    if sub_task:
                        entry["subordinate_status"] = sub_task.status
                        entry["last_updated"] = sub_task.updated_at
                except Exception:
                    entry["subordinate_status"] = "unknown"

            sub_status = entry["subordinate_status"]
            _terminal = {"done", "cancelled", "failed"}
            if status_filter == "active" and sub_status in _terminal:
                continue
            if status_filter == "completed" and sub_status not in _terminal:
                continue

            results.append(entry)

        self._activity.log(
            "tool_use",
            tool="task_tracker",
            summary=t("handler.task_tracker_log", status=status_filter, count=len(results)),
        )

        if not results:
            return t("handler.no_matching_delegated", status=status_filter)

        return _json.dumps(results, ensure_ascii=False, indent=2)
