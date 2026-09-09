from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""DelegationMixin — delegate_task and task_tracker."""

import json as _json
import logging
import uuid
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


def _record_taskboard_delegation(
    *,
    delegated_to: str,
    delegated_task_id: str,
    delegator: str,
    tracking_task_id: str | None = None,
) -> None:
    """Record optional TaskBoard presentation metadata after canonical publication."""
    from core.taskboard.models import AttentionVisibility, BoardColumn
    from core.taskboard.store import TaskBoardStore

    store = TaskBoardStore()
    store.upsert_metadata(
        anima_name=delegated_to,
        task_id=delegated_task_id,
        actor=delegator,
        event_type="metadata_upserted",
        visibility=AttentionVisibility.ACTIVE,
        column=BoardColumn.TODO,
        source_ref=f"task_queue:{delegated_to}:{delegated_task_id}",
    )
    if tracking_task_id:
        store.upsert_metadata(
            anima_name=delegator,
            task_id=tracking_task_id,
            actor=delegator,
            event_type="metadata_upserted",
            visibility=AttentionVisibility.ACTIVE,
            column=BoardColumn.WAITING,
            source_ref=f"task_queue:{delegator}:{tracking_task_id}",
        )


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
        from core.tooling.org_helpers import resolve_anima_name

        target_name = resolve_anima_name(args.get("name", ""))
        instruction = args.get("instruction", "")
        summary = args.get("summary", "") or instruction[:100]
        raw_criteria = args.get("acceptance_criteria")
        acceptance_criteria: list[str] = (
            [c for c in raw_criteria if isinstance(c, str)] if isinstance(raw_criteria, list) else []
        )

        workspace_raw = args.get("workspace", "")
        resolved_wd = ""
        if workspace_raw:
            try:
                from core.workspace import resolve_workspace

                resolved_wd = str(resolve_workspace(workspace_raw))
            except ValueError as e:
                return _error_result(
                    "InvalidArguments",
                    f"Workspace resolution failed: {e}",
                    suggestion=str(e),
                )

        model = args.get("model")
        if model is not None and not isinstance(model, str):
            return _error_result("InvalidArguments", "model must be a string")
        model = model.strip() if isinstance(model, str) else ""

        if not target_name:
            return _error_result("InvalidArguments", "name is required")
        if not instruction:
            return _error_result("InvalidArguments", "instruction is required")

        err = self._check_subordinate(target_name)
        if err:
            return err

        if model:
            from core.config.model_catalog import validate_model_override

            model_err = validate_model_override(target_name, model)
            if model_err:
                return _error_result(
                    "InvalidArguments",
                    t("tooling.model_list_hint", error=model_err),
                )

        from core.company import check_company_boundary
        from core.paths import get_animas_dir

        animas_dir = get_animas_dir()
        boundary = check_company_boundary(
            self._anima_name,
            target_name,
            animas_dir=animas_dir,
        )
        if boundary.cross_company:
            if boundary.resolved_via == "fail_closed":
                return t("handler.company_boundary_unverifiable")
            return t(
                "handler.cross_company_delegation_blocked",
                display_name=boundary.display_name,
            )

        target_dir = animas_dir / target_name

        sub_task_id = uuid.uuid4().hex[:12]
        tracking_task_id = uuid.uuid4().hex[:12]
        from core.tasks_dispatch import publish_delegation

        task_desc = {
            "task_type": "llm",
            "task_id": sub_task_id,
            "title": summary,
            "description": instruction,
            "context": "",
            "acceptance_criteria": acceptance_criteria,
            "constraints": [],
            "file_paths": [],
            "submitted_by": self._anima_name,
            "submitted_at": datetime.now(UTC).isoformat(),
            "reply_to": self._anima_name,
            "source": "delegation",
            "working_directory": resolved_wd,
            "model": model,
        }
        used_server_fallback = False
        try:
            used_server_fallback = publish_delegation(
                target_dir,
                task_desc,
                delegator=self._anima_name,
                tracking_task_id=tracking_task_id,
            )
        except ValueError as exc:
            return _error_result("InvalidArguments", str(exc))
        except Exception as exc:
            logger.exception("delegate_task persistence failed")
            return _error_result("PersistenceFailed", str(exc))

        if not used_server_fallback:
            try:
                _record_taskboard_delegation(
                    delegated_to=target_name,
                    delegated_task_id=sub_task_id,
                    delegator=self._anima_name,
                    tracking_task_id=tracking_task_id,
                )
            except Exception as e:
                logger.warning(
                    "TaskBoard write failed in delegate_task; queue entries remain authoritative: %s",
                    e,
                )

        # Build outgoing origin_chain (provenance Phase 3)
        outgoing_chain = build_outgoing_origin_chain(
            self._session_origin,
            self._session_origin_chain,
        )

        dm_result = ""
        if self._messenger:
            dm_enabled = True
            try:
                from core.config.models import load_config

                dm_enabled = load_config().heartbeat.delegation_dm_enabled
            except Exception as e:
                dm_enabled = True
                logger.warning("Could not read delegation_dm_enabled config: %s", e)
            if not dm_enabled:
                dm_result = t("handler.delegation_dm_skipped")
            else:
                try:
                    self._messenger.send(
                        to=target_name,
                        content=t(
                            "handler.delegation_dm_content",
                            instruction=instruction,
                            task_id=sub_task_id,
                        ),
                        intent="delegation",
                        origin_chain=outgoing_chain,
                        meta={"task_id": sub_task_id},
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

        self._activity.log(
            "tool_use",
            tool="delegate_task",
            summary=t("handler.delegate_log", target_name=target_name, summary=summary[:80]),
            meta={
                "target": target_name,
                "own_task_id": tracking_task_id,
                "sub_task_id": sub_task_id,
            },
        )

        result = t(
            "handler.delegated_success",
            target_name=target_name,
            sub_id=sub_task_id,
            own_id=tracking_task_id,
            dm_result=dm_result,
        )
        return result + process_warning

    def _handle_task_tracker(self, args: dict[str, Any]) -> str:
        """Track progress of delegated tasks."""
        status_filter = args.get("status", "active")

        from core.memory.task_queue import TaskQueueManager

        own_tqm = TaskQueueManager(self._anima_dir)
        delegated = [
            task for task in own_tqm._load_all(include_archived=True).values() if task.meta.get("delegated_to")
        ]

        if not delegated:
            return t("handler.no_delegated_tasks")

        results: list[dict[str, Any]] = []

        for task in delegated:
            meta = task.meta or {}
            delegated_to = meta.get("delegated_to", "")

            entry: dict[str, Any] = {
                "my_task_id": task.task_id,
                "delegated_to": delegated_to,
                "summary": task.summary,
                "delegated_at": task.ts,
                "subordinate_status": meta.get("delegated_status", task.status),
                "last_updated": task.updated_at,
            }

            sub_status = entry["subordinate_status"]
            _terminal = {"done", "cancelled"}
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
