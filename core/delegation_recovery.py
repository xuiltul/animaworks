from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Recovery helpers for disabled or dormant delegated Animas."""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from core.memory.task_queue import TaskQueueManager
from core.schemas import TaskEntry
from core.time_utils import ensure_aware, now_local

logger = logging.getLogger("animaworks.delegation_recovery")

ACTIVE_TASK_STATUSES = frozenset({"pending", "in_progress", "blocked", "delegated"})
DELEGATION_BOUNCE_DEFAULT_DAYS = 14
DORMANT_ANIMA_DEFAULT_DAYS = 60


def _read_status(anima_dir: Path) -> dict[str, Any]:
    status_path = anima_dir / "status.json"
    if not status_path.exists():
        return {}
    try:
        data = json.loads(status_path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        logger.debug("Failed to read status.json for %s", anima_dir.name, exc_info=True)
        return {}


def _is_enabled(anima_dir: Path) -> bool:
    return bool(_read_status(anima_dir).get("enabled", True))


def _supervisor_of(anima_dir: Path) -> str:
    value = _read_status(anima_dir).get("supervisor", "")
    return str(value or "")


def _parse_time(value: str) -> datetime | None:
    try:
        return ensure_aware(datetime.fromisoformat(value))
    except (TypeError, ValueError):
        return None


def _task_age_days(task: TaskEntry, now: datetime) -> float:
    ts = _parse_time(task.updated_at) or _parse_time(task.ts) or now
    return max(0.0, (now - ts).total_seconds() / 86400)


def _find_tracking_task(
    delegator_dir: Path,
    *,
    delegated_to: str,
    delegated_task_id: str,
) -> TaskEntry | None:
    try:
        for task in TaskQueueManager(delegator_dir).get_delegated_tasks():
            meta = task.meta or {}
            if meta.get("delegated_to") == delegated_to and meta.get("delegated_task_id") == delegated_task_id:
                return task
    except Exception:
        logger.debug("Failed to inspect tracking tasks for %s", delegator_dir.name, exc_info=True)
    return None


def _active_alert_exists(
    delegator_dir: Path,
    *,
    kind: str,
    target_name: str,
    delegated_task_id: str = "",
) -> bool:
    try:
        for task in TaskQueueManager(delegator_dir).load_active_tasks().values():
            meta = task.meta or {}
            if meta.get("kind") != kind:
                continue
            if meta.get("target") != target_name:
                continue
            if delegated_task_id and meta.get("delegated_task_id") != delegated_task_id:
                continue
            return True
    except Exception:
        logger.debug("Failed to inspect alert tasks for %s", delegator_dir.name, exc_info=True)
    return False


def _add_alert_task(
    delegator_dir: Path,
    *,
    kind: str,
    target_name: str,
    summary: str,
    instruction: str,
    delegated_task_id: str = "",
    extra_meta: dict[str, Any] | None = None,
) -> TaskEntry | None:
    if _active_alert_exists(
        delegator_dir,
        kind=kind,
        target_name=target_name,
        delegated_task_id=delegated_task_id,
    ):
        return None
    meta: dict[str, Any] = {"kind": kind, "target": target_name}
    if delegated_task_id:
        meta["delegated_task_id"] = delegated_task_id
    if extra_meta:
        meta.update(extra_meta)
    return TaskQueueManager(delegator_dir).add_task(
        source="anima",
        original_instruction=instruction,
        assignee=delegator_dir.name,
        summary=summary,
        relay_chain=[target_name, delegator_dir.name],
        meta=meta,
    )


def surface_disabled_delegations_for_supervisor(
    supervisor_name: str,
    animas_dir: Path,
    *,
    target_name: str | None = None,
) -> list[dict[str, str]]:
    """Create supervisor-visible reassignment alerts for disabled delegatees."""
    supervisor_dir = animas_dir / supervisor_name
    if not supervisor_dir.is_dir():
        return []

    alerts: list[dict[str, str]] = []
    for target_dir in sorted(animas_dir.iterdir()) if animas_dir.is_dir() else []:
        if not target_dir.is_dir():
            continue
        if target_name and target_dir.name != target_name:
            continue
        if _is_enabled(target_dir):
            continue

        try:
            tasks = TaskQueueManager(target_dir).load_active_tasks().values()
        except Exception:
            logger.debug("Failed to load disabled delegatee tasks for %s", target_dir.name, exc_info=True)
            continue

        for task in tasks:
            if task.source != "anima":
                continue
            delegator = task.relay_chain[0] if task.relay_chain else _supervisor_of(target_dir)
            if delegator != supervisor_name:
                continue

            tracking = _find_tracking_task(
                supervisor_dir,
                delegated_to=target_dir.name,
                delegated_task_id=task.task_id,
            )
            if tracking:
                TaskQueueManager(supervisor_dir).update_meta(
                    tracking.task_id,
                    {
                        "needs_reassignment": True,
                        "disabled_delegatee": target_dir.name,
                    },
                    summary=f"{tracking.summary} (needs reassignment: {target_dir.name} disabled)",
                )

            created = _add_alert_task(
                supervisor_dir,
                kind="disabled_delegation_reassignment",
                target_name=target_dir.name,
                delegated_task_id=task.task_id,
                summary=f"Needs reassignment: {task.summary}",
                instruction=(
                    f"Delegated task {task.task_id} assigned to disabled Anima "
                    f"{target_dir.name} needs reassignment or cancellation.\n\n"
                    f"Original instruction:\n{task.original_instruction}"
                ),
                extra_meta={"tracking_task_id": tracking.task_id if tracking else ""},
            )
            alerts.append(
                {
                    "delegatee": target_dir.name,
                    "delegated_task_id": task.task_id,
                    "alert_task_id": created.task_id if created else "",
                }
            )
    return alerts


def bounce_disabled_delegations(
    animas_dir: Path,
    *,
    older_than_days: int = DELEGATION_BOUNCE_DEFAULT_DAYS,
    now: datetime | None = None,
) -> list[dict[str, str]]:
    """Bounce stale open tasks from disabled delegatees back to delegators."""
    current = now or now_local()
    bounced: list[dict[str, str]] = []
    if not animas_dir.is_dir():
        return bounced

    for target_dir in sorted(animas_dir.iterdir()):
        if not target_dir.is_dir() or _is_enabled(target_dir):
            continue
        try:
            target_tqm = TaskQueueManager(target_dir)
            tasks = list(target_tqm.load_active_tasks().values())
        except Exception:
            logger.debug("Failed to load tasks for disabled anima %s", target_dir.name, exc_info=True)
            continue

        for task in tasks:
            if task.source != "anima":
                continue
            if bool((task.meta or {}).get("bounced_back")):
                continue
            if _task_age_days(task, current) < older_than_days:
                continue
            delegator = task.relay_chain[0] if task.relay_chain else _supervisor_of(target_dir)
            if not delegator:
                continue
            delegator_dir = animas_dir / delegator
            if not delegator_dir.is_dir():
                continue

            created = _add_alert_task(
                delegator_dir,
                kind="delegation_bounced",
                target_name=target_dir.name,
                delegated_task_id=task.task_id,
                summary=f"Bounced delegation from disabled {target_dir.name}: {task.summary}",
                instruction=(
                    f"The delegated task {task.task_id} was still open after "
                    f"{older_than_days} days while {target_dir.name} is disabled. "
                    "Decide whether to reassign, cancel, or complete it manually.\n\n"
                    f"Original instruction:\n{task.original_instruction}"
                ),
                extra_meta={"age_days": round(_task_age_days(task, current), 2)},
            )
            target_tqm.update_status(
                task.task_id,
                "blocked",
                summary=f"{task.summary} (bounced to {delegator}: delegatee disabled)",
            )
            target_tqm.update_meta(
                task.task_id,
                {
                    "bounced_back": True,
                    "bounce_delegator": delegator,
                    "bounce_reason": "delegatee_disabled",
                },
            )
            tracking = _find_tracking_task(
                delegator_dir,
                delegated_to=target_dir.name,
                delegated_task_id=task.task_id,
            )
            if tracking:
                TaskQueueManager(delegator_dir).update_meta(
                    tracking.task_id,
                    {
                        "bounced_back": True,
                        "disabled_delegatee": target_dir.name,
                    },
                    summary=f"{tracking.summary} (bounced back: {target_dir.name} disabled)",
                )
            bounced.append(
                {
                    "delegatee": target_dir.name,
                    "delegator": delegator,
                    "delegated_task_id": task.task_id,
                    "bounce_task_id": created.task_id if created else "",
                }
            )
    return bounced


def detect_dormant_animas(
    animas_dir: Path,
    *,
    dormant_days: int = DORMANT_ANIMA_DEFAULT_DAYS,
    now: datetime | None = None,
) -> list[dict[str, str]]:
    """Return enabled Animas whose status last_activity is older than dormant_days."""
    current = now or now_local()
    cutoff = current - timedelta(days=dormant_days)
    proposals: list[dict[str, str]] = []
    if not animas_dir.is_dir():
        return proposals

    for anima_dir in sorted(animas_dir.iterdir()):
        if not anima_dir.is_dir() or not _is_enabled(anima_dir):
            continue
        status = _read_status(anima_dir)
        last_activity = str(status.get("last_activity") or status.get("last_heartbeat") or "")
        activity_ts = _parse_time(last_activity)
        if activity_ts is None or activity_ts >= cutoff:
            continue
        proposals.append(
            {
                "name": anima_dir.name,
                "supervisor": str(status.get("supervisor") or ""),
                "last_activity": activity_ts.isoformat(),
                "age_days": str(int((current - activity_ts).total_seconds() / 86400)),
            }
        )
    return proposals


def record_dormant_offboarding_proposals(
    animas_dir: Path,
    *,
    dormant_days: int = DORMANT_ANIMA_DEFAULT_DAYS,
    now: datetime | None = None,
) -> list[dict[str, str]]:
    """Create one active offboarding proposal task per dormant Anima."""
    recorded: list[dict[str, str]] = []
    for proposal in detect_dormant_animas(animas_dir, dormant_days=dormant_days, now=now):
        supervisor = proposal["supervisor"]
        if not supervisor:
            continue
        supervisor_dir = animas_dir / supervisor
        if not supervisor_dir.is_dir():
            continue
        created = _add_alert_task(
            supervisor_dir,
            kind="dormant_anima_offboarding",
            target_name=proposal["name"],
            summary=f"Offboarding proposal: {proposal['name']} dormant for {proposal['age_days']} days",
            instruction=(
                f"{proposal['name']} has had no recorded activity since "
                f"{proposal['last_activity']} ({proposal['age_days']} days). "
                "Review whether to archive, re-enable, or keep it registered."
            ),
            extra_meta={"last_activity": proposal["last_activity"], "age_days": proposal["age_days"]},
        )
        recorded.append(
            {
                **proposal,
                "proposal_task_id": created.task_id if created else "",
            }
        )
    return recorded


def build_supervision_context(
    supervisor_name: str,
    animas_dir: Path,
    *,
    bounce_days: int = DELEGATION_BOUNCE_DEFAULT_DAYS,
    dormant_days: int = DORMANT_ANIMA_DEFAULT_DAYS,
) -> str:
    """Return heartbeat context for supervision recovery and tool-backed oversight."""
    disabled_alerts = surface_disabled_delegations_for_supervisor(supervisor_name, animas_dir)
    bounced = bounce_disabled_delegations(animas_dir, older_than_days=bounce_days)
    dormant = [
        item for item in record_dormant_offboarding_proposals(animas_dir, dormant_days=dormant_days)
        if item.get("supervisor") == supervisor_name
    ]

    lines: list[str] = [
        "## Subordinate supervision status",
        "",
        "Use task_tracker, org_dashboard, ping_subordinate, and read_subordinate_state before assuming delegated work is healthy.",
    ]
    if disabled_alerts:
        lines.append("")
        lines.append("Disabled delegatees with open tasks needing reassignment:")
        for item in disabled_alerts[:20]:
            lines.append(f"- {item['delegatee']} task={item['delegated_task_id']}")
    if bounced:
        own_bounced = [item for item in bounced if item.get("delegator") == supervisor_name]
        if own_bounced:
            lines.append("")
            lines.append(f"Delegations bounced back after {bounce_days} days:")
            for item in own_bounced[:20]:
                lines.append(f"- {item['delegatee']} task={item['delegated_task_id']}")
    if dormant:
        lines.append("")
        lines.append(f"Dormant Animas with offboarding proposals ({dormant_days}+ days):")
        for item in dormant[:20]:
            lines.append(f"- {item['name']} last_activity={item['last_activity']}")

    if len(lines) == 3:
        return ""
    return "\n".join(lines)
