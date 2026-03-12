from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Auto-generate ``shared/task-board.md`` from task_queue.jsonl files.

The task queue (JSONL) is the single source of truth.  This module reads
every agent's ``state/task_queue.jsonl``, aggregates the tasks, and
renders a Markdown task-board.  It also syncs the rendered board to a
pinned Slack message via ``slack_channel_update``.

Usage from handler hooks::

    from core.memory.taskboard_generator import regenerate_and_sync
    regenerate_and_sync(anima_dir)   # call after update_task / backlog_task
"""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path

from core.memory.task_queue import TaskQueueManager
from core.paths import get_animas_dir, get_shared_dir
from core.schemas import TaskEntry
from core.time_utils import ensure_aware, now_local

logger = logging.getLogger("animaworks.taskboard_generator")

# How many days of completed tasks to show
_COMPLETED_DAYS = 7

# Slack channel for #デジタル伴走事業_tasks
_TASKS_CHANNEL_ID = "C0AJ4J5KK46"

# Statuses treated as "blocked"
_BLOCKED_STATUSES = frozenset({"blocked"})
# Statuses treated as "in progress"
_IN_PROGRESS_STATUSES = frozenset({"in_progress", "delegated"})
# Statuses treated as "pending"
_PENDING_STATUSES = frozenset({"pending"})
# Terminal statuses
_DONE_STATUSES = frozenset({"done", "cancelled", "failed"})


def _collect_all_tasks() -> dict[str, TaskEntry]:
    """Load tasks from all agents' task_queue.jsonl files.

    Returns a dict of task_id -> TaskEntry.  When the same task_id
    exists in multiple agents (delegation), the *non-delegated* copy
    wins so we show the executor's status rather than the delegator's.
    """
    animas_dir = get_animas_dir()
    if not animas_dir.exists():
        return {}

    all_tasks: dict[str, TaskEntry] = {}
    delegated_tasks: dict[str, TaskEntry] = {}

    for agent_dir in sorted(animas_dir.iterdir()):
        if not agent_dir.is_dir():
            continue
        mgr = TaskQueueManager(agent_dir)
        try:
            tasks = mgr._load_all()
        except Exception:
            logger.warning("Failed to load task queue for %s", agent_dir.name)
            continue
        for tid, entry in tasks.items():
            if entry.status == "delegated":
                # Track delegated tasks separately
                delegated_tasks[tid] = entry
                # Also check meta for delegated_task_id
                dtid = entry.meta.get("delegated_task_id")
                if dtid:
                    delegated_tasks[dtid] = entry
            else:
                all_tasks[tid] = entry

    # Add delegated tasks only if no executor-side copy exists
    for tid, entry in delegated_tasks.items():
        if tid not in all_tasks:
            all_tasks[tid] = entry

    return all_tasks


def _safe_deadline_sort_key(entry: TaskEntry) -> tuple[int, str]:
    """Sort key: tasks with deadline first (by deadline), then without."""
    if entry.deadline:
        return (0, entry.deadline)
    return (1, entry.ts)


def _format_deadline_short(deadline: str | None) -> str:
    """Format deadline as short date string (e.g. '3/15')."""
    if not deadline:
        return "—"
    try:
        dt = datetime.fromisoformat(deadline)
        return f"{dt.month}/{dt.day}"
    except (ValueError, TypeError):
        return "—"


def _is_overdue(deadline: str | None, now: datetime) -> bool:
    """Check if a deadline has passed."""
    if not deadline:
        return False
    try:
        dl = ensure_aware(datetime.fromisoformat(deadline))
        return now >= dl
    except (ValueError, TypeError):
        return False


def generate_taskboard(
    tasks: dict[str, TaskEntry] | None = None,
    now: datetime | None = None,
) -> str:
    """Generate task-board.md content from task queue data.

    Args:
        tasks: Pre-loaded tasks dict. If None, loads from all agents.
        now: Current datetime. If None, uses now_local().

    Returns:
        Markdown string for task-board.md.
    """
    if tasks is None:
        tasks = _collect_all_tasks()
    if now is None:
        now = now_local()

    # Classify tasks by status
    blocked: list[TaskEntry] = []
    in_progress: list[TaskEntry] = []
    pending: list[TaskEntry] = []
    completed: list[TaskEntry] = []

    cutoff = now - timedelta(days=_COMPLETED_DAYS)

    for entry in tasks.values():
        if entry.status in _BLOCKED_STATUSES:
            blocked.append(entry)
        elif entry.status in _IN_PROGRESS_STATUSES:
            in_progress.append(entry)
        elif entry.status in _PENDING_STATUSES:
            pending.append(entry)
        elif entry.status in _DONE_STATUSES:
            # Only include recent completions
            try:
                updated = ensure_aware(datetime.fromisoformat(entry.updated_at))
                if updated >= cutoff:
                    completed.append(entry)
            except (ValueError, TypeError):
                pass

    # Sort each section
    blocked.sort(key=_safe_deadline_sort_key)
    in_progress.sort(key=_safe_deadline_sort_key)
    pending.sort(key=_safe_deadline_sort_key)
    completed.sort(key=lambda e: e.updated_at, reverse=True)

    # Truncate long summaries for readability
    _MAX_SUMMARY_LEN = 80
    for entry_list in (blocked, in_progress, pending, completed):
        for entry in entry_list:
            if len(entry.summary) > _MAX_SUMMARY_LEN:
                entry.summary = entry.summary[:_MAX_SUMMARY_LEN] + "…"

    # Build markdown
    lines: list[str] = []
    timestamp = now.strftime("%Y-%m-%d %H:%M")
    lines.append("# タスクボード")
    lines.append("")
    lines.append(f"最終更新: {timestamp}（自動生成）")
    lines.append("")
    lines.append("> このファイルは task_queue.jsonl から自動生成されています。直接編集しないでください。")
    lines.append("")

    # 🔴 Blocked
    lines.append("---")
    lines.append("")
    lines.append("## 🔴 ブロック中")
    lines.append("")
    if blocked:
        lines.append("| # | タスク | 担当 | 状態 | 期限 |")
        lines.append("|---|--------|------|------|------|")
        for entry in blocked:
            overdue = " ⚠️期限超過" if _is_overdue(entry.deadline, now) else ""
            lines.append(
                f"| {entry.task_id[:8]} | {entry.summary} | {entry.assignee} "
                f"| blocked{overdue} | {_format_deadline_short(entry.deadline)} |"
            )
    else:
        lines.append("（なし）")
    lines.append("")

    # 🟡 In Progress
    lines.append("---")
    lines.append("")
    lines.append("## 🟡 進行中")
    lines.append("")
    if in_progress:
        lines.append("| # | タスク | 担当 | 状態 | 期限 |")
        lines.append("|---|--------|------|------|------|")
        for entry in in_progress:
            status_label = entry.status
            if entry.status == "delegated":
                delegated_to = entry.meta.get("delegated_to", "")
                status_label = f"委任中→{delegated_to}" if delegated_to else "委任中"
            overdue = " ⚠️期限超過" if _is_overdue(entry.deadline, now) else ""
            lines.append(
                f"| {entry.task_id[:8]} | {entry.summary} | {entry.assignee} "
                f"| {status_label}{overdue} | {_format_deadline_short(entry.deadline)} |"
            )
    else:
        lines.append("（なし）")
    lines.append("")

    # 📋 Pending
    lines.append("---")
    lines.append("")
    lines.append("## 📋 未着手")
    lines.append("")
    if pending:
        lines.append("| # | タスク | 担当 | 期限 |")
        lines.append("|---|--------|------|------|")
        for entry in pending:
            overdue = " ⚠️期限超過" if _is_overdue(entry.deadline, now) else ""
            lines.append(
                f"| {entry.task_id[:8]} | {entry.summary} | {entry.assignee} "
                f"| {_format_deadline_short(entry.deadline)}{overdue} |"
            )
    else:
        lines.append("（なし）")
    lines.append("")

    # ✅ Completed
    lines.append("---")
    lines.append("")
    lines.append(f"## ✅ 直近{_COMPLETED_DAYS}日間の完了")
    lines.append("")
    if completed:
        lines.append("| タスク | 担当 | 完了日 |")
        lines.append("|--------|------|--------|")
        for entry in completed:
            try:
                done_dt = datetime.fromisoformat(entry.updated_at)
                done_date = f"{done_dt.month}/{done_dt.day}"
            except (ValueError, TypeError):
                done_date = "—"
            status_note = ""
            if entry.status == "cancelled":
                status_note = " （キャンセル）"
            elif entry.status == "failed":
                status_note = " （失敗）"
            lines.append(f"| {entry.summary}{status_note} | {entry.assignee} | {done_date} |")
    else:
        lines.append("（なし）")
    lines.append("")

    # Footer
    lines.append("---")
    lines.append("")
    lines.append("## 運用ルール")
    lines.append("")
    lines.append("- **正本**: 各エージェントの `state/task_queue.jsonl` が SSOT")
    lines.append("- **このファイル**: task_queue.jsonl から自動生成（`backlog_task` / `update_task` 実行時に再生成）")
    lines.append("- **Slack 同期**: 再生成のたびに `#デジタル伴走事業_tasks` のピン留めメッセージを自動更新")
    lines.append("- **直接編集禁止**: このファイルを手動で編集しても次の再生成で上書きされます")
    lines.append("")

    return "\n".join(lines)


def _write_taskboard(markdown: str) -> Path:
    """Write task-board.md to shared directory."""
    shared_dir = get_shared_dir()
    shared_dir.mkdir(parents=True, exist_ok=True)
    path = shared_dir / "task-board.md"
    path.write_text(markdown, encoding="utf-8")
    logger.info("task-board.md regenerated (%d bytes)", len(markdown))
    return path


def _sync_to_slack(markdown: str) -> None:
    """Sync task-board.md content to Slack pinned message.

    Reads pinned_ts from shared/task-board-slack.json.
    Falls back to posting a new message if update fails.
    """
    try:
        from core.tools._base import get_credential
        from core.tools.slack import SlackClient, taskboard_md_to_slack
    except ImportError:
        logger.warning("Slack tools not available, skipping Slack sync")
        return

    shared_dir = get_shared_dir()
    slack_json_path = shared_dir / "task-board-slack.json"

    try:
        token = get_credential("slack", "notification", env_var="SLACK_BOT_TOKEN")
    except Exception:
        logger.warning("Slack token not available, skipping Slack sync")
        return

    client = SlackClient(token=token)
    slack_text = taskboard_md_to_slack(markdown)

    # Slack chat.update has strict size limits (blocks/text).
    # Truncate to safe limit and append footer if needed.
    _SLACK_MAX_LEN = 2000
    if len(slack_text) > _SLACK_MAX_LEN:
        truncated = slack_text[:_SLACK_MAX_LEN].rsplit("\n", 1)[0]
        slack_text = truncated + "\n\n_…省略あり。Full board: `shared/task-board.md`_"

    pinned_ts = ""
    if slack_json_path.exists():
        try:
            data = json.loads(slack_json_path.read_text(encoding="utf-8"))
            pinned_ts = data.get("pinned_ts", "")
        except (json.JSONDecodeError, OSError):
            pass

    new_ts = pinned_ts
    if pinned_ts:
        try:
            client.update_message(_TASKS_CHANNEL_ID, pinned_ts, slack_text)
            logger.info("Slack task-board updated (ts=%s)", pinned_ts)
        except Exception:
            logger.warning("Slack update failed, posting new message", exc_info=True)
            pinned_ts = ""  # fall through to post

    if not pinned_ts:
        try:
            resp = client.post_message(_TASKS_CHANNEL_ID, slack_text, username="kaede")
            new_ts = resp["ts"] if resp else ""
            logger.info("Slack task-board posted (ts=%s)", new_ts)
        except Exception:
            logger.warning("Slack post failed", exc_info=True)
            return

    # Update slack json
    now = now_local()
    slack_data = {
        "channel": _TASKS_CHANNEL_ID,
        "pinned_ts": new_ts,
        "description": "task-board.md Slack同期先（自動更新）",
        "last_synced": now.isoformat(),
    }
    try:
        slack_json_path.write_text(
            json.dumps(slack_data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    except OSError:
        logger.warning("Failed to write task-board-slack.json")


def regenerate_and_sync(anima_dir: Path | None = None) -> str:
    """Regenerate task-board.md from all agents and sync to Slack.

    This is the main entry point called from handler hooks.

    Args:
        anima_dir: Not used directly (all agents are scanned).
            Kept for consistent handler hook signature.

    Returns:
        The generated markdown string.
    """
    try:
        markdown = generate_taskboard()
        _write_taskboard(markdown)
        _sync_to_slack(markdown)
    except Exception:
        logger.exception("Failed to regenerate task-board")
        markdown = ""
    return markdown
