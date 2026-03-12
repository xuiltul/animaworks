from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""AuditAggregator — ActivityLog + TaskQueue aggregation for supervisor audits."""

import json
import logging
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

from core.i18n import t

logger = logging.getLogger(__name__)

AUDIT_EVENT_TYPES: list[str] = [
    "heartbeat_end",
    "heartbeat_reflection",
    "response_sent",
    "cron_executed",
    "tool_use",
    "message_sent",
    "task_exec_end",
    "issue_resolved",
    "error",
]

_SUMMARY_TRUNCATE = 300
_REPORT_ENTRY_TRUNCATE = 300
_BATCH_PER_ANIMA_CHARS = 500
_MAX_ERRORS_SHOWN = 5

_MAX_TOOL_SUMMARY = 10  # tool usage TOP N (summary line, not individual entries)

_REPORT_ICONS: dict[str, str] = {
    "heartbeat_end": "🔄",
    "heartbeat_reflection": "🪞",
    "response_sent": "💬",
    "cron_executed": "⏰",
    "tool_use": "🔧",
    "message_sent": "📨",
    "task_exec_end": "✅",
    "issue_resolved": "🎯",
    "error": "❌",
}


def _audit_title(key: str, *, since: datetime | None = None, **kwargs: Any) -> str:
    """Return the appropriate title string, switching to ``_since`` variant when needed."""
    if since is not None:
        since_str = since.strftime("%H:%M")
        return t(f"{key}_since", since=since_str, **kwargs)
    return t(key, **kwargs)


# ── AuditAggregator ──────────────────────────────────────────


class AuditAggregator:
    """Aggregate ActivityLog + TaskQueue data for supervisor audits."""

    def __init__(self, anima_dir: Path) -> None:
        self._anima_dir = anima_dir
        self._name = anima_dir.name

    def _load_entries(self, hours: int, since: datetime | None = None) -> list[Any]:
        from core.memory.activity import ActivityLogger

        al = ActivityLogger(self._anima_dir)
        return al._load_entries(hours=hours, types=AUDIT_EVENT_TYPES, since=since)

    def _get_status_info(self) -> tuple[str, str]:
        """Return (process_status, model_name) from status.json."""
        status_file = self._anima_dir / "status.json"
        process_status = "unknown"
        model_name = "unknown"
        if status_file.exists():
            try:
                sdata = json.loads(status_file.read_text(encoding="utf-8"))
                process_status = "enabled" if sdata.get("enabled", True) else "disabled"
                model_name = sdata.get("model", "unknown")
            except (json.JSONDecodeError, OSError):
                pass
        return process_status, model_name

    @staticmethod
    def _extract_content(entry: Any) -> str:
        """Extract display content from an entry based on its type."""
        etype = entry.type

        if etype == "heartbeat_end":
            return (entry.summary or entry.content)[:_REPORT_ENTRY_TRUNCATE]

        if etype == "heartbeat_reflection":
            return entry.content[:_REPORT_ENTRY_TRUNCATE]

        if etype == "response_sent":
            return entry.content[:_REPORT_ENTRY_TRUNCATE]

        if etype == "cron_executed":
            return (entry.content or entry.summary)[:500]

        if etype == "tool_use":
            tool_name = entry.tool or "unknown"
            detail = (entry.content or entry.summary)[:200]
            return f"{tool_name}: {detail}" if detail else tool_name

        if etype == "message_sent":
            peer = entry.to_person or "unknown"
            content = entry.content[:200]
            return f"→ {peer}: {content}"

        if etype == "task_exec_end":
            return (entry.summary or entry.content)[:_REPORT_ENTRY_TRUNCATE]

        if etype == "issue_resolved":
            return entry.content[:_REPORT_ENTRY_TRUNCATE]

        if etype == "error":
            text = entry.summary or entry.content[:100]
            phase = entry.meta.get("phase", "") if isinstance(entry.meta, dict) else ""
            return f"(phase: {phase}) {text}" if phase else text

        return (entry.summary or entry.content)[:_REPORT_ENTRY_TRUNCATE]

    # ── Summary mode ─────────────────────────────────────────

    def generate_summary(
        self,
        hours: int = 24,
        *,
        compact: bool = False,
        since: datetime | None = None,
    ) -> str:
        """Generate a statistics summary for the audit period."""
        entries = self._load_entries(hours, since=since)
        process_status, model_name = self._get_status_info()

        type_counts: Counter[str] = Counter()
        peer_sent: Counter[str] = Counter()
        peer_recv: Counter[str] = Counter()
        tool_counts: Counter[str] = Counter()
        error_details: list[str] = []

        for e in entries:
            type_counts[e.type] += 1

            if e.type == "tool_use" and e.tool:
                tool_counts[e.tool] += 1

            if e.type in ("message_sent", "dm_sent"):
                peer = e.to_person or "unknown"
                peer_sent[peer] += 1
            elif e.type in ("message_received", "dm_received"):
                peer = e.from_person or "unknown"
                peer_recv[peer] += 1

            if e.type == "error":
                ts_short = e.ts[11:16] if len(e.ts) >= 16 else e.ts
                phase = e.meta.get("phase", "") if isinstance(e.meta, dict) else ""
                summary = e.summary or e.content[:100]
                detail = f"[{ts_short}]"
                if phase:
                    detail += f" {phase} phase:"
                detail += f" {summary}"
                error_details.append(detail)

        title = _audit_title("handler.audit_summary_title", name=self._name, hours=hours, since=since)
        lines: list[str] = [title, ""]

        if not compact:
            lines.append(t("handler.audit_status_line", status=process_status, model=model_name))
            lines.append("")

        # Activity counts
        lines.append(t("handler.audit_section_activity"))
        msg_recv = type_counts.get("message_received", 0) + type_counts.get("dm_received", 0)
        resp_sent = type_counts.get("response_sent", 0)
        dm_sent = type_counts.get("message_sent", 0) + type_counts.get("dm_sent", 0)
        tool_use = type_counts.get("tool_use", 0)
        hb = type_counts.get("heartbeat_end", 0)
        cron = type_counts.get("cron_executed", 0)
        errors = type_counts.get("error", 0)

        lines.append(
            t(
                "handler.audit_activity_counts",
                msg_recv=msg_recv,
                resp_sent=resp_sent,
                dm_sent=dm_sent,
                tool_use=tool_use,
                hb=hb,
                cron=cron,
                errors=errors,
            )
        )
        lines.append("")

        # Task status
        lines.append(t("handler.audit_section_tasks"))
        task_info = self._get_task_summary(hours)
        lines.append(task_info)
        lines.append("")

        # Communication peers
        if peer_sent or peer_recv:
            lines.append(t("handler.audit_section_comms"))
            all_peers = sorted(set(peer_sent) | set(peer_recv))
            peer_strs = [f"{p} ({peer_sent.get(p, 0)}↑/{peer_recv.get(p, 0)}↓)" for p in all_peers]
            lines.append("  " + ", ".join(peer_strs))
            lines.append("")

        # Error details
        if error_details:
            lines.append(t("handler.audit_section_errors"))
            for d in error_details[-_MAX_ERRORS_SHOWN:]:
                lines.append(f"  {d}")
            lines.append("")

        if compact and len("\n".join(lines)) > _BATCH_PER_ANIMA_CHARS:
            text = "\n".join(lines)[:_BATCH_PER_ANIMA_CHARS]
            last_nl = text.rfind("\n")
            if last_nl > 0:
                text = text[:last_nl]
            return text + "\n  ..."

        return "\n".join(lines)

    # ── Report mode ──────────────────────────────────────────

    def generate_report(self, hours: int = 24, since: datetime | None = None) -> str:
        """Generate a unified timeline report (日報形式).

        All non-tool_use events are displayed chronologically.
        tool_use events are aggregated into a summary at the bottom.
        """
        entries = self._load_entries(hours, since=since)
        process_status, model_name = self._get_status_info()

        title = _audit_title("handler.audit_report_title", name=self._name, hours=hours, since=since)
        lines: list[str] = [
            title,
            "",
            t("handler.audit_status_line", status=process_status, model=model_name),
            "",
        ]

        if not entries:
            lines.append(t("handler.audit_no_activity"))
            return "\n".join(lines)

        type_counts: Counter[str] = Counter()
        tool_counts: Counter[str] = Counter()
        for e in entries:
            type_counts[e.type] += 1
            if e.type == "tool_use" and e.tool:
                tool_counts[e.tool] += 1

        timeline_entries = [e for e in entries if e.type != "tool_use"]

        for e in timeline_entries:
            ts_short = e.ts[11:16] if len(e.ts) >= 16 else e.ts
            icon = _REPORT_ICONS.get(e.type, "📝")
            label = self._event_label(e)
            content = self._extract_content(e)
            lines.append(f"[{ts_short}] {icon} {label}")
            if content:
                for cl in content.split("\n")[:3]:
                    lines.append(f"  {cl}")
            lines.append("")

        # Tool usage summary (not individual entries)
        tool_count = type_counts.get("tool_use", 0)
        if tool_count > 0:
            lines.append(t("handler.audit_section_tool_summary", count=tool_count))
            top_tools = tool_counts.most_common(_MAX_TOOL_SUMMARY)
            summary_parts = [f"{name}: {cnt}" for name, cnt in top_tools]
            lines.append("  " + " | ".join(summary_parts))
            lines.append("")

        # Footer statistics
        total = len(entries)
        hb_count = type_counts.get("heartbeat_end", 0)
        resp_count = type_counts.get("response_sent", 0) + type_counts.get("cron_executed", 0)
        dm_count = type_counts.get("message_sent", 0)
        err_count = type_counts.get("error", 0)
        lines.append(
            t(
                "handler.audit_report_footer",
                total=total,
                tools=tool_count,
                hb=hb_count,
                resp_sent=resp_count,
                dm_sent=dm_count,
                errors=err_count,
            )
        )

        return "\n".join(lines)

    # ── Merged timeline ────────────────────────────────────────

    @classmethod
    def generate_merged_timeline(
        cls,
        anima_dirs: list[Path],
        hours: int = 24,
        since: datetime | None = None,
    ) -> str:
        """Generate a unified cross-anima timeline sorted chronologically.

        Merges events from multiple Animas into a single timeline.
        tool_use events are aggregated per-anima at the bottom.
        """
        from core.memory.activity import ActivityLogger

        tagged: list[tuple[str, Any]] = []
        per_anima_tool_counts: dict[str, Counter[str]] = {}
        per_anima_total_tools: Counter[str] = Counter()
        global_type_counts: Counter[str] = Counter()

        for anima_dir in anima_dirs:
            name = anima_dir.name
            al = ActivityLogger(anima_dir)
            entries = al._load_entries(hours=hours, types=AUDIT_EVENT_TYPES, since=since)
            per_anima_tool_counts[name] = Counter()
            for e in entries:
                global_type_counts[e.type] += 1
                if e.type == "tool_use":
                    if e.tool:
                        per_anima_tool_counts[name][e.tool] += 1
                    per_anima_total_tools[name] += 1
                else:
                    tagged.append((name, e))

        tagged.sort(key=lambda x: x[1].ts)

        title = _audit_title("handler.audit_merged_title", hours=hours, count=len(anima_dirs), since=since)
        lines: list[str] = [title, ""]

        if not tagged and not any(per_anima_total_tools.values()):
            lines.append(t("handler.audit_no_activity"))
            return "\n".join(lines)

        for name, e in tagged:
            ts_short = e.ts[11:16] if len(e.ts) >= 16 else e.ts
            icon = _REPORT_ICONS.get(e.type, "📝")
            label = cls._event_label(e)
            content = cls._extract_content(e)
            lines.append(f"[{ts_short}] {name} {icon} {label}")
            if content:
                for cl in content.split("\n")[:3]:
                    lines.append(f"  {cl}")
            lines.append("")

        has_tools = any(c for c in per_anima_tool_counts.values() if c)
        if has_tools:
            lines.append(t("handler.audit_merged_tool_header"))
            for name in sorted(per_anima_tool_counts):
                tc = per_anima_tool_counts[name]
                total = per_anima_total_tools[name]
                if total == 0:
                    continue
                top = tc.most_common(_MAX_TOOL_SUMMARY)
                parts = [f"{tn}: {cnt}" for tn, cnt in top]
                lines.append("  " + t("handler.audit_tool_line", name=name, total=total) + " | ".join(parts))
            lines.append("")

        total = sum(global_type_counts.values())
        tool_total = sum(per_anima_total_tools.values())
        lines.append(
            t(
                "handler.audit_merged_footer",
                count=len(anima_dirs),
                total=total,
                tools=tool_total,
                hb=global_type_counts.get("heartbeat_end", 0),
                resp_sent=global_type_counts.get("response_sent", 0) + global_type_counts.get("cron_executed", 0),
                dm_sent=global_type_counts.get("message_sent", 0),
                errors=global_type_counts.get("error", 0),
            )
        )

        return "\n".join(lines)

    # ── Helpers ───────────────────────────────────────────────

    @staticmethod
    def _event_label(entry: Any) -> str:
        """Human-readable label for an event entry."""
        etype = entry.type
        if etype == "heartbeat_end":
            return t("handler.audit_label_heartbeat")
        if etype == "heartbeat_reflection":
            return t("handler.audit_label_reflection")
        if etype == "response_sent":
            return t("handler.audit_label_response")
        if etype == "cron_executed":
            task_name = ""
            if isinstance(entry.meta, dict):
                task_name = entry.meta.get("task_name", "")
            return t("handler.audit_label_cron", task_name=task_name)
        if etype == "tool_use":
            return t("handler.audit_label_tool", tool=entry.tool or "unknown")
        if etype == "message_sent":
            return t("handler.audit_label_dm", peer=entry.to_person or "unknown")
        if etype == "task_exec_end":
            return t("handler.audit_label_task_done")
        if etype == "issue_resolved":
            return t("handler.audit_label_resolved")
        if etype == "error":
            phase = entry.meta.get("phase", "") if isinstance(entry.meta, dict) else ""
            return t("handler.audit_label_error", phase=phase)
        return etype

    def _get_task_summary(self, hours: int) -> str:
        """Build a task status line from TaskQueueManager."""
        try:
            from core.memory.task_queue import TaskQueueManager

            tqm = TaskQueueManager(self._anima_dir)
            pending = [tk for tk in tqm.get_pending() if tk.status == "pending"]
            in_progress = [tk for tk in tqm.get_pending() if tk.status == "in_progress"]
            done = tqm.list_tasks(status="done")
            stale = tqm.get_stale_tasks()
            return t(
                "handler.audit_task_counts",
                pending=len(pending),
                in_progress=len(in_progress),
                done=len(done),
                stale=len(stale),
            )
        except Exception:
            return t("handler.audit_task_counts", pending=0, in_progress=0, done=0, stale=0)
