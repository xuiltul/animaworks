from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Organisation-wide audit data collection utility.

Collects structured audit data across all enabled Animas by reading
activity logs, status files, and task queues.  Used by the Activity
Report API and the CLI audit commands.
"""

import asyncio
import json
import logging
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from core.i18n import t
from core.paths import get_animas_dir

logger = logging.getLogger(__name__)

_AUDIT_ENTRY_LIMIT = 10_000

# High-value event types for qualitative extraction (not tool_use)
_QUALITATIVE_EVENT_TYPES = frozenset(
    {
        "heartbeat_end",
        "heartbeat_reflection",
        "response_sent",
        "message_sent",
        "cron_executed",
        "task_exec_end",
        "issue_resolved",
        "error",
    }
)
_MAX_KEY_ACTIVITIES = 15
_KEY_ACTIVITY_TRUNCATE = 200
_TOP_TOOLS_LIMIT = 10
_TIMELINE_CONTENT_TRUNCATE = 300

_THINNABLE_EVENT_TYPES = frozenset({"heartbeat_end", "heartbeat_reflection"})
_PROTECTED_EVENT_TYPES = _QUALITATIVE_EVENT_TYPES - _THINNABLE_EVENT_TYPES - {"cron_executed"}
_COMMAND_CRON_PREFIX = "コマンド:"

_TIMELINE_ICONS: dict[str, str] = {
    "heartbeat_end": "🔄",
    "heartbeat_reflection": "🪞",
    "response_sent": "💬",
    "cron_executed": "⏰",
    "message_sent": "📨",
    "task_exec_end": "✅",
    "issue_resolved": "🎯",
    "error": "❌",
}

_TIMELINE_LABEL_KEYS: dict[str, str] = {
    "heartbeat_end": "audit.timeline_label_heartbeat_end",
    "heartbeat_reflection": "audit.timeline_label_heartbeat_reflection",
    "response_sent": "audit.timeline_label_response_sent",
    "cron_executed": "audit.timeline_label_cron_executed",
    "message_sent": "audit.timeline_label_message_sent",
    "task_exec_end": "audit.timeline_label_task_exec_end",
    "issue_resolved": "audit.timeline_label_issue_resolved",
    "error": "audit.timeline_label_error",
}


def _get_timeline_label(etype: str) -> str:
    """Resolve a timeline event label via i18n."""
    key = _TIMELINE_LABEL_KEYS.get(etype)
    return t(key) if key else etype


# ── Data models ──────────────────────────────────────────────


@dataclass
class AnimaAuditEntry:
    """Audit metrics for a single Anima."""

    name: str
    enabled: bool
    model: str
    supervisor: str | None
    role: str | None
    total_entries: int
    type_counts: dict[str, int]
    messages_sent: int
    messages_received: int
    errors: int
    tasks_total: int
    tasks_pending: int
    tasks_done: int
    peers_sent: dict[str, int]
    peers_received: dict[str, int]
    first_activity: str | None
    last_activity: str | None

    # Qualitative fields for LLM narrative generation
    key_activities: list[dict[str, str]] = field(default_factory=list)
    # Each element: {"ts": "HH:MM", "type": "heartbeat_end", "summary": "...(max 200 chars)"}
    top_tools: list[dict[str, Any]] = field(default_factory=list)
    # [{"name": "Read", "count": 850}, {"name": "Write", "count": 420}, ...]

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class OrgAuditReport:
    """Organisation-wide audit report aggregating all Anima metrics."""

    date: str
    animas: list[AnimaAuditEntry]
    total_entries: int = 0
    total_messages: int = 0
    total_errors: int = 0
    total_tasks_done: int = 0
    active_anima_count: int = 0
    disabled_anima_count: int = 0

    def to_dict(self) -> dict:
        d = asdict(self)
        d["animas"] = [a.to_dict() for a in self.animas]
        return d


# ── Collection logic ─────────────────────────────────────────


def _ts_to_hhmm(ts: str) -> str:
    """Extract HH:MM from ISO timestamp (e.g. 2026-03-07T10:30:00+09:00)."""
    if not ts:
        return ""
    if "T" in ts:
        time_part = ts.split("T")[1]
        # Strip timezone (+09:00 or Z)
        for sep in ("+", "-", "Z"):
            if sep in time_part:
                time_part = time_part.split(sep)[0]
        return time_part[:5] if len(time_part) >= 5 else time_part
    return ts[:5] if len(ts) >= 5 else ts


def _extract_summary_for_entry(e: dict) -> str:
    """Extract summary text for a high-value activity entry."""
    etype = e.get("type", "")
    meta = e.get("meta") or {}
    content = e.get("content", "") or ""
    summary = e.get("summary", "") or ""

    if etype == "heartbeat_end":
        return (summary or content)[:_KEY_ACTIVITY_TRUNCATE]
    if etype == "heartbeat_reflection":
        return content[:_KEY_ACTIVITY_TRUNCATE]
    if etype == "response_sent":
        return content[:_KEY_ACTIVITY_TRUNCATE]
    if etype == "message_sent":
        to_person = e.get("to_person", "") or "unknown"
        full = f"→ {to_person}: {content or summary}"
        return full[:_KEY_ACTIVITY_TRUNCATE]
    if etype == "cron_executed":
        return (content or summary)[:_KEY_ACTIVITY_TRUNCATE]
    if etype == "task_exec_end":
        return (summary or content)[:_KEY_ACTIVITY_TRUNCATE]
    if etype == "issue_resolved":
        return content[:_KEY_ACTIVITY_TRUNCATE]
    if etype == "error":
        base = (summary or content[:100])[:_KEY_ACTIVITY_TRUNCATE]
        phase = meta.get("phase")
        if phase:
            return f"(phase: {phase}) {base}"
        return base
    return (summary or content)[:_KEY_ACTIVITY_TRUNCATE]


def _load_entries_for_date(log_dir: Path, target_date: str) -> list[dict]:
    """Read activity_log/{target_date}.jsonl directly for the exact date."""
    path = log_dir / f"{target_date}.jsonl"
    if not path.exists():
        return []
    entries: list[dict] = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "timestamp" in raw and "ts" not in raw:
                raw["ts"] = raw.pop("timestamp")
            if "from" in raw:
                raw["from_person"] = raw.pop("from")
            if "to" in raw:
                raw["to_person"] = raw.pop("to")
            entries.append(raw)
    except OSError:
        pass
    entries.sort(key=lambda e: e.get("ts", ""))
    return entries


def _collect_single_anima(anima_dir: Path, target_date: str) -> AnimaAuditEntry | None:
    """Collect audit data for one Anima on a specific date."""
    name = anima_dir.name
    status_file = anima_dir / "status.json"

    enabled = True
    model = "unknown"
    supervisor: str | None = None
    role: str | None = None

    if status_file.exists():
        try:
            sdata = json.loads(status_file.read_text(encoding="utf-8"))
            enabled = sdata.get("enabled", True)
            model = sdata.get("model", "unknown")
            supervisor = sdata.get("supervisor") or None
            role = sdata.get("role") or None
        except (json.JSONDecodeError, OSError):
            logger.debug("Failed to read status.json for %s", name)

    log_dir = anima_dir / "activity_log"
    raw_entries = _load_entries_for_date(log_dir, target_date)

    type_counts: Counter[str] = Counter()
    for e in raw_entries:
        type_counts[e.get("type", "unknown")] += 1

    sent = [e for e in raw_entries if e.get("type") in ("message_sent", "dm_sent")]
    received = [e for e in raw_entries if e.get("type") in ("message_received", "dm_received")]
    error_entries = [e for e in raw_entries if e.get("type") == "error"]

    peer_sent: dict[str, int] = {}
    peer_recv: dict[str, int] = {}
    for e in sent:
        peer = e.get("to_person") or "unknown"
        peer_sent[peer] = peer_sent.get(peer, 0) + 1
    for e in received:
        peer = e.get("from_person") or "unknown"
        peer_recv[peer] = peer_recv.get(peer, 0) + 1

    tasks_total = 0
    tasks_pending = 0
    tasks_done = 0
    try:
        from core.memory.task_queue import TaskQueueManager

        tqm = TaskQueueManager(anima_dir)
        all_tasks = tqm.list_tasks()
        tasks_total = len(all_tasks)
        tasks_pending = len([t for t in all_tasks if t.status in ("pending", "in_progress", "blocked")])
        tasks_done = len([t for t in all_tasks if t.status == "done"])
    except Exception:
        logger.debug("Failed to read task queue for %s", name, exc_info=True)

    first_activity = raw_entries[0].get("ts") if raw_entries else None
    last_activity = raw_entries[-1].get("ts") if raw_entries else None

    # Extract qualitative data: key_activities and top_tools
    key_activities: list[dict[str, str]] = []
    for e in raw_entries:
        if len(key_activities) >= _MAX_KEY_ACTIVITIES:
            break
        etype = e.get("type", "")
        if etype not in _QUALITATIVE_EVENT_TYPES:
            continue
        ts = _ts_to_hhmm(e.get("ts", ""))
        summary = _extract_summary_for_entry(e)
        key_activities.append({"ts": ts, "type": etype, "summary": summary})

    tool_counts: Counter[str] = Counter()
    for e in raw_entries:
        if e.get("type") != "tool_use":
            continue
        tool_name = e.get("tool") or (e.get("meta") or {}).get("tool") or "unknown"
        tool_counts[tool_name] += 1
    top_tools = [{"name": name, "count": count} for name, count in tool_counts.most_common(_TOP_TOOLS_LIMIT)]

    return AnimaAuditEntry(
        name=name,
        enabled=enabled,
        model=model,
        supervisor=supervisor,
        role=role,
        total_entries=len(raw_entries),
        type_counts=dict(type_counts),
        messages_sent=len(sent),
        messages_received=len(received),
        errors=len(error_entries),
        tasks_total=tasks_total,
        tasks_pending=tasks_pending,
        tasks_done=tasks_done,
        peers_sent=peer_sent,
        peers_received=peer_recv,
        first_activity=first_activity,
        last_activity=last_activity,
        key_activities=key_activities,
        top_tools=top_tools,
    )


async def collect_org_audit(
    date: str,
) -> OrgAuditReport:
    """Collect audit data for all Animas on the specified date.

    Reads activity_log/{date}.jsonl for each Anima, ensuring historical
    reports reflect the requested date rather than today.

    Args:
        date: Report date string (YYYY-MM-DD).

    Returns:
        OrgAuditReport with per-anima metrics and org-level aggregates.
    """
    animas_dir = get_animas_dir()
    if not animas_dir.exists():
        return OrgAuditReport(date=date, animas=[])

    anima_dirs = sorted([d for d in animas_dir.iterdir() if d.is_dir() and (d / "status.json").exists()])

    loop = asyncio.get_event_loop()
    tasks = [loop.run_in_executor(None, _collect_single_anima, d, date) for d in anima_dirs]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    animas: list[AnimaAuditEntry] = []
    for r in results:
        if isinstance(r, AnimaAuditEntry):
            animas.append(r)
        elif isinstance(r, Exception):
            logger.warning("Audit collection failed for an anima: %s", r)

    total_entries = sum(a.total_entries for a in animas)
    total_messages = sum(a.messages_sent + a.messages_received for a in animas)
    total_errors = sum(a.errors for a in animas)
    total_tasks_done = sum(a.tasks_done for a in animas)
    active_count = sum(1 for a in animas if a.enabled)
    disabled_count = sum(1 for a in animas if not a.enabled)

    return OrgAuditReport(
        date=date,
        animas=animas,
        total_entries=total_entries,
        total_messages=total_messages,
        total_errors=total_errors,
        total_tasks_done=total_tasks_done,
        active_anima_count=active_count,
        disabled_anima_count=disabled_count,
    )


# ── Timeline text generation ─────────────────────────────────


def _extract_timeline_content(e: dict) -> str:
    """Extract display content from a raw activity dict."""
    etype = e.get("type", "")
    content = e.get("content", "") or ""
    summary = e.get("summary", "") or ""

    if etype in ("heartbeat_end", "task_exec_end"):
        return (summary or content)[:_TIMELINE_CONTENT_TRUNCATE]
    if etype in ("heartbeat_reflection", "response_sent", "issue_resolved"):
        return content[:_TIMELINE_CONTENT_TRUNCATE]
    if etype == "cron_executed":
        return (content or summary)[:_TIMELINE_CONTENT_TRUNCATE]
    if etype == "message_sent":
        to_person = e.get("to_person") or "unknown"
        return f"→ {to_person}: {(content or summary)[:250]}"
    if etype == "error":
        text = (summary or content[:100])[:_TIMELINE_CONTENT_TRUNCATE]
        meta = e.get("meta") or {}
        phase = meta.get("phase", "")
        return f"(phase: {phase}) {text}" if phase else text
    return (summary or content)[:_TIMELINE_CONTENT_TRUNCATE]


def _is_command_cron(e: dict) -> bool:
    """True if the event is a command-type cron (no narrative value)."""
    if e.get("type") != "cron_executed":
        return False
    content = (e.get("content", "") or "").strip()
    summary = (e.get("summary", "") or "").strip()
    return content.startswith(_COMMAND_CRON_PREFIX) or summary.startswith(_COMMAND_CRON_PREFIX)


def _render_event_lines(name: str, e: dict) -> list[str]:
    """Render a single timeline event to lines (without trailing blank)."""
    ts = _ts_to_hhmm(e.get("ts", ""))
    etype = e.get("type", "")
    icon = _TIMELINE_ICONS.get(etype, "📝")
    label = _get_timeline_label(etype)
    content = _extract_timeline_content(e)
    out = [f"[{ts}] {name} {icon} {label}"]
    if content:
        for cl in content.split("\n")[:3]:
            out.append(f"  {cl}")
    return out


def _estimate_rendered_size(items: list[tuple[str, dict]]) -> int:
    """Estimate total character count for rendered timeline events."""
    total = 0
    for name, e in items:
        for line in _render_event_lines(name, e):
            total += len(line) + 1  # +1 for newline
        total += 1  # blank separator
    return total


def _thin_to_budget(
    protected: list[tuple[str, dict]],
    thinnable: list[tuple[str, dict]],
    remaining_budget: int,
) -> tuple[list[tuple[str, dict]], int]:
    """Select evenly-spaced thinnable events to fit within remaining_budget.

    Returns (kept_events, original_count).
    """
    original_count = len(thinnable)
    if not thinnable or remaining_budget <= 0:
        return [], original_count

    total_thin_size = _estimate_rendered_size(thinnable)
    if total_thin_size <= remaining_budget:
        return thinnable, original_count

    avg_size = total_thin_size / len(thinnable)
    keep_count = max(1, int(remaining_budget / avg_size))
    if keep_count >= len(thinnable):
        return thinnable, original_count

    if keep_count == 1:
        indices = [0]
    else:
        step = (len(thinnable) - 1) / (keep_count - 1)
        indices = [round(i * step) for i in range(keep_count)]
    kept = [thinnable[i] for i in indices]
    return kept, original_count


def generate_org_timeline(target_date: str, *, max_chars: int | None = None) -> str:
    """Generate a unified cross-anima timeline for a specific date.

    Args:
        target_date: Date string in YYYY-MM-DD format.
        max_chars: Character budget for LLM consumption.  ``None`` (default)
            returns the full timeline for CLI display.  When set, command-type
            cron events are removed and heartbeat/reflection events are
            evenly thinned to fit within the budget.

    Returns:
        Plain text suitable for direct LLM consumption or CLI display.
    """
    animas_dir = get_animas_dir()
    if not animas_dir.exists():
        return ""

    anima_dirs = sorted([d for d in animas_dir.iterdir() if d.is_dir() and (d / "status.json").exists()])

    protected: list[tuple[str, dict]] = []
    thinnable: list[tuple[str, dict]] = []
    per_anima_tool_counts: dict[str, Counter[str]] = {}
    per_anima_total_tools: Counter[str] = Counter()
    global_type_counts: Counter[str] = Counter()
    command_cron_count = 0

    for anima_dir in anima_dirs:
        name = anima_dir.name
        log_dir = anima_dir / "activity_log"
        raw_entries = _load_entries_for_date(log_dir, target_date)
        per_anima_tool_counts[name] = Counter()
        for e in raw_entries:
            etype = e.get("type", "unknown")
            global_type_counts[etype] += 1
            if etype == "tool_use":
                tool_name = e.get("tool") or (e.get("meta") or {}).get("tool") or "unknown"
                per_anima_tool_counts[name][tool_name] += 1
                per_anima_total_tools[name] += 1
            elif etype in _QUALITATIVE_EVENT_TYPES:
                if max_chars is not None and _is_command_cron(e):
                    command_cron_count += 1
                    continue
                if max_chars is not None and (etype in _THINNABLE_EVENT_TYPES or etype == "cron_executed"):
                    thinnable.append((name, e))
                else:
                    protected.append((name, e))

    all_tagged = protected + thinnable
    all_tagged.sort(key=lambda x: x[1].get("ts", ""))

    active_names = [d.name for d in anima_dirs]
    lines: list[str] = [
        t("audit.org_timeline_title", date=target_date, count=len(active_names)),
        "",
    ]

    if not all_tagged and not any(per_anima_total_tools.values()):
        lines.append(t("audit.org_timeline_no_activity"))
        return "\n".join(lines)

    # Budget-aware thinning when max_chars is set
    thinned_original = 0
    thinned_kept = 0
    if max_chars is not None and thinnable:
        overhead = 500  # header + footer + tool summary
        protected_size = _estimate_rendered_size(protected)
        remaining = max_chars - protected_size - overhead
        kept, thinned_original = _thin_to_budget(protected, thinnable, remaining)
        thinned_kept = len(kept)
        display_items = protected + kept
        display_items.sort(key=lambda x: x[1].get("ts", ""))
    else:
        display_items = all_tagged

    for name, e in display_items:
        for line in _render_event_lines(name, e):
            lines.append(line)
        lines.append("")

    has_tools = any(c for c in per_anima_tool_counts.values() if c)
    if has_tools:
        lines.append(t("audit.org_timeline_tool_header"))
        for name in sorted(per_anima_tool_counts):
            tc = per_anima_tool_counts[name]
            total = per_anima_total_tools[name]
            if total == 0:
                continue
            top = tc.most_common(10)
            parts = [f"{tn}: {cnt}" for tn, cnt in top]
            lines.append("  " + t("audit.org_timeline_tool_line", name=name, total=total) + " | ".join(parts))
        lines.append("")

    if max_chars is not None and (thinned_original > thinned_kept or command_cron_count > 0):
        lines.append(
            t(
                "audit.org_timeline_thinned_notice",
                hb_original=thinned_original,
                hb_kept=thinned_kept,
                cmd_cron=command_cron_count,
            )
        )

    total = sum(global_type_counts.values())
    tool_total = sum(per_anima_total_tools.values())
    hb = global_type_counts.get("heartbeat_end", 0)
    resp = global_type_counts.get("response_sent", 0) + global_type_counts.get("cron_executed", 0)
    dm = global_type_counts.get("message_sent", 0)
    err = global_type_counts.get("error", 0)
    lines.append(
        t(
            "audit.org_timeline_footer",
            count=len(active_names),
            total=total,
            tools=tool_total,
            hb=hb,
            resp=resp,
            dm=dm,
            err=err,
        )
    )

    return "\n".join(lines)
