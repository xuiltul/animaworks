from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Semantic replay projection helpers for activity timeline groups."""

from collections.abc import Iterable
from typing import Any

VISIBLE_TOOL_NAMES = frozenset(
    {
        "delegate_task",
        "update_task",
        "backlog_task",
        "submit_tasks",
        "call_human",
        "post_channel",
        "send_message",
    }
)
_COMPLETED_STATUSES = frozenset({"done", "completed", "closed", "resolved", "success", "succeeded", "finished"})


def build_semantic_replay_events(groups: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Project trigger groups into deterministic user-facing replay events."""
    projected: list[dict[str, Any]] = []
    for group_index, group in enumerate(groups):
        projected.extend(_project_group(group, group_index))
    projected.sort(key=lambda event: (event["ts"], event["id"]))
    return projected


def resolve_semantic_group_id(group: dict[str, Any]) -> str:
    """Resolve a stable semantic group id from the first matching source field."""
    events = _events(group)
    for key, prefix in (
        ("task_id", "task"),
        ("task_name", "task"),
        ("issue_id", "issue"),
        ("github_issue", "issue"),
    ):
        for event in events:
            value = _first(_meta(event).get(key))
            if value:
                return f"{prefix}:{value}"

    for event in events:
        meta = _meta(event)
        repo = _first(meta.get("repo"))
        number = _first(meta.get("number"))
        if repo and number:
            return f"issue:{repo}#{number}"

    for key in ("thread_id", "message_id", "conversation_id"):
        for event in events:
            value = _first(_meta(event).get(key))
            if value:
                return f"thread:{value}"

    for event in events:
        channel = _channel(event)
        if channel:
            return f"channel:{channel.lstrip('#')}"

    return f"group:{_first(group.get('id')) or 'unknown'}"


def _project_group(group: dict[str, Any], group_index: int) -> list[dict[str, Any]]:
    events = _events(group)
    if not events:
        return []

    base_ctx = {
        "group": group,
        "group_index": group_index,
        "group_id": resolve_semantic_group_id(group),
        "group_type": _first(group.get("type")) or "single",
        "raw_event_count": _raw_event_count(events),
        "suppressed_count": _suppressed_count(events),
        "source_types": _source_types(events),
    }
    semantic_events: list[dict[str, Any]] = []
    fallback_source: dict[str, Any] | None = None
    for event_index, event in enumerate(events):
        ctx = {**base_ctx, "event_index": event_index}
        item = _project_event(event, ctx)
        if item is None:
            fallback_source = fallback_source or event
        else:
            semantic_events.append(item)

    if semantic_events:
        return semantic_events
    return [_fallback_event(fallback_source or events[0], base_ctx)]


def _project_event(event: dict[str, Any], ctx: dict[str, Any]) -> dict[str, Any] | None:
    event_type = _event_type(event)
    actor = _actor(event, ctx["group"])
    target = _target(event)
    channel = _channel(event)
    tool = _tool(event)
    summary = _summary(event)

    if event_type == "tool_use":
        return _project_tool_use(event, ctx, actor, target, channel, tool, summary)
    if event_type == "tool_result" or (event_type.startswith("tool_") and event_type != "tool_use"):
        return None

    if event_type in ("message_received", "dm_received"):
        return _emit(
            event,
            ctx,
            actor=actor,
            target=target or _first(event.get("anima"), ctx["group"].get("anima")),
            kind="message",
            label=f"Message from {actor or 'unknown'}",
            summary=summary,
            importance=4,
            status="started",
            line_type=_line_type("message", event, actor, target),
        )

    if event_type in ("message_sent", "dm_sent"):
        is_delegation = _first(_meta(event).get("intent"), event.get("intent")) == "delegation"
        return _emit(
            event,
            ctx,
            actor=actor,
            target=target,
            kind="delegation" if is_delegation else "message",
            label=f"Delegated to {target}" if is_delegation and target else f"Message to {target or 'recipient'}",
            summary=summary,
            importance=4 if is_delegation else 3,
            status="progress",
            line_type="delegation" if is_delegation else _line_type("message", event, actor, target),
        )

    if event_type == "response_sent":
        return _emit(
            event,
            ctx,
            actor=actor,
            target=target,
            kind="response",
            label=f"Responded to {target}" if target else "Response sent",
            summary=summary,
            importance=3,
            status="completed",
            line_type=_line_type("response", event, actor, target),
        )

    if event_type == "channel_post":
        return _emit(
            event,
            ctx,
            actor=actor,
            target=f"#{channel}" if channel else target,
            kind="channel",
            label=f"Posted to #{channel}" if channel else "Channel post",
            summary=summary,
            importance=4,
            status="progress",
            line_type="channel",
            channel=channel,
            tool=tool,
        )

    if event_type in ("task_created", "task_updated"):
        completed = _is_completed(event)
        label = "Task created" if event_type == "task_created" else "Task completed" if completed else "Task updated"
        return _emit(
            event,
            ctx,
            actor=actor,
            target=target,
            kind="task",
            label=label,
            summary=summary or _first(_meta(event).get("task_name")),
            importance=5,
            status="started" if event_type == "task_created" else "completed" if completed else "progress",
            line_type=_line_type("task", event, actor, target),
        )

    if event_type in ("inbox_processing_start", "inbox_processing_end"):
        done = event_type.endswith("_end")
        return _emit(
            event,
            ctx,
            actor=actor,
            target=target,
            kind="inbox",
            label="Inbox processing completed" if done else "Inbox processing started",
            summary=summary,
            importance=4,
            status="completed" if done else "started",
            line_type=_line_type("inbox", event, actor, target),
            channel=channel,
        )

    if event_type in ("heartbeat_start", "heartbeat_end"):
        done = event_type == "heartbeat_end"
        return _emit(
            event,
            ctx,
            actor=actor,
            target="",
            kind="heartbeat",
            label="Heartbeat completed" if done else "Heartbeat started",
            summary=summary,
            importance=3 if done else 2,
            status="completed" if done else "started",
        )

    if event_type == "cron_executed":
        task_name = _first(_meta(event).get("task_name"))
        return _emit(
            event,
            ctx,
            actor=actor,
            target=target,
            kind="cron",
            label=f"Cron: {task_name}" if task_name else "Cron executed",
            summary=summary,
            importance=3,
            status="started",
        )

    if event_type == "human_notify":
        return _emit(
            event,
            ctx,
            actor=actor,
            target=target,
            kind="external",
            label="Human notification",
            summary=summary,
            importance=5,
            status="progress",
            line_type="external",
            channel=channel,
            tool=tool,
        )

    if event_type == "memory_write":
        return _emit(
            event,
            ctx,
            actor=actor,
            target=target,
            kind="memory",
            label="Memory updated",
            summary=summary,
            importance=2,
            status="progress",
        )

    if event_type == "error" or _first(event.get("level"), _meta(event).get("level")).lower() == "error":
        return _emit(
            event,
            ctx,
            actor=actor,
            target=target,
            kind="error",
            label="Error",
            summary=summary,
            importance=5,
            status="failed",
            line_type=_line_type("error", event, actor, target),
            tool=tool,
        )

    return None


def _project_tool_use(
    event: dict[str, Any],
    ctx: dict[str, Any],
    actor: str,
    target: str,
    channel: str,
    tool: str,
    summary: str,
) -> dict[str, Any] | None:
    if tool not in VISIBLE_TOOL_NAMES:
        return None

    kind = "tool"
    label = f"Tool: {tool}"
    importance = 2
    line_type = ""
    event_channel = ""
    if tool == "delegate_task":
        kind, label, importance, line_type = (
            "delegation",
            f"Delegated to {target}" if target else "Delegated task",
            4,
            "delegation",
        )
    elif tool in {"update_task", "backlog_task", "submit_tasks"}:
        kind, label, importance = "task", f"Task tool: {tool}", 4
    elif tool == "post_channel":
        kind, label, importance, line_type = (
            "channel",
            f"Posted to #{channel}" if channel else "Posted to channel",
            4,
            "channel",
        )
        target, event_channel = (f"#{channel}" if channel else target), channel
    elif tool in {"send_message", "call_human"}:
        kind, label, importance, line_type = (
            "external",
            "External message" if tool == "send_message" else "Human call",
            4,
            "external",
        )

    return _emit(
        event,
        ctx,
        actor=actor,
        target=target,
        kind=kind,
        label=label,
        summary=summary or _tool_result_summary(event),
        importance=importance,
        status="progress",
        line_type=line_type,
        channel=event_channel,
        tool=tool,
    )


def _emit(
    event: dict[str, Any],
    ctx: dict[str, Any],
    *,
    actor: str,
    target: str,
    kind: str,
    label: str,
    summary: str,
    importance: int,
    status: str,
    line_type: str = "",
    channel: str = "",
    tool: str = "",
) -> dict[str, Any]:
    group_index = ctx["group_index"]
    event_index = ctx["event_index"]
    source_id = _source_id(event, group_index, event_index)
    item = {
        "id": f"sem:{ctx['group_id']}:{source_id}",
        "ts": _first(event.get("ts"), event.get("timestamp")),
        "actor": _trim(actor, 80),
        "target": _trim(target, 80),
        "kind": kind,
        "label": _trim(label, 48),
        "summary": _trim(summary, 180),
        "importance": max(1, min(5, int(importance))),
        "group_id": ctx["group_id"],
        "group_type": ctx["group_type"],
        "status": status,
        "source_event_ids": _source_event_ids(event, group_index, event_index),
        "raw_event_count": ctx["raw_event_count"],
        "line_type": line_type,
        "channel": _trim(channel, 80),
        "tool": _trim(tool, 80),
        "debug": {
            "suppressed_count": ctx["suppressed_count"],
            "source_types": ctx["source_types"],
        },
    }
    activity_ctx = _first(event.get("ctx"))
    if activity_ctx:
        item["ctx"] = activity_ctx
    return item


def _fallback_event(event: dict[str, Any], ctx: dict[str, Any]) -> dict[str, Any]:
    source_ids: list[str] = []
    for index, source_event in enumerate(_events(ctx["group"])):
        source_ids.extend(_source_event_ids(source_event, ctx["group_index"], index))
    item = _emit(
        event,
        {**ctx, "event_index": 0},
        actor=_actor(event, ctx["group"]),
        target=_target(event),
        kind="other",
        label="Tool activity" if _event_type(event).startswith("tool") else "Activity",
        summary=_summary(event) or _first(ctx["group"].get("summary")),
        importance=1,
        status="progress",
        tool=_tool(event),
    )
    item["source_event_ids"] = [source_id for source_id in source_ids if source_id]
    return item


def _events(group: dict[str, Any]) -> list[dict[str, Any]]:
    return [event for event in group.get("events", []) if isinstance(event, dict)]


def _meta(event: dict[str, Any]) -> dict[str, Any]:
    meta = event.get("meta")
    return meta if isinstance(meta, dict) else {}


def _event_type(event: dict[str, Any]) -> str:
    return _first(event.get("type"), event.get("name"))


def _actor(event: dict[str, Any], group: dict[str, Any]) -> str:
    meta = _meta(event)
    return _first(
        meta.get("actor"),
        meta.get("from_person"),
        event.get("from_person"),
        event.get("from"),
        event.get("anima"),
        group.get("anima"),
        event.get("tool"),
    )


def _target(event: dict[str, Any]) -> str:
    meta = _meta(event)
    return _first(
        meta.get("target"),
        meta.get("to_person"),
        meta.get("assignee"),
        meta.get("assigned_to"),
        meta.get("delegatee"),
        event.get("to_person"),
        event.get("to"),
    )


def _channel(event: dict[str, Any]) -> str:
    return _first(_meta(event).get("channel"), event.get("channel")).lstrip("#")


def _tool(event: dict[str, Any]) -> str:
    return _first(_meta(event).get("tool"), event.get("tool"), event.get("tool_name"))


def _summary(event: dict[str, Any]) -> str:
    meta = _meta(event)
    return _first(
        event.get("summary"),
        meta.get("summary"),
        meta.get("text"),
        event.get("content"),
        meta.get("task_name"),
        meta.get("title"),
    )


def _tool_result_summary(event: dict[str, Any]) -> str:
    result = event.get("tool_result")
    return _first(result.get("summary"), result.get("content")) if isinstance(result, dict) else ""


def _line_type(kind: str, event: dict[str, Any], actor: str, target: str) -> str:
    if kind in {"delegation", "channel", "external"}:
        return kind
    if _first(_meta(event).get("from_type"), event.get("from_type")) == "external":
        return "external"
    return "internal" if actor and target else ""


def _is_completed(event: dict[str, Any]) -> bool:
    status = _first(_meta(event).get("status"), _meta(event).get("state"), event.get("status"))
    return status.lower() in _COMPLETED_STATUSES


def _source_id(event: dict[str, Any], group_index: int, event_index: int) -> str:
    return _first(event.get("id")) or f"{group_index}:{event_index}"


def _source_event_ids(event: dict[str, Any], group_index: int, event_index: int) -> list[str]:
    ids = [_source_id(event, group_index, event_index)]
    result = event.get("tool_result")
    if isinstance(result, dict):
        result_id = _first(result.get("id"))
        if result_id:
            ids.append(result_id)
    return ids


def _raw_event_count(events: list[dict[str, Any]]) -> int:
    return len(events) + sum(1 for event in events if isinstance(event.get("tool_result"), dict))


def _source_types(events: list[dict[str, Any]]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for event in events:
        for event_type in (_event_type(event), "tool_result" if isinstance(event.get("tool_result"), dict) else ""):
            if event_type and event_type not in seen:
                seen.add(event_type)
                out.append(event_type)
    return out


def _suppressed_count(events: list[dict[str, Any]]) -> int:
    count = 0
    for event in events:
        event_type = _event_type(event)
        if (
            (event_type == "tool_use" and _tool(event) not in VISIBLE_TOOL_NAMES)
            or event_type == "tool_result"
            or (event_type.startswith("tool_") and event_type != "tool_use")
        ):
            count += 1
        if isinstance(event.get("tool_result"), dict):
            count += 1
    return count


def _first(*values: Any) -> str:
    for value in values:
        if value is None:
            continue
        text = value.strip() if isinstance(value, str) else str(value).strip()
        if text:
            return text
    return ""


def _trim(value: str, limit: int) -> str:
    text = " ".join(_first(value).split())
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)].rstrip() + "..." if limit > 3 else text[:limit]
