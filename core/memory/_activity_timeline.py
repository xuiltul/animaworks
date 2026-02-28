from __future__ import annotations
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.

"""Timeline trigger-based grouping mixin for ActivityLogger.

Internal module — import from :mod:`core.memory.activity` instead.
"""

from typing import Any

from core.memory._activity_models import ActivityEntry, find_tool_result_fallback


class TimelineMixin:
    """Mixin providing trigger-based grouping for timeline UI."""

    _TRIGGER_TYPES = frozenset({
        "heartbeat_start", "message_received", "cron_executed",
        "task_created", "task_updated",
        "inbox_processing_start", "task_exec_start",
    })

    _CLOSE_MAP: dict[str, frozenset[str]] = {
        "heartbeat": frozenset({"heartbeat_end"}),
        "chat": frozenset({"response_sent"}),
        "dm": frozenset({"response_sent", "message_sent", "dm_sent"}),
        "inbox": frozenset({"inbox_processing_end"}),
    }

    @staticmethod
    def group_by_trigger(
        entries: list[ActivityEntry],
    ) -> list[dict[str, Any]]:
        """Group entries by trigger events for timeline display.

        Trigger events (heartbeat_start, message_received, cron_executed,
        task_created, inbox_processing_start, task_exec_start) open a new
        group.  Subsequent events are absorbed into the open group until a
        closing event or the next trigger *for the same Anima*.

        Each Anima's open group is tracked independently so that
        cross-Anima interleaving does not break grouping.

        tool_use / tool_result pairs are merged into a single entry
        with ``tool_result`` field attached.

        This is a static method — no ``ActivityLogger`` instance is needed.
        """
        _TM = TimelineMixin
        paired = _TM._pair_tool_results(entries)
        groups: list[dict[str, Any]] = []
        current_by_anima: dict[str, dict[str, Any]] = {}

        for entry in paired:
            etype = entry.type
            anima = entry._anima_name
            evt_dict = _TM._entry_to_event_dict(entry)
            is_trigger = etype in _TM._TRIGGER_TYPES
            cur = current_by_anima.get(anima)

            if is_trigger:
                if cur is not None:
                    cur["is_open"] = False
                    _TM._finalize_group(cur)
                    groups.append(cur)
                current_by_anima[anima] = _TM._open_group(entry, evt_dict)
                continue

            if etype == "heartbeat_end":
                if cur and cur["type"] == "heartbeat":
                    cur["events"].append(evt_dict)
                    cur["end_ts"] = entry.ts
                    cur["is_open"] = False
                    if entry.summary:
                        cur["summary"] = entry.summary
                    _TM._finalize_group(cur)
                    groups.append(cur)
                    del current_by_anima[anima]
                    continue
                retrogrp = _TM._find_recent_group(
                    groups, anima, "heartbeat",
                )
                if retrogrp is not None:
                    retrogrp["events"].append(evt_dict)
                    retrogrp["end_ts"] = entry.ts
                    retrogrp["is_open"] = False
                    if entry.summary:
                        retrogrp["summary"] = entry.summary
                    retrogrp["event_count"] = len(retrogrp["events"])
                    continue
                if cur is not None:
                    cur["events"].append(evt_dict)
                    cur["end_ts"] = entry.ts
                    continue

            if cur is not None:
                close_types = _TM._CLOSE_MAP.get(cur["type"], frozenset())
                cur["events"].append(evt_dict)
                cur["end_ts"] = entry.ts
                if etype in close_types:
                    cur["is_open"] = False
                    _TM._finalize_group(cur)
                    groups.append(cur)
                    del current_by_anima[anima]
                continue

            retrogrp = _TM._find_recent_group(groups, anima)
            if retrogrp is not None:
                retrogrp["events"].append(evt_dict)
                retrogrp["end_ts"] = entry.ts
                retrogrp["event_count"] = len(retrogrp["events"])
                continue

            groups.append(_TM._make_single_group(entry, evt_dict))

        for cur in current_by_anima.values():
            _TM._finalize_group(cur)
            groups.append(cur)

        return groups

    @staticmethod
    def _pair_tool_results(
        entries: list[ActivityEntry],
    ) -> list[ActivityEntry]:
        """Attach tool_result data to tool_use entries, remove paired results.

        Sets ``_tool_result_data`` on tool_use entries and returns a
        filtered list excluding consumed tool_result entries.
        """
        result_by_id: dict[str, ActivityEntry] = {}
        for e in entries:
            if e.type == "tool_result":
                tid = e.meta.get("tool_use_id", "")
                if tid:
                    result_by_id[tid] = e

        paired_ids: set[int] = set()
        for e in entries:
            if e.type == "tool_use":
                tid = e.meta.get("tool_use_id", "")
                result_entry = result_by_id.get(tid) if tid else None
                if not result_entry:
                    result_entry = find_tool_result_fallback(entries, e)
                if result_entry:
                    e._tool_result_data = {
                        "content": result_entry.content or result_entry.summary,
                        "is_error": result_entry.meta.get("is_error", False),
                    }
                    paired_ids.add(id(result_entry))

        return [e for e in entries if id(e) not in paired_ids]

    @staticmethod
    def _entry_to_event_dict(entry: ActivityEntry) -> dict[str, Any]:
        """Convert an entry to API dict, attaching tool_result if paired."""
        d = entry.to_api_dict()
        if entry.type == "tool_use" and entry._tool_result_data is not None:
            d["tool_result"] = entry._tool_result_data
        return d

    @staticmethod
    def _open_group(entry: ActivityEntry, evt_dict: dict[str, Any]) -> dict[str, Any]:
        """Create a new group dict from a trigger entry."""
        etype = entry.type
        anima = entry._anima_name

        if etype == "heartbeat_start":
            gtype = "heartbeat"
        elif etype == "message_received":
            from_type = entry.meta.get("from_type", "human")
            gtype = "dm" if from_type == "anima" else "chat"
        elif etype == "cron_executed":
            gtype = "cron"
        elif etype in ("task_created", "task_updated"):
            gtype = "task"
        elif etype == "inbox_processing_start":
            gtype = "inbox"
        elif etype == "task_exec_start":
            gtype = "task_exec"
        else:
            gtype = "single"

        summary = entry.summary or entry.meta.get("task_name", "")
        if gtype == "dm":
            summary = entry.from_person or entry.to_person or summary

        return {
            "id": f"grp-{anima}:{entry.ts}:{gtype}",
            "type": gtype,
            "anima": anima,
            "start_ts": entry.ts,
            "end_ts": entry.ts,
            "summary": summary,
            "event_count": 1,
            "is_open": True,
            "events": [evt_dict],
        }

    @staticmethod
    def _make_single_group(entry: ActivityEntry, evt_dict: dict[str, Any]) -> dict[str, Any]:
        anima = entry._anima_name
        return {
            "id": f"grp-{anima}:{entry.ts}:single",
            "type": "single",
            "anima": anima,
            "start_ts": entry.ts,
            "end_ts": entry.ts,
            "summary": entry.summary,
            "event_count": 1,
            "is_open": False,
            "events": [evt_dict],
        }

    @staticmethod
    def _find_recent_group(
        groups: list[dict[str, Any]],
        anima: str,
        gtype: str | None = None,
    ) -> dict[str, Any] | None:
        """Find the most recently finalized group for *anima*.

        Searches *groups* in reverse.  If *gtype* is given, only groups of
        that type are considered.
        """
        for grp in reversed(groups):
            if grp["anima"] != anima:
                continue
            if gtype is not None and grp["type"] != gtype:
                continue
            return grp
        return None

    @staticmethod
    def _finalize_group(group: dict[str, Any]) -> None:
        """Update event_count before returning the group."""
        group["event_count"] = len(group["events"])
