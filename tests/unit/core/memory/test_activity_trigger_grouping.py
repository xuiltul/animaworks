"""Unit tests for trigger-based activity grouping (group_by_trigger)."""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import pytest

from core.memory.activity import ActivityEntry, ActivityLogger


# ── Helpers ───────────────────────────────────────────────


def _make(type: str, ts: str, **kwargs) -> ActivityEntry:
    anima_name = kwargs.pop("anima_name", "yuki")
    entry = ActivityEntry(ts=ts, type=type, **kwargs)
    entry._anima_name = anima_name
    return entry


def _ts(base: str, offset_minutes: int = 0) -> str:
    dt = datetime.fromisoformat(base) + timedelta(minutes=offset_minutes)
    return dt.isoformat()


BASE = "2026-02-25T14:00:00+09:00"


def _group(entries: list) -> list:
    return ActivityLogger.group_by_trigger(entries)


# ── Heartbeat grouping ──────────────────────────────────


class TestHeartbeatTriggerGrouping:
    def test_heartbeat_full_cycle(self) -> None:
        """heartbeat_start → tool_use → heartbeat_end → 1 heartbeat group."""
        entries = [
            _make("heartbeat_start", _ts(BASE, 0), summary="定期巡回開始"),
            _make("channel_read", _ts(BASE, 1), channel="general"),
            _make("tool_use", _ts(BASE, 2), tool="web_search"),
            _make("heartbeat_end", _ts(BASE, 3), summary="異常なし"),
        ]
        groups = _group(entries)
        assert len(groups) == 1
        g = groups[0]
        assert g["type"] == "heartbeat"
        assert g["event_count"] == 4
        assert g["is_open"] is False
        assert g["summary"] == "異常なし"

    def test_heartbeat_open(self) -> None:
        """heartbeat_start without end → is_open=True."""
        entries = [
            _make("heartbeat_start", _ts(BASE, 0)),
            _make("channel_read", _ts(BASE, 1)),
        ]
        groups = _group(entries)
        assert len(groups) == 1
        assert groups[0]["is_open"] is True

    def test_consecutive_heartbeats(self) -> None:
        """Two complete heartbeat cycles → 2 groups."""
        entries = [
            _make("heartbeat_start", _ts(BASE, 0)),
            _make("heartbeat_end", _ts(BASE, 2), summary="ok1"),
            _make("heartbeat_start", _ts(BASE, 30)),
            _make("heartbeat_end", _ts(BASE, 32), summary="ok2"),
        ]
        groups = _group(entries)
        hb = [g for g in groups if g["type"] == "heartbeat"]
        assert len(hb) == 2
        assert hb[0]["summary"] == "ok1"
        assert hb[1]["summary"] == "ok2"


# ── Chat grouping ────────────────────────────────────────


class TestChatTriggerGrouping:
    def test_user_chat_grouped(self) -> None:
        """message_received(human) → tool_use → response_sent → 1 chat group."""
        entries = [
            _make("message_received", _ts(BASE, 0), from_person="admin",
                  content="hello", meta={"from_type": "human"}),
            _make("tool_use", _ts(BASE, 1), tool="web_search"),
            _make("response_sent", _ts(BASE, 2), content="result"),
        ]
        groups = _group(entries)
        assert len(groups) == 1
        assert groups[0]["type"] == "chat"
        assert groups[0]["event_count"] == 3
        assert groups[0]["is_open"] is False

    def test_chat_open_without_response(self) -> None:
        """message_received without response_sent → is_open=True."""
        entries = [
            _make("message_received", _ts(BASE, 0), from_person="admin",
                  content="hello", meta={"from_type": "human"}),
            _make("tool_use", _ts(BASE, 1), tool="read_file"),
        ]
        groups = _group(entries)
        assert len(groups) == 1
        assert groups[0]["is_open"] is True


# ── DM grouping ──────────────────────────────────────────


class TestDmTriggerGrouping:
    def test_anima_dm_grouped(self) -> None:
        """message_received(anima) → response_sent → 1 dm group."""
        entries = [
            _make("message_received", _ts(BASE, 0), from_person="taro",
                  content="task done", meta={"from_type": "anima"}),
            _make("response_sent", _ts(BASE, 1), content="thanks"),
        ]
        groups = _group(entries)
        assert len(groups) == 1
        assert groups[0]["type"] == "dm"
        assert groups[0]["is_open"] is False


# ── Cron grouping ────────────────────────────────────────


class TestCronTriggerGrouping:
    def test_cron_with_subsequent_events(self) -> None:
        """cron_executed absorbs tool_use until next trigger."""
        entries = [
            _make("cron_executed", _ts(BASE, 0), meta={"task_name": "check_mail"}),
            _make("tool_use", _ts(BASE, 1), tool="gmail"),
            _make("memory_write", _ts(BASE, 2)),
        ]
        groups = _group(entries)
        assert len(groups) == 1
        assert groups[0]["type"] == "cron"
        assert groups[0]["event_count"] == 3

    def test_cron_closed_by_next_trigger(self) -> None:
        """cron_executed + tool_use | heartbeat_start → 2 groups."""
        entries = [
            _make("cron_executed", _ts(BASE, 0), meta={"task_name": "check_mail"}),
            _make("tool_use", _ts(BASE, 1), tool="gmail"),
            _make("heartbeat_start", _ts(BASE, 5)),
            _make("heartbeat_end", _ts(BASE, 7)),
        ]
        groups = _group(entries)
        assert len(groups) == 2
        assert groups[0]["type"] == "cron"
        assert groups[0]["event_count"] == 2
        assert groups[0]["is_open"] is False
        assert groups[1]["type"] == "heartbeat"


# ── Tool pairing ─────────────────────────────────────────


class TestToolPairing:
    def test_tool_use_result_paired(self) -> None:
        """tool_use + tool_result with same tool_use_id → merged."""
        entries = [
            _make("heartbeat_start", _ts(BASE, 0)),
            _make("tool_use", _ts(BASE, 1), tool="web_search",
                  meta={"tool_use_id": "tu_001"}),
            _make("tool_result", _ts(BASE, 1), tool="web_search",
                  content="3 results", meta={"tool_use_id": "tu_001"}),
            _make("heartbeat_end", _ts(BASE, 2)),
        ]
        groups = _group(entries)
        assert len(groups) == 1
        events = groups[0]["events"]
        tool_events = [e for e in events if e["type"] == "tool_use"]
        assert len(tool_events) == 1
        assert tool_events[0].get("tool_result") is not None
        assert tool_events[0]["tool_result"]["content"] == "3 results"
        # tool_result should not appear as standalone event
        result_events = [e for e in events if e["type"] == "tool_result"]
        assert len(result_events) == 0

    def test_tool_use_without_result(self) -> None:
        """tool_use without matching tool_result → tool_result=None in event dict."""
        entries = [
            _make("heartbeat_start", _ts(BASE, 0)),
            _make("tool_use", _ts(BASE, 1), tool="web_search",
                  meta={"tool_use_id": "tu_orphan"}),
            _make("heartbeat_end", _ts(BASE, 2)),
        ]
        groups = _group(entries)
        events = groups[0]["events"]
        tool_events = [e for e in events if e["type"] == "tool_use"]
        assert len(tool_events) == 1
        assert "tool_result" not in tool_events[0]

    def test_orphan_tool_result_becomes_single(self) -> None:
        """tool_result without matching tool_use → single group."""
        entries = [
            _make("tool_result", _ts(BASE, 0), tool="web_search",
                  content="orphan result", meta={"tool_use_id": "tu_missing"}),
        ]
        groups = _group(entries)
        assert len(groups) == 1
        assert groups[0]["type"] == "single"


# ── Orphan events ────────────────────────────────────────


class TestOrphanEvents:
    def test_orphan_becomes_single(self) -> None:
        """Events not belonging to any trigger → single group."""
        entries = [
            _make("channel_post", _ts(BASE, 0), channel="general"),
        ]
        groups = _group(entries)
        assert len(groups) == 1
        assert groups[0]["type"] == "single"
        assert groups[0]["event_count"] == 1
        assert groups[0]["is_open"] is False

    def test_empty_entries(self) -> None:
        """Empty list → empty groups."""
        groups = _group([])
        assert groups == []


# ── Mixed scenario ───────────────────────────────────────


class TestMixedTriggerGrouping:
    def test_full_scenario(self) -> None:
        """HB + Chat + DM + Cron → correct groups (channel_post absorbed into heartbeat)."""
        entries = [
            # Heartbeat (1 group)
            _make("heartbeat_start", _ts(BASE, 0)),
            _make("channel_read", _ts(BASE, 1)),
            _make("heartbeat_end", _ts(BASE, 2), summary="ok"),
            # channel_post at +5 is within time window of heartbeat → absorbed
            _make("channel_post", _ts(BASE, 5), channel="ops"),
            # Chat (1 group)
            _make("message_received", _ts(BASE, 10), from_person="admin",
                  meta={"from_type": "human"}, content="hi"),
            _make("response_sent", _ts(BASE, 11), content="hello"),
            # DM (1 group)
            _make("message_received", _ts(BASE, 20), from_person="taro",
                  meta={"from_type": "anima"}, content="report"),
            _make("response_sent", _ts(BASE, 21), content="acknowledged"),
            # Cron (1 group)
            _make("cron_executed", _ts(BASE, 30), meta={"task_name": "backup"}),
            _make("tool_use", _ts(BASE, 31), tool="aws"),
        ]
        groups = _group(entries)
        types = [g["type"] for g in groups]
        assert types == ["heartbeat", "chat", "dm", "cron"]
        assert groups[0]["event_count"] == 4  # start, channel_read, end, channel_post
        assert groups[0]["is_open"] is False
        assert groups[1]["event_count"] == 2  # message_received, response_sent
        assert groups[2]["event_count"] == 2
        assert groups[3]["event_count"] == 2
        assert groups[3]["is_open"] is True  # cron has no explicit close


# ── Group ID format ──────────────────────────────────────


class TestGroupIdFormat:
    def test_group_id_format(self) -> None:
        """Group IDs follow grp-{anima}:{ts}:{type} format."""
        entries = [
            _make("heartbeat_start", _ts(BASE, 0), anima_name="yuki"),
            _make("heartbeat_end", _ts(BASE, 1)),
        ]
        groups = _group(entries)
        gid = groups[0]["id"]
        assert gid.startswith("grp-yuki:")
        assert ":heartbeat" in gid


# ── Anima field propagation ──────────────────────────────


class TestAnimaFieldPropagation:
    def test_events_have_anima_field(self) -> None:
        """Events within groups should carry the anima name."""
        entries = [
            _make("heartbeat_start", _ts(BASE, 0), anima_name="yuki"),
            _make("heartbeat_end", _ts(BASE, 1), anima_name="yuki"),
        ]
        groups = _group(entries)
        assert groups[0]["anima"] == "yuki"
        for evt in groups[0]["events"]:
            assert evt["anima"] == "yuki"
