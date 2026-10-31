"""Unit tests for trigger-based activity grouping (group_by_trigger)."""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from datetime import datetime, timedelta

from core.activity.logger import ActivityEntry, ActivityLogger

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
            _make("message_received", _ts(BASE, 0), from_person="admin", content="hello", meta={"from_type": "human"}),
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
            _make("message_received", _ts(BASE, 0), from_person="admin", content="hello", meta={"from_type": "human"}),
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
            _make(
                "message_received", _ts(BASE, 0), from_person="taro", content="task done", meta={"from_type": "anima"}
            ),
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
            _make("tool_use", _ts(BASE, 1), tool="web_search", meta={"tool_use_id": "tu_001"}),
            _make("tool_result", _ts(BASE, 1), tool="web_search", content="3 results", meta={"tool_use_id": "tu_001"}),
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
            _make("tool_use", _ts(BASE, 1), tool="web_search", meta={"tool_use_id": "tu_orphan"}),
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
            _make(
                "tool_result",
                _ts(BASE, 0),
                tool="web_search",
                content="orphan result",
                meta={"tool_use_id": "tu_missing"},
            ),
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
            _make("message_received", _ts(BASE, 10), from_person="admin", meta={"from_type": "human"}, content="hi"),
            _make("response_sent", _ts(BASE, 11), content="hello"),
            # DM (1 group)
            _make("message_received", _ts(BASE, 20), from_person="taro", meta={"from_type": "anima"}, content="report"),
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


# ── Ctx-aware parallel grouping ───────────────────────────


def _ts_sec(base: str, offset_seconds: int = 0) -> str:
    dt = datetime.fromisoformat(base) + timedelta(seconds=offset_seconds)
    return dt.isoformat()


class TestCtxAwareParallelGrouping:
    def test_two_interleaved_tasks_overlap(self) -> None:
        """2 task_exec streams interleaved → 2 overlapping groups by ctx."""
        entries = [
            _make(
                "task_exec_start",
                _ts_sec(BASE, 0),
                summary="task A",
                ctx="task:aaa",
                meta={"task_id": "aaa", "title": "A"},
            ),
            _make(
                "task_exec_start",
                _ts_sec(BASE, 5),
                summary="task B",
                ctx="task:bbb",
                meta={"task_id": "bbb", "title": "B"},
            ),
            _make("tool_use", _ts_sec(BASE, 10), tool="search", ctx="task:aaa"),
            _make("tool_use", _ts_sec(BASE, 15), tool="read", ctx="task:bbb"),
            _make(
                "task_exec_end",
                _ts_sec(BASE, 40),
                summary="A done",
                ctx="task:aaa",
            ),
            _make(
                "task_exec_end",
                _ts_sec(BASE, 50),
                summary="B done",
                ctx="task:bbb",
            ),
        ]
        groups = _group(entries)
        task_groups = [g for g in groups if g["type"] == "task_exec"]
        assert len(task_groups) == 2

        by_ctx = {g["ctx"]: g for g in task_groups}
        assert set(by_ctx) == {"task:aaa", "task:bbb"}

        a, b = by_ctx["task:aaa"], by_ctx["task:bbb"]
        # Events stay on the correct task
        assert all(e.get("ctx") == "task:aaa" for e in a["events"])
        assert all(e.get("ctx") == "task:bbb" for e in b["events"])
        assert a["event_count"] == 3  # start, tool, end
        assert b["event_count"] == 3

        # Time ranges overlap (A: 0-40s, B: 5-50s)
        a_start, a_end = datetime.fromisoformat(a["start_ts"]), datetime.fromisoformat(a["end_ts"])
        b_start, b_end = datetime.fromisoformat(b["start_ts"]), datetime.fromisoformat(b["end_ts"])
        assert a_start < b_start < a_end < b_end

    def test_five_tasks_plus_heartbeat(self) -> None:
        """5 parallel task_exec + concurrent heartbeat → 6 groups, ctx-correct."""
        entries: list = []
        for i in range(5):
            tid = f"t{i}"
            entries.append(
                _make(
                    "task_exec_start",
                    _ts_sec(BASE, i),
                    summary=f"task {i}",
                    ctx=f"task:{tid}",
                    meta={"task_id": tid},
                )
            )
        # Heartbeat opens while tasks are running (empty ctx — real log pattern)
        entries.append(_make("heartbeat_start", _ts_sec(BASE, 10)))
        entries.append(_make("channel_read", _ts_sec(BASE, 11), channel="ops"))
        entries.append(_make("heartbeat_end", _ts_sec(BASE, 12), summary="ok"))

        for i in range(5):
            tid = f"t{i}"
            entries.append(
                _make(
                    "tool_use",
                    _ts_sec(BASE, 20 + i),
                    tool=f"tool_{i}",
                    ctx=f"task:{tid}",
                )
            )
            entries.append(
                _make(
                    "task_exec_end",
                    _ts_sec(BASE, 60 + i),
                    summary=f"done {i}",
                    ctx=f"task:{tid}",
                )
            )

        groups = _group(entries)
        task_groups = [g for g in groups if g["type"] == "task_exec"]
        hb_groups = [g for g in groups if g["type"] == "heartbeat"]
        assert len(task_groups) == 5
        assert len(hb_groups) == 1
        assert len(groups) == 6

        for g in sorted(task_groups, key=lambda x: x["ctx"]):
            tid = g["ctx"].removeprefix("task:")
            assert all(e.get("ctx") == g["ctx"] for e in g["events"])
            assert any(e["type"] == "task_exec_start" for e in g["events"])
            tools = [e for e in g["events"] if e["type"] == "tool_use"]
            assert len(tools) == 1
            assert tools[0]["tool"] == f"tool_{tid[1:]}"

        hb = hb_groups[0]
        assert hb["event_count"] == 3
        assert hb["summary"] == "ok"
        # Heartbeat must not swallow task events
        assert not any((e.get("ctx") or "").startswith("task:") for e in hb["events"])

    def test_empty_ctx_legacy_serial_matches_prior_behaviour(self) -> None:
        """Empty-ctx stream remains anima-serial (golden: one open slot)."""
        entries = [
            _make("task_exec_start", _ts_sec(BASE, 0), summary="A"),
            _make("tool_use", _ts_sec(BASE, 5), tool="a1"),
            # Second start forces-close the first (legacy serial)
            _make("task_exec_start", _ts_sec(BASE, 10), summary="B"),
            _make("tool_use", _ts_sec(BASE, 15), tool="b1"),
            _make("heartbeat_start", _ts_sec(BASE, 20)),
            _make("heartbeat_end", _ts_sec(BASE, 25), summary="hb"),
        ]
        groups = _group(entries)
        types = [g["type"] for g in groups]
        assert types == ["task_exec", "task_exec", "heartbeat"]
        assert groups[0]["is_open"] is False
        assert groups[0]["event_count"] == 2  # start A + tool a1
        assert groups[0]["ctx"] == ""
        assert groups[1]["event_count"] == 2  # start B + tool b1
        assert groups[1]["is_open"] is False  # closed by heartbeat trigger
        assert groups[2]["type"] == "heartbeat"
        assert groups[2]["summary"] == "hb"

    def test_mixed_empty_and_nonempty_ctx(self) -> None:
        """Non-empty parallel groups stay isolated; empty-ctx events serialise."""
        entries = [
            _make(
                "task_exec_start",
                _ts_sec(BASE, 0),
                summary="parallel",
                ctx="task:p1",
            ),
            _make("tool_use", _ts_sec(BASE, 5), tool="p_tool", ctx="task:p1"),
            # Empty-ctx heartbeat must not close the parallel task group
            _make("heartbeat_start", _ts_sec(BASE, 10)),
            _make("heartbeat_end", _ts_sec(BASE, 12), summary="ok"),
            _make(
                "task_exec_end",
                _ts_sec(BASE, 30),
                summary="p done",
                ctx="task:p1",
            ),
            # Orphan empty-ctx event after everything closed
            _make("channel_post", _ts_sec(BASE, 40), channel="ops"),
        ]
        groups = _group(entries)
        by_type = {}
        for g in groups:
            by_type.setdefault(g["type"], []).append(g)

        assert len(by_type["task_exec"]) == 1
        te = by_type["task_exec"][0]
        assert te["ctx"] == "task:p1"
        assert te["is_open"] is False or te["end_ts"] == _ts_sec(BASE, 30)
        assert all(e.get("ctx") == "task:p1" for e in te["events"])
        # start + tool + end
        assert te["event_count"] == 3

        assert len(by_type["heartbeat"]) == 1
        assert by_type["heartbeat"][0]["summary"] == "ok"

        # channel_post absorbed into most recent finalized (heartbeat) or single
        non_task = [g for g in groups if g["type"] != "task_exec"]
        assert any(any(e["type"] == "channel_post" for e in g["events"]) for g in non_task)

    def test_group_exposes_ctx_field(self) -> None:
        entries = [
            _make(
                "task_exec_start",
                _ts_sec(BASE, 0),
                ctx="task:xyz",
                summary="x",
            ),
        ]
        groups = _group(entries)
        assert groups[0]["ctx"] == "task:xyz"
        assert groups[0]["id"].endswith(":task_exec")
        # ID format stays grp-{anima}:{ts}:{type} (no ctx suffix — microsecond
        # timestamps keep IDs unique; find_group_by_id stays backward-compatible).
        assert groups[0]["id"].count(":") >= 2
