"""Unit tests for activity entry grouping and group formatting."""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import pytest

from core.memory.activity import ActivityEntry, ActivityLogger, EntryGroup


# ── Helpers ───────────────────────────────────────────────


def _make_entry(type: str, ts: str, **kwargs) -> ActivityEntry:
    return ActivityEntry(ts=ts, type=type, **kwargs)


def _ts(base: str, offset_minutes: int = 0) -> str:
    """Offset *base* ISO timestamp by *offset_minutes* and return ISO string."""
    dt = datetime.fromisoformat(base) + timedelta(minutes=offset_minutes)
    return dt.isoformat()


BASE_TS = "2026-02-18T10:00:00"


# ── DM grouping ──────────────────────────────────────────


class TestDmGrouping:
    def test_dm_entries_grouped_by_peer(self) -> None:
        """3 DM entries to/from same peer within 5 min -> 1 group."""
        entries = [
            _make_entry("dm_sent", _ts(BASE_TS, 0), to_person="yuki", content="hi"),
            _make_entry("dm_received", _ts(BASE_TS, 2), from_person="yuki", content="hey"),
            _make_entry("dm_sent", _ts(BASE_TS, 5), to_person="yuki", content="ok"),
        ]
        groups = ActivityLogger._group_entries(entries)
        assert len(groups) == 1
        assert groups[0].type == "dm"
        assert len(groups[0].entries) == 3

    def test_dm_entries_split_by_time_gap(self) -> None:
        """2 DM entries to same peer but 31+ min apart -> 2 groups."""
        entries = [
            _make_entry("dm_sent", _ts(BASE_TS, 0), to_person="yuki", content="first"),
            _make_entry("dm_sent", _ts(BASE_TS, 31), to_person="yuki", content="second"),
        ]
        groups = ActivityLogger._group_entries(entries)
        dm_groups = [g for g in groups if g.type == "dm"]
        assert len(dm_groups) == 2

    def test_dm_entries_split_by_different_peer(self) -> None:
        """DM to 'yuki' then DM to 'taro' -> 2 separate dm groups."""
        entries = [
            _make_entry("dm_sent", _ts(BASE_TS, 0), to_person="yuki", content="hi yuki"),
            _make_entry("dm_sent", _ts(BASE_TS, 1), to_person="taro", content="hi taro"),
        ]
        groups = ActivityLogger._group_entries(entries)
        dm_groups = [g for g in groups if g.type == "dm"]
        assert len(dm_groups) == 2


# ── HB grouping ──────────────────────────────────────────


class TestHeartbeatGrouping:
    def test_heartbeat_grouped(self) -> None:
        """heartbeat_start + heartbeat_end -> 1 hb group with 2 entries."""
        entries = [
            _make_entry("heartbeat_start", _ts(BASE_TS, 0)),
            _make_entry("heartbeat_end", _ts(BASE_TS, 5), summary="all clear"),
        ]
        groups = ActivityLogger._group_entries(entries)
        hb_groups = [g for g in groups if g.type == "hb"]
        assert len(hb_groups) == 1
        assert len(hb_groups[0].entries) == 2

    def test_heartbeat_excludes_interleaved_events(self) -> None:
        """HB start, dm_sent, HB end -> 3 groups (hb, single, hb).

        The HB group does NOT absorb non-HB events.
        """
        entries = [
            _make_entry("heartbeat_start", _ts(BASE_TS, 0)),
            _make_entry("dm_sent", _ts(BASE_TS, 1), to_person="yuki", content="mid-hb"),
            _make_entry("heartbeat_end", _ts(BASE_TS, 5), summary="done"),
        ]
        groups = ActivityLogger._group_entries(entries)
        assert len(groups) == 3
        assert groups[0].type == "hb"
        assert len(groups[0].entries) == 1  # only heartbeat_start
        assert groups[1].type in ("single", "dm")
        assert groups[2].type == "hb"
        assert len(groups[2].entries) == 1  # only heartbeat_end


# ── CRON grouping ─────────────────────────────────────────


class TestCronGrouping:
    def test_cron_entries_grouped_by_task_name(self) -> None:
        """2 cron_executed with same task_name -> 1 cron group."""
        entries = [
            _make_entry(
                "cron_executed", _ts(BASE_TS, 0),
                meta={"task_name": "daily-report"},
            ),
            _make_entry(
                "cron_executed", _ts(BASE_TS, 1),
                meta={"task_name": "daily-report"},
            ),
        ]
        groups = ActivityLogger._group_entries(entries)
        cron_groups = [g for g in groups if g.type == "cron"]
        assert len(cron_groups) == 1
        assert len(cron_groups[0].entries) == 2


# ── Mixed types ───────────────────────────────────────────


class TestMixedGrouping:
    def test_mixed_types_grouped(self) -> None:
        """DM + HB + CRON + single message -> correct number of groups."""
        entries = [
            # DM conversation (1 group)
            _make_entry("dm_sent", _ts(BASE_TS, 0), to_person="yuki", content="hi"),
            _make_entry("dm_received", _ts(BASE_TS, 1), from_person="yuki", content="hey"),
            # single message (1 group)
            _make_entry("message_received", _ts(BASE_TS, 5), content="user msg"),
            # HB (1 group)
            _make_entry("heartbeat_start", _ts(BASE_TS, 10)),
            _make_entry("heartbeat_end", _ts(BASE_TS, 12), summary="ok"),
            # CRON (1 group)
            _make_entry(
                "cron_executed", _ts(BASE_TS, 15),
                meta={"task_name": "backup"},
            ),
        ]
        groups = ActivityLogger._group_entries(entries)
        assert len(groups) == 4
        types = [g.type for g in groups]
        assert types == ["dm", "single", "hb", "cron"]


# ── Source lines ──────────────────────────────────────────


class TestSourceLines:
    def test_group_has_source_lines(self) -> None:
        """Entries with _line_number -> source_lines contain 'L' and '.jsonl'."""
        entries = [
            _make_entry("dm_sent", "2026-02-18T10:00:00", to_person="yuki", content="a"),
            _make_entry("dm_received", "2026-02-18T10:01:00", from_person="yuki", content="b"),
        ]
        entries[0]._line_number = 3
        entries[1]._line_number = 4
        groups = ActivityLogger._group_entries(entries)
        assert len(groups) == 1
        sl = groups[0].source_lines
        assert "L" in sl
        assert ".jsonl" in sl

    def test_group_cross_day_source_lines(self) -> None:
        """Entries with different dates -> source_lines contain ' + '."""
        entries = [
            _make_entry("dm_sent", "2026-02-17T23:59:00", to_person="yuki", content="late"),
            _make_entry("dm_received", "2026-02-18T00:01:00", from_person="yuki", content="early"),
        ]
        entries[0]._line_number = 10
        entries[1]._line_number = 1
        groups = ActivityLogger._group_entries(entries)
        assert len(groups) == 1
        sl = groups[0].source_lines
        assert " + " in sl
        assert "2026-02-17" in sl
        assert "2026-02-18" in sl


# ── Group formatting ──────────────────────────────────────


class TestFormatGroup:
    def test_format_group_dm(self) -> None:
        """DM group should have 'DM' header, indented child lines, and pointer."""
        entries = [
            _make_entry("dm_sent", "2026-02-18T10:00:00", to_person="yuki", content="hello"),
            _make_entry("dm_received", "2026-02-18T10:02:00", from_person="yuki", content="hi back"),
        ]
        entries[0]._line_number = 1
        entries[1]._line_number = 2
        groups = ActivityLogger._group_entries(entries)
        assert len(groups) == 1
        output = ActivityLogger._format_group(groups[0])
        lines = output.splitlines()
        # Header should contain "DM"
        assert "DM" in lines[0]
        # Child lines should be indented and contain DM direction labels
        child_lines = [ln for ln in lines[1:] if not ln.strip().startswith("->")]
        directions = {"DM<", "DM>"}
        for cl in child_lines:
            assert cl.startswith("  "), f"Child line not indented: {cl!r}"
            assert any(d in cl for d in directions), f"No DM direction in: {cl!r}"
        # Should have a pointer line
        pointer_lines = [ln for ln in lines if "->" in ln]
        assert len(pointer_lines) >= 1

    def test_format_group_hb(self) -> None:
        """HB group with summary should contain 'HB:' and the summary text."""
        entries = [
            _make_entry("heartbeat_start", "2026-02-18T10:00:00"),
            _make_entry(
                "heartbeat_end", "2026-02-18T10:05:00",
                summary="channels checked, no issues",
            ),
        ]
        groups = ActivityLogger._group_entries(entries)
        hb_groups = [g for g in groups if g.type == "hb"]
        assert len(hb_groups) == 1
        output = ActivityLogger._format_group(hb_groups[0])
        assert "HB:" in output
        assert "channels checked" in output

    def test_format_group_single(self) -> None:
        """Single-entry group formatting must match _format_entry output."""
        entry = _make_entry(
            "message_received", "2026-02-18T10:00:00",
            content="hello", from_person="user",
        )
        # Create a single-entry group via _group_entries
        groups = ActivityLogger._group_entries([entry])
        assert len(groups) == 1
        assert groups[0].type == "single"
        group_output = ActivityLogger._format_group(groups[0])
        entry_output = ActivityLogger._format_entry(entry)
        assert group_output == entry_output


# ── Budget-based group cutting ─────────────────────────────


class TestFormatForPrimingBudget:
    def test_format_for_priming_budget_cuts_old_groups(self) -> None:
        """With a small budget, newest groups appear and oldest are cut."""
        # Create enough entries to generate many groups (single-entry groups)
        entries = []
        for i in range(30):
            entries.append(
                _make_entry(
                    "message_received",
                    _ts(BASE_TS, i),
                    content=f"Message {i:03d} with some padding to use up budget space quickly",
                )
            )
        logger = ActivityLogger.__new__(ActivityLogger)
        # Very small budget: only a few groups should fit
        result = logger.format_for_priming(entries, budget_tokens=100)
        # The newest message should be present
        assert "Message 029" in result
        # The oldest message should be cut
        assert "Message 000" not in result
