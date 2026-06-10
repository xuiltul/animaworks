"""Unit tests for ConsolidationEngine multipass activity consolidation helpers."""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from unittest.mock import MagicMock, patch
from zoneinfo import ZoneInfo

from core.memory.consolidation import ConsolidationEngine


@dataclass
class FakeEntry:
    ts: str = "2026-03-29T09:30:00+09:00"
    type: str = "tool_result"
    tool: str | None = None
    content: str | None = None
    summary: str | None = None
    from_person: str | None = None
    to_person: str | None = None
    channel: str | None = None
    meta: dict | None = field(default_factory=dict)


class TestComputeActivityBudget:
    """Tests for ConsolidationEngine.compute_activity_budget."""

    @patch("core.prompt.context.resolve_context_window")
    def test_budget_calculation_200k(self, mock_resolve: MagicMock) -> None:
        mock_resolve.return_value = 200_000
        # int(200000 * 0.80) - 25000 = 160000 - 25000 = 135000; * 3 = 405000
        expected = (160_000 - ConsolidationEngine._OVERHEAD_TOKENS) * ConsolidationEngine._CHARS_PER_TOKEN
        assert ConsolidationEngine.compute_activity_budget("any-model") == expected
        mock_resolve.assert_called_once_with("any-model")

    @patch("core.prompt.context.resolve_context_window")
    def test_budget_calculation_small_model(self, mock_resolve: MagicMock) -> None:
        mock_resolve.return_value = 8192
        # int(8192 * 0.80) = 6553; max(6553 - 25000, 10000) = 10000; * 3 = 30000
        assert ConsolidationEngine.compute_activity_budget("ollama/gemma3") == 30_000

    @patch("core.prompt.context.resolve_context_window")
    def test_budget_scales_with_context(self, mock_resolve: MagicMock) -> None:
        mock_resolve.return_value = 100_000
        low = ConsolidationEngine.compute_activity_budget("m1")
        mock_resolve.return_value = 200_000
        high = ConsolidationEngine.compute_activity_budget("m2")
        assert high > low
        assert low == (80_000 - ConsolidationEngine._OVERHEAD_TOKENS) * ConsolidationEngine._CHARS_PER_TOKEN


class TestIsExcludedTool:
    """Tests for ConsolidationEngine._is_excluded_tool."""

    def test_excluded_read_memory_file(self) -> None:
        e = FakeEntry(tool="read_memory_file")
        assert ConsolidationEngine._is_excluded_tool(e) is True

    def test_excluded_search_memory(self) -> None:
        e = FakeEntry(tool="search_memory")
        assert ConsolidationEngine._is_excluded_tool(e) is True

    def test_excluded_tool_search(self) -> None:
        e = FakeEntry(tool="ToolSearch")
        assert ConsolidationEngine._is_excluded_tool(e) is True

    def test_excluded_mcp_prefix(self) -> None:
        e = FakeEntry(tool="mcp__aw__send_message")
        assert ConsolidationEngine._is_excluded_tool(e) is True

    def test_not_excluded_bash(self) -> None:
        e = FakeEntry(tool="Bash")
        assert ConsolidationEngine._is_excluded_tool(e) is False

    def test_not_excluded_gmail(self) -> None:
        e = FakeEntry(tool="gmail")
        assert ConsolidationEngine._is_excluded_tool(e) is False

    def test_not_excluded_none(self) -> None:
        e = FakeEntry(tool=None)
        assert ConsolidationEngine._is_excluded_tool(e) is False


class TestFormatEntryFull:
    """Tests for ConsolidationEngine._format_entry_full."""

    def test_short_message(self) -> None:
        e = FakeEntry(
            ts="2026-03-29T09:30:00+09:00",
            type="response_sent",
            content="Hello",
        )
        out = ConsolidationEngine._format_entry_full(e)
        assert out == "[09:30] RESPONSE: Hello"

    def test_long_content_indented(self) -> None:
        body = "x" * 201
        e = FakeEntry(
            ts="2026-03-29T10:00:00+09:00",
            type="response_sent",
            content=body,
        )
        out = ConsolidationEngine._format_entry_full(e)
        assert out.startswith("[10:00] RESPONSE:\n")
        assert "  " + body[:50] in out

    def test_tool_result_with_content(self) -> None:
        e = FakeEntry(
            ts="2026-03-29T11:15:00+09:00",
            type="tool_result",
            tool="github",
            content='{"ok": true}',
            meta={"result_status": "ok"},
        )
        out = ConsolidationEngine._format_entry_full(e)
        assert "TOOL_RESULT:github" in out
        assert '{"ok": true}' in out
        assert "[FAIL]" not in out

    def test_tool_result_fail(self) -> None:
        e = FakeEntry(
            ts="2026-03-29T12:00:00+09:00",
            type="tool_result",
            tool="slack",
            content="timeout",
            meta={"result_status": "fail"},
        )
        out = ConsolidationEngine._format_entry_full(e)
        assert "[FAIL]" in out
        assert "TOOL_RESULT:slack" in out

    def test_comm_entry_with_metadata(self) -> None:
        e = FakeEntry(
            ts="2026-03-29T13:05:00+09:00",
            type="message_received",
            content="Ping",
            from_person="alice",
            to_person="bob",
            channel="general",
        )
        out = ConsolidationEngine._format_entry_full(e)
        assert "from:alice" in out
        assert "to:bob" in out
        assert "#general" in out

    def test_multiline_content(self) -> None:
        e = FakeEntry(
            ts="2026-03-29T14:00:00+09:00",
            type="response_sent",
            content="line1\nline2",
        )
        out = ConsolidationEngine._format_entry_full(e)
        assert ":\n" in out
        assert "  line1" in out
        assert "  line2" in out

    def test_no_content(self) -> None:
        e = FakeEntry(
            ts="2026-03-29T15:00:00+09:00",
            type="heartbeat_end",
            content=None,
            summary=None,
        )
        out = ConsolidationEngine._format_entry_full(e)
        assert "(no content)" in out


class TestSplitIntoChunks:
    """Tests for ConsolidationEngine._split_into_chunks."""

    def test_single_chunk(self) -> None:
        entries = [("09", "[09:00] A: hello"), ("09", "[09:01] B: world")]
        chunks = ConsolidationEngine._split_into_chunks(entries, budget=10_000)
        assert len(chunks) == 1
        assert "hello" in chunks[0]
        assert "world" in chunks[0]

    def test_splits_at_hour_boundary(self) -> None:
        line_a = "a" * 30
        line_b = "b" * 30
        line_c = "c" * 30
        entries = [("09", line_a), ("09", line_b), ("10", line_c)]
        budget = 50
        chunks = ConsolidationEngine._split_into_chunks(entries, budget=budget)
        assert len(chunks) == 2
        assert line_a in chunks[0] and line_b in chunks[0]
        assert line_c in chunks[1]

    def test_large_entry_truncated(self) -> None:
        huge = "Z" * 500
        entries = [("09", huge)]
        budget = 200
        chunks = ConsolidationEngine._split_into_chunks(entries, budget=budget)
        assert len(chunks) == 1
        assert "... (truncated)" in chunks[0]
        assert len(chunks[0]) <= budget + 50

    def test_empty_entries(self) -> None:
        assert ConsolidationEngine._split_into_chunks([], budget=100) == []

    def test_multiple_hours(self) -> None:
        entries = [
            ("08", "h8"),
            ("09", "h9a"),
            ("09", "h9b"),
            ("10", "h10"),
        ]
        chunks = ConsolidationEngine._split_into_chunks(entries, budget=12)
        assert len(chunks) >= 2
        joined = "\n".join(chunks)
        assert "h8" in joined
        assert "h10" in joined


class TestMergeTimelineParts:
    """Tests for ConsolidationEngine.merge_timeline_parts."""

    def test_single_part(self) -> None:
        assert ConsolidationEngine.merge_timeline_parts(["only"]) == "only"

    def test_merge_two_parts(self) -> None:
        merged = ConsolidationEngine.merge_timeline_parts(["line1", "line2"])
        assert "line1" in merged
        assert "line2" in merged

    def test_dedup_headers(self) -> None:
        p1 = "## 09:00-10:00 Morning\nfoo"
        p2 = "## 09:00-10:00 Morning\nbar"
        merged = ConsolidationEngine.merge_timeline_parts([p1, p2])
        assert merged.count("## 09:00-10:00 Morning") == 1
        assert "foo" in merged
        assert "bar" in merged

    def test_empty_parts(self) -> None:
        assert ConsolidationEngine.merge_timeline_parts([]) == ""


class TestCommTypes:
    """Tests for ConsolidationEngine._COMM_TYPES."""

    def test_message_sent_included(self) -> None:
        assert "message_sent" in ConsolidationEngine._COMM_TYPES

    def test_human_notify_included(self) -> None:
        assert "human_notify" in ConsolidationEngine._COMM_TYPES


class TestCollectActivityChunks:
    """Tests for ConsolidationEngine.collect_activity_chunks."""

    @patch("core.memory.consolidation.ConsolidationEngine.compute_activity_budget")
    @patch("core.memory.activity.ActivityLogger")
    @patch("core.memory.consolidation.now_local")
    def test_collect_respects_exclusions_and_formats(
        self,
        mock_now_local: MagicMock,
        mock_logger_cls: MagicMock,
        mock_budget: MagicMock,
    ) -> None:
        fixed_now = datetime(2026, 3, 29, 15, 0, 0, tzinfo=UTC)
        mock_now_local.return_value = fixed_now
        mock_budget.return_value = 50_000

        included = FakeEntry(
            ts="2026-03-29T14:30:00+00:00",
            type="response_sent",
            content="keep me",
        )
        excluded_tool = FakeEntry(
            ts="2026-03-29T14:31:00+00:00",
            type="tool_result",
            tool="read_memory_file",
            content="skip",
        )
        mock_activity = MagicMock()
        mock_activity.recent.return_value = [included, excluded_tool]
        mock_logger_cls.return_value = mock_activity

        engine = ConsolidationEngine(Path("/tmp/x"), "t")
        chunks = engine.collect_activity_chunks(hours=24, model="claude-test")

        mock_activity.recent.assert_called_once()
        text = "\n".join(chunks)
        assert "keep me" in text
        assert "read_memory_file" not in text
        assert "skip" not in text

    @patch("core.memory.consolidation.ConsolidationEngine.compute_activity_budget")
    @patch("core.memory.activity.ActivityLogger")
    @patch("core.memory.consolidation.now_local")
    def test_collect_empty_returns_empty(
        self,
        mock_now_local: MagicMock,
        mock_logger_cls: MagicMock,
        mock_budget: MagicMock,
    ) -> None:
        mock_now_local.return_value = datetime(2026, 3, 29, 12, 0, 0, tzinfo=UTC)
        mock_budget.return_value = 10_000
        mock_activity = MagicMock()
        mock_activity.recent.return_value = []
        mock_logger_cls.return_value = mock_activity

        engine = ConsolidationEngine(Path("/tmp/y"), "t")
        assert engine.collect_activity_chunks(hours=24, model="m") == []

    @patch("core.memory.consolidation.ConsolidationEngine.compute_activity_budget")
    @patch("core.memory.activity.ActivityLogger")
    def test_collect_uses_fixed_date_window(
        self,
        mock_logger_cls: MagicMock,
        mock_budget: MagicMock,
    ) -> None:
        mock_budget.return_value = 50_000
        start = datetime(2026, 6, 9, 0, 0, 0, tzinfo=UTC)
        end = datetime(2026, 6, 10, 0, 0, 0, tzinfo=UTC)
        inside = FakeEntry(
            ts="2026-06-09T23:50:00+00:00",
            type="response_sent",
            content="previous day activity",
        )
        boundary = FakeEntry(
            ts="2026-06-10T00:00:00+00:00",
            type="response_sent",
            content="today boundary activity",
        )
        mock_activity = MagicMock()
        mock_activity._load_entries.return_value = [inside, boundary]
        mock_logger_cls.return_value = mock_activity

        engine = ConsolidationEngine(Path("/tmp/window"), "t")
        chunks = engine.collect_activity_chunks(model="m", since=start, until=end)

        mock_activity._load_entries.assert_called_once_with(since=start, until=end)
        text = "\n".join(chunks)
        assert "previous day activity" in text
        assert "today boundary activity" not in text


class TestDailyEpisodeWriteHelpers:
    """Tests for previous-day target and merge/archive episode helpers."""

    def test_previous_local_day_window_uses_configured_timezone(self) -> None:
        tokyo = ZoneInfo("Asia/Tokyo")
        target, start, end = ConsolidationEngine.previous_local_day_window(datetime(2026, 6, 10, 2, 0, tzinfo=tokyo))

        assert target == date(2026, 6, 9)
        assert start.isoformat() == "2026-06-09T00:00:00+09:00"
        assert end.isoformat() == "2026-06-10T00:00:00+09:00"

        new_york = ZoneInfo("America/New_York")
        target, start, end = ConsolidationEngine.previous_local_day_window(datetime(2026, 3, 9, 2, 0, tzinfo=new_york))

        assert target == date(2026, 3, 8)
        assert start.date() == date(2026, 3, 8)
        assert end.date() == date(2026, 3, 9)
        assert start.tzinfo is new_york

    def test_build_merged_episode_preserves_existing_raw_notes(self) -> None:
        existing = "# 2026-06-09\n\n## 14:00 — Raw note\n\nDaytime note."
        timeline = "## 14:00-15:00 Work\n\n- 14:10 Consolidated event"

        merged = ConsolidationEngine.build_merged_episode_content(existing, timeline)

        assert ConsolidationEngine.RAW_NOTES_HEADER in merged
        assert existing in merged
        assert ConsolidationEngine.CONSOLIDATED_TIMELINE_HEADER in merged
        assert timeline in merged
        assert merged.index(existing) < merged.index(timeline)

    def test_write_consolidated_episode_archives_before_replacing(self, tmp_path: Path) -> None:
        anima_dir = tmp_path / "animas" / "ritsu"
        engine = ConsolidationEngine(anima_dir, "ritsu")
        target = date(2026, 6, 9)
        episode_path = engine.episode_path_for_date(target)
        original = "# 2026-06-09\n\n## 12:00 — Raw\n\nOriginal raw note."
        episode_path.write_text(original, encoding="utf-8")

        with patch(
            "core.memory.consolidation.now_local",
            return_value=datetime(2026, 6, 10, 2, 0, tzinfo=ZoneInfo("Asia/Tokyo")),
        ):
            written = engine.write_consolidated_episode(
                target,
                "## 12:00-13:00 Consolidated\n\n- 12:05 Kept event",
            )

        assert written == episode_path
        archived = list((anima_dir / "archive" / "episodes").glob("2026-06-09_*.md"))
        assert len(archived) == 1
        assert archived[0].read_text(encoding="utf-8") == original
        updated = episode_path.read_text(encoding="utf-8")
        assert "## Raw notes (preserved)" in updated
        assert "Original raw note." in updated
        assert "## Consolidated timeline" in updated
        assert "Kept event" in updated

    def test_write_consolidated_episode_creates_new_file_without_archive(self, tmp_path: Path) -> None:
        anima_dir = tmp_path / "animas" / "ritsu"
        engine = ConsolidationEngine(anima_dir, "ritsu")
        target = date(2026, 6, 9)

        engine.write_consolidated_episode(target, "## 09:00-10:00 Work\n\n- event")

        assert (engine.episodes_dir / "2026-06-09.md").read_text(encoding="utf-8") == (
            "## 09:00-10:00 Work\n\n- event\n"
        )
        assert not (anima_dir / "archive" / "episodes").exists()


class TestPhaseBCarryover:
    """Tests for Phase B timeout carry-over state."""

    def test_record_phase_b_carryover_caps_to_three_days(self, tmp_path: Path) -> None:
        engine = ConsolidationEngine(tmp_path / "animas" / "ritsu", "ritsu")

        for day in range(1, 5):
            engine.record_phase_b_carryover(
                f"episode source {day}",
                target_date=date(2026, 6, day),
                reason="phase_b_pending",
            )

        items = engine.load_phase_b_carryover()

        assert [item["date"] for item in items] == ["2026-06-02", "2026-06-03", "2026-06-04"]
        assert all("episode source 1" not in item["episodes_summary"] for item in items)

    def test_format_phase_b_carryover(self, tmp_path: Path) -> None:
        engine = ConsolidationEngine(tmp_path / "animas" / "ritsu", "ritsu")
        items = engine.record_phase_b_carryover(
            "## 14:00\nImportant source",
            target_date=date(2026, 6, 9),
            reason="phase_b_pending",
        )

        formatted = engine.format_phase_b_carryover(items)

        assert "Carry-over from 2026-06-09" in formatted
        assert "Important source" in formatted

    def test_load_phase_b_carryover_accepts_legacy_list_state(self, tmp_path: Path) -> None:
        engine = ConsolidationEngine(tmp_path / "animas" / "ritsu", "ritsu")
        path = engine.phase_b_carryover_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            '[{"date":"2026-06-09","reason":"phase_b_pending","episodes_summary":"legacy source"}]\n',
            encoding="utf-8",
        )

        items = engine.load_phase_b_carryover()

        assert len(items) == 1
        assert items[0]["date"] == "2026-06-09"
        assert items[0]["episodes_summary"] == "legacy source"

    def test_clear_phase_b_carryover(self, tmp_path: Path) -> None:
        engine = ConsolidationEngine(tmp_path / "animas" / "ritsu", "ritsu")
        engine.record_phase_b_carryover(
            "episode source",
            target_date=date(2026, 6, 9),
            reason="phase_b_pending",
        )

        assert engine.phase_b_carryover_path().exists()

        engine.clear_phase_b_carryover()

        assert engine.load_phase_b_carryover() == []
        assert not engine.phase_b_carryover_path().exists()
