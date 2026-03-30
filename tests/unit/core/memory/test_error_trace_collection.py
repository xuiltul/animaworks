from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.


"""Tests for error trace collection in consolidation engine."""

import json
from datetime import timedelta
from pathlib import Path

import pytest

from core.time_utils import now_iso, now_local


@pytest.fixture
def temp_anima_dir(tmp_path: Path) -> Path:
    anima_dir = tmp_path / "test_anima"
    for sub in ("episodes", "knowledge", "activity_log"):
        (anima_dir / sub).mkdir(parents=True)
    return anima_dir


@pytest.fixture
def consolidation_engine(temp_anima_dir: Path):
    from core.memory.consolidation import ConsolidationEngine

    return ConsolidationEngine(
        anima_dir=temp_anima_dir,
        anima_name="test_anima",
    )


def _write_activity(anima_dir: Path, entries: list[dict]) -> None:
    """Write activity entries to today's JSONL file."""
    log_dir = anima_dir / "activity_log"
    today = now_local().strftime("%Y-%m-%d")
    path = log_dir / f"{today}.jsonl"
    with path.open("a", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


class TestCollectErrorEntries:

    def test_no_activity_log(self, consolidation_engine):
        result = consolidation_engine._collect_error_entries(hours=24)
        assert "エラーなし" in result or "No errors" in result

    def test_no_errors(self, temp_anima_dir, consolidation_engine):
        _write_activity(
            temp_anima_dir,
            [
                {"ts": now_iso(), "type": "message_received", "summary": "hello"},
                {"ts": now_iso(), "type": "response_sent", "summary": "hi there"},
            ],
        )
        result = consolidation_engine._collect_error_entries(hours=24)
        assert "エラーなし" in result or "No errors" in result

    def test_collects_error_events(self, temp_anima_dir, consolidation_engine):
        _write_activity(
            temp_anima_dir,
            [
                {
                    "ts": now_iso(),
                    "type": "error",
                    "summary": "APIError — rate limit",
                    "meta": {"phase": "process_message", "error": "rate limit exceeded"},
                },
            ],
        )
        result = consolidation_engine._collect_error_entries(hours=24)
        assert "ERR" in result
        assert "process_message" in result
        assert "rate limit" in result

    def test_collects_tool_result_fail(self, temp_anima_dir, consolidation_engine):
        _write_activity(
            temp_anima_dir,
            [
                {
                    "ts": now_iso(),
                    "type": "tool_result",
                    "tool": "web_search",
                    "content": "ConnectionError: timeout after 30s",
                    "meta": {"result_status": "fail"},
                },
            ],
        )
        result = consolidation_engine._collect_error_entries(hours=24)
        assert "FAIL" in result
        assert "web_search" in result
        assert "ConnectionError" in result

    def test_skips_tool_result_ok(self, temp_anima_dir, consolidation_engine):
        _write_activity(
            temp_anima_dir,
            [
                {
                    "ts": now_iso(),
                    "type": "tool_result",
                    "tool": "web_search",
                    "content": "found 5 results",
                    "meta": {"result_status": "ok", "result_count": 5},
                },
            ],
        )
        result = consolidation_engine._collect_error_entries(hours=24)
        assert "エラーなし" in result or "No errors" in result

    def test_mixed_events(self, temp_anima_dir, consolidation_engine):
        _write_activity(
            temp_anima_dir,
            [
                {"ts": now_iso(), "type": "message_received", "summary": "hello"},
                {
                    "ts": now_iso(),
                    "type": "error",
                    "summary": "Timeout",
                    "meta": {"phase": "heartbeat", "error": "hard timeout"},
                },
                {
                    "ts": now_iso(),
                    "type": "tool_result",
                    "tool": "slack",
                    "content": "401 Unauthorized",
                    "meta": {"result_status": "fail"},
                },
                {
                    "ts": now_iso(),
                    "type": "tool_result",
                    "tool": "github",
                    "content": "ok",
                    "meta": {"result_status": "ok"},
                },
            ],
        )
        result = consolidation_engine._collect_error_entries(hours=24)
        lines = [l for l in result.split("\n") if l.strip()]
        assert len(lines) == 2
        assert "ERR" in lines[0] or "ERR" in lines[1]
        assert "FAIL" in lines[0] or "FAIL" in lines[1]

    def test_char_budget_limit(self, temp_anima_dir, consolidation_engine):
        entries = []
        for i in range(100):
            entries.append(
                {
                    "ts": now_iso(),
                    "type": "error",
                    "summary": f"Error {i}: " + "x" * 100,
                    "meta": {"phase": f"phase_{i}", "error": "x" * 100},
                }
            )
        _write_activity(temp_anima_dir, entries)
        result = consolidation_engine._collect_error_entries(hours=24)
        assert len(result) <= consolidation_engine._ERROR_CHAR_BUDGET + 200

    def test_entry_limit(self, temp_anima_dir, consolidation_engine):
        entries = []
        for i in range(80):
            entries.append(
                {
                    "ts": now_iso(),
                    "type": "error",
                    "summary": f"Error {i}",
                    "meta": {"phase": "test", "error": f"err{i}"},
                }
            )
        _write_activity(temp_anima_dir, entries)
        result = consolidation_engine._collect_error_entries(hours=24)
        lines = [l for l in result.split("\n") if l.strip()]
        assert len(lines) <= consolidation_engine._ERROR_ENTRY_LIMIT


class TestFormatErrorEntry:

    def test_format_error_type(self):
        from core.memory._activity_models import ActivityEntry
        from core.memory.consolidation import ConsolidationEngine

        entry = ActivityEntry(
            ts="2026-03-29T15:30:00+09:00",
            type="error",
            summary="APIError",
            meta={"phase": "process_message", "error": "rate limit exceeded"},
        )
        result = ConsolidationEngine._format_error_entry(entry)
        assert result == "[15:30] ERR phase=process_message: rate limit exceeded"

    def test_format_tool_fail(self):
        from core.memory._activity_models import ActivityEntry
        from core.memory.consolidation import ConsolidationEngine

        entry = ActivityEntry(
            ts="2026-03-29T16:00:00+09:00",
            type="tool_result",
            tool="slack",
            content="401 Unauthorized",
            meta={"result_status": "fail"},
        )
        result = ConsolidationEngine._format_error_entry(entry)
        assert result == "[16:00] FAIL tool=slack: 401 Unauthorized"

    def test_format_tool_ok_returns_none(self):
        from core.memory._activity_models import ActivityEntry
        from core.memory.consolidation import ConsolidationEngine

        entry = ActivityEntry(
            ts="2026-03-29T16:00:00+09:00",
            type="tool_result",
            tool="slack",
            meta={"result_status": "ok"},
        )
        assert ConsolidationEngine._format_error_entry(entry) is None

    def test_format_truncates_long_error(self):
        from core.memory._activity_models import ActivityEntry
        from core.memory.consolidation import ConsolidationEngine

        entry = ActivityEntry(
            ts="2026-03-29T15:30:00+09:00",
            type="error",
            meta={"phase": "test", "error": "x" * 200},
        )
        result = ConsolidationEngine._format_error_entry(entry)
        assert result is not None
        assert len(result) < 200
        assert result.endswith("...")
