from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for core.audit qualitative fields (key_activities, top_tools)."""

import json
from pathlib import Path

import pytest

from core.audit import AnimaAuditEntry, _collect_single_anima

_DATE = "2026-03-07"


# ── Fixtures ──────────────────────────────────────────────────


def _write_activity_log(anima_dir: Path, date: str, entries: list[dict]) -> None:
    log_dir = anima_dir / "activity_log"
    log_dir.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(e, ensure_ascii=False) for e in entries]
    (log_dir / f"{date}.jsonl").write_text("\n".join(lines), encoding="utf-8")


@pytest.fixture()
def anima_dir(tmp_path: Path) -> Path:
    d = tmp_path / "test-anima"
    d.mkdir()
    (d / "status.json").write_text(
        json.dumps(
            {
                "enabled": True,
                "model": "claude-sonnet-4-6",
                "supervisor": "boss",
                "role": "engineer",
            }
        ),
        encoding="utf-8",
    )
    (d / "state").mkdir(parents=True)
    (d / "activity_log").mkdir()
    return d


# ── key_activities extraction ─────────────────────────────────


class TestKeyActivitiesExtraction:
    def test_extracts_key_activities_from_mock_data(self, anima_dir: Path):
        _write_activity_log(
            anima_dir,
            _DATE,
            [
                {"ts": "2026-03-07T09:00:00+09:00", "type": "heartbeat_end", "summary": "Observed 3 tasks"},
                {"ts": "2026-03-07T10:30:00+09:00", "type": "response_sent", "content": "Here is my reply"},
                {"ts": "2026-03-07T11:00:00+09:00", "type": "message_sent", "to_person": "bob", "content": "Hi bob"},
            ],
        )
        result = _collect_single_anima(anima_dir, _DATE)
        assert result is not None
        assert len(result.key_activities) == 3
        assert result.key_activities[0]["type"] == "heartbeat_end"
        assert result.key_activities[0]["summary"] == "Observed 3 tasks"
        assert result.key_activities[0]["ts"] == "09:00"
        assert result.key_activities[1]["type"] == "response_sent"
        assert result.key_activities[1]["summary"] == "Here is my reply"
        assert result.key_activities[2]["type"] == "message_sent"
        assert "→ bob:" in result.key_activities[2]["summary"]

    def test_excludes_tool_use_from_key_activities(self, anima_dir: Path):
        _write_activity_log(
            anima_dir,
            _DATE,
            [
                {"ts": "2026-03-07T09:00:00+09:00", "type": "heartbeat_end", "summary": "Done"},
                {"ts": "2026-03-07T09:01:00+09:00", "type": "tool_use", "tool": "Read", "summary": "read file"},
                {"ts": "2026-03-07T09:02:00+09:00", "type": "tool_use", "tool": "Write", "summary": "wrote"},
            ],
        )
        result = _collect_single_anima(anima_dir, _DATE)
        assert result is not None
        assert len(result.key_activities) == 1
        assert result.key_activities[0]["type"] == "heartbeat_end"
        assert all(a["type"] != "tool_use" for a in result.key_activities)

    def test_response_sent_excludes_thinking_text_from_summary(self, anima_dir: Path):
        _write_activity_log(
            anima_dir,
            _DATE,
            [
                {
                    "ts": "2026-03-07T10:00:00+09:00",
                    "type": "response_sent",
                    "content": "The answer is 42",
                    "meta": {"thinking_text": "Let me think... internal reasoning..."},
                },
            ],
        )
        result = _collect_single_anima(anima_dir, _DATE)
        assert result is not None
        assert len(result.key_activities) == 1
        assert result.key_activities[0]["summary"] == "The answer is 42"
        assert "thinking" not in result.key_activities[0]["summary"].lower()

    def test_respects_15_entry_limit(self, anima_dir: Path):
        entries = [
            {
                "ts": f"2026-03-07T{9 + i // 60:02d}:{i % 60:02d}:00+09:00",
                "type": "heartbeat_end",
                "summary": f"HB {i}",
            }
            for i in range(20)
        ]
        _write_activity_log(anima_dir, _DATE, entries)
        result = _collect_single_anima(anima_dir, _DATE)
        assert result is not None
        assert len(result.key_activities) == 15

    def test_truncates_summary_to_200_chars(self, anima_dir: Path):
        long_content = "x" * 300
        _write_activity_log(
            anima_dir,
            _DATE,
            [
                {"ts": "2026-03-07T10:00:00+09:00", "type": "heartbeat_end", "summary": long_content},
            ],
        )
        result = _collect_single_anima(anima_dir, _DATE)
        assert result is not None
        assert len(result.key_activities) == 1
        assert len(result.key_activities[0]["summary"]) == 200


# ── top_tools extraction ───────────────────────────────────────


class TestTopToolsExtraction:
    def test_extracts_top_tools_with_correct_counts(self, anima_dir: Path):
        _write_activity_log(
            anima_dir,
            _DATE,
            [
                {"ts": "2026-03-07T09:00:00+09:00", "type": "tool_use", "tool": "Read"},
                {"ts": "2026-03-07T09:01:00+09:00", "type": "tool_use", "tool": "Read"},
                {"ts": "2026-03-07T09:02:00+09:00", "type": "tool_use", "tool": "Read"},
                {"ts": "2026-03-07T09:03:00+09:00", "type": "tool_use", "tool": "Write"},
                {"ts": "2026-03-07T09:04:00+09:00", "type": "tool_use", "tool": "Write"},
                {"ts": "2026-03-07T09:05:00+09:00", "type": "tool_use", "tool": "Bash"},
            ],
        )
        result = _collect_single_anima(anima_dir, _DATE)
        assert result is not None
        assert result.top_tools == [
            {"name": "Read", "count": 3},
            {"name": "Write", "count": 2},
            {"name": "Bash", "count": 1},
        ]

    def test_tool_from_meta_when_not_at_top_level(self, anima_dir: Path):
        _write_activity_log(
            anima_dir,
            _DATE,
            [
                {"ts": "2026-03-07T09:00:00+09:00", "type": "tool_use", "meta": {"tool": "search_memory"}},
                {"ts": "2026-03-07T09:01:00+09:00", "type": "tool_use", "meta": {"tool": "search_memory"}},
            ],
        )
        result = _collect_single_anima(anima_dir, _DATE)
        assert result is not None
        assert result.top_tools == [{"name": "search_memory", "count": 2}]

    def test_top_tools_limited_to_10(self, anima_dir: Path):
        entries = [
            {"ts": f"2026-03-07T09:{i:02d}:00+09:00", "type": "tool_use", "tool": f"Tool{i}"}
            for i in range(15)
        ]
        _write_activity_log(anima_dir, _DATE, entries)
        result = _collect_single_anima(anima_dir, _DATE)
        assert result is not None
        assert len(result.top_tools) == 10


# ── Backward compatibility ───────────────────────────────────


class TestBackwardCompatibility:
    def test_anima_audit_entry_without_new_fields_works(self):
        """Creating AnimaAuditEntry without key_activities/top_tools uses defaults."""
        entry = AnimaAuditEntry(
            name="alice",
            enabled=True,
            model="test",
            supervisor=None,
            role=None,
            total_entries=0,
            type_counts={},
            messages_sent=0,
            messages_received=0,
            errors=0,
            tasks_total=0,
            tasks_pending=0,
            tasks_done=0,
            peers_sent={},
            peers_received={},
            first_activity=None,
            last_activity=None,
        )
        assert entry.key_activities == []
        assert entry.top_tools == []

    def test_to_dict_includes_new_fields(self, anima_dir: Path):
        _write_activity_log(
            anima_dir,
            _DATE,
            [
                {"ts": "2026-03-07T09:00:00+09:00", "type": "heartbeat_end", "summary": "Done"},
                {"ts": "2026-03-07T09:01:00+09:00", "type": "tool_use", "tool": "Read"},
            ],
        )
        result = _collect_single_anima(anima_dir, _DATE)
        assert result is not None
        d = result.to_dict()
        assert "key_activities" in d
        assert "top_tools" in d
        assert len(d["key_activities"]) == 1
        assert len(d["top_tools"]) == 1
        assert d["top_tools"][0]["name"] == "Read"
        assert d["top_tools"][0]["count"] == 1
