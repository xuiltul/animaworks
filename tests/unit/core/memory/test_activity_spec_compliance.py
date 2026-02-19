"""Unit tests for activity log spec compliance fixes (Fix 1, 2, 3)."""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

from core.memory.activity import ActivityEntry, ActivityLogger


# ── Fix 3: JSONL field name unification ────────────────────


class TestFieldNameUnification:
    """to_dict() outputs 'from'/'to', recent() reads them back correctly."""

    def test_to_dict_uses_from_and_to_keys(self):
        entry = ActivityEntry(
            ts="2026-02-17T10:00:00",
            type="dm_sent",
            from_person="alice",
            to_person="bob",
        )
        d = entry.to_dict()
        assert "from" in d
        assert "to" in d
        assert "from_person" not in d
        assert "to_person" not in d
        assert d["from"] == "alice"
        assert d["to"] == "bob"

    def test_to_dict_omits_empty_from_to(self):
        entry = ActivityEntry(ts="2026-02-17T10:00:00", type="heartbeat_start")
        d = entry.to_dict()
        assert "from" not in d
        assert "to" not in d
        assert "from_person" not in d
        assert "to_person" not in d

    def test_roundtrip_from_to_fields(self, tmp_path):
        """Log with from/to, read back, verify fields match."""
        anima_dir = tmp_path / "anima"
        anima_dir.mkdir()
        logger = ActivityLogger(anima_dir)
        logger.log("dm_sent", from_person="alice", to_person="bob", content="hello")

        entries = logger.recent(days=1)
        assert len(entries) == 1
        assert entries[0].from_person == "alice"
        assert entries[0].to_person == "bob"

    def test_jsonl_on_disk_uses_from_to(self, tmp_path):
        """Verify the actual JSONL file uses 'from'/'to' keys."""
        anima_dir = tmp_path / "anima"
        anima_dir.mkdir()
        logger = ActivityLogger(anima_dir)
        logger.log("dm_sent", from_person="alice", to_person="bob")

        today = date.today().isoformat()
        path = anima_dir / "activity_log" / f"{today}.jsonl"
        raw = json.loads(path.read_text(encoding="utf-8").strip())
        assert "from" in raw
        assert "to" in raw
        assert "from_person" not in raw
        assert "to_person" not in raw

    def test_involves_filter_with_new_keys(self, tmp_path):
        """_involves works with the new 'from'/'to' JSONL keys."""
        anima_dir = tmp_path / "anima"
        anima_dir.mkdir()
        logger = ActivityLogger(anima_dir)
        logger.log("dm_sent", from_person="alice", to_person="bob")
        logger.log("dm_sent", from_person="charlie", to_person="dave")

        entries = logger.recent(days=1, involving="alice")
        assert len(entries) == 1
        assert entries[0].from_person == "alice"


# ── Fix 1: run_cron_command activity logging ──────────────


class TestCronCommandActivityLog:
    """Verify run_cron_command records cron_executed events."""

    def test_cron_command_log_entry(self, tmp_path):
        """ActivityLogger.log("cron_executed") with command metadata."""
        anima_dir = tmp_path / "anima"
        anima_dir.mkdir()
        logger = ActivityLogger(anima_dir)
        logger.log(
            "cron_executed",
            summary="コマンド: daily-backup",
            meta={"task_name": "daily-backup", "exit_code": 0, "command": "tar czf backup.tgz .", "tool": ""},
        )

        entries = logger.recent(days=1, types=["cron_executed"])
        assert len(entries) == 1
        assert entries[0].type == "cron_executed"
        assert "daily-backup" in entries[0].summary
        assert entries[0].meta["exit_code"] == 0


# ── Fix 2: memory_write and error events ──────────────────


class TestMemoryWriteEvent:
    """Verify memory_write events are properly formatted."""

    def test_memory_write_log_entry(self, tmp_path):
        anima_dir = tmp_path / "anima"
        anima_dir.mkdir()
        logger = ActivityLogger(anima_dir)
        logger.log(
            "memory_write",
            summary="knowledge/tech.md (overwrite)",
            meta={"path": "knowledge/tech.md", "mode": "overwrite"},
        )

        entries = logger.recent(days=1, types=["memory_write"])
        assert len(entries) == 1
        assert entries[0].type == "memory_write"
        assert "knowledge/tech.md" in entries[0].summary

    def test_memory_write_format_has_label(self):
        entry = ActivityEntry(
            ts="2026-02-17T10:00:00",
            type="memory_write",
            summary="knowledge/tech.md (overwrite)",
        )
        logger = ActivityLogger.__new__(ActivityLogger)
        line = logger._format_entry(entry)
        assert "MEM" in line


class TestErrorEvent:
    """Verify error events are properly formatted and recorded."""

    def test_error_log_entry(self, tmp_path):
        anima_dir = tmp_path / "anima"
        anima_dir.mkdir()
        logger = ActivityLogger(anima_dir)
        logger.log(
            "error",
            summary="process_messageエラー: ValueError",
            meta={"phase": "process_message", "error": "invalid input"},
        )

        entries = logger.recent(days=1, types=["error"])
        assert len(entries) == 1
        assert entries[0].type == "error"
        assert "process_message" in entries[0].summary
        assert entries[0].meta["phase"] == "process_message"

    def test_error_format_has_label(self):
        entry = ActivityEntry(
            ts="2026-02-17T10:00:00",
            type="error",
            summary="run_heartbeatエラー: RuntimeError",
        )
        logger = ActivityLogger.__new__(ActivityLogger)
        line = logger._format_entry(entry)
        assert "ERR" in line

    def test_multiple_error_types_recorded(self, tmp_path):
        """Multiple error events from different phases can be recorded and filtered."""
        anima_dir = tmp_path / "anima"
        anima_dir.mkdir()
        logger = ActivityLogger(anima_dir)
        logger.log("error", summary="process_messageエラー", meta={"phase": "process_message"})
        logger.log("error", summary="run_heartbeatエラー", meta={"phase": "run_heartbeat"})
        logger.log("error", summary="run_cron_taskエラー", meta={"phase": "run_cron_task"})
        logger.log("message_received", content="normal event")

        errors = logger.recent(days=1, types=["error"])
        assert len(errors) == 3
        phases = {e.meta["phase"] for e in errors}
        assert phases == {"process_message", "run_heartbeat", "run_cron_task"}
