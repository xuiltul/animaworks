# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
"""E2E tests: activity log spec compliance fixes.

Validates Fix 1 (cron_command logging), Fix 2 (memory_write/error events),
Fix 3 (JSONL from/to field names), Fix 5 (dead code removed),
and Fix 6 (heartbeat history from activity log).
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from unittest.mock import patch

import pytest

from core.memory.activity import ActivityEntry, ActivityLogger


# ── Fixtures ──────────────────────────────────────────────────


@pytest.fixture
def anima_dir(tmp_path: Path) -> Path:
    d = tmp_path / "animas" / "test-anima"
    for subdir in ("episodes", "knowledge", "skills", "activity_log"):
        (d / subdir).mkdir(parents=True)
    return d


# ── Fix 3: JSONL field names E2E ──────────────────────────────


@pytest.mark.asyncio
async def test_from_to_fields_in_priming_pipeline(anima_dir: Path, tmp_path: Path) -> None:
    """Field name 'from'/'to' flows correctly through the full priming pipeline.

    Records events with from_person/to_person, then verifies:
    1. JSONL on disk uses 'from'/'to' keys
    2. PrimingEngine._channel_b_recent_activity reads them correctly
    3. Formatted output contains the person names
    """
    from core.memory.priming import PrimingEngine

    shared_dir = tmp_path / "shared"
    (shared_dir / "channels").mkdir(parents=True)
    (shared_dir / "users").mkdir(parents=True)
    common_skills = tmp_path / "common_skills"
    common_skills.mkdir()

    # Record events
    al = ActivityLogger(anima_dir)
    al.log("dm_sent", from_person="sakura", to_person="kotoha", content="テスト送信")
    al.log("message_received", from_person="taro", content="テスト受信")

    # Verify JSONL on disk
    today = date.today().isoformat()
    path = anima_dir / "activity_log" / f"{today}.jsonl"
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    for line in lines:
        raw = json.loads(line)
        assert "from_person" not in raw, "JSONL should not contain 'from_person'"
        assert "to_person" not in raw, "JSONL should not contain 'to_person'"
        if raw.get("type") == "dm_sent":
            assert raw["from"] == "sakura"
            assert raw["to"] == "kotoha"

    # Verify priming pipeline reads correctly
    with (
        patch("core.paths.get_shared_dir", return_value=shared_dir),
        patch("core.paths.get_common_skills_dir", return_value=common_skills),
    ):
        engine = PrimingEngine(anima_dir, shared_dir)
        result = await engine._channel_b_recent_activity(
            sender_name="taro", keywords=[],
        )

    # DM group header shows the peer name (to_person), not the sender
    assert "kotoha" in result
    # message_received single entry shows from_person
    assert "taro" in result
    # "sakura" (dm_sent from_person) is not shown in the DM group header
    # because DM groups display the peer name; verify it was stored on disk instead
    assert "DM" in result  # DM group exists


# ── Fix 2: memory_write event E2E ─────────────────────────────


def test_memory_write_event_recorded(anima_dir: Path) -> None:
    """memory_write event is properly recorded and retrievable."""
    al = ActivityLogger(anima_dir)
    al.log(
        "memory_write",
        summary="knowledge/tech.md (overwrite)",
        meta={"path": "knowledge/tech.md", "mode": "overwrite"},
    )

    entries = al.recent(days=1, types=["memory_write"])
    assert len(entries) == 1
    assert entries[0].type == "memory_write"
    assert entries[0].meta["path"] == "knowledge/tech.md"

    # Verify it appears in formatted priming output
    formatted = al.format_for_priming(entries)
    assert "MEM" in formatted
    assert "memory_write" in formatted


# ── Fix 2: error event E2E ────────────────────────────────────


def test_error_events_across_phases(anima_dir: Path) -> None:
    """Error events from different phases are recorded and filterable."""
    al = ActivityLogger(anima_dir)

    phases = ["process_message", "process_message_stream", "run_heartbeat", "run_cron_task", "run_cron_command"]
    for phase in phases:
        al.log(
            "error",
            summary=f"{phase}エラー: TestException",
            meta={"phase": phase, "error": "test error message"},
        )

    # Also log some non-error events
    al.log("message_received", content="normal msg")
    al.log("heartbeat_start")

    # Filter only errors
    errors = al.recent(days=1, types=["error"])
    assert len(errors) == 5
    recorded_phases = {e.meta["phase"] for e in errors}
    assert recorded_phases == set(phases)

    # Verify formatted output
    formatted = al.format_for_priming(errors)
    assert "ERR" in formatted
    for phase in phases:
        assert phase in formatted


# ── Fix 6: heartbeat history from activity log E2E ────────────


def test_heartbeat_history_from_activity_log(anima_dir: Path) -> None:
    """_load_heartbeat_history reads from activity_log, not legacy files."""
    # Import Anima dependencies minimally
    from core.anima import DigitalAnima

    # Record heartbeat_end events to activity log
    al = ActivityLogger(anima_dir)
    al.log("heartbeat_end", summary="チェックリスト完了: inbox確認済み")
    al.log("heartbeat_end", summary="チェックリスト完了: タスク状況確認済み")
    al.log("heartbeat_end", summary="チェックリスト完了: 全員の状態確認済み")

    # Also create a legacy heartbeat_history file to verify it's NOT read
    legacy_dir = anima_dir / "shortterm" / "heartbeat_history"
    legacy_dir.mkdir(parents=True, exist_ok=True)
    legacy_entry = json.dumps({
        "timestamp": "2026-02-17T09:00:00",
        "trigger": "heartbeat",
        "action": "checked",
        "summary": "LEGACY_SHOULD_NOT_APPEAR",
    })
    (legacy_dir / f"{date.today().isoformat()}.jsonl").write_text(
        legacy_entry + "\n", encoding="utf-8",
    )

    # Create minimal Anima-like object to test _load_heartbeat_history
    # We'll create a simple mock that has the method
    class _MinimalAnima:
        def __init__(self, anima_dir):
            self.anima_dir = anima_dir
            self.name = "test-anima"
            self._HEARTBEAT_HISTORY_N = 3

    # Bind the method
    import types
    obj = _MinimalAnima(anima_dir)
    obj._load_heartbeat_history = types.MethodType(
        DigitalAnima._load_heartbeat_history, obj,
    )

    text = obj._load_heartbeat_history()
    assert text != ""
    assert "LEGACY_SHOULD_NOT_APPEAR" not in text
    assert "チェックリスト完了" in text
    lines = text.strip().splitlines()
    assert len(lines) == 3


# ── Fix 5: dead code removed E2E ──────────────────────────────


def test_append_dm_log_restored() -> None:
    """_append_dm_log method restored on Messenger for legacy fallback writes."""
    from core.messenger import Messenger
    assert hasattr(Messenger, "_append_dm_log"), \
        "_append_dm_log should exist for parallel dm_logs/ writes"


def test_append_transcript_removed() -> None:
    """_append_transcript method no longer exists on ConversationMemory."""
    from core.memory.conversation import ConversationMemory
    assert not hasattr(ConversationMemory, "_append_transcript"), \
        "_append_transcript should have been removed"


# ── Fix 1: cron_command logging E2E ───────────────────────────


def test_cron_command_activity_log_format(anima_dir: Path) -> None:
    """cron_executed event from command-type cron has expected format."""
    al = ActivityLogger(anima_dir)
    al.log(
        "cron_executed",
        summary="コマンド: daily-report",
        meta={
            "task_name": "daily-report",
            "exit_code": 0,
            "command": "python3 report.py",
            "tool": "",
        },
    )

    entries = al.recent(days=1, types=["cron_executed"])
    assert len(entries) == 1
    e = entries[0]
    assert e.meta["task_name"] == "daily-report"
    assert e.meta["exit_code"] == 0
    assert e.meta["command"] == "python3 report.py"

    # Verify priming format
    formatted = al.format_for_priming(entries)
    assert "CRON" in formatted
    assert "daily-report" in formatted
