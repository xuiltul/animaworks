"""E2E tests for DM flood prevention — Messenger → cascade_limiter → ActivityLogger.

Exercises the real flow without mocking core components.
"""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from core.cascade_limiter import ConversationDepthLimiter
from core.background import _rotate_dm_logs_sync
from core.messenger import Messenger
from core.time_utils import now_jst


@pytest.fixture
def workspace(tmp_path):
    """Create a complete workspace with shared and animas dirs."""
    shared = tmp_path / "shared"
    shared.mkdir()
    (shared / "inbox" / "alice").mkdir(parents=True)
    (shared / "inbox" / "bob").mkdir(parents=True)
    (shared / "dm_logs").mkdir(parents=True, exist_ok=True)

    animas = tmp_path / "animas"
    animas.mkdir()
    (animas / "alice").mkdir()
    (animas / "bob").mkdir()

    return tmp_path


def _write_dm_sent_entry(log_path: Path, ts: str, to: str = "bob") -> None:
    """Write a dm_sent activity log entry."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "ts": ts,
        "type": "dm_sent",
        "content": "test msg",
        "to": to,
    }
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _make_limiter(max_per_hour: int = 5, max_per_day: int = 10, max_depth: int = 6) -> ConversationDepthLimiter:
    """Create a limiter with explicit limits (bypasses load_config)."""
    with patch("core.cascade_limiter.load_config") as mock_cfg:
        mock_cfg.return_value = MagicMock()
        mock_cfg.return_value.heartbeat.depth_window_s = 600
        mock_cfg.return_value.heartbeat.max_depth = max_depth
        mock_cfg.return_value.heartbeat.max_messages_per_hour = max_per_hour
        mock_cfg.return_value.heartbeat.max_messages_per_day = max_per_day
        return ConversationDepthLimiter(
            max_per_hour=max_per_hour,
            max_per_day=max_per_day,
            max_depth=max_depth,
        )


# ── Test 1: Global hourly limit ─────────────────────────────────────────────


@pytest.mark.e2e
def test_global_hourly_limit_blocks_excess_messages(workspace: Path) -> None:
    """Create real activity_log with dm_sent entries exceeding hourly limit.
    messenger.send() returns error Message with GlobalOutboundLimitExceeded.
    """
    shared_dir = workspace / "shared"
    animas_dir = workspace / "animas"
    alice_dir = animas_dir / "alice"
    activity_log_dir = alice_dir / "activity_log"
    activity_log_dir.mkdir(parents=True, exist_ok=True)
    today = now_jst().strftime("%Y-%m-%d")
    log_file = activity_log_dir / f"{today}.jsonl"

    # Write 6 dm_sent entries in the last hour (exceeds limit of 5)
    base = now_jst()
    for i in range(6):
        ts = (base - timedelta(minutes=10 * i)).isoformat()
        _write_dm_sent_entry(log_file, ts)

    limiter = _make_limiter(max_per_hour=5, max_per_day=10)
    messenger = Messenger(shared_dir, "alice")

    with (
        patch("core.paths.get_animas_dir", return_value=animas_dir),
        patch("core.cascade_limiter.depth_limiter", limiter),
    ):
        result = messenger.send("bob", "should be blocked")

    assert result.type == "error"
    assert "GlobalOutboundLimitExceeded" in result.content


# ── Test 2: Global daily limit ───────────────────────────────────────────────


@pytest.mark.e2e
def test_global_daily_limit_blocks_excess_messages(workspace: Path) -> None:
    """Write entries spread across 24h exceeding daily limit.
    Verify blocking.
    """
    shared_dir = workspace / "shared"
    animas_dir = workspace / "animas"
    alice_dir = animas_dir / "alice"
    activity_log_dir = alice_dir / "activity_log"
    activity_log_dir.mkdir(parents=True, exist_ok=True)
    today = now_jst().strftime("%Y-%m-%d")
    yesterday = (now_jst() - timedelta(days=1)).strftime("%Y-%m-%d")
    log_today = activity_log_dir / f"{today}.jsonl"
    log_yesterday = activity_log_dir / f"{yesterday}.jsonl"

    # Write 6 entries in the last 24h (but spread so hourly stays under 5)
    base = now_jst()
    for i in range(6):
        ts = (base - timedelta(hours=2 * i)).isoformat()
        target = log_today if i < 4 else log_yesterday
        _write_dm_sent_entry(target, ts)

    limiter = _make_limiter(max_per_hour=10, max_per_day=5)
    messenger = Messenger(shared_dir, "alice")

    with (
        patch("core.paths.get_animas_dir", return_value=animas_dir),
        patch("core.cascade_limiter.depth_limiter", limiter),
    ):
        result = messenger.send("bob", "should be blocked by daily limit")

    assert result.type == "error"
    assert "GlobalOutboundLimitExceeded" in result.content


# ── Test 3: Depth check fail-closed ──────────────────────────────────────────


@pytest.mark.e2e
def test_depth_check_fail_closed(workspace: Path) -> None:
    """Activity log read error triggers fail-closed (returns False).

    When reading the activity log raises an exception, check_depth
    returns False (fail-closed) to block rather than allow on uncertainty.
    """
    animas_dir = workspace / "animas"
    alice_dir = animas_dir / "alice"

    limiter = _make_limiter(max_depth=6)

    with patch("core.memory.activity.ActivityLogger.recent", side_effect=OSError("disk error")):
        result = limiter.check_depth("alice", "bob", alice_dir)

    assert result is False, (
        "check_depth should be fail-closed when activity_log read fails"
    )


# ── Test 4: DM log rotation end-to-end ───────────────────────────────────────


@pytest.mark.e2e
def test_dm_log_rotation_end_to_end(workspace: Path) -> None:
    """Create real dm_log files with old and recent entries.
    Call _rotate_dm_logs_sync, verify archive and retained entries.
    """
    shared_dir = workspace / "shared"
    dm_logs_dir = shared_dir / "dm_logs"
    dm_logs_dir.mkdir(parents=True, exist_ok=True)

    base = now_jst()
    old_ts = (base - timedelta(days=10)).isoformat()
    recent_ts = (base - timedelta(days=2)).isoformat()

    filepath = dm_logs_dir / "alice-bob.jsonl"
    entries = [
        {"ts": old_ts, "from": "alice", "to": "bob", "text": "old message"},
        {"ts": recent_ts, "from": "bob", "to": "alice", "text": "recent message"},
    ]
    with filepath.open("w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    result = _rotate_dm_logs_sync(shared_dir, max_age_days=7)

    assert "alice-bob.jsonl" in result
    assert result["alice-bob.jsonl"]["archived"] == 1
    assert result["alice-bob.jsonl"]["kept"] == 1

    date_str = base.strftime("%Y%m%d")
    archive_path = dm_logs_dir / f"alice-bob.{date_str}.archive.jsonl"
    assert archive_path.exists()
    archive_lines = archive_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(archive_lines) == 1
    archived = json.loads(archive_lines[0])
    assert archived["text"] == "old message"

    main_content = filepath.read_text(encoding="utf-8").strip().splitlines()
    assert len(main_content) == 1
    kept = json.loads(main_content[0])
    assert kept["text"] == "recent message"


# ── Test 5: Ack messages bypass limits ──────────────────────────────────────


@pytest.mark.e2e
def test_ack_messages_bypass_all_limits(workspace: Path) -> None:
    """Set up exceeded global limit, send ack message, verify it succeeds."""
    shared_dir = workspace / "shared"
    animas_dir = workspace / "animas"
    alice_dir = animas_dir / "alice"
    activity_log_dir = alice_dir / "activity_log"
    activity_log_dir.mkdir(parents=True, exist_ok=True)
    today = now_jst().strftime("%Y-%m-%d")
    log_file = activity_log_dir / f"{today}.jsonl"

    # Exceed hourly limit
    base = now_jst()
    for i in range(6):
        ts = (base - timedelta(minutes=5 * i)).isoformat()
        _write_dm_sent_entry(log_file, ts)

    limiter = _make_limiter(max_per_hour=5, max_per_day=10)
    messenger = Messenger(shared_dir, "alice")

    with (
        patch("core.paths.get_animas_dir", return_value=animas_dir),
        patch("core.cascade_limiter.depth_limiter", limiter),
    ):
        result = messenger.send(
            "bob",
            "[既読通知] 受信しました",
            msg_type="ack",
            skip_logging=True,
        )

    assert result.type == "ack"
    assert result.from_person == "alice"
    assert result.to_person == "bob"
