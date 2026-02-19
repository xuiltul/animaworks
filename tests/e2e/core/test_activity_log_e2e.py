# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
"""E2E tests: unified activity log + priming integration.

Verifies that ActivityLogger recordings are correctly surfaced by
PrimingEngine's Channel B (recent activity) and that the old episodes/
fallback path works when no activity_log entries exist.
"""
from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from unittest.mock import patch

import pytest

from core.memory.activity import ActivityLogger
from core.memory.priming import PrimingEngine, PrimingResult, format_priming_section


# ── Fixtures ──────────────────────────────────────────────────


@pytest.fixture
def anima_dir(tmp_path: Path) -> Path:
    """Create a minimal anima directory with required subdirs."""
    d = tmp_path / "animas" / "test-anima"
    for subdir in ("episodes", "knowledge", "skills", "activity_log"):
        (d / subdir).mkdir(parents=True)
    return d


@pytest.fixture
def shared_dir(tmp_path: Path) -> Path:
    """Create a shared directory with channels and users."""
    d = tmp_path / "shared"
    (d / "channels").mkdir(parents=True)
    (d / "users").mkdir(parents=True)
    return d


# ── Helpers ───────────────────────────────────────────────────


def _record_events(anima_dir: Path) -> list[str]:
    """Record a variety of activity events and return their types."""
    logger = ActivityLogger(anima_dir)

    logger.log(
        "message_received",
        content="sakuraからのメッセージです。お疲れ様です。",
        from_person="sakura",
        channel="chat",
    )
    logger.log(
        "response_sent",
        content="sakuraへの返答です。了解しました。",
        to_person="sakura",
        channel="chat",
    )
    logger.log(
        "dm_sent",
        content="taroへのDMです。明日の会議について確認です。",
        to_person="taro",
    )
    logger.log(
        "channel_post",
        content="generalチャネルへの投稿です。全員に周知します。",
        channel="general",
    )
    logger.log(
        "tool_use",
        summary="web_searchでAnimaWorks関連の最新情報を検索",
        tool="web_search",
    )

    return [
        "message_received",
        "response_sent",
        "dm_sent",
        "channel_post",
        "tool_use",
    ]


def _patch_paths(shared_dir: Path, common_skills_dir: Path | None = None):
    """Return a context manager that patches get_shared_dir and get_common_skills_dir."""
    import contextlib

    skills_dir = common_skills_dir or (shared_dir.parent / "common_skills")
    skills_dir.mkdir(parents=True, exist_ok=True)

    return contextlib.ExitStack()


# ── Tests ─────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_activity_log_priming_integration(
    anima_dir: Path,
    shared_dir: Path,
    tmp_path: Path,
) -> None:
    """ActivityLogger events are surfaced by PrimingEngine Channel B.

    Steps:
        1. Record multiple event types via ActivityLogger
        2. Call PrimingEngine._channel_b_recent_activity() directly
        3. Verify all recorded events appear in the priming output
    """
    event_types = _record_events(anima_dir)

    common_skills = tmp_path / "common_skills"
    common_skills.mkdir(exist_ok=True)

    with (
        patch("core.paths.get_shared_dir", return_value=shared_dir),
        patch("core.paths.get_common_skills_dir", return_value=common_skills),
    ):
        engine = PrimingEngine(anima_dir, shared_dir)
        result = await engine._channel_b_recent_activity(
            sender_name="sakura",
            keywords=[],
        )

    # The result should be a non-empty string containing formatted entries
    assert result, "Channel B should return non-empty priming text"

    # Verify each event type appears in the formatted output.
    # DM entries are grouped under a "DM" header (not shown as "dm_sent").
    for etype in event_types:
        if etype == "dm_sent":
            assert "DM" in result, "DM group header not found in priming output"
        else:
            assert etype in result, f"Event type '{etype}' not found in priming output"

    # Verify person/channel context appears
    assert "sakura" in result
    assert "taro" in result
    assert "general" in result
    assert "web_search" in result


@pytest.mark.asyncio
async def test_activity_log_replaces_episodes(
    anima_dir: Path,
    shared_dir: Path,
    tmp_path: Path,
) -> None:
    """When activity_log has entries, old episodes/ are NOT read.

    When activity_log is empty, episodes/ fallback IS used.
    """
    common_skills = tmp_path / "common_skills"
    common_skills.mkdir(exist_ok=True)

    # --- Part 1: Write an old episode AND activity_log entries ---
    today = date.today()
    episode_file = anima_dir / "episodes" / f"{today.isoformat()}.md"
    episode_file.write_text(
        f"# {today} 行動ログ\n\n## 09:00 — OLD_EPISODE_MARKER\n\n",
        encoding="utf-8",
    )

    # Record an activity entry
    al = ActivityLogger(anima_dir)
    al.log("message_received", content="ACTIVITY_LOG_MARKER", from_person="alice")

    with (
        patch("core.paths.get_shared_dir", return_value=shared_dir),
        patch("core.paths.get_common_skills_dir", return_value=common_skills),
    ):
        engine = PrimingEngine(anima_dir, shared_dir)

        result_with_activity = await engine._channel_b_recent_activity(
            sender_name="alice",
            keywords=[],
        )

    # Activity log content should appear
    assert "ACTIVITY_LOG_MARKER" in result_with_activity
    # Old episode content should NOT appear (activity_log takes precedence)
    assert "OLD_EPISODE_MARKER" not in result_with_activity

    # --- Part 2: Remove activity_log, keep episode ---
    # Clear activity_log directory
    for f in (anima_dir / "activity_log").iterdir():
        f.unlink()

    with (
        patch("core.paths.get_shared_dir", return_value=shared_dir),
        patch("core.paths.get_common_skills_dir", return_value=common_skills),
    ):
        engine2 = PrimingEngine(anima_dir, shared_dir)

        result_fallback = await engine2._channel_b_recent_activity(
            sender_name="alice",
            keywords=[],
        )

    # Now the fallback should read old episodes
    assert "OLD_EPISODE_MARKER" in result_fallback
    # Activity log content should NOT appear (files deleted)
    assert "ACTIVITY_LOG_MARKER" not in result_fallback


@pytest.mark.asyncio
async def test_priming_result_format(
    anima_dir: Path,
    shared_dir: Path,
    tmp_path: Path,
) -> None:
    """prime_memories() produces a PrimingResult whose recent_activity is
    formatted into a '直近のアクティビティ' section by format_priming_section().
    """
    # Record some events
    _record_events(anima_dir)

    common_skills = tmp_path / "common_skills"
    common_skills.mkdir(exist_ok=True)

    with (
        patch("core.paths.get_shared_dir", return_value=shared_dir),
        patch("core.paths.get_common_skills_dir", return_value=common_skills),
    ):
        engine = PrimingEngine(anima_dir, shared_dir)

        priming_result = await engine.prime_memories(
            message="sakuraさんとの会話について確認したい",
            sender_name="human",
            channel="chat",
        )

    # PrimingResult should have recent_activity populated
    assert isinstance(priming_result, PrimingResult)
    assert priming_result.recent_activity, (
        "recent_activity should be non-empty when activity_log has entries"
    )

    # Format the priming section
    section = format_priming_section(priming_result, sender_name="human")

    # Top-level heading
    assert "あなたが思い出していること" in section

    # The activity channel heading must appear
    assert "直近のアクティビティ" in section

    # At least one event's content should appear in the section
    assert "sakura" in section or "message_received" in section


@pytest.mark.asyncio
async def test_activity_log_prioritises_sender(
    anima_dir: Path,
    shared_dir: Path,
    tmp_path: Path,
) -> None:
    """Entries involving the current sender are prioritised in priming output."""
    al = ActivityLogger(anima_dir)

    # Record many unrelated events first
    for i in range(30):
        al.log(
            "channel_post",
            content=f"Unrelated post #{i}",
            channel="random",
        )

    # Then record a sender-specific event
    al.log(
        "message_received",
        content="IMPORTANT_SENDER_MSG from target sender",
        from_person="target-sender",
    )

    common_skills = tmp_path / "common_skills"
    common_skills.mkdir(exist_ok=True)

    with (
        patch("core.paths.get_shared_dir", return_value=shared_dir),
        patch("core.paths.get_common_skills_dir", return_value=common_skills),
    ):
        engine = PrimingEngine(anima_dir, shared_dir)
        result = await engine._channel_b_recent_activity(
            sender_name="target-sender",
            keywords=[],
        )

    # The sender-specific event should appear
    assert "IMPORTANT_SENDER_MSG" in result
    assert "target-sender" in result


@pytest.mark.asyncio
async def test_activity_log_empty_produces_empty_priming(
    anima_dir: Path,
    shared_dir: Path,
    tmp_path: Path,
) -> None:
    """When both activity_log and episodes are empty, priming returns empty."""
    common_skills = tmp_path / "common_skills"
    common_skills.mkdir(exist_ok=True)

    with (
        patch("core.paths.get_shared_dir", return_value=shared_dir),
        patch("core.paths.get_common_skills_dir", return_value=common_skills),
    ):
        engine = PrimingEngine(anima_dir, shared_dir)

        priming_result = await engine.prime_memories(
            message="hello",
            sender_name="nobody",
            channel="chat",
        )

    # No activity and no episodes => recent_activity empty
    assert priming_result.recent_activity == ""

    # format_priming_section on a fully empty result returns ""
    section = format_priming_section(priming_result, sender_name="nobody")
    if priming_result.is_empty():
        assert section == ""
    else:
        # May have other channels (skills, etc.) but no activity section
        assert "直近のアクティビティ" not in section
