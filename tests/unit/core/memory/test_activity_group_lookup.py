"""Tests for direct activity-group lookup by stable group ID."""

from __future__ import annotations

from datetime import timedelta
from unittest.mock import patch

from core.memory.activity import ActivityEntry, ActivityLogger
from core.time_utils import now_local


def _entry(event_type: str, ts: str, **kwargs) -> ActivityEntry:
    return ActivityEntry(ts=ts, type=event_type, **kwargs)


def _create_target_log(logger: ActivityLogger, ts: str) -> None:
    logger._log_dir.mkdir(parents=True, exist_ok=True)
    (logger._log_dir / f"{ts[:10]}.jsonl").touch()


def test_find_group_parses_iso_timestamp_and_rebuilds_complete_group(tmp_path) -> None:
    logger = ActivityLogger(tmp_path / "alice")
    start = now_local().replace(microsecond=0)
    start_ts = start.isoformat()
    entries = [
        _entry(
            "message_received",
            start_ts,
            content="Please inspect this",
            meta={"from_type": "human"},
        ),
        _entry(
            "tool_use",
            (start + timedelta(minutes=1)).isoformat(),
            tool="read_file",
            meta={"tool_use_id": "tool-1"},
        ),
        _entry(
            "tool_result",
            (start + timedelta(minutes=2)).isoformat(),
            tool="read_file",
            content="file contents",
            meta={"tool_use_id": "tool-1"},
        ),
        _entry(
            "response_sent",
            (start + timedelta(minutes=3)).isoformat(),
            content="Done",
        ),
    ]
    group_id = f"grp-alice:{start_ts}:chat"
    _create_target_log(logger, start_ts)

    with patch.object(logger, "_load_entries", return_value=entries) as load_entries:
        group = logger.find_group_by_id(group_id)

    assert group is not None
    assert group["id"] == group_id
    assert group["anima"] == "alice"
    assert group["event_count"] == 3
    assert group["is_open"] is False
    assert group["events"][1]["tool_result"]["content"] == "file contents"
    assert group["events"][1]["tool_result"]["ts"] == (start + timedelta(minutes=2)).isoformat()
    assert load_entries.call_args.kwargs == {
        "since": start - timedelta(days=1),
        "until": start + timedelta(days=1),
    }


def test_find_group_returns_updated_open_group_on_each_lookup(tmp_path) -> None:
    logger = ActivityLogger(tmp_path / "alice")
    start = now_local().replace(microsecond=0)
    group_id = f"grp-alice:{start.isoformat()}:chat"
    first_entries = [
        _entry(
            "message_received",
            start.isoformat(),
            content="Still working?",
            meta={"from_type": "human"},
        )
    ]
    updated_entries = [
        *first_entries,
        _entry(
            "tool_use",
            (start + timedelta(seconds=1)).isoformat(),
            tool="status_check",
        ),
    ]
    _create_target_log(logger, start.isoformat())

    with patch.object(logger, "_load_entries", side_effect=[first_entries, updated_entries]):
        initial = logger.find_group_by_id(group_id)
        updated = logger.find_group_by_id(group_id)

    assert initial is not None and initial["is_open"] is True
    assert initial["event_count"] == 1
    assert updated is not None and updated["is_open"] is True
    assert updated["event_count"] == 2


def test_find_group_maps_handler_failure_status_to_tool_error(tmp_path) -> None:
    logger = ActivityLogger(tmp_path / "alice")
    start = now_local().replace(microsecond=0)
    group_id = f"grp-alice:{start.isoformat()}:chat"
    entries = [
        _entry(
            "message_received",
            start.isoformat(),
            meta={"from_type": "human"},
        ),
        _entry(
            "tool_use",
            (start + timedelta(seconds=1)).isoformat(),
            tool="read_file",
            meta={"tool_use_id": "tool-fail"},
        ),
        _entry(
            "tool_result",
            (start + timedelta(seconds=2)).isoformat(),
            tool="read_file",
            content="Error: unavailable",
            meta={"tool_use_id": "tool-fail", "result_status": "fail"},
        ),
    ]
    _create_target_log(logger, start.isoformat())

    with patch.object(logger, "_load_entries", return_value=entries):
        group = logger.find_group_by_id(group_id)

    assert group is not None
    assert group["events"][1]["tool_result"]["is_error"] is True


def test_find_group_rejects_malformed_or_wrong_anima_ids(tmp_path) -> None:
    logger = ActivityLogger(tmp_path / "alice")
    invalid_ids = [
        "not-a-group",
        "grp-alice:not-a-timestamp:chat",
        "grp-alice:2026-07-21T10:00:00+09:00",
        "grp-bob:2026-07-21T10:00:00+09:00:chat",
    ]

    with patch.object(logger, "_load_entries") as load_entries:
        assert all(logger.find_group_by_id(group_id) is None for group_id in invalid_ids)

    load_entries.assert_not_called()


def test_find_group_returns_none_when_rebuilt_groups_do_not_match(tmp_path) -> None:
    logger = ActivityLogger(tmp_path / "alice")
    start = now_local().replace(microsecond=0)
    requested_id = f"grp-alice:{start.isoformat()}:chat"
    _create_target_log(logger, start.isoformat())

    with patch.object(logger, "_load_entries", return_value=[]):
        assert logger.find_group_by_id(requested_id) is None


def test_find_group_rejects_missing_ancient_log_without_scanning(tmp_path) -> None:
    logger = ActivityLogger(tmp_path / "alice")

    with patch.object(logger, "_load_entries") as load_entries:
        group = logger.find_group_by_id("grp-alice:0001-01-01T00:00:00+00:00:chat")

    assert group is None
    load_entries.assert_not_called()
