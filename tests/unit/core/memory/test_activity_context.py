from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from core.execution.session_context import RuntimeSessionContext, runtime_session_scope
from core.memory.activity import (
    ActivityEntry,
    ActivityLogger,
    activity_context_from_trigger,
    build_semantic_replay_events,
)
from core.time_utils import now_local
from core.tooling.handler import ToolHandler


@pytest.mark.parametrize(
    ("trigger", "session_type", "expected"),
    [
        ("task:abc123", "task", "task:abc123"),
        ("heartbeat", "heartbeat", "heartbeat"),
        ("cron:daily", "cron", "cron:daily"),
        ("inbox:alice", "inbox", "inbox"),
        ("message:taka", "chat", "chat"),
        ("manual", "chat", "chat"),
    ],
)
def test_activity_context_from_runtime_trigger(trigger: str, session_type: str, expected: str) -> None:
    assert activity_context_from_trigger(trigger, session_type) == expected


def test_bound_context_is_written_loaded_and_returned_by_api(tmp_path: Path) -> None:
    anima_dir = tmp_path / "animas" / "alice"
    activity = ActivityLogger(anima_dir)
    activity.bind_runtime_session(
        RuntimeSessionContext.create(
            session_type="task",
            thread_id="task-42",
            trigger="task:task-42",
        )
    )

    written = activity.log("tool_use", tool="search_memory")
    loaded = activity.recent(days=1)

    assert written.ctx == "task:task-42"
    assert loaded[0].ctx == "task:task-42"
    assert loaded[0].to_api_dict("alice")["ctx"] == "task:task-42"
    log_path = anima_dir / "activity_log" / f"{now_local().date().isoformat()}.jsonl"
    assert json.loads(log_path.read_text(encoding="utf-8"))["ctx"] == "task:task-42"


def test_legacy_entry_without_context_remains_readable(tmp_path: Path) -> None:
    anima_dir = tmp_path / "animas" / "alice"
    log_dir = anima_dir / "activity_log"
    log_dir.mkdir(parents=True)
    today = now_local().date().isoformat()
    (log_dir / f"{today}.jsonl").write_text(
        json.dumps({"ts": f"{today}T10:00:00+09:00", "type": "tool_use", "tool": "read_file"}) + "\n",
        encoding="utf-8",
    )

    entry = ActivityLogger(anima_dir).recent(days=1)[0]

    assert entry.ctx == ""
    assert entry.to_api_dict("alice")["ctx"] == ""


def test_tool_handler_session_bind_propagates_context_to_activity_logger(tmp_path: Path) -> None:
    handler = ToolHandler(anima_dir=tmp_path / "alice", memory=MagicMock())
    ctx = RuntimeSessionContext.create(
        session_type="cron",
        thread_id="daily",
        trigger="cron:daily",
    )

    handler.bind_runtime_session(ctx)
    handler._log_tool_activity("search_memory", {"query": "synthetic"})

    entry = ActivityLogger(tmp_path / "alice").recent(days=1)[0]
    assert entry.type == "tool_use"
    assert entry.ctx == "cron:daily"


def test_context_is_preserved_in_grouped_and_semantic_views() -> None:
    entry = ActivityEntry(
        ts="2026-07-13T10:00:00+09:00",
        type="tool_use",
        tool="search_memory",
        ctx="task:parallel-1",
    )
    entry._anima_name = "alice"

    groups = ActivityLogger.group_by_trigger([entry])
    semantic = build_semantic_replay_events(groups)

    assert groups[0]["events"][0]["ctx"] == "task:parallel-1"
    assert semantic[0]["ctx"] == "task:parallel-1"


def test_context_is_included_in_live_activity_event(tmp_path: Path) -> None:
    anima_dir = tmp_path / "animas" / "alice"
    activity = ActivityLogger(anima_dir)
    activity.bind_runtime_session(
        RuntimeSessionContext.create(
            session_type="task",
            thread_id="parallel-2",
            trigger="task:parallel-2",
        )
    )

    with patch("core.memory.activity.get_data_dir", return_value=tmp_path):
        activity.log("tool_use", tool="delegate_task")

    event_file = next((tmp_path / "run" / "events" / "alice").glob("ta_*.json"))
    payload = json.loads(event_file.read_text(encoding="utf-8"))
    assert payload["data"]["ctx"] == "task:parallel-2"


def test_unbound_logger_uses_active_runtime_context(tmp_path: Path) -> None:
    activity = ActivityLogger(tmp_path / "alice")
    ctx = RuntimeSessionContext.create(
        session_type="heartbeat",
        thread_id="default",
        trigger="heartbeat",
    )

    with runtime_session_scope(ctx):
        entry = activity.log("heartbeat_end")

    assert entry.ctx == "heartbeat"
