"""Phase A noop-cron filter tests (consolidation input shrink)."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

from core.memory.maintenance.activity_compaction import ActivityCompactionSettings
from core.memory.maintenance.consolidation import (
    _INPUT_PROFILE_KEY,
    _NOOP_FILTER_KEY,
    ConsolidationEngine,
)
from core.memory.maintenance.cron_noop import (
    NoopStats,
    filter_noop_cron_entries,
    is_noop_cron_execution,
)
from core.time_utils import now_local


def _entry(
    ts: str,
    type_: str,
    *,
    content: str = "",
    tool: str = "",
    meta: dict | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        ts=ts,
        type=type_,
        content=content,
        summary="",
        from_person="",
        to_person="",
        channel="",
        tool=tool,
        via="",
        meta=meta or {},
        origin="",
        origin_chain=[],
        ctx="",
    )


def _iso(day: date, hour: int, minute: int = 0, second: int = 0, tz: str = "+00:00") -> str:
    return f"{day.isoformat()}T{hour:02d}:{minute:02d}:{second:02d}{tz}"


# ── Pure filter function ────────────────────────────────────


def test_command_cron_exit0_excluded_nonzero_kept() -> None:
    day = date(2026, 9, 30)
    ok = _entry(
        _iso(day, 10, 0),
        "cron_executed",
        meta={"task_name": "poll", "exit_code": 0, "command": "/x.sh", "tool": ""},
    )
    fail = _entry(
        _iso(day, 10, 5),
        "cron_executed",
        content="something went wrong",
        meta={"task_name": "poll", "exit_code": 1, "command": "/x.sh", "tool": ""},
    )

    kept, stats = filter_noop_cron_entries([ok, fail])

    assert ok not in kept
    assert fail in kept
    assert stats.command_excluded == 1
    assert stats.llm_excluded == 0


def test_llm_noop_excluded_together_with_in_window_tools() -> None:
    day = date(2026, 9, 30)
    cron = _entry(
        _iso(day, 10, 0),
        "cron_executed",
        content="新しい投稿はなかったよ",
        meta={"task_name": "watch", "duration_ms": 20000, "status": "completed"},
    )
    # ts is the *end* time; window = [09:59:40, 10:00:00]
    tool_use = _entry(_iso(day, 9, 59, 50), "tool_use", tool="read_channel")
    tool_result = _entry(_iso(day, 9, 59, 55), "tool_result", tool="read_channel")
    outside = _entry(_iso(day, 10, 1), "tool_use", tool="read_channel")

    kept, stats = filter_noop_cron_entries([cron, tool_use, tool_result, outside])

    assert cron not in kept
    assert tool_use not in kept
    assert tool_result not in kept
    assert outside in kept
    assert stats.llm_excluded == 1
    assert stats.tool_entries_excluded == 2


def test_llm_window_with_message_kept() -> None:
    day = date(2026, 9, 30)
    cron = _entry(
        _iso(day, 10, 0),
        "cron_executed",
        content="ok",
        meta={"task_name": "watch", "duration_ms": 20000, "status": "completed"},
    )
    msg = _entry(_iso(day, 9, 59, 50), "message_sent", content="hi")

    kept, _ = filter_noop_cron_entries([cron, msg])

    assert cron in kept


def test_llm_send_type_tool_kept() -> None:
    day = date(2026, 9, 30)
    cron = _entry(
        _iso(day, 10, 0),
        "cron_executed",
        meta={"task_name": "watch", "duration_ms": 20000, "status": "completed"},
    )
    tool_use = _entry(_iso(day, 9, 59, 50), "tool_use", tool="send_message")

    kept, _ = filter_noop_cron_entries([cron, tool_use])

    assert cron in kept


def test_llm_mcp_prefixed_send_tool_kept() -> None:
    day = date(2026, 9, 30)
    cron = _entry(
        _iso(day, 10, 0),
        "cron_executed",
        meta={"task_name": "watch", "duration_ms": 20000, "status": "completed"},
    )
    tool_use = _entry(_iso(day, 9, 59, 50), "tool_use", tool="mcp__aw__post_channel")

    kept, _ = filter_noop_cron_entries([cron, tool_use])

    assert cron in kept


def test_llm_bash_external_verb_kept() -> None:
    day = date(2026, 9, 30)
    cron = _entry(
        _iso(day, 10, 0),
        "cron_executed",
        meta={"task_name": "watch", "duration_ms": 20000, "status": "completed"},
    )
    tool_use = _entry(
        _iso(day, 9, 59, 50),
        "tool_use",
        tool="Bash",
        content="animaworks-tool chatwork send 'hi there'",
    )

    kept, _ = filter_noop_cron_entries([cron, tool_use])

    assert cron in kept


def test_memory_write_is_activity_for_episode_summary_by_default() -> None:
    day = date(2026, 9, 30)
    cron = _entry(
        _iso(day, 10, 0),
        "cron_executed",
        meta={"task_name": "watch", "duration_ms": 20000, "status": "completed"},
    )
    tool_use = _entry(_iso(day, 9, 59, 50), "tool_use", tool="write_memory_file")
    memory_event = _entry(_iso(day, 9, 59, 55), "memory_write")

    kept, _ = filter_noop_cron_entries([cron, tool_use, memory_event])

    # Unchanged historical behaviour: writing memory counts as activity.
    assert cron in kept


def test_memory_write_is_noop_when_ignoring_memory_writes() -> None:
    day = date(2026, 9, 30)
    cron = _entry(
        _iso(day, 10, 0),
        "cron_executed",
        meta={"task_name": "watch", "duration_ms": 20000, "status": "completed"},
    )
    tool_use = _entry(_iso(day, 9, 59, 50), "tool_use", tool="write_memory_file")
    knowledge = _entry(_iso(day, 9, 59, 52), "knowledge_outcome")
    memory_event = _entry(_iso(day, 9, 59, 55), "memory_write")

    kept, stats = filter_noop_cron_entries(
        [cron, tool_use, knowledge, memory_event],
        ignore_memory_writes=True,
    )

    assert cron not in kept
    assert stats.llm_excluded == 1


def test_is_noop_cron_execution_ignores_duplicate_target_entry() -> None:
    day = date(2026, 9, 30)
    cron = _entry(
        _iso(day, 10, 0),
        "cron_executed",
        meta={"task_name": "watch", "duration_ms": 20000, "status": "completed"},
    )
    duplicate = _entry(
        _iso(day, 10, 0),
        "cron_executed",
        meta={"task_name": "watch", "duration_ms": 20000, "status": "completed"},
    )

    assert is_noop_cron_execution(cron, [cron, duplicate]) is True


def test_llm_overlapping_windows_kept() -> None:
    day = date(2026, 9, 30)
    a = _entry(
        _iso(day, 10, 0),
        "cron_executed",
        content="a",
        meta={"task_name": "t1", "duration_ms": 60000, "status": "completed"},
    )
    # window [09:59:00, 10:00:00]
    b = _entry(
        _iso(day, 9, 59, 30),
        "cron_executed",
        content="b",
        meta={"task_name": "t2", "duration_ms": 60000, "status": "completed"},
    )
    # window [09:58:30, 09:59:30] — overlaps a's window

    kept, stats = filter_noop_cron_entries([a, b])

    assert a in kept
    assert b in kept
    assert stats.llm_excluded == 0


def test_llm_status_not_completed_kept() -> None:
    day = date(2026, 9, 30)
    cron = _entry(
        _iso(day, 10, 0),
        "cron_executed",
        meta={"task_name": "watch", "duration_ms": 20000, "status": "failed"},
    )

    kept, stats = filter_noop_cron_entries([cron])

    assert cron in kept
    assert stats.llm_excluded == 0


def test_llm_without_duration_kept() -> None:
    day = date(2026, 9, 30)
    cron = _entry(
        _iso(day, 10, 0),
        "cron_executed",
        content="ok",
        meta={"task_name": "watch", "status": "completed"},
    )

    kept, _ = filter_noop_cron_entries([cron])

    assert cron in kept


def test_out_of_window_tool_remains() -> None:
    day = date(2026, 9, 30)
    cron = _entry(
        _iso(day, 10, 0),
        "cron_executed",
        content="noop",
        meta={"task_name": "watch", "duration_ms": 20000, "status": "completed"},
    )
    tool_use = _entry(_iso(day, 9, 0), "tool_use", tool="read_channel")

    kept, stats = filter_noop_cron_entries([cron, tool_use])

    assert cron not in kept
    assert tool_use in kept
    assert stats.tool_entries_excluded == 0


def test_out_of_window_send_tool_does_not_keep_noop_cron() -> None:
    """A send elsewhere in the day must not mark every LLM cron as active."""
    day = date(2026, 9, 30)
    cron = _entry(
        _iso(day, 10, 0),
        "cron_executed",
        content="noop",
        meta={"task_name": "watch", "duration_ms": 20000, "status": "completed"},
    )
    send = _entry(_iso(day, 9, 0), "tool_use", tool="send_message")
    message = _entry(_iso(day, 9, 0, 5), "message_sent")

    kept, stats = filter_noop_cron_entries([send, message, cron])

    assert cron not in kept
    assert send in kept
    assert message in kept
    assert stats.llm_excluded == 1


def test_stats_report_totals() -> None:
    day = date(2026, 9, 30)
    ok = _entry(
        _iso(day, 10, 0),
        "cron_executed",
        meta={"task_name": "p", "exit_code": 0, "command": "/x.sh", "tool": ""},
    )
    normal = _entry(_iso(day, 10, 1), "message_sent", content="x")

    _, stats = filter_noop_cron_entries([ok, ok, normal])

    assert isinstance(stats, NoopStats)
    assert stats.entries_total == 3
    assert stats.command_excluded == 2


# ── Activity chunk wiring (exclude_noop_cron) ─────────────────


def _write_activity(anima_dir: Path, lines: list[dict]) -> None:
    log_dir = anima_dir / "activity_log"
    log_dir.mkdir(parents=True, exist_ok=True)
    day = now_local().date().isoformat()
    with (log_dir / f"{day}.jsonl").open("a", encoding="utf-8") as fh:
        for line in lines:
            fh.write(json.dumps(line, ensure_ascii=False) + "\n")


def test_collect_activity_chunks_filters_noop_cron_when_requested(tmp_path: Path) -> None:
    engine = ConsolidationEngine(tmp_path, "test-anima")
    day = now_local().date()
    start, end = engine.local_day_window(day)

    _write_activity(
        tmp_path,
        [
            {
                "ts": _iso(day, 10, 0),
                "type": "cron_executed",
                "meta": {"task_name": "poll", "exit_code": 0, "command": "/x.sh", "tool": ""},
            },
            {
                "ts": _iso(day, 11, 0),
                "type": "message_sent",
                "content": "some real event",
                "meta": {},
            },
        ],
    )

    unfiltered = engine.collect_activity_chunks(model="test-model", since=start, until=end, exclude_noop_cron=False)
    text_unfiltered = "\n".join(unfiltered)
    assert "CRON" in text_unfiltered

    filtered = engine.collect_activity_chunks(model="test-model", since=start, until=end, exclude_noop_cron=True)
    text_filtered = "\n".join(filtered)
    assert "CRON" not in text_filtered
    assert "MSG_OUT" in text_filtered


# ── Checkpoint compatibility ─────────────────────────────────


def _craft_noop_llm_lines(day: date) -> list[dict]:
    return [
        {
            "ts": _iso(day, 10, 0),
            "type": "cron_executed",
            "content": "nothing new",
            "meta": {"task_name": "watch", "duration_ms": 20000, "status": "completed"},
        },
        {
            "ts": _iso(day, 9, 59, 50),
            "type": "tool_use",
            "tool": "read_channel",
            "meta": {},
        },
        {
            "ts": _iso(day, 11, 0),
            "type": "message_sent",
            "content": "real event",
            "meta": {},
        },
    ]


def test_checkpoint_old_processed_date_uses_no_filter(tmp_path: Path) -> None:
    engine = ConsolidationEngine(tmp_path, "test-anima")
    day = now_local().date()
    _write_activity(tmp_path, _craft_noop_llm_lines(day))
    # Simulate the old format: the date already has recorded hashes but no
    # _noop_cron_filtered entry.
    engine.record_consolidated_chunks(day, ["legacy chunk"])

    pending, applied = engine.collect_pending_activity_chunks(day, model="test-model", exclude_noop_cron=True)
    assert applied is False
    # Legacy path: the recorded hash is still skipped, nothing new processed.
    assert all("legacy" not in chunk for chunk in pending)


def test_checkpoint_new_date_uses_filter_and_records_flag(tmp_path: Path) -> None:
    engine = ConsolidationEngine(tmp_path, "test-anima")
    day = now_local().date()
    _write_activity(tmp_path, _craft_noop_llm_lines(day))

    pending, applied = engine.collect_pending_activity_chunks(day, model="test-model", exclude_noop_cron=True)
    assert applied is True
    assert pending, "expected pending chunks after filtering"

    engine.record_consolidated_chunks(day, pending, noop_cron_filtered=applied)

    checkpoint = engine._load_episode_checkpoint()
    assert day.isoformat() in checkpoint[_NOOP_FILTER_KEY]
    # The date now has hashes, so it stays filter-applied going forward.
    pending2, applied2 = engine.collect_pending_activity_chunks(day, model="test-model", exclude_noop_cron=True)
    assert applied2 is True
    assert pending2 == []


def test_checkpoint_config_false_is_legacy(tmp_path: Path) -> None:
    engine = ConsolidationEngine(tmp_path, "test-anima")
    day = now_local().date()
    _write_activity(tmp_path, _craft_noop_llm_lines(day))

    pending, applied = engine.collect_pending_activity_chunks(day, model="test-model", exclude_noop_cron=False)
    assert applied is False

    engine.record_consolidated_chunks(day, pending, noop_cron_filtered=applied)
    checkpoint = engine._load_episode_checkpoint()
    assert _NOOP_FILTER_KEY not in checkpoint


def test_checkpoint_uses_compact_profile_for_unprocessed_date(tmp_path: Path) -> None:
    engine = ConsolidationEngine(tmp_path, "test-anima")
    day = now_local().date()
    engine.collect_activity_chunks = MagicMock(return_value=["new compact chunk"])

    pending, _applied = engine.collect_pending_activity_chunks(
        day,
        model="test-model",
        compaction_settings=ActivityCompactionSettings(profile="compact"),
    )

    assert pending == ["new compact chunk"]
    assert engine.collect_activity_chunks.call_args.kwargs["compaction_settings"].profile == "compact_v2"


def test_checkpoint_keeps_recorded_compact_profile(tmp_path: Path) -> None:
    engine = ConsolidationEngine(tmp_path, "test-anima")
    day = now_local().date()
    engine.record_consolidated_chunks(day, ["old compact chunk"], input_profile="compact")
    engine.collect_activity_chunks = MagicMock(return_value=["old compact chunk"])

    pending, _applied = engine.collect_pending_activity_chunks(
        day,
        model="test-model",
        compaction_settings=ActivityCompactionSettings(profile="compact"),
    )

    assert pending == []
    assert engine.collect_activity_chunks.call_args.kwargs["compaction_settings"].profile == "compact"
    engine.record_consolidated_chunks(day, ["v2 chunk"], input_profile="compact_v2")
    assert engine.resolve_input_profile_for_date(day, "compact") == "compact_v2"


def test_checkpoint_legacy_hash_uses_full_profile_and_matching_chunk_hash(tmp_path: Path) -> None:
    engine = ConsolidationEngine(tmp_path, "test-anima")
    day = now_local().date()
    engine.record_consolidated_chunks(day, ["legacy full chunk"])
    engine.collect_activity_chunks = MagicMock(return_value=["legacy full chunk"])

    pending, _applied = engine.collect_pending_activity_chunks(
        day,
        model="test-model",
        compaction_settings=ActivityCompactionSettings(profile="compact"),
    )

    assert pending == []
    assert engine.collect_activity_chunks.call_args.kwargs["compaction_settings"].profile == "full"
    assert _INPUT_PROFILE_KEY not in engine._load_episode_checkpoint()


def test_checkpoint_uses_recorded_profile_for_processed_date(tmp_path: Path) -> None:
    engine = ConsolidationEngine(tmp_path, "test-anima")
    day = now_local().date()
    engine.record_consolidated_chunks(day, ["compact chunk"], input_profile="compact")
    engine.collect_activity_chunks = MagicMock(return_value=["compact chunk"])

    pending, _applied = engine.collect_pending_activity_chunks(
        day,
        model="test-model",
        compaction_settings=ActivityCompactionSettings(profile="full"),
    )

    assert pending == []
    assert engine.collect_activity_chunks.call_args.kwargs["compaction_settings"].profile == "compact"
    assert engine._load_episode_checkpoint()[_INPUT_PROFILE_KEY] == {day.isoformat(): "compact"}


def test_load_episode_checkpoint_preserves_special_key(tmp_path: Path) -> None:
    engine = ConsolidationEngine(tmp_path, "test-anima")
    state_dir = tmp_path / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / "consolidation_episode_checkpoint.json").write_text(
        json.dumps({_NOOP_FILTER_KEY: ["2026-09-30"], "2026-09-30": ["abc", "def"]}),
        encoding="utf-8",
    )

    checkpoint = engine._load_episode_checkpoint()
    assert checkpoint[_NOOP_FILTER_KEY] == ["2026-09-30"]
    assert checkpoint["2026-09-30"] == ["abc", "def"]
