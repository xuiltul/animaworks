# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Company timeline activity recording (heartbeat / cron / task shared writer)."""

from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

from core.hub.timeline import list_items
from core.hub.timeline_activity import record_activity, record_cron_activity
from core.i18n import t
from core.time_utils import now_local


def _set_hub(monkeypatch, *, timeline_enabled: bool = True, company: str = "acme") -> None:
    hub = SimpleNamespace(timeline_enabled=timeline_enabled, company=company)
    monkeypatch.setattr("core.config.models.load_config", lambda: SimpleNamespace(hub=hub))


def _set_company(anima_dir: Path, company: str) -> None:
    (anima_dir / "status.json").write_text(json.dumps({"company": company}), encoding="utf-8")


class _WindowActivity:
    """ActivityLogger stub exposing the window read used by the timeline."""

    def __init__(self, entries: list[Any]) -> None:
        self.entries = entries

    def _load_entries(self, *, since: Any, until: Any) -> list[Any]:
        return list(self.entries)


def _entry(
    ts: str,
    entry_type: str,
    *,
    tool: str = "",
    content: str = "",
    meta: dict[str, Any] | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        ts=ts,
        type=entry_type,
        tool=tool,
        content=content,
        summary="",
        meta=meta or {},
    )


def _cron_entry(
    ts: str,
    *,
    task_name: str = "daily",
    duration_ms: int = 20_000,
    status: str = "completed",
) -> SimpleNamespace:
    return _entry(
        ts,
        "cron_executed",
        meta={"task_name": task_name, "duration_ms": duration_ms, "status": status},
    )


def _within(entry: SimpleNamespace, cron: SimpleNamespace, seconds_before: int = 10) -> None:
    end = now_local()
    entry.ts = (end - timedelta(seconds=seconds_before)).isoformat()
    cron.ts = end.isoformat()


# ── record_activity ─────────────────────────────────────────


def test_record_activity_writes_activity_item(data_dir, make_anima, monkeypatch) -> None:
    anima_dir = make_anima("alice")
    _set_company(anima_dir, "acme")
    _set_hub(monkeypatch, company="acme")

    item_id = record_activity("alice", via="heartbeat", text="Sent the weekly report", label="heartbeat")

    assert item_id is not None
    items = list_items(data_dir / "shared", company="acme")
    assert len(items) == 1
    item = items[0]
    assert item["id"] == item_id
    assert item["kind"] == "activity"
    assert item["source"] == "anima"
    assert item["from"] == "alice"
    assert item["company"] == "acme"
    assert item["via"] == "heartbeat"
    assert item["label"] == "heartbeat"
    assert item["text"] == "Sent the weekly report"


def test_record_activity_skips_when_timeline_disabled(data_dir, make_anima, monkeypatch) -> None:
    anima_dir = make_anima("alice")
    _set_company(anima_dir, "acme")
    _set_hub(monkeypatch, timeline_enabled=False, company="acme")

    assert record_activity("alice", via="heartbeat", text="anything") is None
    assert list_items(data_dir / "shared", company="acme") == []


def test_record_activity_skips_other_company(data_dir, make_anima, monkeypatch) -> None:
    anima_dir = make_anima("alice")
    _set_company(anima_dir, "other")
    _set_hub(monkeypatch, company="acme")

    assert record_activity("alice", via="heartbeat", text="anything") is None
    assert list_items(data_dir / "shared", company="acme") == []


def test_record_activity_marks_failure_at_start(data_dir, make_anima, monkeypatch) -> None:
    anima_dir = make_anima("alice")
    _set_company(anima_dir, "acme")
    _set_hub(monkeypatch, company="acme")

    assert record_activity("alice", via="cron", text="the run broke", failed=True) is not None

    item = list_items(data_dir / "shared", company="acme")[0]
    assert item["text"].startswith(t("hub.activity_failed_prefix"))
    assert "the run broke" in item["text"]


def test_record_activity_never_raises(data_dir, make_anima, monkeypatch) -> None:
    anima_dir = make_anima("alice")
    _set_company(anima_dir, "acme")
    _set_hub(monkeypatch, company="acme")

    with patch("core.hub.timeline.append_item", side_effect=OSError("disk full")):
        assert record_activity("alice", via="heartbeat", text="anything") is None


def test_issue_resolution_is_recorded_with_issue_via(tmp_path, monkeypatch) -> None:
    from core.memory.conversation.state_update import _record_resolutions

    anima_dir = tmp_path / "alice"
    anima_dir.mkdir()
    calls: list[tuple[tuple, dict]] = []

    def fake_record(*args, **kwargs):
        calls.append((args, kwargs))
        return "tl_issue"

    monkeypatch.setattr("core.hub.timeline_activity.record_activity", fake_record)
    memory_mgr = SimpleNamespace(append_resolution=lambda **kwargs: None)

    _record_resolutions(anima_dir, memory_mgr, ["the open question"])

    assert calls == [(("alice",), {"via": "issue", "text": "the open question"})]


# ── cron ───────────────────────────────────────────────────


def test_cron_external_action_is_recorded_with_full_text(data_dir, make_anima, monkeypatch) -> None:
    anima_dir = make_anima("alice")
    _set_company(anima_dir, "acme")
    _set_hub(monkeypatch, company="acme")

    cron = _cron_entry("")
    tool = _entry("", "tool_use", tool="call_human")
    _within(tool, cron)
    text = "通知しました。" + ("x" * 1200)

    item_id = record_cron_activity(
        "alice",
        activity=_WindowActivity([tool, cron]),
        cron_entry=cron,
        task_name="daily",
        text=text,
    )

    assert item_id is not None
    item = list_items(data_dir / "shared", company="acme")[0]
    assert item["via"] == "cron"
    assert item["label"] == "daily"
    assert item["text"] == text
    assert len(item["text"]) > 500


def test_cron_bash_external_action_is_recorded(data_dir, make_anima, monkeypatch) -> None:
    anima_dir = make_anima("alice")
    _set_company(anima_dir, "acme")
    _set_hub(monkeypatch, company="acme")

    cron = _cron_entry("")
    tool = _entry(
        "",
        "tool_use",
        tool="Bash",
        content="animaworks-tool chatwork send --text hello",
    )
    _within(tool, cron)

    item_id = record_cron_activity(
        "alice",
        activity=_WindowActivity([tool, cron]),
        cron_entry=cron,
        task_name="daily",
        text="chatwork に送信した",
    )

    assert item_id is not None
    assert list_items(data_dir / "shared", company="acme")[0]["text"] == "chatwork に送信した"


def test_cron_read_only_run_is_not_recorded(data_dir, make_anima, monkeypatch) -> None:
    anima_dir = make_anima("alice")
    _set_company(anima_dir, "acme")
    _set_hub(monkeypatch, company="acme")

    cron = _cron_entry("")
    tool = _entry("", "tool_use", tool="read_channel")
    result = _entry("", "tool_result", tool="read_channel")
    _within(tool, cron)
    result.ts = tool.ts

    assert (
        record_cron_activity(
            "alice",
            activity=_WindowActivity([tool, result, cron]),
            cron_entry=cron,
            task_name="daily",
            text="新着なし",
        )
        is None
    )
    assert list_items(data_dir / "shared", company="acme") == []


def test_cron_memory_write_only_run_is_not_recorded(data_dir, make_anima, monkeypatch) -> None:
    anima_dir = make_anima("alice")
    _set_company(anima_dir, "acme")
    _set_hub(monkeypatch, company="acme")

    cron = _cron_entry("")
    tool = _entry("", "tool_use", tool="write_memory_file")
    memory_event = _entry("", "memory_write")
    _within(tool, cron)
    memory_event.ts = tool.ts

    assert (
        record_cron_activity(
            "alice",
            activity=_WindowActivity([tool, memory_event, cron]),
            cron_entry=cron,
            task_name="daily",
            text="知識を1件書いた",
        )
        is None
    )
    assert list_items(data_dir / "shared", company="acme") == []


def test_cron_failed_run_is_recorded(data_dir, make_anima, monkeypatch) -> None:
    anima_dir = make_anima("alice")
    _set_company(anima_dir, "acme")
    _set_hub(monkeypatch, company="acme")

    cron = _cron_entry("", status="failed")
    tool = _entry("", "tool_use", tool="read_channel")
    _within(tool, cron)

    item_id = record_cron_activity(
        "alice",
        activity=_WindowActivity([tool, cron]),
        cron_entry=cron,
        task_name="daily",
        text="stream error",
        failed=True,
    )

    assert item_id is not None
    item = list_items(data_dir / "shared", company="acme")[0]
    assert item["text"].startswith(t("hub.activity_failed_prefix"))
    assert "stream error" in item["text"]


def test_cron_budget_skipped_is_not_recorded(data_dir, make_anima, monkeypatch) -> None:
    anima_dir = make_anima("alice")
    _set_company(anima_dir, "acme")
    _set_hub(monkeypatch, company="acme")

    cron = _cron_entry("")
    tool = _entry("", "tool_use", tool="call_human")
    _within(tool, cron)

    assert (
        record_cron_activity(
            "alice",
            activity=_WindowActivity([tool, cron]),
            cron_entry=cron,
            task_name="daily",
            text="should not be recorded",
            stop_kind="budget_skipped",
        )
        is None
    )
    assert list_items(data_dir / "shared", company="acme") == []
