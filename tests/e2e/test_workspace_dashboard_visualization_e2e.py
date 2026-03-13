"""E2E tests for dashboard visualization enhancement.

Tests the integration between server-side event emission and frontend
card stream / message line rendering for the new dashboard features:
- New event types (message_sent, response_sent, channel_post, task events)
- Direction icons in card stream
- External channel node mapping
- Message line type variants

Uses Python simulation of JS logic + actual server API testing.
"""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from core.memory.activity import ActivityLogger

REPO_ROOT = Path(__file__).resolve().parents[2]
ORG_DASHBOARD_JS = (
    REPO_ROOT / "server" / "static" / "workspace" / "modules" / "org-dashboard.js"
)


# ── Python simulation of enhanced JS card stream logic ──────────

MAX_STREAM_ENTRIES = 4

TOOL_TO_EXTERNAL = {
    "slack": "ext_slack",
    "chatwork": "ext_chatwork",
    "github": "ext_github",
    "gmail": "ext_gmail",
    "web_search": "ext_web",
    "x_search": "ext_web",
}


class EnhancedCardStreamSimulator:
    """Python simulation of the enhanced updateCardActivity logic."""

    def __init__(self):
        self.streams: dict[str, list[dict]] = {}

    def update(self, name: str, data: dict) -> list[dict]:
        entries = list(self.streams.get(name, []))
        event_type = data.get("eventType", "")

        if event_type == "message_sent":
            to = data.get("to_person", "")
            summary = data.get("summary") or data.get("content", "")
            entries.append({
                "id": str(len(entries)),
                "type": "msg_out",
                "text": f"↑ {to}: {summary[:60]}",
                "status": "done",
                "ts": 1000,
            })
        elif event_type == "message_received":
            frm = data.get("from_person", "")
            summary = data.get("summary") or data.get("content", "")
            entries.append({
                "id": str(len(entries)),
                "type": "msg_in",
                "text": f"↓ {frm}: {summary[:60]}",
                "status": "done",
                "ts": 1000,
            })
        elif event_type == "response_sent":
            summary = data.get("summary") or data.get("content", "")
            entries.append({
                "id": str(len(entries)),
                "type": "msg_out",
                "text": f"↑ {summary[:80]}",
                "status": "done",
                "ts": 1000,
            })
        elif event_type == "channel_post":
            ch = data.get("channel", "?")
            summary = data.get("summary") or data.get("content", "")
            entries.append({
                "id": str(len(entries)),
                "type": "board",
                "text": f"↑ #{ch}: {summary[:60]}",
                "status": "done",
                "ts": 1000,
            })
        elif event_type in ("task_created", "task_updated"):
            summary = data.get("summary") or event_type
            entries.append({
                "id": str(len(entries)),
                "type": "task",
                "text": f"⚙ {summary[:80]}",
                "status": "done",
                "ts": 1000,
            })
        elif event_type == "tool_start":
            entries.append({
                "id": data.get("toolId", str(len(entries))),
                "type": "tool",
                "text": data.get("toolName", "tool"),
                "status": "running",
                "ts": 1000,
            })

        if len(entries) > MAX_STREAM_ENTRIES * 2:
            entries = entries[-MAX_STREAM_ENTRIES:]

        self.streams[name] = entries
        return entries[-MAX_STREAM_ENTRIES:]


# ── Card Stream: New Event Types ──────────────────────

class TestEnhancedCardStream:
    """Test new event types produce correct card stream entries."""

    def test_message_sent_creates_msg_out(self):
        sim = EnhancedCardStreamSimulator()
        entries = sim.update("sakura", {
            "eventType": "message_sent",
            "to_person": "hinata",
            "summary": "→ hinata: 顧客対応完了を報告",
        })
        assert len(entries) == 1
        assert entries[0]["type"] == "msg_out"
        assert "↑" in entries[0]["text"]
        assert "hinata" in entries[0]["text"]

    def test_message_received_creates_msg_in(self):
        sim = EnhancedCardStreamSimulator()
        entries = sim.update("sakura", {
            "eventType": "message_received",
            "from_person": "taka",
            "summary": "新しいタスクをお願いします",
        })
        assert len(entries) == 1
        assert entries[0]["type"] == "msg_in"
        assert "↓" in entries[0]["text"]
        assert "taka" in entries[0]["text"]

    def test_response_sent_creates_msg_out(self):
        sim = EnhancedCardStreamSimulator()
        entries = sim.update("sakura", {
            "eventType": "response_sent",
            "summary": "見積もり送付しました。添付をご確認ください。",
        })
        assert len(entries) == 1
        assert entries[0]["type"] == "msg_out"
        assert "↑" in entries[0]["text"]
        assert "見積もり" in entries[0]["text"]

    def test_channel_post_creates_board_entry(self):
        sim = EnhancedCardStreamSimulator()
        entries = sim.update("sakura", {
            "eventType": "channel_post",
            "channel": "general",
            "summary": "本日の進捗報告",
        })
        assert len(entries) == 1
        assert entries[0]["type"] == "board"
        assert "#general" in entries[0]["text"]
        assert "↑" in entries[0]["text"]

    def test_task_created_creates_task_entry(self):
        sim = EnhancedCardStreamSimulator()
        entries = sim.update("sakura", {
            "eventType": "task_created",
            "summary": "nagiにタスク委譲: 調査",
        })
        assert len(entries) == 1
        assert entries[0]["type"] == "task"
        assert "⚙" in entries[0]["text"]

    def test_task_updated_creates_task_entry(self):
        sim = EnhancedCardStreamSimulator()
        entries = sim.update("sakura", {
            "eventType": "task_updated",
            "summary": "タスク完了: 調査レポート提出",
        })
        assert len(entries) == 1
        assert entries[0]["type"] == "task"
        assert "完了" in entries[0]["text"]

    def test_mixed_events_maintain_order(self):
        sim = EnhancedCardStreamSimulator()
        sim.update("sakura", {"eventType": "message_received", "from_person": "human", "summary": "見積もりをください"})
        sim.update("sakura", {"eventType": "tool_start", "toolId": "t1", "toolName": "slack"})
        sim.update("sakura", {"eventType": "response_sent", "summary": "承知しました。作成中です。"})
        entries = sim.update("sakura", {"eventType": "message_sent", "to_person": "hinata", "summary": "→ hinata: 見積もり依頼の件"})

        assert len(entries) == 4
        assert entries[0]["type"] == "msg_in"
        assert entries[1]["type"] == "tool"
        assert entries[2]["type"] == "msg_out"
        assert entries[3]["type"] == "msg_out"

    def test_content_fallback_when_summary_empty(self):
        sim = EnhancedCardStreamSimulator()
        entries = sim.update("sakura", {
            "eventType": "message_sent",
            "to_person": "bob",
            "summary": "",
            "content": "Fallback content here",
        })
        assert "Fallback content" in entries[0]["text"]


# ── External Channel Mapping ──────────────────────

class TestExternalChannelMapping:
    """Tool names correctly map to external node IDs."""

    def test_slack_maps(self):
        assert TOOL_TO_EXTERNAL["slack"] == "ext_slack"

    def test_chatwork_maps(self):
        assert TOOL_TO_EXTERNAL["chatwork"] == "ext_chatwork"

    def test_github_maps(self):
        assert TOOL_TO_EXTERNAL["github"] == "ext_github"

    def test_gmail_maps(self):
        assert TOOL_TO_EXTERNAL["gmail"] == "ext_gmail"

    def test_web_search_maps(self):
        assert TOOL_TO_EXTERNAL["web_search"] == "ext_web"

    def test_unknown_tool_returns_none(self):
        assert TOOL_TO_EXTERNAL.get("unknown_tool") is None

    def test_js_mapping_matches_python(self):
        """Verify JS source contains the same tool-to-external mappings."""
        src = ORG_DASHBOARD_JS.read_text(encoding="utf-8")
        for tool, ext_id in TOOL_TO_EXTERNAL.items():
            assert ext_id in src, f"Missing {ext_id} for tool {tool} in JS"


# ── Server-side Live Event Emission ──────────────────────

class TestLiveEventEmissionForDashboard:
    """Activity logger emits live events for dashboard-relevant event types."""

    @pytest.fixture()
    def activity_logger(self, tmp_path: Path) -> ActivityLogger:
        anima_dir = tmp_path / "animas" / "sakura"
        anima_dir.mkdir(parents=True)
        return ActivityLogger(anima_dir)

    def _get_events(self, tmp_path: Path) -> list[dict]:
        event_dir = tmp_path / "run" / "events" / "sakura"
        if not event_dir.exists():
            return []
        return [
            json.loads(f.read_text(encoding="utf-8"))
            for f in sorted(event_dir.glob("ta_*.json"))
        ]

    def test_message_sent_event_has_to_person(self, activity_logger, tmp_path):
        with patch("core.memory.activity.get_data_dir", return_value=tmp_path):
            activity_logger.log(
                "message_sent",
                content="報告: 顧客対応完了",
                to_person="hinata",
                summary="→ hinata: 報告: 顧客対応完了",
                meta={"intent": "report"},
            )
        events = self._get_events(tmp_path)
        assert len(events) == 1
        data = events[0]["data"]
        assert data["type"] == "message_sent"
        assert data["to_person"] == "hinata"
        assert "報告" in data["summary"]

    def test_response_sent_event_has_content(self, activity_logger, tmp_path):
        with patch("core.memory.activity.get_data_dir", return_value=tmp_path):
            activity_logger.log(
                "response_sent",
                content="見積もりを送付いたします。",
                to_person="human",
                channel="chat",
                summary="見積もりを送付いたします。",
            )
        events = self._get_events(tmp_path)
        data = events[0]["data"]
        assert data["channel"] == "chat"
        assert "見積もり" in data["content"]

    def test_channel_post_event_has_channel(self, activity_logger, tmp_path):
        with patch("core.memory.activity.get_data_dir", return_value=tmp_path):
            activity_logger.log(
                "channel_post",
                channel="general",
                content="進捗報告",
                summary="Board post",
            )
        events = self._get_events(tmp_path)
        data = events[0]["data"]
        assert data["channel"] == "general"

    def test_full_flow_multiple_events(self, activity_logger, tmp_path):
        """Simulate a realistic Anima session with multiple dashboard events."""
        with patch("core.memory.activity.get_data_dir", return_value=tmp_path):
            activity_logger.log(
                "tool_use", tool="slack", summary="slack: read_messages"
            )
            activity_logger.log(
                "message_sent",
                content="hinataに報告",
                to_person="hinata",
                summary="→ hinata: hinataに報告",
            )
            activity_logger.log(
                "response_sent",
                content="お客様への返信を完了しました",
                to_person="human",
                summary="お客様への返信を完了しました",
                channel="chat",
            )
        events = self._get_events(tmp_path)
        assert len(events) == 3
        types = sorted(e["data"]["type"] for e in events)
        assert types == ["message_sent", "response_sent", "tool_use"]
