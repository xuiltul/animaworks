"""Unit tests for PrimingEngine pending human notifications channel."""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.memory.priming import PrimingEngine, PrimingResult
from core.time_utils import now_iso


@pytest.fixture
def anima_dir(tmp_path: Path) -> Path:
    d = tmp_path / "animas" / "rin"
    d.mkdir(parents=True)
    (d / "episodes").mkdir()
    (d / "knowledge").mkdir()
    (d / "skills").mkdir()
    (d / "activity_log").mkdir()
    return d


def _write_activity(anima_dir: Path, entries: list[dict]) -> None:
    log_dir = anima_dir / "activity_log"
    log_dir.mkdir(exist_ok=True)
    date_str = entries[0]["ts"][:10] if entries else "2026-03-06"
    path = log_dir / f"{date_str}.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


# ── PrimingResult field ─────────────────────────────────────


class TestPrimingResultPendingHumanNotifications:
    def test_field_default_empty(self):
        result = PrimingResult()
        assert result.pending_human_notifications == ""

    def test_is_empty_false_with_notifications(self):
        result = PrimingResult(pending_human_notifications="some notification")
        assert not result.is_empty()

    def test_is_empty_true_without_notifications(self):
        result = PrimingResult()
        assert result.is_empty()

    def test_total_chars_includes_notifications(self):
        result = PrimingResult(pending_human_notifications="12345")
        assert result.total_chars() == 5


# ── _collect_pending_human_notifications ───────────────────


class TestCollectPendingHumanNotifications:
    @pytest.mark.asyncio
    async def test_chat_channel_returns_notifications(self, anima_dir: Path):
        ts = now_iso()
        _write_activity(anima_dir, [
            {
                "ts": ts,
                "type": "human_notify",
                "content": "Win11 VM: IP=192.168.1.100, user=admin, pass=secret123",
                "via": "slack",
            },
        ])
        engine = PrimingEngine(anima_dir)
        result = await engine._collect_pending_human_notifications(channel="chat")
        assert "## Pending Human Notifications" in result
        assert "192.168.1.100" in result
        assert "slack" in result

    @pytest.mark.asyncio
    async def test_heartbeat_channel_returns_notifications(self, anima_dir: Path):
        ts = now_iso()
        _write_activity(anima_dir, [
            {
                "ts": ts,
                "type": "human_notify",
                "content": "Server is ready.",
                "via": "ntfy",
            },
        ])
        engine = PrimingEngine(anima_dir)
        result = await engine._collect_pending_human_notifications(channel="heartbeat")
        assert "Server is ready." in result

    @pytest.mark.asyncio
    async def test_message_channel_returns_notifications(self, anima_dir: Path):
        ts = now_iso()
        _write_activity(anima_dir, [
            {
                "ts": ts,
                "type": "human_notify",
                "content": "Notification body",
                "via": "line",
            },
        ])
        engine = PrimingEngine(anima_dir)
        result = await engine._collect_pending_human_notifications(channel="message:owner")
        assert "Notification body" in result

    @pytest.mark.asyncio
    async def test_cron_channel_returns_empty(self, anima_dir: Path):
        ts = now_iso()
        _write_activity(anima_dir, [
            {
                "ts": ts,
                "type": "human_notify",
                "content": "should not appear",
                "via": "slack",
            },
        ])
        engine = PrimingEngine(anima_dir)
        result = await engine._collect_pending_human_notifications(channel="cron:daily")
        assert result == ""

    @pytest.mark.asyncio
    async def test_inbox_channel_returns_empty(self, anima_dir: Path):
        ts = now_iso()
        _write_activity(anima_dir, [
            {
                "ts": ts,
                "type": "human_notify",
                "content": "should not appear",
                "via": "slack",
            },
        ])
        engine = PrimingEngine(anima_dir)
        result = await engine._collect_pending_human_notifications(channel="inbox:someone")
        assert result == ""

    @pytest.mark.asyncio
    async def test_task_channel_returns_empty(self, anima_dir: Path):
        engine = PrimingEngine(anima_dir)
        result = await engine._collect_pending_human_notifications(channel="task:abc")
        assert result == ""

    @pytest.mark.asyncio
    async def test_empty_when_no_notifications(self, anima_dir: Path):
        engine = PrimingEngine(anima_dir)
        result = await engine._collect_pending_human_notifications(channel="chat")
        assert result == ""

    @pytest.mark.asyncio
    async def test_budget_truncation(self, anima_dir: Path):
        ts = now_iso()
        entries = []
        for i in range(15):
            entries.append({
                "ts": ts,
                "type": "human_notify",
                "content": f"Notification {i}: " + "x" * 300,
                "via": "slack",
            })
        _write_activity(anima_dir, entries)
        engine = PrimingEngine(anima_dir)
        result = await engine._collect_pending_human_notifications(channel="chat")
        assert len(result) <= 500 * 4 + 200

    @pytest.mark.asyncio
    async def test_multiple_notifications_chronological(self, anima_dir: Path):
        from core.time_utils import now_jst

        today = now_jst().date().isoformat()
        _write_activity(anima_dir, [
            {
                "ts": f"{today}T10:00:00+09:00",
                "type": "human_notify",
                "content": "First notification",
                "via": "slack",
            },
            {
                "ts": f"{today}T11:00:00+09:00",
                "type": "human_notify",
                "content": "Second notification",
                "via": "ntfy",
            },
        ])
        engine = PrimingEngine(anima_dir)
        result = await engine._collect_pending_human_notifications(channel="chat")
        first_idx = result.index("First notification")
        second_idx = result.index("Second notification")
        assert first_idx < second_idx

    @pytest.mark.asyncio
    async def test_uses_summary_when_content_empty(self, anima_dir: Path):
        ts = now_iso()
        _write_activity(anima_dir, [
            {
                "ts": ts,
                "type": "human_notify",
                "content": "",
                "summary": "Fallback summary text",
                "via": "slack",
            },
        ])
        engine = PrimingEngine(anima_dir)
        result = await engine._collect_pending_human_notifications(channel="chat")
        assert "Fallback summary text" in result


# ── Integration with prime_memories ──────────────────────────


class TestPrimeMemoriesIntegration:
    @pytest.mark.asyncio
    async def test_prime_memories_includes_notifications_for_chat(self, anima_dir: Path):
        ts = now_iso()
        _write_activity(anima_dir, [
            {
                "ts": ts,
                "type": "human_notify",
                "content": "VM credentials sent",
                "via": "slack",
            },
        ])
        engine = PrimingEngine(anima_dir)
        result = await engine.prime_memories("hello", channel="chat")
        assert "VM credentials sent" in result.pending_human_notifications

    @pytest.mark.asyncio
    async def test_prime_memories_excludes_notifications_for_cron(self, anima_dir: Path):
        ts = now_iso()
        _write_activity(anima_dir, [
            {
                "ts": ts,
                "type": "human_notify",
                "content": "should not appear",
                "via": "slack",
            },
        ])
        engine = PrimingEngine(anima_dir)
        result = await engine.prime_memories("hello", channel="cron:daily")
        assert result.pending_human_notifications == ""
