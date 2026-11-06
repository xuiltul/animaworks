"""Unit tests for PrimingEngine unified activity channel and fallback channels."""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import tempfile
from datetime import timedelta
from pathlib import Path

import pytest

from core.memory.priming import PrimingEngine, PrimingResult, format_priming_section
from core.memory.priming.channel_b import read_old_channels
from core.messaging.messenger import ChannelMeta, save_channel_meta
from core.time_utils import now_jst


@pytest.fixture
def anima_dir(tmp_path: Path) -> Path:
    d = tmp_path / "animas" / "sakura"
    d.mkdir(parents=True)
    (d / "status.json").write_text("{}", encoding="utf-8")
    (d / "episodes").mkdir()
    (d / "knowledge").mkdir()
    (d / "skills").mkdir()
    return d


@pytest.fixture
def shared_dir(tmp_path: Path) -> Path:
    d = tmp_path / "shared"
    d.mkdir()
    (d / "channels").mkdir(parents=True)
    return d


def _write_channel(shared_dir: Path, channel: str, entries: list[dict]) -> None:
    filepath = shared_dir / "channels" / f"{channel}.jsonl"
    with filepath.open("w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


# ── PrimingResult ────────────────────────────────────────


class TestPrimingResultRecentActivity:
    def test_recent_activity_field(self):
        result = PrimingResult(recent_activity="some activity")
        assert result.recent_activity == "some activity"

    def test_is_empty_false_with_recent_activity(self):
        result = PrimingResult(recent_activity="activity")
        assert not result.is_empty()

    def test_is_empty_true_without_recent_activity(self):
        result = PrimingResult()
        assert result.is_empty()

    def test_total_chars_includes_recent_activity(self):
        result = PrimingResult(recent_activity="12345")
        assert result.total_chars() == 5

    def test_estimated_tokens(self):
        result = PrimingResult(recent_activity="1234")
        assert result.estimated_tokens() == 1


# ── Fallback: _read_old_channels ─────────────────────────


class TestFallbackChannels:
    async def test_empty_when_no_shared_dir(self, anima_dir):
        result = await read_old_channels(anima_dir, None)
        assert result == ""

    async def test_empty_when_no_channels_dir(self, anima_dir, tmp_path):
        # shared_dir exists but no channels subdir
        empty_shared = tmp_path / "empty_shared"
        empty_shared.mkdir()
        result = await read_old_channels(anima_dir, empty_shared)
        assert result == ""

    @pytest.mark.parametrize(
        "meta",
        [ChannelMeta(members=["bob"]), ChannelMeta(members=[], closed=True)],
        ids=["non-member", "closed"],
    )
    async def test_old_channel_fallback_skips_non_member_channels(self, tmp_path: Path, meta: ChannelMeta, monkeypatch):
        anima_dir = tmp_path / "animas" / "alice"
        anima_dir.mkdir(parents=True)
        shared_dir = tmp_path / "shared"
        channels_dir = shared_dir / "channels"
        channels_dir.mkdir(parents=True)
        channel_file = channels_dir / "general.jsonl"
        channel_file.write_text(
            json.dumps({"ts": now_jst().isoformat(), "from": "bob", "text": "secret post", "source": "anima"}) + "\n",
            encoding="utf-8",
        )
        save_channel_meta(shared_dir, "general", meta)
        original_read_text = Path.read_text

        def reject_channel_read(path: Path, *args, **kwargs):
            if path == channel_file:
                raise AssertionError("ACL-rejected channel JSONL must not be read")
            return original_read_text(path, *args, **kwargs)

        monkeypatch.setattr(Path, "read_text", reject_channel_read)

        assert await read_old_channels(anima_dir, shared_dir) == ""

    async def test_reads_general_channel(self, anima_dir, shared_dir):
        now = now_jst()
        _write_channel(
            shared_dir,
            "general",
            [
                {"ts": now.isoformat(), "from": "kotoha", "text": "Hello!", "source": "anima"},
            ],
        )
        result = await read_old_channels(anima_dir, shared_dir)
        assert "kotoha" in result
        assert "Hello!" in result
        assert "#general" in result

    async def test_reads_ops_channel(self, anima_dir, shared_dir):
        now = now_jst()
        _write_channel(
            shared_dir,
            "ops",
            [
                {"ts": now.isoformat(), "from": "yuki", "text": "Server down", "source": "anima"},
            ],
        )
        result = await read_old_channels(anima_dir, shared_dir)
        assert "#ops" in result
        assert "Server down" in result

    async def test_last_5_entries(self, anima_dir, shared_dir):
        now = now_jst()
        entries = [
            {
                "ts": (now - timedelta(minutes=10 - i)).isoformat(),
                "from": f"anima{i}",
                "text": f"msg{i}",
                "source": "anima",
            }
            for i in range(10)
        ]
        _write_channel(shared_dir, "general", entries)
        result = await read_old_channels(anima_dir, shared_dir)
        # Should include at least the last 5
        assert "msg5" in result
        assert "msg9" in result

    async def test_human_messages_within_24h(self, anima_dir, shared_dir):
        now = now_jst()
        entries = [
            {
                "ts": (now - timedelta(hours=2)).isoformat(),
                "from": "owner",
                "text": "Error resolved",
                "source": "human",
            },
            {"ts": (now - timedelta(hours=30)).isoformat(), "from": "owner", "text": "Old message", "source": "human"},
        ]
        _write_channel(shared_dir, "general", entries)
        result = await read_old_channels(anima_dir, shared_dir)
        assert "Error resolved" in result
        assert "[human]" in result

    async def test_mentions_included(self, anima_dir, shared_dir):
        now = now_jst()
        entries = [
            {
                "ts": (now - timedelta(hours=48)).isoformat(),
                "from": "mio",
                "text": "@sakura please check",
                "source": "anima",
            },
            # Add recent entries to push this beyond last-5 window
        ] + [
            {"ts": (now - timedelta(minutes=i)).isoformat(), "from": f"a{i}", "text": f"filler{i}", "source": "anima"}
            for i in range(10)
        ]
        _write_channel(shared_dir, "general", entries)
        result = await read_old_channels(anima_dir, shared_dir)
        assert "@sakura please check" in result

    async def test_empty_channel_files(self, anima_dir, shared_dir):
        (shared_dir / "channels" / "general.jsonl").touch()
        result = await read_old_channels(anima_dir, shared_dir)
        assert result == ""


# ── prime_memories integration ───────────────────────────


class TestPrimeMemoriesWithActivity:
    async def test_includes_recent_activity(self, anima_dir, shared_dir, monkeypatch):
        now = now_jst()
        _write_channel(
            shared_dir,
            "general",
            [
                {"ts": now.isoformat(), "from": "kotoha", "text": "Test msg", "source": "anima"},
            ],
        )

        # Patch channels A/C to avoid real filesystem / RAG dependency
        async def _stub_a(self, name):
            return ""

        async def _stub_b(self, sender_name, keywords, *, channel=""):
            return "some activity content"

        async def _stub_c(self, kw, **kwargs):
            return ("", "")

        monkeypatch.setattr("core.memory.priming.PrimingEngine._channel_a_sender_profile", _stub_a)
        monkeypatch.setattr("core.memory.priming.PrimingEngine._channel_b_recent_activity", _stub_b)
        monkeypatch.setattr("core.memory.priming.PrimingEngine._channel_c_related_knowledge", _stub_c)

        engine = PrimingEngine(anima_dir, shared_dir=shared_dir)
        result = await engine.prime_memories("hello", sender_name="owner", channel="heartbeat")
        assert result.recent_activity != ""

    async def test_fallback_populates_recent_activity(self, anima_dir, shared_dir, monkeypatch):
        """When _channel_b_recent_activity falls back to old channels, result goes into recent_activity."""
        now = now_jst()
        _write_channel(
            shared_dir,
            "general",
            [
                {"ts": now.isoformat(), "from": "kotoha", "text": "Fallback msg", "source": "anima"},
            ],
        )

        # Stub A/C, but let B run its real fallback logic
        async def _stub_a(self, name):
            return ""

        async def _stub_c(self, kw, **kwargs):
            return ("", "")

        # Stub _channel_b to simulate the fallback path returning old channel data
        async def _stub_b_fallback(self, sender_name, keywords, *, channel=""):
            from core.memory.priming.channel_b import fallback_episodes_and_channels

            return await fallback_episodes_and_channels(self.anima_dir, self.shared_dir)

        monkeypatch.setattr("core.memory.priming.PrimingEngine._channel_a_sender_profile", _stub_a)
        monkeypatch.setattr("core.memory.priming.PrimingEngine._channel_b_recent_activity", _stub_b_fallback)
        monkeypatch.setattr("core.memory.priming.PrimingEngine._channel_c_related_knowledge", _stub_c)

        engine = PrimingEngine(anima_dir, shared_dir=shared_dir)
        result = await engine.prime_memories("hello", sender_name="owner", channel="heartbeat")
        assert result.recent_activity != ""
        assert "Fallback msg" in result.recent_activity


# ── format_priming_section ────────────────────────────────


class TestFormatPrimingSectionActivity:
    def test_includes_activity_section(self):
        result = PrimingResult(recent_activity="### #general\nsome messages")
        text = format_priming_section(result)
        assert "直近のアクティビティ" in text
        assert "#general" in text

    def test_no_activity_section_when_empty(self):
        result = PrimingResult(sender_profile="profile")
        text = format_priming_section(result)
        assert "直近のアクティビティ" not in text


@pytest.fixture
def temp_anima_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        anima_dir = Path(tmpdir) / "animas" / "test"
        for directory in ["episodes", "knowledge", "skills", "state"]:
            (anima_dir / directory).mkdir(parents=True)
        yield anima_dir


class TestPrimingResultPendingTasks:
    def test_pending_tasks_field(self):
        result = PrimingResult(pending_tasks="test tasks")
        assert result.pending_tasks == "test tasks"

    def test_is_empty_with_pending_tasks(self):
        result = PrimingResult(pending_tasks="task")
        assert not result.is_empty()

    def test_total_chars_includes_pending(self):
        result = PrimingResult(pending_tasks="12345")
        assert result.total_chars() >= 5


class TestChannelEPendingTasks:
    async def test_channel_e_no_tasks(self, temp_anima_dir):
        engine = PrimingEngine(temp_anima_dir)
        result = await engine._channel_e_pending_tasks()
        assert result == ""

    async def test_channel_e_with_tasks(self, temp_anima_dir):
        from core.tasks.queue import TaskQueueManager

        manager = TaskQueueManager(temp_anima_dir)
        manager.add_task(
            source="human",
            original_instruction="Test task",
            assignee="rin",
            summary="Test summary",
        )
        engine = PrimingEngine(temp_anima_dir)
        result = await engine._channel_e_pending_tasks()
        assert "🔴 HIGH" in result
        assert "Test summary" in result

    def test_budget_constant(self):
        from core.memory.priming.engine import _BUDGET_PENDING_TASKS

        assert _BUDGET_PENDING_TASKS == 500

    async def test_empty_overflow_dir(self, temp_anima_dir):
        (temp_anima_dir / "state" / "overflow_inbox").mkdir(parents=True)
        engine = PrimingEngine(temp_anima_dir)
        result = await engine._channel_e_pending_tasks()
        assert "overflow_inbox" not in result

    async def test_overflow_files_shown(self, temp_anima_dir):
        overflow_dir = temp_anima_dir / "state" / "overflow_inbox"
        overflow_dir.mkdir(parents=True)
        for i in range(3):
            (overflow_dir / f"20260318_1200_sender{i}.md").write_text(
                f"---\nfrom: sender{i}\n---\ntest message {i}",
                encoding="utf-8",
            )
        engine = PrimingEngine(temp_anima_dir)
        result = await engine._channel_e_pending_tasks()
        assert "3" in result
        assert "overflow_inbox" in result
        assert "read_memory_file" in result
        assert "archive_memory_file" in result

    async def test_overflow_more_than_5_shows_count(self, temp_anima_dir):
        overflow_dir = temp_anima_dir / "state" / "overflow_inbox"
        overflow_dir.mkdir(parents=True)
        for i in range(8):
            (overflow_dir / f"20260318_1200_sender{i}.md").write_text(
                f"---\nfrom: sender{i}\n---\nmsg",
                encoding="utf-8",
            )
        engine = PrimingEngine(temp_anima_dir)
        result = await engine._channel_e_pending_tasks()
        assert "未処理メッセージ 8件" in result
        assert "read_memory_file" in result
        assert "archive_memory_file" in result
        assert "sender0" not in result
        assert "sender7" not in result

    async def test_non_md_files_ignored(self, temp_anima_dir):
        overflow_dir = temp_anima_dir / "state" / "overflow_inbox"
        overflow_dir.mkdir(parents=True)
        (overflow_dir / "not_a_message.txt").write_text("ignored")
        (overflow_dir / "actual.md").write_text("---\nfrom: a\n---\nmsg")
        engine = PrimingEngine(temp_anima_dir)
        result = await engine._channel_e_pending_tasks()
        assert "1 " in result or "1件" in result

    async def test_prime_memories_returns_pending_tasks(self, temp_anima_dir, monkeypatch):
        from core.tasks.queue import TaskQueueManager

        manager = TaskQueueManager(temp_anima_dir)
        manager.add_task(
            source="human",
            original_instruction="Important task",
            assignee="test",
            summary="Important task summary",
        )

        engine = PrimingEngine(temp_anima_dir)
        monkeypatch.setattr(engine, "_channel_a_sender_profile", lambda sender_name: _empty_text())
        monkeypatch.setattr(engine, "_channel_b_recent_activity", lambda sender_name, keywords, **kwargs: _empty_text())
        monkeypatch.setattr(engine, "_channel_c0_important_knowledge", _empty_text)
        monkeypatch.setattr(engine, "_channel_c_related_knowledge", lambda keywords, **kwargs: _empty_pair())
        monkeypatch.setattr(engine, "_collect_recent_outbound", _empty_text)
        monkeypatch.setattr(engine, "_channel_f_episodes", lambda keywords, **kwargs: _empty_text())
        monkeypatch.setattr(engine, "_collect_pending_human_notifications", lambda **kwargs: _empty_text())
        result = await engine.prime_memories("hello", sender_name="test")
        assert result.pending_tasks != ""
        assert "Important task" in result.pending_tasks


async def _empty_text(*args, **kwargs) -> str:
    return ""


async def _empty_pair() -> tuple[str, str]:
    return ("", "")
