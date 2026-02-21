"""Unit tests for PrimingEngine unified activity channel and fallback channels."""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from core.memory.priming import PrimingEngine, PrimingResult, format_priming_section


@pytest.fixture
def anima_dir(tmp_path: Path) -> Path:
    d = tmp_path / "animas" / "sakura"
    d.mkdir(parents=True)
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
        result = PrimingResult(recent_activity="1234")  # 4 chars / 4 = 1 token
        assert result.estimated_tokens() == 1


# ── Fallback: _read_old_channels ─────────────────────────


class TestFallbackChannels:
    async def test_empty_when_no_shared_dir(self, anima_dir):
        engine = PrimingEngine(anima_dir, shared_dir=None)
        result = await engine._read_old_channels()
        assert result == ""

    async def test_empty_when_no_channels_dir(self, anima_dir, tmp_path):
        # shared_dir exists but no channels subdir
        empty_shared = tmp_path / "empty_shared"
        empty_shared.mkdir()
        engine = PrimingEngine(anima_dir, shared_dir=empty_shared)
        result = await engine._read_old_channels()
        assert result == ""

    async def test_reads_general_channel(self, anima_dir, shared_dir):
        now = datetime.now()
        _write_channel(shared_dir, "general", [
            {"ts": now.isoformat(), "from": "kotoha", "text": "Hello!", "source": "anima"},
        ])
        engine = PrimingEngine(anima_dir, shared_dir=shared_dir)
        result = await engine._read_old_channels()
        assert "kotoha" in result
        assert "Hello!" in result
        assert "#general" in result

    async def test_reads_ops_channel(self, anima_dir, shared_dir):
        now = datetime.now()
        _write_channel(shared_dir, "ops", [
            {"ts": now.isoformat(), "from": "yuki", "text": "Server down", "source": "anima"},
        ])
        engine = PrimingEngine(anima_dir, shared_dir=shared_dir)
        result = await engine._read_old_channels()
        assert "#ops" in result
        assert "Server down" in result

    async def test_last_5_entries(self, anima_dir, shared_dir):
        now = datetime.now()
        entries = [
            {"ts": (now - timedelta(minutes=10 - i)).isoformat(), "from": f"anima{i}", "text": f"msg{i}", "source": "anima"}
            for i in range(10)
        ]
        _write_channel(shared_dir, "general", entries)
        engine = PrimingEngine(anima_dir, shared_dir=shared_dir)
        result = await engine._read_old_channels()
        # Should include at least the last 5
        assert "msg5" in result
        assert "msg9" in result

    async def test_human_messages_within_24h(self, anima_dir, shared_dir):
        now = datetime.now()
        entries = [
            {"ts": (now - timedelta(hours=2)).isoformat(), "from": "taka", "text": "Error resolved", "source": "human"},
            {"ts": (now - timedelta(hours=30)).isoformat(), "from": "taka", "text": "Old message", "source": "human"},
        ]
        _write_channel(shared_dir, "general", entries)
        engine = PrimingEngine(anima_dir, shared_dir=shared_dir)
        result = await engine._read_old_channels()
        assert "Error resolved" in result
        assert "[human]" in result

    async def test_mentions_included(self, anima_dir, shared_dir):
        now = datetime.now()
        entries = [
            {"ts": (now - timedelta(hours=48)).isoformat(), "from": "mio", "text": "@sakura please check", "source": "anima"},
            # Add recent entries to push this beyond last-5 window
        ] + [
            {"ts": (now - timedelta(minutes=i)).isoformat(), "from": f"a{i}", "text": f"filler{i}", "source": "anima"}
            for i in range(10)
        ]
        _write_channel(shared_dir, "general", entries)
        engine = PrimingEngine(anima_dir, shared_dir=shared_dir)
        result = await engine._read_old_channels()
        assert "@sakura please check" in result

    async def test_empty_channel_files(self, anima_dir, shared_dir):
        (shared_dir / "channels" / "general.jsonl").touch()
        engine = PrimingEngine(anima_dir, shared_dir=shared_dir)
        result = await engine._read_old_channels()
        assert result == ""


# ── prime_memories integration ───────────────────────────


class TestPrimeMemoriesWithActivity:
    async def test_includes_recent_activity(self, anima_dir, shared_dir, monkeypatch):
        now = datetime.now()
        _write_channel(shared_dir, "general", [
            {"ts": now.isoformat(), "from": "kotoha", "text": "Test msg", "source": "anima"},
        ])

        # Patch channels A/C/D to avoid real filesystem / RAG dependency
        async def _stub_a(self, name):
            return ""

        async def _stub_b(self, sender_name, keywords):
            return "some activity content"

        async def _stub_c(self, kw):
            return ""

        async def _stub_d(self, msg, kw, channel="chat"):
            return []

        monkeypatch.setattr("core.memory.priming.PrimingEngine._channel_a_sender_profile", _stub_a)
        monkeypatch.setattr("core.memory.priming.PrimingEngine._channel_b_recent_activity", _stub_b)
        monkeypatch.setattr("core.memory.priming.PrimingEngine._channel_c_related_knowledge", _stub_c)
        monkeypatch.setattr("core.memory.priming.PrimingEngine._channel_d_skill_match", _stub_d)

        engine = PrimingEngine(anima_dir, shared_dir=shared_dir)
        result = await engine.prime_memories("hello", sender_name="taka")
        assert result.recent_activity != ""

    async def test_fallback_populates_recent_activity(self, anima_dir, shared_dir, monkeypatch):
        """When _channel_b_recent_activity falls back to old channels, result goes into recent_activity."""
        now = datetime.now()
        _write_channel(shared_dir, "general", [
            {"ts": now.isoformat(), "from": "kotoha", "text": "Fallback msg", "source": "anima"},
        ])

        # Stub A/C/D, but let B run its real fallback logic
        async def _stub_a(self, name):
            return ""

        async def _stub_c(self, kw):
            return ""

        async def _stub_d(self, msg, kw, channel="chat"):
            return []

        # Stub _channel_b to simulate the fallback path returning old channel data
        async def _stub_b_fallback(self, sender_name, keywords):
            return await self._fallback_episodes_and_channels()

        monkeypatch.setattr("core.memory.priming.PrimingEngine._channel_a_sender_profile", _stub_a)
        monkeypatch.setattr("core.memory.priming.PrimingEngine._channel_b_recent_activity", _stub_b_fallback)
        monkeypatch.setattr("core.memory.priming.PrimingEngine._channel_c_related_knowledge", _stub_c)
        monkeypatch.setattr("core.memory.priming.PrimingEngine._channel_d_skill_match", _stub_d)

        engine = PrimingEngine(anima_dir, shared_dir=shared_dir)
        result = await engine.prime_memories("hello", sender_name="taka")
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
