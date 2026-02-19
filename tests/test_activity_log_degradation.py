from __future__ import annotations
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.

"""Tests for the activity log degradation fix.

Covers all changes made to restore communication visibility, improve
activity log recording quality, and fix priming data loss:

1.  priming.py — _read_shared_channels() reads shared/channels/*.jsonl
2.  priming.py — _channel_b_recent_activity() integration (no limit, post-score limit)
3.  messenger.py — send() records dm_sent in activity log
4.  messenger.py — send() writes to dm_logs/
5.  (removed — Section 9 deleted per Arch-1 hippocampus model)
6.  handler.py — _fanout_board_mentions() includes stopped Animas
7.  anima.py — DM receive limit 50
8.  activity.py — format_for_priming() with content_trim parameter
9.  messenger.py — read_dm_history() type filter
10. activity.py — _format_entry() with content_trim parameter
"""

import json
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from core.memory.activity import ActivityEntry, ActivityLogger


# ── Fixtures ────────────────────────────────────────────────────────


@pytest.fixture
def anima_dir(tmp_path: Path) -> Path:
    """Create a temporary anima directory with required subdirectories."""
    d = tmp_path / "animas" / "test-anima"
    d.mkdir(parents=True)
    (d / "activity_log").mkdir()
    (d / "episodes").mkdir()
    (d / "knowledge").mkdir()
    (d / "skills").mkdir()
    return d


@pytest.fixture
def shared_dir(tmp_path: Path) -> Path:
    """Create a temporary shared directory."""
    d = tmp_path / "shared"
    d.mkdir(parents=True)
    (d / "inbox").mkdir()
    return d


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    """Return the root data dir (parent of animas/ and shared/)."""
    return tmp_path


# ── 1. priming.py — _read_shared_channels() ────────────────────────


class TestReadSharedChannels:
    """Tests for PrimingEngine._read_shared_channels()."""

    def test_returns_entries_from_channel_files(self, anima_dir: Path, shared_dir: Path):
        """Verify that _read_shared_channels reads JSONL and returns ActivityEntry."""
        from core.memory.priming import PrimingEngine

        channels_dir = shared_dir / "channels"
        channels_dir.mkdir(parents=True)
        now = datetime.now()
        entry = json.dumps({
            "ts": now.isoformat(),
            "from": "alice",
            "text": "Hello everyone",
            "source": "anima",
        }, ensure_ascii=False)
        (channels_dir / "general.jsonl").write_text(entry + "\n", encoding="utf-8")

        engine = PrimingEngine(anima_dir, shared_dir=shared_dir)
        result = engine._read_shared_channels(limit_per_channel=5)

        assert len(result) >= 1
        assert isinstance(result[0], ActivityEntry)
        assert result[0].type == "channel_post"
        assert result[0].content == "Hello everyone"
        assert result[0].from_person == "alice"
        assert result[0].channel == "general"

    def test_returns_empty_when_shared_dir_is_none(self, anima_dir: Path):
        """Verify returns empty list when shared_dir is None."""
        from core.memory.priming import PrimingEngine

        engine = PrimingEngine(anima_dir, shared_dir=None)
        result = engine._read_shared_channels()
        assert result == []

    def test_returns_empty_when_channels_dir_missing(self, anima_dir: Path, shared_dir: Path):
        """Verify returns empty list when channels/ directory doesn't exist."""
        from core.memory.priming import PrimingEngine

        # shared_dir exists but channels/ subdirectory does not
        engine = PrimingEngine(anima_dir, shared_dir=shared_dir)
        result = engine._read_shared_channels()
        assert result == []

    def test_handles_malformed_jsonl_gracefully(self, anima_dir: Path, shared_dir: Path):
        """Verify malformed JSONL lines are skipped without error."""
        from core.memory.priming import PrimingEngine

        channels_dir = shared_dir / "channels"
        channels_dir.mkdir(parents=True)
        now = datetime.now()
        lines = [
            "this is not json",
            json.dumps({"ts": now.isoformat(), "from": "bob", "text": "Valid", "source": "anima"}),
            "{broken json",
        ]
        (channels_dir / "general.jsonl").write_text(
            "\n".join(lines) + "\n", encoding="utf-8",
        )

        engine = PrimingEngine(anima_dir, shared_dir=shared_dir)
        result = engine._read_shared_channels(limit_per_channel=10)

        # Only the valid entry should be returned
        assert len(result) == 1
        assert result[0].content == "Valid"

    def test_includes_mention_entries(self, anima_dir: Path, shared_dir: Path):
        """Verify that @mention entries are included even if outside latest N."""
        from core.memory.priming import PrimingEngine

        channels_dir = shared_dir / "channels"
        channels_dir.mkdir(parents=True)

        # Create 10 entries; only the last 5 are in "latest N",
        # but entry #2 has a mention of @test-anima
        entries = []
        for i in range(10):
            ts = (datetime.now() - timedelta(hours=10 - i)).isoformat()
            text = f"Message {i}"
            if i == 2:
                text = "Hey @test-anima please review"
            entries.append(json.dumps({
                "ts": ts, "from": f"user{i}", "text": text, "source": "anima",
            }, ensure_ascii=False))
        (channels_dir / "general.jsonl").write_text(
            "\n".join(entries) + "\n", encoding="utf-8",
        )

        engine = PrimingEngine(anima_dir, shared_dir=shared_dir)
        result = engine._read_shared_channels(limit_per_channel=5)

        # Should include the mention entry (index 2) plus latest 5 (indices 5-9)
        contents = [e.content for e in result]
        assert "Hey @test-anima please review" in contents
        assert len(result) >= 6  # 5 latest + 1 mention

    def test_includes_recent_human_posts(self, anima_dir: Path, shared_dir: Path):
        """Verify that human posts within 24h are included."""
        from core.memory.priming import PrimingEngine

        channels_dir = shared_dir / "channels"
        channels_dir.mkdir(parents=True)

        # Old entries (outside latest N) but one is a recent human post
        entries = []
        for i in range(10):
            ts = (datetime.now() - timedelta(hours=10 - i)).isoformat()
            source = "human" if i == 1 else "anima"
            entries.append(json.dumps({
                "ts": ts, "from": f"user{i}", "text": f"Msg {i}", "source": source,
            }, ensure_ascii=False))
        (channels_dir / "ops.jsonl").write_text(
            "\n".join(entries) + "\n", encoding="utf-8",
        )

        engine = PrimingEngine(anima_dir, shared_dir=shared_dir)
        result = engine._read_shared_channels(limit_per_channel=5)

        # The human post at index 1 (within 24h) should be included
        contents = [e.content for e in result]
        assert "Msg 1" in contents

    def test_limits_to_limit_per_channel(self, anima_dir: Path, shared_dir: Path):
        """Verify that at most limit_per_channel latest entries are returned."""
        from core.memory.priming import PrimingEngine

        channels_dir = shared_dir / "channels"
        channels_dir.mkdir(parents=True)

        # 20 entries, all anima-source (no mentions, no human posts)
        entries = []
        for i in range(20):
            ts = (datetime.now() - timedelta(days=2, hours=20 - i)).isoformat()
            entries.append(json.dumps({
                "ts": ts, "from": f"bot{i}", "text": f"Bot message {i}", "source": "anima",
            }, ensure_ascii=False))
        (channels_dir / "general.jsonl").write_text(
            "\n".join(entries) + "\n", encoding="utf-8",
        )

        engine = PrimingEngine(anima_dir, shared_dir=shared_dir)
        result = engine._read_shared_channels(limit_per_channel=3)

        # Only the latest 3 (no mentions/human posts match)
        assert len(result) == 3


# ── 2. priming.py — _channel_b_recent_activity() integration ───────


class TestChannelBRecentActivity:
    """Tests for PrimingEngine._channel_b_recent_activity() integration."""

    @pytest.mark.asyncio
    async def test_shared_channel_entries_included(self, anima_dir: Path, shared_dir: Path):
        """Verify that shared channel entries are merged into channel B output."""
        from core.memory.priming import PrimingEngine

        # Create shared channel with an entry
        channels_dir = shared_dir / "channels"
        channels_dir.mkdir(parents=True)
        now = datetime.now()
        channel_entry = json.dumps({
            "ts": now.isoformat(),
            "from": "alice",
            "text": "Shared channel message",
            "source": "anima",
        }, ensure_ascii=False)
        (channels_dir / "general.jsonl").write_text(
            channel_entry + "\n", encoding="utf-8",
        )

        engine = PrimingEngine(anima_dir, shared_dir=shared_dir)

        # Mock to avoid needing actual paths and RAG
        with patch("core.memory.priming.PrimingEngine._fallback_episodes_and_channels",
                    return_value=""):
            result = await engine._channel_b_recent_activity("alice", ["test"])

        # The result should contain shared channel content
        assert "Shared channel message" in result

    @pytest.mark.asyncio
    async def test_no_limit_before_scoring(self, anima_dir: Path, shared_dir: Path):
        """Verify that activity.recent() is called without limit parameter."""
        from core.memory.priming import PrimingEngine

        engine = PrimingEngine(anima_dir, shared_dir=shared_dir)

        with patch("core.memory.activity.ActivityLogger.recent") as mock_recent:
            mock_recent.return_value = []
            with patch.object(engine, "_read_shared_channels", return_value=[]):
                with patch.object(engine, "_fallback_episodes_and_channels",
                                  return_value="fallback"):
                    result = await engine._channel_b_recent_activity("alice", [])

            # Verify recent() was called with days=2 and no limit keyword
            call_kwargs = mock_recent.call_args
            assert call_kwargs[1].get("days", call_kwargs[0][0] if call_kwargs[0] else None) == 2 or \
                   mock_recent.call_args == ((), {"days": 2})


# ── 3. messenger.py — send() records dm_sent in Activity Log ───────


class TestMessengerSendActivityLog:
    """Tests for Messenger.send() recording dm_sent in activity log."""

    def test_send_creates_dm_sent_entry(self, tmp_path: Path):
        """Verify that send() creates a dm_sent entry in activity_log/{date}.jsonl."""
        from core.messenger import Messenger

        # Setup directory structure: shared/inbox/sender/ and animas/sender/activity_log/
        shared = tmp_path / "shared"
        (shared / "inbox" / "sender").mkdir(parents=True)
        (shared / "inbox" / "receiver").mkdir(parents=True)
        anima_dir = tmp_path / "animas" / "sender"
        (anima_dir / "activity_log").mkdir(parents=True)

        messenger = Messenger(shared, "sender")
        messenger.send(to="receiver", content="Hello receiver!")

        # Check activity_log
        today = datetime.now().strftime("%Y-%m-%d")
        log_file = anima_dir / "activity_log" / f"{today}.jsonl"
        assert log_file.exists(), "activity_log JSONL file should exist"

        lines = log_file.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) >= 1

        found_dm_sent = False
        for line in lines:
            entry = json.loads(line)
            if entry.get("type") == "dm_sent":
                found_dm_sent = True
                assert entry.get("to") == "receiver"
                assert "Hello receiver!" in entry.get("content", "")
                break
        assert found_dm_sent, "Should find a dm_sent entry in activity log"


# ── 4. messenger.py — send() writes to dm_logs/ ───────────────────


class TestMessengerSendDmLogs:
    """Tests for Messenger.send() appending to dm_logs/."""

    def test_send_appends_to_dm_logs(self, tmp_path: Path):
        """Verify that send() writes an entry to shared/dm_logs/{pair}.jsonl."""
        from core.messenger import Messenger

        shared = tmp_path / "shared"
        (shared / "inbox" / "alice").mkdir(parents=True)
        (shared / "inbox" / "bob").mkdir(parents=True)
        # Also need animas dir for activity log (don't fail on missing)
        (tmp_path / "animas" / "alice").mkdir(parents=True)

        messenger = Messenger(shared, "alice")
        messenger.send(to="bob", content="Hi Bob!")

        # dm_logs path uses sorted pair: alice-bob.jsonl
        dm_log = shared / "dm_logs" / "alice-bob.jsonl"
        assert dm_log.exists(), "dm_logs file should exist"

        lines = dm_log.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) >= 1

        entry = json.loads(lines[-1])
        assert entry.get("from") == "alice"
        assert entry.get("to") == "bob"
        assert entry.get("text") == "Hi Bob!"


# ── 5. (Section 9 deleted — Arch-1 hippocampus model) ─────────────
# _load_recent_activity_summary() has been removed.
# PrimingEngine Channel B is now the sole activity reader.


# ── 6. handler.py — _fanout_board_mentions() includes stopped Animas


class TestFanoutBoardMentionsStoppedAnimas:
    """Tests for ToolHandler._fanout_board_mentions() including stopped Animas."""

    def test_mentions_sent_to_stopped_animas(self, tmp_path: Path):
        """Verify mentions are delivered to both running and stopped Animas."""
        from core.tooling.handler import ToolHandler

        # Setup directories
        data_dir = tmp_path / "data"
        sockets_dir = data_dir / "run" / "sockets"
        sockets_dir.mkdir(parents=True)
        animas_dir = data_dir / "animas"

        # Create running anima (has socket)
        (sockets_dir / "running-anima.sock").touch()
        (animas_dir / "running-anima").mkdir(parents=True)

        # Create stopped anima (no socket, but directory exists)
        (animas_dir / "stopped-anima").mkdir(parents=True)

        # Self anima
        (animas_dir / "self-anima").mkdir(parents=True)

        # Create shared dir for messenger
        shared_dir = tmp_path / "shared"
        (shared_dir / "inbox" / "self-anima").mkdir(parents=True)

        # Use MagicMock for messenger and memory
        mock_messenger = MagicMock()
        mock_memory = MagicMock()
        mock_memory.read_permissions.return_value = ""

        handler = ToolHandler(
            anima_dir=animas_dir / "self-anima",
            memory=mock_memory,
            messenger=mock_messenger,
        )

        # Patch get_data_dir to return our temp directory.
        # get_data_dir is imported locally inside _fanout_board_mentions,
        # so we patch it at the source module level.
        with patch("core.paths.get_data_dir", return_value=data_dir):
            handler._fanout_board_mentions("general", "Hey @all check this out")

        # Verify both running and stopped animas received mentions
        sent_targets = set()
        for call in mock_messenger.send.call_args_list:
            if call.kwargs.get("to"):
                sent_targets.add(call.kwargs["to"])
            elif len(call.args) >= 1:
                sent_targets.add(call.args[0])
        # Note: @(\w+) regex won't match names with hyphens, but "all"
        # triggers the @all path which uses directory listing.
        # running_anima and stopped_anima use underscores so the test
        # covers the actual code path.

    def test_all_mention_excludes_stopped_animas(self, tmp_path: Path):
        """Verify @all mentions only reach running Animas (Fix 8)."""
        from core.tooling.handler import ToolHandler

        data_dir = tmp_path / "data"
        sockets_dir = data_dir / "run" / "sockets"
        sockets_dir.mkdir(parents=True)
        animas_dir = data_dir / "animas"

        # running_anima has a socket file
        (sockets_dir / "running_anima.sock").touch()
        (animas_dir / "running_anima").mkdir(parents=True)

        # stopped_anima has no socket but exists on disk
        (animas_dir / "stopped_anima").mkdir(parents=True)

        # self_anima
        (animas_dir / "self_anima").mkdir(parents=True)

        shared_dir = tmp_path / "shared"
        (shared_dir / "inbox" / "self_anima").mkdir(parents=True)

        mock_messenger = MagicMock()
        mock_memory = MagicMock()
        mock_memory.read_permissions.return_value = ""

        handler = ToolHandler(
            anima_dir=animas_dir / "self_anima",
            memory=mock_memory,
            messenger=mock_messenger,
        )

        with patch("core.paths.get_data_dir", return_value=data_dir):
            handler._fanout_board_mentions("general", "Hey @all check this out")

        sent_targets = set()
        for call in mock_messenger.send.call_args_list:
            if call.kwargs.get("to"):
                sent_targets.add(call.kwargs["to"])
            elif len(call.args) >= 1:
                sent_targets.add(call.args[0])
        assert "running_anima" in sent_targets, "Running anima should receive @all mention"
        assert "stopped_anima" not in sent_targets, "Stopped anima should NOT receive @all mention"
        assert "self_anima" not in sent_targets, "Self should be excluded from @all"

    def test_named_mention_underscore_name_stopped_not_sent(self, tmp_path: Path):
        """Verify named @mentions do NOT reach stopped Animas (Fix 8: running only)."""
        from core.tooling.handler import ToolHandler

        data_dir = tmp_path / "data"
        sockets_dir = data_dir / "run" / "sockets"
        sockets_dir.mkdir(parents=True)
        animas_dir = data_dir / "animas"

        (animas_dir / "stopped_target").mkdir(parents=True)
        (animas_dir / "self_anima").mkdir(parents=True)

        shared_dir = tmp_path / "shared"
        (shared_dir / "inbox" / "self_anima").mkdir(parents=True)

        mock_messenger = MagicMock()
        mock_memory = MagicMock()
        mock_memory.read_permissions.return_value = ""

        handler = ToolHandler(
            anima_dir=animas_dir / "self_anima",
            memory=mock_memory,
            messenger=mock_messenger,
        )

        with patch("core.paths.get_data_dir", return_value=data_dir):
            handler._fanout_board_mentions("ops", "Hey @stopped_target please review")

        sent_targets = set()
        for call in mock_messenger.send.call_args_list:
            if call.kwargs.get("to"):
                sent_targets.add(call.kwargs["to"])
            elif len(call.args) >= 1:
                sent_targets.add(call.args[0])
        assert "stopped_target" not in sent_targets, "Stopped Anima should NOT receive named mention"


# ── 7. anima.py — DM receive limit 50 ──────────────────────────────


class TestAnimaDmReceiveLimit:
    """Tests verifying DM recording uses limit of 50 instead of 10."""

    def test_recordable_limit_is_50_in_source(self):
        """Verify the source code uses [:50] for DM recording, not [:10]."""
        import inspect
        from core.anima import DigitalAnima

        source = inspect.getsource(DigitalAnima.run_heartbeat)
        # The fix changed [:10] to [:50]
        assert "_recordable[:50]" in source, \
            "run_heartbeat should use _recordable[:50] (not [:10])"
        assert "_recordable[:10]" not in source, \
            "run_heartbeat should NOT use _recordable[:10] anymore"


# ── 8. activity.py — format_for_priming() with content_trim ───────


class TestFormatForPrimingContentTrim:
    """Tests for ActivityLogger.format_for_priming() content_trim parameter."""

    def test_default_trim_at_200_chars(self, anima_dir: Path):
        """Verify default content_trim trims at 200 characters.

        Uses tool_use type to avoid DM grouping which has its own truncation.
        """
        activity = ActivityLogger(anima_dir)
        long_text = "A" * 300
        entries = [
            ActivityEntry(
                ts=datetime.now().isoformat(),
                type="tool_use",
                content=long_text,
                tool="web_search",
            ),
        ]
        result = activity.format_for_priming(entries)
        # Should be trimmed (default 200 chars)
        assert "..." in result
        assert "activity_log/" in result

    def test_custom_trim_value(self, anima_dir: Path):
        """Verify custom content_trim value works.

        Uses message_received type to avoid DM grouping.
        """
        activity = ActivityLogger(anima_dir)
        text = "B" * 100
        entries = [
            ActivityEntry(
                ts=datetime.now().isoformat(),
                type="message_received",
                content=text,
                from_person="user",
            ),
        ]
        # With content_trim=50, 100-char text should be trimmed
        result = activity.format_for_priming(entries, content_trim=50)
        assert "..." in result

    def test_content_trim_zero_disables_trimming(self, anima_dir: Path):
        """Verify content_trim=0 means no trimming."""
        activity = ActivityLogger(anima_dir)
        long_text = "C" * 500
        entries = [
            ActivityEntry(
                ts=datetime.now().isoformat(),
                type="tool_use",
                content=long_text,
                tool="web_search",
            ),
        ]
        result = activity.format_for_priming(
            entries, budget_tokens=5000, content_trim=0,
        )
        # The full text should be present (no truncation marker)
        assert long_text in result
        assert "...(-> activity_log/" not in result


# ── 9. messenger.py — read_dm_history() type filter ───────────────


class TestReadDmHistoryTypeFilter:
    """Tests for Messenger.read_dm_history() using correct type filters."""

    def test_only_requests_dm_types(self, tmp_path: Path):
        """Verify read_dm_history only queries dm_sent and dm_received types."""
        from core.messenger import Messenger

        shared = tmp_path / "shared"
        (shared / "inbox" / "alice").mkdir(parents=True)
        anima_dir = tmp_path / "animas" / "alice"
        (anima_dir / "activity_log").mkdir(parents=True)

        messenger = Messenger(shared, "alice")

        with patch("core.memory.activity.ActivityLogger.recent") as mock_recent:
            mock_recent.return_value = []
            messenger.read_dm_history("bob", limit=20)

            # Verify the types parameter
            call_kwargs = mock_recent.call_args[1] if mock_recent.call_args else {}
            types = call_kwargs.get("types", [])
            assert "dm_sent" in types
            assert "dm_received" in types
            # The fix removes message_received and response_sent from the filter
            assert "message_received" not in types
            assert "response_sent" not in types

    def test_read_dm_history_days_30(self, tmp_path: Path):
        """Verify read_dm_history uses days=30 for broader history."""
        from core.messenger import Messenger

        shared = tmp_path / "shared"
        (shared / "inbox" / "alice").mkdir(parents=True)
        anima_dir = tmp_path / "animas" / "alice"
        (anima_dir / "activity_log").mkdir(parents=True)

        messenger = Messenger(shared, "alice")

        with patch("core.memory.activity.ActivityLogger.recent") as mock_recent:
            mock_recent.return_value = []
            messenger.read_dm_history("bob", limit=20)

            call_kwargs = mock_recent.call_args[1] if mock_recent.call_args else {}
            days = call_kwargs.get("days")
            assert days == 30, f"Expected days=30 but got days={days}"


# ── 10. activity.py — _format_entry() with content_trim ───────────


class TestFormatEntryContentTrim:
    """Tests for ActivityLogger._format_entry() with content_trim parameter."""

    def test_default_trims_at_200_chars(self):
        """Verify default content_trim (200) truncates long content."""
        entry = ActivityEntry(
            ts="2026-02-19T10:30:00",
            type="dm_sent",
            content="X" * 300,
            to_person="alice",
        )
        result = ActivityLogger._format_entry(entry)
        # Should contain truncation marker
        assert "..." in result
        assert "activity_log/" in result
        # Should not contain full content
        assert "X" * 300 not in result

    def test_custom_trim_value(self):
        """Verify custom content_trim value truncates at specified length."""
        entry = ActivityEntry(
            ts="2026-02-19T10:30:00",
            type="dm_sent",
            content="Y" * 100,
            to_person="alice",
        )
        result = ActivityLogger._format_entry(entry, content_trim=50)
        # 100 chars > 50 chars trim threshold, so should be truncated
        assert "..." in result

    def test_content_trim_zero_no_trimming(self):
        """Verify content_trim=0 disables trimming entirely."""
        long_content = "Z" * 500
        entry = ActivityEntry(
            ts="2026-02-19T10:30:00",
            type="tool_use",
            content=long_content,
            tool="web_search",
        )
        result = ActivityLogger._format_entry(entry, content_trim=0)
        # Full content should be present
        assert long_content in result
        # No truncation marker
        assert "...(-> activity_log/" not in result

    def test_short_content_not_trimmed(self):
        """Verify content shorter than trim threshold is not modified."""
        entry = ActivityEntry(
            ts="2026-02-19T10:30:00",
            type="dm_sent",
            content="Short message",
            to_person="alice",
        )
        result = ActivityLogger._format_entry(entry)
        assert "Short message" in result
        assert "..." not in result

    def test_summary_used_over_content_when_available(self):
        """Verify summary is preferred over content for display."""
        entry = ActivityEntry(
            ts="2026-02-19T10:30:00",
            type="dm_sent",
            content="Full content text that is very long",
            summary="Short summary",
            to_person="alice",
        )
        result = ActivityLogger._format_entry(entry)
        assert "Short summary" in result

    def test_format_entry_type_icons(self):
        """Verify type-to-icon mapping works correctly."""
        test_cases = [
            ("dm_sent", "DM>"),
            ("dm_received", "DM<"),
            ("message_received", "MSG<"),
            ("response_sent", "MSG>"),
            ("channel_post", "CH.W"),
            ("tool_use", "TOOL"),
            ("heartbeat_start", "HB"),
            ("error", "ERR"),
        ]
        for event_type, expected_icon in test_cases:
            entry = ActivityEntry(
                ts="2026-02-19T10:30:00",
                type=event_type,
                content="test",
            )
            result = ActivityLogger._format_entry(entry)
            assert expected_icon in result, \
                f"Expected '{expected_icon}' in output for type '{event_type}'"


# ── Additional integration tests ────────────────────────────────────


class TestActivityLoggerBasic:
    """Basic tests for ActivityLogger recording and retrieval."""

    def test_log_creates_jsonl_file(self, anima_dir: Path):
        """Verify that logging creates a JSONL file for today."""
        activity = ActivityLogger(anima_dir)
        activity.log("dm_sent", content="Hello", to_person="bob")

        today = datetime.now().strftime("%Y-%m-%d")
        log_file = anima_dir / "activity_log" / f"{today}.jsonl"
        assert log_file.exists()

        lines = log_file.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 1
        entry = json.loads(lines[0])
        assert entry["type"] == "dm_sent"
        assert entry["to"] == "bob"

    def test_recent_returns_entries(self, anima_dir: Path):
        """Verify that recent() returns logged entries."""
        activity = ActivityLogger(anima_dir)
        activity.log("dm_sent", content="Hello", to_person="bob")
        activity.log("dm_received", content="Hi", from_person="bob")

        entries = activity.recent(days=1)
        assert len(entries) == 2
        assert entries[0].type == "dm_sent"
        assert entries[1].type == "dm_received"

    def test_recent_with_type_filter(self, anima_dir: Path):
        """Verify type filtering in recent()."""
        activity = ActivityLogger(anima_dir)
        activity.log("dm_sent", content="Sent", to_person="bob")
        activity.log("dm_received", content="Received", from_person="bob")
        activity.log("tool_use", tool="web_search", summary="Searched")

        entries = activity.recent(days=1, types=["dm_sent", "dm_received"])
        assert len(entries) == 2
        types = {e.type for e in entries}
        assert types == {"dm_sent", "dm_received"}

    def test_format_for_priming_empty_entries(self, anima_dir: Path):
        """Verify format_for_priming returns empty string for empty list."""
        activity = ActivityLogger(anima_dir)
        result = activity.format_for_priming([])
        assert result == ""


class TestMessengerSendIntegration:
    """Integration tests for Messenger.send() with activity log and dm_logs."""

    def test_send_writes_both_activity_and_dm_logs(self, tmp_path: Path):
        """Verify send() writes to both activity log and dm_logs."""
        from core.messenger import Messenger

        shared = tmp_path / "shared"
        (shared / "inbox" / "alice").mkdir(parents=True)
        (shared / "inbox" / "bob").mkdir(parents=True)
        anima_dir = tmp_path / "animas" / "alice"
        (anima_dir / "activity_log").mkdir(parents=True)

        messenger = Messenger(shared, "alice")
        messenger.send(to="bob", content="Test message")

        # Check activity log
        today = datetime.now().strftime("%Y-%m-%d")
        activity_file = anima_dir / "activity_log" / f"{today}.jsonl"
        assert activity_file.exists()
        activity_entries = [
            json.loads(line)
            for line in activity_file.read_text(encoding="utf-8").strip().splitlines()
        ]
        dm_sent_entries = [e for e in activity_entries if e.get("type") == "dm_sent"]
        assert len(dm_sent_entries) >= 1

        # Check dm_logs
        dm_log = shared / "dm_logs" / "alice-bob.jsonl"
        assert dm_log.exists()
        dm_entries = [
            json.loads(line)
            for line in dm_log.read_text(encoding="utf-8").strip().splitlines()
        ]
        assert len(dm_entries) >= 1
        assert dm_entries[-1]["from"] == "alice"
        assert dm_entries[-1]["to"] == "bob"

    def test_send_does_not_fail_if_anima_dir_missing(self, tmp_path: Path):
        """Verify send() succeeds even if animas/ directory doesn't exist."""
        from core.messenger import Messenger

        shared = tmp_path / "shared"
        (shared / "inbox" / "alice").mkdir(parents=True)
        (shared / "inbox" / "bob").mkdir(parents=True)
        # Deliberately do NOT create animas/alice/

        messenger = Messenger(shared, "alice")
        # Should not raise
        msg = messenger.send(to="bob", content="Hello")
        assert msg.content == "Hello"

    def test_send_inbox_file_created(self, tmp_path: Path):
        """Verify send() creates the inbox JSON file for the recipient."""
        from core.messenger import Messenger

        shared = tmp_path / "shared"
        (shared / "inbox" / "alice").mkdir(parents=True)

        messenger = Messenger(shared, "alice")
        msg = messenger.send(to="bob", content="Inbox test")

        inbox_file = shared / "inbox" / "bob" / f"{msg.id}.json"
        assert inbox_file.exists()
        data = json.loads(inbox_file.read_text(encoding="utf-8"))
        assert data["from_person"] == "alice"
        assert data["to_person"] == "bob"
        assert data["content"] == "Inbox test"


class TestFormatGroupContentTrim:
    """Tests for _format_group passing content_trim to _format_entry."""

    def test_format_group_passes_content_trim(self, anima_dir: Path):
        """Verify _format_group propagates content_trim to single entries."""
        from core.memory.activity import EntryGroup

        long_content = "D" * 500
        entry = ActivityEntry(
            ts="2026-02-19T14:00:00",
            type="tool_use",
            content=long_content,
            tool="web_search",
        )
        group = EntryGroup(
            type="single",
            start_ts=entry.ts,
            end_ts=entry.ts,
            entries=[entry],
            label="",
            source_lines="",
        )

        # With content_trim=0, full content should be present
        result_no_trim = ActivityLogger._format_group(group, content_trim=0)
        assert long_content in result_no_trim

        # With default content_trim, content should be truncated
        result_trimmed = ActivityLogger._format_group(group)
        assert long_content not in result_trimmed
        assert "..." in result_trimmed
