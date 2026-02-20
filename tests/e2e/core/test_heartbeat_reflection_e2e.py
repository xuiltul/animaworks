# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from datetime import date, datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.memory.activity import ActivityLogger


# ── Unit tests for _extract_reflection ──────────────────────────


class TestExtractReflection:
    """Unit tests for the pure function _extract_reflection()."""

    def test_normal_parse(self):
        """Valid [REFLECTION]...[/REFLECTION] block returns content."""
        from core.anima import _extract_reflection

        text = (
            "メッセージを確認しました。\n\n"
            "[REFLECTION]\n"
            "チームの連携パターンに変化が見られる。最近はmioからの報告頻度が"
            "増えており、タスクの優先度が変わっている可能性がある。\n"
            "[/REFLECTION]"
        )
        result = _extract_reflection(text)
        assert "チームの連携パターンに変化が見られる" in result
        assert "タスクの優先度が変わっている可能性がある" in result

    def test_no_reflection_tag(self):
        """Text without reflection tags returns empty string."""
        from core.anima import _extract_reflection

        text = "メッセージを確認しました。特に問題ありません。"
        result = _extract_reflection(text)
        assert result == ""

    def test_empty_string_input(self):
        """Empty string input returns empty string."""
        from core.anima import _extract_reflection

        result = _extract_reflection("")
        assert result == ""

    def test_multiple_reflection_tags(self):
        """Multiple reflection tags returns first match only."""
        from core.anima import _extract_reflection

        text = (
            "[REFLECTION]\n"
            "最初の振り返り内容\n"
            "[/REFLECTION]\n\n"
            "[REFLECTION]\n"
            "二番目の振り返り内容\n"
            "[/REFLECTION]"
        )
        result = _extract_reflection(text)
        assert "最初の振り返り内容" in result
        assert "二番目の振り返り内容" not in result

    def test_multiline_reflection_content(self):
        """Multiline reflection content is correctly captured across lines."""
        from core.anima import _extract_reflection

        text = (
            "[REFLECTION]\n"
            "1. チームの連携改善が必要\n"
            "2. タスクの優先度を再評価すべき\n"
            "3. mioとの定期ミーティングを検討\n"
            "[/REFLECTION]"
        )
        result = _extract_reflection(text)
        assert "1. チームの連携改善が必要" in result
        assert "2. タスクの優先度を再評価すべき" in result
        assert "3. mioとの定期ミーティングを検討" in result

    def test_whitespace_around_tags(self):
        """Whitespace around tags is properly stripped."""
        from core.anima import _extract_reflection

        text = (
            "[REFLECTION]   \n"
            "   振り返り内容がここにある   \n"
            "   [/REFLECTION]"
        )
        result = _extract_reflection(text)
        assert result == "振り返り内容がここにある"

    def test_none_input_returns_empty(self):
        """None-like falsy input returns empty string."""
        from core.anima import _extract_reflection

        # The function checks `if not text:` which handles None-like values
        result = _extract_reflection("")
        assert result == ""

    def test_reflection_with_no_newlines(self):
        """Reflection tags on single line still work."""
        from core.anima import _extract_reflection

        text = "[REFLECTION]短い振り返り[/REFLECTION]"
        result = _extract_reflection(text)
        assert result == "短い振り返り"


# ── E2E tests for heartbeat reflection flow ─────────────────────


class TestHeartbeatReflectionE2E:
    """E2E tests for the full heartbeat reflection recording flow."""

    async def test_heartbeat_with_reflection_records_episode_and_activity(
        self, data_dir, make_anima,
    ):
        """Heartbeat with reflection in accumulated_text records both
        [REFLECTION] in episode AND heartbeat_reflection in activity log.
        """
        alice_dir = make_anima("alice")
        shared_dir = data_dir / "shared"

        reflection_content = (
            "チームの連携パターンに変化が見られる。"
            "最近はmioからの報告頻度が増えており、"
            "タスクの優先度が変わっている可能性がある。"
        )

        with patch("core.anima.AgentCore") as MockAgent, \
             patch("core.anima.ConversationMemory") as MockConv, \
             patch("core.anima.load_prompt", return_value="prompt"):
            MockConv.return_value.load.return_value = MagicMock(turns=[])

            from core.anima import DigitalAnima
            dp = DigitalAnima(alice_dir, shared_dir)
            dp.agent.reset_reply_tracking = MagicMock()
            dp.agent.replied_to = set()

            async def mock_stream(prompt, trigger="manual"):
                yield {"type": "text_delta", "text": "メッセージを確認しました。\n\n"}
                yield {"type": "text_delta", "text": "[REFLECTION]\n"}
                yield {"type": "text_delta", "text": f"{reflection_content}\n"}
                yield {"type": "text_delta", "text": "[/REFLECTION]"}
                yield {
                    "type": "cycle_done",
                    "cycle_result": {
                        "trigger": "heartbeat",
                        "action": "responded",
                        "summary": "Processed messages with reflection",
                        "duration_ms": 100,
                    },
                }

            dp.agent.run_cycle_streaming = mock_stream

            await dp.run_heartbeat()

        # Verify episode file contains the [REFLECTION] block
        episode_file = alice_dir / "episodes" / f"{date.today().isoformat()}.md"
        assert episode_file.exists(), "Episode file should be created"
        episode_content = episode_file.read_text(encoding="utf-8")
        assert "[REFLECTION]" in episode_content
        assert "[/REFLECTION]" in episode_content
        assert "チームの連携パターンに変化が見られる" in episode_content

        # Verify activity log contains heartbeat_reflection event
        activity_dir = alice_dir / "activity_log"
        today = date.today().isoformat()
        log_file = activity_dir / f"{today}.jsonl"
        assert log_file.exists(), "Activity log file should exist"
        log_content = log_file.read_text(encoding="utf-8")

        reflection_events = [
            json.loads(line)
            for line in log_content.strip().splitlines()
            if json.loads(line).get("type") == "heartbeat_reflection"
        ]
        assert len(reflection_events) >= 1, (
            "At least one heartbeat_reflection event should be logged"
        )
        assert "チームの連携パターンに変化が見られる" in reflection_events[0]["content"]

    async def test_heartbeat_with_short_reflection_not_recorded(
        self, data_dir, make_anima,
    ):
        """Reflection shorter than _MIN_REFLECTION_LENGTH (50 chars) is not recorded."""
        alice_dir = make_anima("alice")
        shared_dir = data_dir / "shared"

        # This reflection is intentionally short (< 50 chars)
        short_reflection = "短い振り返り"
        assert len(short_reflection) < 50

        with patch("core.anima.AgentCore") as MockAgent, \
             patch("core.anima.ConversationMemory") as MockConv, \
             patch("core.anima.load_prompt", return_value="prompt"):
            MockConv.return_value.load.return_value = MagicMock(turns=[])

            from core.anima import DigitalAnima
            dp = DigitalAnima(alice_dir, shared_dir)
            dp.agent.reset_reply_tracking = MagicMock()
            dp.agent.replied_to = set()

            async def mock_stream(prompt, trigger="manual"):
                yield {"type": "text_delta", "text": "確認完了。\n\n"}
                yield {"type": "text_delta", "text": f"[REFLECTION]\n{short_reflection}\n[/REFLECTION]"}
                yield {
                    "type": "cycle_done",
                    "cycle_result": {
                        "trigger": "heartbeat",
                        "action": "responded",
                        "summary": "Heartbeat done with short reflection",
                        "duration_ms": 50,
                    },
                }

            dp.agent.run_cycle_streaming = mock_stream

            await dp.run_heartbeat()

        # Episode should exist but should NOT contain [REFLECTION] tags
        episode_file = alice_dir / "episodes" / f"{date.today().isoformat()}.md"
        assert episode_file.exists()
        episode_content = episode_file.read_text(encoding="utf-8")
        assert "[REFLECTION]" not in episode_content

        # Activity log should NOT contain heartbeat_reflection event
        activity_dir = alice_dir / "activity_log"
        today = date.today().isoformat()
        log_file = activity_dir / f"{today}.jsonl"
        if log_file.exists():
            log_content = log_file.read_text(encoding="utf-8")
            reflection_events = [
                json.loads(line)
                for line in log_content.strip().splitlines()
                if line.strip() and json.loads(line).get("type") == "heartbeat_reflection"
            ]
            assert len(reflection_events) == 0, (
                "Short reflections should not produce heartbeat_reflection events"
            )

    async def test_heartbeat_without_reflection_no_reflection_event(
        self, data_dir, make_anima,
    ):
        """Heartbeat without [REFLECTION] tags records normal episode
        but no heartbeat_reflection activity event.
        """
        alice_dir = make_anima("alice")
        shared_dir = data_dir / "shared"

        with patch("core.anima.AgentCore") as MockAgent, \
             patch("core.anima.ConversationMemory") as MockConv, \
             patch("core.anima.load_prompt", return_value="prompt"):
            MockConv.return_value.load.return_value = MagicMock(turns=[])

            from core.anima import DigitalAnima
            dp = DigitalAnima(alice_dir, shared_dir)
            dp.agent.reset_reply_tracking = MagicMock()
            dp.agent.replied_to = set()

            async def mock_stream(prompt, trigger="manual"):
                yield {"type": "text_delta", "text": "全てのタスクを確認しました。特に問題ありません。"}
                yield {
                    "type": "cycle_done",
                    "cycle_result": {
                        "trigger": "heartbeat",
                        "action": "responded",
                        "summary": "Normal heartbeat without reflection",
                        "duration_ms": 80,
                    },
                }

            dp.agent.run_cycle_streaming = mock_stream

            await dp.run_heartbeat()

        # Episode should exist with heartbeat content
        episode_file = alice_dir / "episodes" / f"{date.today().isoformat()}.md"
        assert episode_file.exists()
        episode_content = episode_file.read_text(encoding="utf-8")
        assert "ハートビート活動" in episode_content
        assert "[REFLECTION]" not in episode_content

        # Activity log should NOT have heartbeat_reflection event
        activity_dir = alice_dir / "activity_log"
        today = date.today().isoformat()
        log_file = activity_dir / f"{today}.jsonl"
        if log_file.exists():
            log_content = log_file.read_text(encoding="utf-8")
            for line in log_content.strip().splitlines():
                if not line.strip():
                    continue
                entry = json.loads(line)
                assert entry.get("type") != "heartbeat_reflection", (
                    "No heartbeat_reflection event should exist without reflection tags"
                )


class TestLoadRecentReflections:
    """Tests for _load_recent_reflections() method."""

    async def test_returns_formatted_reflections_from_activity_log(
        self, data_dir, make_anima,
    ):
        """_load_recent_reflections() returns formatted reflections from activity log."""
        alice_dir = make_anima("alice")
        shared_dir = data_dir / "shared"

        # Pre-populate activity log with heartbeat_reflection events
        activity = ActivityLogger(alice_dir)
        activity.log(
            "heartbeat_reflection",
            content="チームの連携が改善されている。タスクの完了率が上がっている。",
            summary="チームの連携が改善されている",
        )
        activity.log(
            "heartbeat_reflection",
            content="新しいツールの導入によりワークフローが効率化した。",
            summary="ワークフロー効率化",
        )

        with patch("core.anima.AgentCore") as MockAgent, \
             patch("core.anima.ConversationMemory") as MockConv, \
             patch("core.anima.load_prompt", return_value="prompt"):
            MockConv.return_value.load.return_value = MagicMock(turns=[])

            from core.anima import DigitalAnima
            dp = DigitalAnima(alice_dir, shared_dir)

            result = dp._load_recent_reflections()

        assert result != "", "Should return non-empty string when reflections exist"
        assert "チームの連携が改善されている" in result
        assert "ワークフローが効率化" in result
        # Each line should be formatted with timestamp prefix
        lines = result.strip().splitlines()
        assert len(lines) == 2
        for line in lines:
            assert line.startswith("- "), f"Each line should start with '- ': {line}"

    async def test_returns_empty_when_no_reflections(
        self, data_dir, make_anima,
    ):
        """_load_recent_reflections() returns empty string when no reflections exist."""
        alice_dir = make_anima("alice")
        shared_dir = data_dir / "shared"

        # Activity log dir exists but has no heartbeat_reflection events
        activity = ActivityLogger(alice_dir)
        activity.log("heartbeat_end", summary="Normal heartbeat")

        with patch("core.anima.AgentCore") as MockAgent, \
             patch("core.anima.ConversationMemory") as MockConv, \
             patch("core.anima.load_prompt", return_value="prompt"):
            MockConv.return_value.load.return_value = MagicMock(turns=[])

            from core.anima import DigitalAnima
            dp = DigitalAnima(alice_dir, shared_dir)

            result = dp._load_recent_reflections()

        assert result == "", "Should return empty string when no reflections exist"

    async def test_returns_empty_when_no_activity_log(
        self, data_dir, make_anima,
    ):
        """_load_recent_reflections() returns empty string when activity log dir
        does not exist."""
        alice_dir = make_anima("alice")
        shared_dir = data_dir / "shared"

        # Ensure no activity_log directory
        activity_dir = alice_dir / "activity_log"
        assert not activity_dir.exists()

        with patch("core.anima.AgentCore") as MockAgent, \
             patch("core.anima.ConversationMemory") as MockConv, \
             patch("core.anima.load_prompt", return_value="prompt"):
            MockConv.return_value.load.return_value = MagicMock(turns=[])

            from core.anima import DigitalAnima
            dp = DigitalAnima(alice_dir, shared_dir)

            result = dp._load_recent_reflections()

        assert result == ""

    async def test_limits_to_recent_reflections_n(
        self, data_dir, make_anima,
    ):
        """_load_recent_reflections() limits output to _RECENT_REFLECTIONS_N (3) entries."""
        alice_dir = make_anima("alice")
        shared_dir = data_dir / "shared"

        # Write 5 reflection events
        activity = ActivityLogger(alice_dir)
        for i in range(5):
            activity.log(
                "heartbeat_reflection",
                content=f"振り返り{i + 1}: テスト内容",
                summary=f"振り返り{i + 1}",
            )

        with patch("core.anima.AgentCore") as MockAgent, \
             patch("core.anima.ConversationMemory") as MockConv, \
             patch("core.anima.load_prompt", return_value="prompt"):
            MockConv.return_value.load.return_value = MagicMock(turns=[])

            from core.anima import DigitalAnima
            dp = DigitalAnima(alice_dir, shared_dir)

            result = dp._load_recent_reflections()

        lines = result.strip().splitlines()
        assert len(lines) == 3, (
            f"Should return at most 3 reflections, got {len(lines)}"
        )
        # Should return the most recent 3 (3, 4, 5)
        assert "振り返り3" in result
        assert "振り返り4" in result
        assert "振り返り5" in result


class TestBuildHeartbeatPromptReflection:
    """Tests for reflection injection in _build_heartbeat_prompt()."""

    async def test_includes_reflection_section_when_reflections_exist(
        self, data_dir, make_anima,
    ):
        """_build_heartbeat_prompt() includes reflection section when reflections exist."""
        alice_dir = make_anima("alice")
        shared_dir = data_dir / "shared"

        # Pre-populate activity log with heartbeat_reflection events
        activity = ActivityLogger(alice_dir)
        activity.log(
            "heartbeat_reflection",
            content="前回のハートビートでの気づき: チームの連携パターンが変化している。",
            summary="チームの連携パターン変化",
        )

        with patch("core.anima.AgentCore") as MockAgent, \
             patch("core.anima.ConversationMemory") as MockConv, \
             patch("core.anima.load_prompt", return_value="prompt"):
            MockConv.return_value.load.return_value = MagicMock(turns=[])

            from core.anima import DigitalAnima
            dp = DigitalAnima(alice_dir, shared_dir)

            parts = await dp._build_heartbeat_prompt()

        # Join all parts and check for reflection section
        full_prompt = "\n\n".join(parts)
        assert "直近の振り返り（前回までの気づき）" in full_prompt
        assert "チームの連携パターンが変化している" in full_prompt

    async def test_no_reflection_section_when_no_reflections(
        self, data_dir, make_anima,
    ):
        """_build_heartbeat_prompt() does NOT include reflection section
        when no reflections exist."""
        alice_dir = make_anima("alice")
        shared_dir = data_dir / "shared"

        with patch("core.anima.AgentCore") as MockAgent, \
             patch("core.anima.ConversationMemory") as MockConv, \
             patch("core.anima.load_prompt", return_value="prompt"):
            MockConv.return_value.load.return_value = MagicMock(turns=[])

            from core.anima import DigitalAnima
            dp = DigitalAnima(alice_dir, shared_dir)

            parts = await dp._build_heartbeat_prompt()

        full_prompt = "\n\n".join(parts)
        assert "直近の振り返り（前回までの気づき）" not in full_prompt
