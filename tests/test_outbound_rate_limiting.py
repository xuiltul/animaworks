"""Unit tests for outbound rate limiting — per-run guard, cross-run guard,
messenger last_post_by, cascade_limiter file-based, and priming outbound section.
"""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from core.memory import MemoryManager
from core.memory.activity import ActivityEntry, ActivityLogger
from core.messenger import Messenger
from core.time_utils import now_iso, now_jst
from core.tooling.handler import ToolHandler


# ── Helpers ──────────────────────────────────────────────────


def _make_handler(
    tmp_path: Path,
    anima_name: str = "alice",
) -> tuple[ToolHandler, Messenger, Path]:
    """Build a ToolHandler with a real Messenger and mock MemoryManager."""
    shared_dir = tmp_path / "shared"
    shared_dir.mkdir(parents=True, exist_ok=True)
    (shared_dir / "inbox").mkdir(exist_ok=True)
    (shared_dir / "channels").mkdir(exist_ok=True)

    anima_dir = tmp_path / "animas" / anima_name
    anima_dir.mkdir(parents=True, exist_ok=True)
    (anima_dir / "identity.md").write_text(f"# {anima_name}\n", encoding="utf-8")
    (anima_dir / "activity_log").mkdir(exist_ok=True)

    memory = MagicMock(spec=MemoryManager)
    memory.read_permissions.return_value = ""
    messenger = Messenger(shared_dir, anima_name)

    handler = ToolHandler(
        anima_dir=anima_dir,
        memory=memory,
        messenger=messenger,
    )
    return handler, messenger, shared_dir


# ── post_channel per-run guard ───────────────────────────────


class TestPostChannelPerRunGuard:
    """post_channel per-run ガードのユニットテスト。"""

    def test_post_channel_per_run_blocks_second_post(self, tmp_path: Path) -> None:
        """同一チャネルに2回投稿すると2回目がエラーになる。"""
        handler, _, _ = _make_handler(tmp_path)

        result1 = handler.handle("post_channel", {"channel": "general", "text": "First"})
        assert "Posted to #general" in result1

        result2 = handler.handle("post_channel", {"channel": "general", "text": "Second"})
        assert "Error" in result2
        assert "投稿済み" in result2

    def test_post_channel_per_run_allows_different_channels(self, tmp_path: Path) -> None:
        """異なるチャネルへの投稿は許可される。"""
        handler, _, _ = _make_handler(tmp_path)

        result1 = handler.handle("post_channel", {"channel": "general", "text": "Hello"})
        assert "Posted to #general" in result1

        result2 = handler.handle("post_channel", {"channel": "ops", "text": "Status OK"})
        assert "Posted to #ops" in result2

    def test_reset_posted_channels_clears_tracking(self, tmp_path: Path) -> None:
        """reset_posted_channels後は再投稿可能になる。"""
        handler, _, _ = _make_handler(tmp_path)

        result1 = handler.handle("post_channel", {"channel": "general", "text": "First"})
        assert "Posted to #general" in result1

        handler.reset_posted_channels()

        # Cross-run guard could block, so mock load_config to set cooldown=0
        with patch("core.config.models.load_config") as mock_cfg:
            mock_cfg.return_value = MagicMock()
            mock_cfg.return_value.heartbeat.channel_post_cooldown_s = 0
            result2 = handler.handle("post_channel", {"channel": "general", "text": "After reset"})
        assert "Posted to #general" in result2


# ── post_channel cross-run guard ────────────────────────────


class TestPostChannelCrossRunGuard:
    """post_channel cross-run ファイルベース cooldown のユニットテスト。"""

    def test_post_channel_cross_run_blocks_within_cooldown(self, tmp_path: Path) -> None:
        """cooldown期間内で別runからの投稿がブロックされる。"""
        handler1, messenger, shared_dir = _make_handler(tmp_path)

        # First run: post to general
        with patch("core.config.models.load_config") as mock_cfg:
            mock_cfg.return_value = MagicMock()
            mock_cfg.return_value.heartbeat.channel_post_cooldown_s = 300
            result1 = handler1.handle("post_channel", {"channel": "general", "text": "First run"})
        assert "Posted to #general" in result1

        # Simulate a second run: new handler instance
        handler2, _, _ = _make_handler(tmp_path, "alice")

        with patch("core.config.models.load_config") as mock_cfg:
            mock_cfg.return_value = MagicMock()
            mock_cfg.return_value.heartbeat.channel_post_cooldown_s = 300
            result2 = handler2.handle("post_channel", {"channel": "general", "text": "Second run"})
        assert "Error" in result2
        assert "クールダウン" in result2

    def test_post_channel_cross_run_allows_after_cooldown(self, tmp_path: Path) -> None:
        """cooldown経過後は投稿が許可される。"""
        handler1, messenger, shared_dir = _make_handler(tmp_path)

        # First run: post to general
        with patch("core.config.models.load_config") as mock_cfg:
            mock_cfg.return_value = MagicMock()
            mock_cfg.return_value.heartbeat.channel_post_cooldown_s = 300
            handler1.handle("post_channel", {"channel": "general", "text": "First run"})

        # Simulate time passing by rewriting the channel file with an old timestamp
        channel_file = shared_dir / "channels" / "general.jsonl"
        old_ts = (now_jst() - timedelta(seconds=600)).isoformat()
        channel_file.write_text(
            json.dumps({"ts": old_ts, "from": "alice", "text": "First run", "source": "anima"}, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

        # Second run: new handler
        handler2, _, _ = _make_handler(tmp_path, "alice")

        with patch("core.config.models.load_config") as mock_cfg:
            mock_cfg.return_value = MagicMock()
            mock_cfg.return_value.heartbeat.channel_post_cooldown_s = 300
            result2 = handler2.handle("post_channel", {"channel": "general", "text": "Second run"})
        assert "Posted to #general" in result2


# ── messenger last_post_by ──────────────────────────────────


class TestLastPostBy:
    """Messenger.last_post_by のユニットテスト。"""

    def test_last_post_by_returns_most_recent(self, tmp_path: Path) -> None:
        """最新の自分の投稿を正しく返す。"""
        shared_dir = tmp_path / "shared"
        shared_dir.mkdir()
        (shared_dir / "inbox").mkdir()
        (shared_dir / "channels").mkdir()

        messenger = Messenger(shared_dir, "alice")
        messenger.post_channel("general", "First post")
        messenger.post_channel("general", "Second post")

        last = messenger.last_post_by("alice", "general")
        assert last is not None
        assert last["text"] == "Second post"
        assert last["from"] == "alice"

    def test_last_post_by_returns_none_when_no_posts(self, tmp_path: Path) -> None:
        """投稿がない場合はNoneを返す。"""
        shared_dir = tmp_path / "shared"
        shared_dir.mkdir()
        (shared_dir / "inbox").mkdir()
        (shared_dir / "channels").mkdir()

        messenger = Messenger(shared_dir, "alice")

        result = messenger.last_post_by("alice", "general")
        assert result is None


# ── cascade_limiter file-based ──────────────────────────────


class TestCascadeLimiterFileBased:
    """cascade_limiter ファイルベース化のユニットテスト。"""

    def test_depth_limiter_blocks_on_exceeded(self, tmp_path: Path) -> None:
        """depth超過でFalseを返す。"""
        with patch("core.cascade_limiter.load_config") as mock_cfg:
            mock_cfg.return_value = MagicMock()
            mock_cfg.return_value.heartbeat.depth_window_s = 600
            mock_cfg.return_value.heartbeat.max_depth = 3

            from core.cascade_limiter import ConversationDepthLimiter

            limiter = ConversationDepthLimiter(window_s=600, max_depth=3)

        # Create anima dir with activity log entries exceeding depth
        anima_dir = tmp_path / "animas" / "alice"
        (anima_dir / "activity_log").mkdir(parents=True)

        today = now_jst().strftime("%Y-%m-%d")
        log_file = anima_dir / "activity_log" / f"{today}.jsonl"

        # Write 4 entries (exceeds max_depth=3)
        entries = []
        for i in range(4):
            ts = (now_jst() - timedelta(seconds=60 * (4 - i))).isoformat()
            entry = {"ts": ts, "type": "dm_sent", "content": f"msg {i}", "to": "bob"}
            entries.append(json.dumps(entry, ensure_ascii=False))
        log_file.write_text("\n".join(entries) + "\n", encoding="utf-8")

        result = limiter.check_depth("alice", "bob", anima_dir)
        assert result is False

    def test_depth_limiter_allows_under_limit(self, tmp_path: Path) -> None:
        """limit内でTrueを返す。"""
        with patch("core.cascade_limiter.load_config") as mock_cfg:
            mock_cfg.return_value = MagicMock()
            mock_cfg.return_value.heartbeat.depth_window_s = 600
            mock_cfg.return_value.heartbeat.max_depth = 6

            from core.cascade_limiter import ConversationDepthLimiter

            limiter = ConversationDepthLimiter(window_s=600, max_depth=6)

        anima_dir = tmp_path / "animas" / "alice"
        (anima_dir / "activity_log").mkdir(parents=True)

        today = now_jst().strftime("%Y-%m-%d")
        log_file = anima_dir / "activity_log" / f"{today}.jsonl"

        # Write 2 entries (under max_depth=6)
        entries = []
        for i in range(2):
            ts = (now_jst() - timedelta(seconds=60 * (2 - i))).isoformat()
            entry = {"ts": ts, "type": "dm_sent", "content": f"msg {i}", "to": "bob"}
            entries.append(json.dumps(entry, ensure_ascii=False))
        log_file.write_text("\n".join(entries) + "\n", encoding="utf-8")

        result = limiter.check_depth("alice", "bob", anima_dir)
        assert result is True

    def test_depth_limiter_fail_open_on_missing_log(self, tmp_path: Path) -> None:
        """アクティビティログがない場合はTrue（fail-open）を返す。"""
        with patch("core.cascade_limiter.load_config") as mock_cfg:
            mock_cfg.return_value = MagicMock()
            mock_cfg.return_value.heartbeat.depth_window_s = 600
            mock_cfg.return_value.heartbeat.max_depth = 3

            from core.cascade_limiter import ConversationDepthLimiter

            limiter = ConversationDepthLimiter(window_s=600, max_depth=3)

        # anima_dir exists but has no activity_log
        anima_dir = tmp_path / "animas" / "alice"
        anima_dir.mkdir(parents=True)

        result = limiter.check_depth("alice", "bob", anima_dir)
        assert result is True


# ── Priming outbound section (via PrimingEngine) ────────────


class TestCollectRecentOutbound:
    """PrimingEngine._collect_recent_outbound のユニットテスト。"""

    @pytest.mark.asyncio
    async def test_collect_recent_outbound_with_entries(self, tmp_path: Path) -> None:
        """エントリありで正しいフォーマットのセクションを生成する。"""
        from core.memory.priming import PrimingEngine

        anima_dir = tmp_path / "animas" / "alice"
        (anima_dir / "activity_log").mkdir(parents=True)
        (anima_dir / "knowledge").mkdir(parents=True)

        today = now_jst().strftime("%Y-%m-%d")
        log_file = anima_dir / "activity_log" / f"{today}.jsonl"

        ts1 = (now_jst() - timedelta(minutes=30)).isoformat()
        ts2 = (now_jst() - timedelta(minutes=15)).isoformat()

        entries = [
            json.dumps({"ts": ts1, "type": "channel_post", "content": "Hello general", "channel": "general"}, ensure_ascii=False),
            json.dumps({"ts": ts2, "type": "message_sent", "content": "Hi bob", "to": "bob"}, ensure_ascii=False),
        ]
        log_file.write_text("\n".join(entries) + "\n", encoding="utf-8")

        engine = PrimingEngine(anima_dir)
        result = await engine._collect_recent_outbound()

        assert "直近のアウトバウンド行動" in result
        assert "#general に投稿済み" in result
        assert "bob にメッセージ送信済み" in result

    @pytest.mark.asyncio
    async def test_collect_recent_outbound_empty(self, tmp_path: Path) -> None:
        """エントリなしで空文字列を返す。"""
        from core.memory.priming import PrimingEngine

        anima_dir = tmp_path / "animas" / "alice"
        (anima_dir / "activity_log").mkdir(parents=True)
        (anima_dir / "knowledge").mkdir(parents=True)

        engine = PrimingEngine(anima_dir)
        result = await engine._collect_recent_outbound()
        assert result == ""
