"""E2E tests for outbound rate limiting — file I/O integration tests.

Tests post_channel rate limiting (per-run + cross-run), cascade_limiter
file-based depth counting, and priming outbound section generation
using real filesystem operations in temporary directories.
"""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from core.memory import MemoryManager
from core.memory.activity import ActivityLogger
from core.messenger import Messenger
from core.time_utils import now_iso, now_jst
from core.tooling.handler import ToolHandler


# ── Helpers ──────────────────────────────────────────────────


def _make_anima_dir(tmp_path: Path, name: str) -> Path:
    """Create a minimal anima directory with required files."""
    anima_dir = tmp_path / "animas" / name
    anima_dir.mkdir(parents=True, exist_ok=True)
    (anima_dir / "identity.md").write_text(
        f"# {name}\n\nテスト用Anima。\n", encoding="utf-8",
    )
    (anima_dir / "permissions.md").write_text(
        "# Permissions\n\n## メッセージング\n- send_message: OK\n- post_channel: OK\n",
        encoding="utf-8",
    )
    (anima_dir / "activity_log").mkdir(exist_ok=True)
    return anima_dir


def _make_shared_dir(tmp_path: Path) -> Path:
    """Create a shared directory with required subdirectories."""
    shared_dir = tmp_path / "shared"
    shared_dir.mkdir(parents=True, exist_ok=True)
    (shared_dir / "inbox").mkdir(exist_ok=True)
    (shared_dir / "channels").mkdir(exist_ok=True)
    (shared_dir / "dm_logs").mkdir(exist_ok=True)
    return shared_dir


def _make_tool_handler(
    anima_dir: Path,
    shared_dir: Path,
) -> ToolHandler:
    """Build a ToolHandler with a real Messenger and mock MemoryManager."""
    name = anima_dir.name
    memory = MagicMock(spec=MemoryManager)
    memory.read_permissions.return_value = ""
    messenger = Messenger(shared_dir, name)
    return ToolHandler(
        anima_dir=anima_dir,
        memory=memory,
        messenger=messenger,
    )


# ── E2E Test: post_channel rate limiting ────────────────────


@pytest.mark.e2e
class TestPostChannelRateLimitingE2E:
    """Messenger + ToolHandler を実際にインスタンス化し、
    post_channel の per-run + cross-run ガードが動作することを確認する。
    """

    def test_post_channel_rate_limiting_e2e(self, tmp_path: Path) -> None:
        """post_channel の per-run ガードと cross-run ガードが
        実際のファイルI/Oで正しく動作する。
        """
        shared_dir = _make_shared_dir(tmp_path)
        alice_dir = _make_anima_dir(tmp_path, "alice")

        handler1 = _make_tool_handler(alice_dir, shared_dir)

        # ── Per-run guard: 1回目は成功 ──
        result1 = handler1.handle("post_channel", {
            "channel": "general",
            "text": "Hello from first post",
        })
        assert "Posted to #general" in result1

        # ── Per-run guard: 同一チャネルへの2回目はブロック ──
        result2 = handler1.handle("post_channel", {
            "channel": "general",
            "text": "Second post attempt",
        })
        assert "Error" in result2
        assert "投稿済み" in result2

        # ── Per-run guard: 異なるチャネルは許可 ──
        result3 = handler1.handle("post_channel", {
            "channel": "ops",
            "text": "Ops message",
        })
        assert "Posted to #ops" in result3

        # ── Cross-run guard: 新しいハンドラ（新しいrun）でも cooldown 内はブロック ──
        handler2 = _make_tool_handler(alice_dir, shared_dir)

        with patch("core.config.models.load_config") as mock_cfg:
            mock_cfg.return_value = MagicMock()
            mock_cfg.return_value.locale = "ja"
            mock_cfg.return_value.heartbeat.channel_post_cooldown_s = 300
            result4 = handler2.handle("post_channel", {
                "channel": "general",
                "text": "New run post",
            })
        assert "Error" in result4
        assert "クールダウン" in result4

        # ── Verify channel file actually has the posts ──
        channel_file = shared_dir / "channels" / "general.jsonl"
        assert channel_file.exists()
        lines = channel_file.read_text(encoding="utf-8").strip().splitlines()
        # Only 1 successful post to general (second was blocked)
        general_posts = [json.loads(line) for line in lines]
        assert len(general_posts) == 1
        assert general_posts[0]["text"] == "Hello from first post"


# ── E2E Test: cascade_limiter file-based ────────────────────


@pytest.mark.e2e
class TestCascadeLimiterFileBasedE2E:
    """activity_log に JSONL を書き込み、
    depth_limiter が正しくカウントすることを確認する。
    """

    def test_cascade_limiter_file_based_e2e(self, tmp_path: Path) -> None:
        """実際のファイルI/Oで cascade_limiter が正しくdepthをカウントする。"""
        anima_dir = _make_anima_dir(tmp_path, "alice")

        # Write activity log entries manually
        today = now_jst().strftime("%Y-%m-%d")
        log_file = anima_dir / "activity_log" / f"{today}.jsonl"

        # Write exactly max_depth entries (should block)
        max_depth = 4
        entries = []
        for i in range(max_depth):
            ts = (now_jst() - timedelta(seconds=60 * (max_depth - i))).isoformat()
            entry_type = "dm_sent" if i % 2 == 0 else "dm_received"
            entry = {
                "ts": ts,
                "type": entry_type,
                "content": f"Message {i}",
                "to": "bob" if entry_type == "dm_sent" else "",
                "from": "bob" if entry_type == "dm_received" else "",
            }
            entries.append(json.dumps(entry, ensure_ascii=False))
        log_file.write_text("\n".join(entries) + "\n", encoding="utf-8")

        # Create limiter with low max_depth
        with patch("core.cascade_limiter.load_config") as mock_cfg:
            mock_cfg.return_value = MagicMock()
            mock_cfg.return_value.heartbeat.depth_window_s = 600
            mock_cfg.return_value.heartbeat.max_depth = max_depth

            from core.cascade_limiter import ConversationDepthLimiter

            limiter = ConversationDepthLimiter(window_s=600, max_depth=max_depth)

        # ── Should block (count == max_depth) ──
        assert limiter.check_depth("alice", "bob", anima_dir) is False

        # ── current_depth should report correct count ──
        depth = limiter.current_depth("alice", "bob", anima_dir)
        assert depth == max_depth

        # ── With higher limit, should allow ──
        with patch("core.cascade_limiter.load_config") as mock_cfg:
            mock_cfg.return_value = MagicMock()
            mock_cfg.return_value.heartbeat.depth_window_s = 600
            mock_cfg.return_value.heartbeat.max_depth = max_depth + 10

            limiter_lenient = ConversationDepthLimiter(
                window_s=600, max_depth=max_depth + 10,
            )

        assert limiter_lenient.check_depth("alice", "bob", anima_dir) is True

        # ── Entries outside window should not be counted ──
        old_entries = []
        for i in range(10):
            ts = (now_jst() - timedelta(seconds=1200 + 60 * i)).isoformat()
            entry = {"ts": ts, "type": "dm_sent", "content": f"Old msg {i}", "to": "bob"}
            old_entries.append(json.dumps(entry, ensure_ascii=False))
        # Prepend old entries to log file (they should be ignored by window filter)
        existing_content = log_file.read_text(encoding="utf-8")
        log_file.write_text(
            "\n".join(old_entries) + "\n" + existing_content,
            encoding="utf-8",
        )

        # depth should still be max_depth (old entries are outside window)
        depth_after = limiter.current_depth("alice", "bob", anima_dir)
        assert depth_after == max_depth


# ── E2E Test: priming outbound section ──────────────────────


@pytest.mark.e2e
class TestPrimingOutboundSectionE2E:
    """activity_log に channel_post と message_sent を書き込み、
    PrimingEngine._collect_recent_outbound が正しくセクションを生成することを確認する。
    """

    @pytest.mark.asyncio
    async def test_priming_outbound_section_e2e(self, tmp_path: Path) -> None:
        """実際のファイルI/Oで outbound section が正しいセクションを生成する。"""
        from core.memory.priming import PrimingEngine

        anima_dir = _make_anima_dir(tmp_path, "alice")
        (anima_dir / "knowledge").mkdir(exist_ok=True)

        today = now_jst().strftime("%Y-%m-%d")
        log_file = anima_dir / "activity_log" / f"{today}.jsonl"

        ts1 = (now_jst() - timedelta(minutes=45)).isoformat()
        ts2 = (now_jst() - timedelta(minutes=30)).isoformat()
        ts3 = (now_jst() - timedelta(minutes=10)).isoformat()

        entries = [
            json.dumps({
                "ts": ts1,
                "type": "channel_post",
                "content": "進捗報告です",
                "channel": "general",
            }, ensure_ascii=False),
            json.dumps({
                "ts": ts2,
                "type": "message_sent",
                "content": "確認お願いします",
                "to": "bob",
            }, ensure_ascii=False),
            json.dumps({
                "ts": ts3,
                "type": "message_sent",
                "content": "ありがとうございました",
                "to": "charlie",
            }, ensure_ascii=False),
        ]
        log_file.write_text("\n".join(entries) + "\n", encoding="utf-8")

        engine = PrimingEngine(anima_dir)
        result = await engine._collect_recent_outbound()

        # ── Section header present ──
        assert "直近のアウトバウンド行動" in result

        # ── channel_post entry formatted correctly ──
        assert "#general に投稿済み" in result
        assert "進捗報告です" in result

        # ── message_sent entries formatted correctly ──
        assert "bob にメッセージ送信済み" in result or "charlie にメッセージ送信済み" in result

        # ── max_entries=3 respected (all 3 should appear) ──
        assert "bob" in result
        assert "charlie" in result

    @pytest.mark.asyncio
    async def test_priming_outbound_section_old_entries_excluded(self, tmp_path: Path) -> None:
        """2時間以上前のエントリはセクションに含まれない。"""
        from core.memory.priming import PrimingEngine

        anima_dir = _make_anima_dir(tmp_path, "alice")
        (anima_dir / "knowledge").mkdir(exist_ok=True)

        today = now_jst().strftime("%Y-%m-%d")
        log_file = anima_dir / "activity_log" / f"{today}.jsonl"

        old_ts = (now_jst() - timedelta(hours=3)).isoformat()
        entries = [
            json.dumps({
                "ts": old_ts,
                "type": "channel_post",
                "content": "Ancient post",
                "channel": "general",
            }, ensure_ascii=False),
        ]
        log_file.write_text("\n".join(entries) + "\n", encoding="utf-8")

        engine = PrimingEngine(anima_dir)
        result = await engine._collect_recent_outbound()

        assert result == ""

    @pytest.mark.asyncio
    async def test_priming_outbound_section_empty_log(self, tmp_path: Path) -> None:
        """アクティビティログが空の場合は空文字列を返す。"""
        from core.memory.priming import PrimingEngine

        anima_dir = _make_anima_dir(tmp_path, "alice")
        (anima_dir / "knowledge").mkdir(exist_ok=True)

        engine = PrimingEngine(anima_dir)
        result = await engine._collect_recent_outbound()
        assert result == ""
