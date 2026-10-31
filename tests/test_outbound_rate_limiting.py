"""Unit tests for per-run Board behavior and priming outbound sections."""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from core.memory import MemoryManager
from core.messaging.messenger import Messenger
from core.time_utils import now_jst
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
    channels_dir = shared_dir / "channels"
    channels_dir.mkdir(exist_ok=True)
    # Pre-create common board channels (post_channel no longer auto-creates)
    for name in ("general", "ops"):
        path = channels_dir / f"{name}.jsonl"
        if not path.exists():
            path.write_text("", encoding="utf-8")

    anima_dir = tmp_path / "animas" / anima_name
    anima_dir.mkdir(parents=True, exist_ok=True)
    (anima_dir / "IDENTITY.md").write_text(f"# {anima_name}\n", encoding="utf-8")
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

        result2 = handler.handle("post_channel", {"channel": "general", "text": "After reset"})
        assert "Posted to #general" in result2


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
            json.dumps(
                {"ts": ts1, "type": "channel_post", "content": "Hello general", "channel": "general"},
                ensure_ascii=False,
            ),
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
