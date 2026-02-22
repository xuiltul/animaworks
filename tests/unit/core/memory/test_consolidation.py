from __future__ import annotations
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.


"""Tests for memory consolidation engine."""

import os
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import pytest


@pytest.fixture
def temp_anima_dir(tmp_path: Path) -> Path:
    """Create a temporary anima directory structure."""
    anima_dir = tmp_path / "test_anima"
    episodes_dir = anima_dir / "episodes"
    knowledge_dir = anima_dir / "knowledge"
    episodes_dir.mkdir(parents=True)
    knowledge_dir.mkdir(parents=True)
    return anima_dir


@pytest.fixture
def consolidation_engine(temp_anima_dir: Path):
    """Create a ConsolidationEngine instance."""
    from core.memory.consolidation import ConsolidationEngine

    return ConsolidationEngine(
        anima_dir=temp_anima_dir,
        anima_name="test_anima",
    )


class TestEpisodeCollection:
    """Test episode collection functionality."""

    def test_collect_recent_episodes_empty(self, consolidation_engine):
        """Test collecting episodes when no episode files exist."""
        entries = consolidation_engine._collect_recent_episodes(hours=24)
        assert entries == []

    def test_collect_recent_episodes_with_data(self, consolidation_engine):
        """Test collecting episodes from existing files."""
        # Create today's episode file
        today = datetime.now().date()
        episode_file = consolidation_engine.episodes_dir / f"{today}.md"

        episode_content = """# エピソード記憶

## 09:00 — 朝のミーティング

**相手**: 山田
**トピック**: プロジェクト進捗確認
**要点**:
- Phase 2の実装が完了
- 次はPhase 3の計画

## 14:30 — バグ修正対応

**相手**: システム
**トピック**: 不具合調査
**要点**:
- メモリリークを発見
- 修正完了
"""
        episode_file.write_text(episode_content, encoding="utf-8")

        entries = consolidation_engine._collect_recent_episodes(hours=24)

        assert len(entries) == 2
        assert entries[0]["time"] == "14:30"  # Newest first
        assert entries[1]["time"] == "09:00"

    def test_collect_recent_episodes_with_cutoff(self, consolidation_engine):
        """Test that old episodes are excluded."""
        # Create yesterday's episode file
        yesterday = datetime.now().date() - timedelta(days=1)
        episode_file = consolidation_engine.episodes_dir / f"{yesterday}.md"

        episode_content = """# エピソード記憶

## 09:00 — 古いエピソード

**要点**: これは古いエピソード
"""
        episode_file.write_text(episode_content, encoding="utf-8")

        # Collect only last 1 hour
        entries = consolidation_engine._collect_recent_episodes(hours=1)

        # Should be empty since episode is from yesterday
        assert len(entries) == 0


class TestKnowledgeManagement:
    """Test knowledge file management."""

    def test_list_knowledge_files_empty(self, consolidation_engine):
        """Test listing knowledge files when none exist."""
        files = consolidation_engine._list_knowledge_files()
        assert files == []

    def test_list_knowledge_files_with_data(self, consolidation_engine):
        """Test listing existing knowledge files."""
        # Create some knowledge files
        (consolidation_engine.knowledge_dir / "test-knowledge.md").write_text(
            "# Test Knowledge", encoding="utf-8"
        )
        (consolidation_engine.knowledge_dir / "another-topic.md").write_text(
            "# Another Topic", encoding="utf-8"
        )

        files = consolidation_engine._list_knowledge_files()

        assert len(files) == 2
        assert "another-topic.md" in files
        assert "test-knowledge.md" in files


class TestEpisodeCollectionGlobAndFallback:
    """Test glob-based episode file discovery, mtime fallback, and deduplication."""

    def test_collect_suffixed_episode_files(self, consolidation_engine):
        """Suffixed files with ## HH:MM headers are discovered and parsed."""
        today = datetime.now().date()
        episode_file = (
            consolidation_engine.episodes_dir / f"{today}_heartbeat_check.md"
        )
        episode_file.write_text(
            "# Heartbeat Check\n\n"
            "## 11:00 — サーバー監視\n\n"
            "**要点**: CPU使用率は正常\n\n"
            "## 11:30 — メモリチェック\n\n"
            "**要点**: メモリ使用量は50%以下\n",
            encoding="utf-8",
        )

        entries = consolidation_engine._collect_recent_episodes(hours=24)

        assert len(entries) == 2
        times = {e["time"] for e in entries}
        assert "11:00" in times
        assert "11:30" in times
        assert entries[0]["time"] == "11:30"  # newest first
        assert entries[1]["time"] == "11:00"

    def test_collect_suffixed_only_no_standard(self, consolidation_engine):
        """Collection works when only suffixed files exist (no YYYY-MM-DD.md)."""
        today = datetime.now().date()
        # No standard file — only suffixed
        suffixed = consolidation_engine.episodes_dir / f"{today}_cron_run.md"
        suffixed.write_text(
            "## 08:00 — Cronバッチ処理\n\n"
            "**要点**: 全ジョブ成功\n",
            encoding="utf-8",
        )

        entries = consolidation_engine._collect_recent_episodes(hours=24)

        assert len(entries) == 1
        assert entries[0]["time"] == "08:00"
        assert "Cronバッチ処理" in entries[0]["content"]

    def test_collect_mixed_standard_and_suffixed(self, consolidation_engine):
        """Both standard and suffixed files are collected together."""
        today = datetime.now().date()

        # Standard file
        standard = consolidation_engine.episodes_dir / f"{today}.md"
        standard.write_text(
            "## 09:00 — 朝会\n\n"
            "**要点**: 進捗共有\n",
            encoding="utf-8",
        )

        # Suffixed file
        suffixed = consolidation_engine.episodes_dir / f"{today}_heartbeat.md"
        suffixed.write_text(
            "## 12:00 — 定期巡回\n\n"
            "**要点**: 異常なし\n",
            encoding="utf-8",
        )

        entries = consolidation_engine._collect_recent_episodes(hours=24)

        assert len(entries) == 2
        times = {e["time"] for e in entries}
        assert "09:00" in times
        assert "12:00" in times

    def test_fallback_no_time_headers(self, consolidation_engine):
        """Files without ## HH:MM headers use mtime; entire content is one entry."""
        today = datetime.now().date()
        episode_file = (
            consolidation_engine.episodes_dir / f"{today}_raw_notes.md"
        )
        raw_content = "手動メモ: 今日は特に問題なし。全システム正常稼働中。"
        episode_file.write_text(raw_content, encoding="utf-8")

        # Set mtime to a recent time (within the cutoff window)
        recent_ts = (datetime.now() - timedelta(hours=1)).timestamp()
        os.utime(episode_file, (recent_ts, recent_ts))

        entries = consolidation_engine._collect_recent_episodes(hours=24)

        assert len(entries) == 1
        assert entries[0]["content"] == raw_content
        assert entries[0]["date"] == str(today)
        # The time should come from the file mtime
        expected_time = datetime.fromtimestamp(recent_ts).strftime("%H:%M")
        assert entries[0]["time"] == expected_time

    def test_fallback_mtime_respects_cutoff(self, consolidation_engine):
        """Files whose mtime falls outside the cutoff window are excluded."""
        today = datetime.now().date()
        episode_file = (
            consolidation_engine.episodes_dir / f"{today}_old_notes.md"
        )
        episode_file.write_text(
            "古いメモ: これは昨日の内容",
            encoding="utf-8",
        )

        # Set mtime to 48 hours ago — outside the default 24h window
        old_ts = (datetime.now() - timedelta(hours=48)).timestamp()
        os.utime(episode_file, (old_ts, old_ts))

        entries = consolidation_engine._collect_recent_episodes(hours=24)

        assert len(entries) == 0

    def test_dedup_identical_content(self, consolidation_engine):
        """Duplicate entries (same first 200 chars) are deduplicated to one."""
        today = datetime.now().date()
        shared_body = "重複テスト内容: " + "A" * 200

        # Standard file
        standard = consolidation_engine.episodes_dir / f"{today}.md"
        standard.write_text(
            f"## 10:00 — イベントA\n\n{shared_body}\n",
            encoding="utf-8",
        )

        # Suffixed file with identical entry body
        suffixed = consolidation_engine.episodes_dir / f"{today}_copy.md"
        suffixed.write_text(
            f"## 10:00 — イベントA\n\n{shared_body}\n",
            encoding="utf-8",
        )

        entries = consolidation_engine._collect_recent_episodes(hours=24)

        # Only one entry should survive deduplication
        assert len(entries) == 1

    def test_dedup_different_content(self, consolidation_engine):
        """Entries with different content (first 200 chars) are both kept."""
        today = datetime.now().date()

        # Standard file
        standard = consolidation_engine.episodes_dir / f"{today}.md"
        standard.write_text(
            "## 10:00 — イベントA\n\n"
            "内容A: これはイベントAの詳細です。\n",
            encoding="utf-8",
        )

        # Suffixed file with different body
        suffixed = consolidation_engine.episodes_dir / f"{today}_other.md"
        suffixed.write_text(
            "## 10:30 — イベントB\n\n"
            "内容B: これはイベントBの詳細です。\n",
            encoding="utf-8",
        )

        entries = consolidation_engine._collect_recent_episodes(hours=24)

        assert len(entries) == 2
        contents = {e["content"] for e in entries}
        assert any("イベントA" in c for c in contents)
        assert any("イベントB" in c for c in contents)

    def test_collect_multiple_suffixed_files(self, consolidation_engine):
        """Multiple suffixed files for the same date are all collected."""
        today = datetime.now().date()

        files_data = [
            (f"{today}_heartbeat.md", "## 08:00 — ハートビート1\n\nチェック完了\n"),
            (f"{today}_cron.md", "## 09:00 — Cronジョブ\n\nバッチ成功\n"),
            (f"{today}_alert.md", "## 10:00 — アラート対応\n\nアラート解消\n"),
        ]

        for filename, content in files_data:
            (consolidation_engine.episodes_dir / filename).write_text(
                content, encoding="utf-8"
            )

        entries = consolidation_engine._collect_recent_episodes(hours=24)

        assert len(entries) == 3
        times = sorted(e["time"] for e in entries)
        assert times == ["08:00", "09:00", "10:00"]


# ── Resolved Events Collection Tests ─────────────────────────


class TestCollectResolvedEventsMeta:
    """Test that _collect_resolved_events includes meta field."""

    def test_resolved_events_include_meta(self, temp_anima_dir: Path) -> None:
        """_collect_resolved_events should return dicts with 'meta' key."""
        from dataclasses import dataclass, field
        from typing import Any

        from core.memory.consolidation import ConsolidationEngine

        engine = ConsolidationEngine(
            anima_dir=temp_anima_dir, anima_name="test_anima",
        )

        @dataclass
        class FakeEntry:
            ts: str = "2026-02-22T10:00:00"
            type: str = "issue_resolved"
            content: str = "問題を解決した"
            summary: str = "解決完了"
            meta: dict[str, Any] = field(default_factory=lambda: {
                "issue_type": "server_down",
                "severity": "high",
            })

        fake_entries = [FakeEntry()]

        with patch(
            "core.memory.activity.ActivityLogger.recent",
            return_value=fake_entries,
        ):
            result = engine._collect_resolved_events(hours=24)

        # Result should contain at least one entry
        assert len(result) == 1
        # The 'meta' field should be included in the result dict
        assert "meta" in result[0]
        assert result[0]["meta"]["issue_type"] == "server_down"
        assert result[0]["meta"]["severity"] == "high"

    def test_resolved_events_empty_meta(self, temp_anima_dir: Path) -> None:
        """_collect_resolved_events should handle entries with None meta."""
        from dataclasses import dataclass, field
        from typing import Any

        from core.memory.consolidation import ConsolidationEngine

        engine = ConsolidationEngine(
            anima_dir=temp_anima_dir, anima_name="test_anima",
        )

        @dataclass
        class FakeEntry:
            ts: str = "2026-02-22T10:00:00"
            type: str = "issue_resolved"
            content: str = "修正完了"
            summary: str = "バグ修正"
            meta: dict[str, Any] | None = None

        fake_entries = [FakeEntry()]

        with patch(
            "core.memory.activity.ActivityLogger.recent",
            return_value=fake_entries,
        ):
            result = engine._collect_resolved_events(hours=24)

        assert len(result) == 1
        # meta should default to empty dict when None (via `e.meta or {}`)
        assert result[0]["meta"] == {}

    def test_resolved_events_returns_empty_on_error(
        self, temp_anima_dir: Path,
    ) -> None:
        """_collect_resolved_events should return [] on exception."""
        from core.memory.consolidation import ConsolidationEngine

        engine = ConsolidationEngine(
            anima_dir=temp_anima_dir, anima_name="test_anima",
        )

        with patch(
            "core.memory.activity.ActivityLogger.recent",
            side_effect=RuntimeError("Activity log unavailable"),
        ):
            result = engine._collect_resolved_events(hours=24)

        # Errors should be caught and return empty list
        assert result == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
