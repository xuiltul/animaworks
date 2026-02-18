from __future__ import annotations
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.


"""Tests for memory consolidation engine."""

import asyncio
import os
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

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

    def test_merge_to_knowledge_create_new(self, consolidation_engine):
        """Test creating new knowledge files."""
        llm_output = """## 既存ファイル更新
(なし)

## 新規ファイル作成
- ファイル名: knowledge/test-lesson.md
  内容: # テスト教訓

これは新しい教訓です。

- 重要なポイント1
- 重要なポイント2
"""

        files_created, files_updated = consolidation_engine._merge_to_knowledge(llm_output)

        assert len(files_created) == 1
        assert "test-lesson.md" in files_created
        assert len(files_updated) == 0

        # Verify file was created
        new_file = consolidation_engine.knowledge_dir / "test-lesson.md"
        assert new_file.exists()
        content = new_file.read_text(encoding="utf-8")
        assert "テスト教訓" in content
        # New format uses YAML frontmatter instead of text marker
        assert "auto_consolidated: true" in content or "AUTO-CONSOLIDATED" in content

    def test_merge_to_knowledge_update_existing(self, consolidation_engine):
        """Test updating existing knowledge files."""
        # Create existing file
        existing_file = consolidation_engine.knowledge_dir / "existing.md"
        existing_file.write_text("# 既存の知識\n\n元の内容", encoding="utf-8")

        llm_output = """## 既存ファイル更新
- ファイル名: knowledge/existing.md
  追加内容: ## 新しいセクション

追加された内容です。

## 新規ファイル作成
(なし)
"""

        files_created, files_updated = consolidation_engine._merge_to_knowledge(llm_output)

        assert len(files_created) == 0
        assert len(files_updated) == 1
        assert "existing.md" in files_updated

        # Verify file was updated
        content = existing_file.read_text(encoding="utf-8")
        assert "元の内容" in content
        assert "新しいセクション" in content
        assert "AUTO-CONSOLIDATED" in content


class TestConsolidationFlow:
    """Test the full consolidation flow."""

    @pytest.mark.asyncio
    async def test_daily_consolidate_skip_no_episodes(self, consolidation_engine):
        """Test that consolidation is skipped when no episodes exist."""
        result = await consolidation_engine.daily_consolidate(min_episodes=1)

        assert result["skipped"] is True
        assert result["episodes_processed"] == 0

    @pytest.mark.asyncio
    async def test_daily_consolidate_with_episodes(self, consolidation_engine):
        """Test daily consolidation with episodes."""
        # Create episode data
        today = datetime.now().date()
        episode_file = consolidation_engine.episodes_dir / f"{today}.md"

        episode_content = """## 10:00 — テストエピソード

**相手**: テストユーザー
**要点**: テスト内容
"""
        episode_file.write_text(episode_content, encoding="utf-8")

        # Mock LLM response
        mock_llm_response = """## 既存ファイル更新
(なし)

## 新規ファイル作成
- ファイル名: knowledge/test-auto-knowledge.md
  内容: # 自動生成された知識

テストから学んだ教訓
"""

        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_llm:
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = mock_llm_response
            mock_llm.return_value = mock_response

            result = await consolidation_engine.daily_consolidate(min_episodes=1)

        assert result["skipped"] is False
        assert result["episodes_processed"] == 1
        assert len(result["knowledge_files_created"]) == 1
        assert "test-auto-knowledge.md" in result["knowledge_files_created"]

        # Verify knowledge file was created
        knowledge_file = consolidation_engine.knowledge_dir / "test-auto-knowledge.md"
        assert knowledge_file.exists()


class TestWeeklyIntegration:
    """Test weekly integration (Phase 3)."""

    @pytest.mark.asyncio
    async def test_weekly_integrate_no_duplicates(self, consolidation_engine):
        """Test weekly integration when no duplicates are found."""
        # Create a knowledge file
        (consolidation_engine.knowledge_dir / "unique-knowledge.md").write_text(
            "# Unique Knowledge\n\nThis is unique content.",
            encoding="utf-8"
        )

        # Mock RAG to return no duplicates
        with patch.object(consolidation_engine, "_detect_duplicates", return_value=[]):
            with patch.object(consolidation_engine, "_compress_old_episodes", return_value=0):
                result = await consolidation_engine.weekly_integrate()

        assert result["skipped"] is False
        assert len(result["knowledge_files_merged"]) == 0
        assert result["episodes_compressed"] == 0

    @pytest.mark.asyncio
    async def test_compress_old_episodes(self, consolidation_engine):
        """Test episode compression for old episodes."""
        # Create old episode (35 days ago)
        old_date = datetime.now().date() - timedelta(days=35)
        old_episode = consolidation_engine.episodes_dir / f"{old_date}.md"

        old_content = """# エピソード記憶

## 09:00 — 古いミーティング

**相手**: テストユーザー
**トピック**: 古い話題
**要点**:
- 重要でない内容1
- 重要でない内容2
- 重要でない内容3
"""
        old_episode.write_text(old_content, encoding="utf-8")

        # Create recent episode (should not be compressed)
        recent_date = datetime.now().date() - timedelta(days=10)
        recent_episode = consolidation_engine.episodes_dir / f"{recent_date}.md"
        recent_episode.write_text(old_content, encoding="utf-8")

        # Mock LLM response
        mock_llm_response = f"""## {old_date} 要約
- 古いミーティングで話題を確認
- 重要でない内容のみ
"""

        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_llm:
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = mock_llm_response
            mock_llm.return_value = mock_response

            compressed_count = await consolidation_engine._compress_old_episodes(
                retention_days=30,
                model="test-model",
            )

        assert compressed_count == 1

        # Verify old episode was compressed
        compressed_content = old_episode.read_text(encoding="utf-8")
        assert "[COMPRESSED:" in compressed_content
        assert "要約" in compressed_content

        # Verify recent episode was not compressed
        recent_content = recent_episode.read_text(encoding="utf-8")
        assert "[COMPRESSED:" not in recent_content

    @pytest.mark.asyncio
    async def test_compress_old_episodes_skip_important(self, consolidation_engine):
        """Test that episodes with [IMPORTANT] tag are not compressed."""
        # Create old episode with IMPORTANT tag
        old_date = datetime.now().date() - timedelta(days=35)
        important_episode = consolidation_engine.episodes_dir / f"{old_date}.md"

        important_content = """# エピソード記憶

## 09:00 — 重要なミーティング [IMPORTANT]

**相手**: CEO
**トピック**: 重要な決定事項
**要点**:
- 重要な戦略決定
"""
        important_episode.write_text(important_content, encoding="utf-8")

        compressed_count = await consolidation_engine._compress_old_episodes(
            retention_days=30,
            model="test-model",
        )

        assert compressed_count == 0

        # Verify episode was not compressed
        content = important_episode.read_text(encoding="utf-8")
        assert "[COMPRESSED:" not in content
        assert "[IMPORTANT]" in content

    @pytest.mark.asyncio
    async def test_merge_knowledge_files(self, consolidation_engine):
        """Test merging duplicate knowledge files."""
        # Create duplicate files
        file1 = consolidation_engine.knowledge_dir / "chatwork-policy.md"
        file2 = consolidation_engine.knowledge_dir / "chatwork-response.md"

        file1.write_text("""# Chatwork対応方針

## 基本方針
- 迅速に返信する
- 丁寧な言葉遣い
""", encoding="utf-8")

        file2.write_text("""# Chatwork返信ルール

## 返信タイミング
- 営業時間内は30分以内に返信
- 緊急時は即座に返信
""", encoding="utf-8")

        duplicates = [
            ("chatwork-policy.md", "chatwork-response.md", 0.92)
        ]

        # Mock LLM response
        mock_llm_response = """## 統合ファイル名
chatwork-guidelines.md

## 統合内容
# Chatwork対応ガイドライン

## 基本方針
- 迅速に返信する
- 丁寧な言葉遣いを心がける

## 返信タイミング
- 営業時間内は30分以内に返信
- 緊急時は即座に返信
"""

        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_llm:
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = mock_llm_response
            mock_llm.return_value = mock_response

            merged_files = await consolidation_engine._merge_knowledge_files(
                duplicates,
                model="test-model",
            )

        assert len(merged_files) == 1
        assert "chatwork-policy.md + chatwork-response.md → chatwork-guidelines.md" in merged_files

        # Verify merged file was created
        merged_file = consolidation_engine.knowledge_dir / "chatwork-guidelines.md"
        assert merged_file.exists()
        content = merged_file.read_text(encoding="utf-8")
        assert "AUTO-MERGED" in content
        assert "Chatwork対応ガイドライン" in content

        # Verify original files were deleted
        assert not file1.exists()
        assert not file2.exists()

    @pytest.mark.asyncio
    async def test_weekly_integrate_full_flow(self, consolidation_engine):
        """Test full weekly integration flow."""
        # Create some knowledge files
        (consolidation_engine.knowledge_dir / "topic1.md").write_text(
            "# Topic 1\n\nContent 1",
            encoding="utf-8"
        )

        # Create old episode
        old_date = datetime.now().date() - timedelta(days=35)
        old_episode = consolidation_engine.episodes_dir / f"{old_date}.md"
        old_episode.write_text("## 09:00 — Old episode\nContent", encoding="utf-8")

        # Mock all methods
        with patch.object(consolidation_engine, "_detect_duplicates", return_value=[]):
            with patch.object(consolidation_engine, "_compress_old_episodes", return_value=1):
                with patch.object(consolidation_engine, "_rebuild_rag_index"):
                    result = await consolidation_engine.weekly_integrate()

        assert result["skipped"] is False
        assert result["episodes_compressed"] == 1


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


class TestDetectDuplicates:
    """Test _detect_duplicates with mocked RAG components."""

    @pytest.mark.asyncio
    async def test_detect_duplicates_finds_similar_files(self, consolidation_engine):
        """Duplicate detection correctly identifies similar knowledge files."""
        from dataclasses import dataclass

        @dataclass
        class FakeRetrievalResult:
            doc_id: str
            content: str
            score: float
            metadata: dict
            source_scores: dict

        # Create two knowledge files
        (consolidation_engine.knowledge_dir / "topic-a.md").write_text(
            "# Topic A\n\nSimilar content about API design.",
            encoding="utf-8",
        )
        (consolidation_engine.knowledge_dir / "topic-b.md").write_text(
            "# Topic B\n\nSimilar content about API design patterns.",
            encoding="utf-8",
        )

        mock_result = FakeRetrievalResult(
            doc_id="test/knowledge/topic-b.md#0",
            content="Similar content about API design patterns.",
            score=0.92,
            metadata={"source_file": "knowledge/topic-b.md"},
            source_scores={"vector": 0.92},
        )

        mock_retriever = MagicMock()
        mock_retriever.search.return_value = [mock_result]

        mock_indexer = MagicMock()
        mock_vector_store = MagicMock()

        with patch(
            "core.memory.rag.singleton.get_vector_store",
            return_value=mock_vector_store,
        ):
            with patch(
                "core.memory.rag.indexer.MemoryIndexer",
                return_value=mock_indexer,
            ):
                with patch(
                    "core.memory.rag.retriever.MemoryRetriever",
                    return_value=mock_retriever,
                ):
                    duplicates = await consolidation_engine._detect_duplicates(
                        threshold=0.85,
                    )

        assert len(duplicates) == 1
        assert duplicates[0][0] == "topic-a.md"
        assert duplicates[0][1] == "topic-b.md"
        assert duplicates[0][2] == 0.92

        # Verify search was called with anima_name keyword argument
        call_kwargs = mock_retriever.search.call_args
        assert call_kwargs.kwargs.get("anima_name") == "test_anima"

    @pytest.mark.asyncio
    async def test_detect_duplicates_below_threshold(self, consolidation_engine):
        """Files below the similarity threshold are not reported as duplicates."""
        from dataclasses import dataclass

        @dataclass
        class FakeRetrievalResult:
            doc_id: str
            content: str
            score: float
            metadata: dict
            source_scores: dict

        # Create two knowledge files
        (consolidation_engine.knowledge_dir / "file-x.md").write_text(
            "# File X\n\nContent about X.",
            encoding="utf-8",
        )
        (consolidation_engine.knowledge_dir / "file-y.md").write_text(
            "# File Y\n\nContent about Y.",
            encoding="utf-8",
        )

        mock_result = FakeRetrievalResult(
            doc_id="test/knowledge/file-y.md#0",
            content="Content about Y.",
            score=0.50,
            metadata={"source_file": "knowledge/file-y.md"},
            source_scores={"vector": 0.50},
        )

        mock_retriever = MagicMock()
        mock_retriever.search.return_value = [mock_result]
        mock_indexer = MagicMock()
        mock_vector_store = MagicMock()

        with patch(
            "core.memory.rag.singleton.get_vector_store",
            return_value=mock_vector_store,
        ):
            with patch(
                "core.memory.rag.indexer.MemoryIndexer",
                return_value=mock_indexer,
            ):
                with patch(
                    "core.memory.rag.retriever.MemoryRetriever",
                    return_value=mock_retriever,
                ):
                    duplicates = await consolidation_engine._detect_duplicates(
                        threshold=0.85,
                    )

        assert len(duplicates) == 0

    @pytest.mark.asyncio
    async def test_detect_duplicates_single_file(self, consolidation_engine):
        """No duplicates returned when only one knowledge file exists."""
        (consolidation_engine.knowledge_dir / "only-file.md").write_text(
            "# Only\n\nSingle file.",
            encoding="utf-8",
        )

        with patch(
            "core.memory.rag.singleton.get_vector_store",
        ):
            with patch(
                "core.memory.rag.indexer.MemoryIndexer",
            ):
                with patch(
                    "core.memory.rag.retriever.MemoryRetriever",
                ):
                    duplicates = await consolidation_engine._detect_duplicates(
                        threshold=0.85,
                    )

        assert duplicates == []

    @pytest.mark.asyncio
    async def test_detect_duplicates_rag_unavailable(self, consolidation_engine):
        """Returns empty list when RAG modules are not importable."""
        (consolidation_engine.knowledge_dir / "a.md").write_text("A", encoding="utf-8")
        (consolidation_engine.knowledge_dir / "b.md").write_text("B", encoding="utf-8")

        with patch.dict("sys.modules", {"core.memory.rag.singleton": None}):
            duplicates = await consolidation_engine._detect_duplicates()

        assert duplicates == []


class TestPathTraversalSanitization:
    """Test _sanitize_filepath and its integration in _merge_to_knowledge."""

    def test_sanitize_filepath_normal_name(self, consolidation_engine):
        """Normal filenames pass through unchanged."""
        from core.memory.consolidation import _sanitize_filepath

        base = consolidation_engine.knowledge_dir
        result = _sanitize_filepath(base, "valid-topic.md")
        assert result == base / "valid-topic.md"

    def test_sanitize_filepath_traversal_dotdot(self, consolidation_engine):
        """Path traversal with ../../ is caught and sanitized."""
        from core.memory.consolidation import _sanitize_filepath

        base = consolidation_engine.knowledge_dir
        result = _sanitize_filepath(base, "../../evil.md")
        assert result.parent == base
        assert result.name.endswith(".md")
        assert ".." not in str(result)

    def test_sanitize_filepath_complex_traversal(self, consolidation_engine):
        """Complex traversal like knowledge/../../../etc/passwd is sanitized."""
        from core.memory.consolidation import _sanitize_filepath

        base = consolidation_engine.knowledge_dir
        result = _sanitize_filepath(base, "knowledge/../../../etc/passwd")
        assert result.parent == base
        assert result.name.endswith(".md")
        assert "etc" not in str(result.parent)
        assert "passwd" in result.name  # the basename is kept but sanitized

    def test_sanitize_filepath_ensures_md_suffix(self, consolidation_engine):
        """Non-.md filenames get .md appended after sanitization."""
        from core.memory.consolidation import _sanitize_filepath

        base = consolidation_engine.knowledge_dir
        result = _sanitize_filepath(base, "../evil")
        assert result.name.endswith(".md")
        assert result.parent == base

    def test_merge_to_knowledge_traversal_in_update(self, consolidation_engine):
        """Path traversal in update filename is sanitized during merge."""
        # Create a file at the sanitized name location
        safe_name = consolidation_engine.knowledge_dir / "evil.md"
        safe_name.write_text("# Existing\n\nContent.", encoding="utf-8")

        llm_output = """## 既存ファイル更新
- ファイル名: ../../evil.md
  追加内容: Malicious content

## 新規ファイル作成
(なし)
"""
        files_created, files_updated = consolidation_engine._merge_to_knowledge(
            llm_output,
        )

        # Should NOT have written outside knowledge_dir
        # The sanitized filename should be used
        assert len(files_updated) <= 1
        assert len(files_created) <= 1
        # Verify no file was created outside knowledge_dir
        parent_dir = consolidation_engine.knowledge_dir.parent
        for f in parent_dir.glob("evil.md"):
            assert f.parent == consolidation_engine.knowledge_dir

    def test_merge_to_knowledge_traversal_in_create(self, consolidation_engine):
        """Path traversal in create filename is sanitized during merge."""
        llm_output = """## 既存ファイル更新
(なし)

## 新規ファイル作成
- ファイル名: knowledge/../../../etc/passwd
  内容: Malicious content
"""
        files_created, files_updated = consolidation_engine._merge_to_knowledge(
            llm_output,
        )

        # The file should be created inside knowledge_dir, not at /etc/passwd
        assert not Path("/etc/passwd").exists() or Path("/etc/passwd").read_text() != "Malicious content"
        for created in files_created:
            filepath = consolidation_engine.knowledge_dir / created
            assert filepath.parent == consolidation_engine.knowledge_dir

    def test_merge_to_knowledge_normal_filenames_unchanged(self, consolidation_engine):
        """Normal filenames are not altered by sanitization."""
        llm_output = """## 既存ファイル更新
(なし)

## 新規ファイル作成
- ファイル名: knowledge/my-topic.md
  内容: # My Topic

Some knowledge content.
"""
        files_created, files_updated = consolidation_engine._merge_to_knowledge(
            llm_output,
        )

        assert len(files_created) == 1
        assert "my-topic.md" in files_created
        assert (consolidation_engine.knowledge_dir / "my-topic.md").exists()

    @pytest.mark.asyncio
    async def test_merge_knowledge_files_traversal_sanitized(
        self, consolidation_engine,
    ):
        """Path traversal in merge filename from LLM is sanitized."""
        file1 = consolidation_engine.knowledge_dir / "topic-a.md"
        file2 = consolidation_engine.knowledge_dir / "topic-b.md"
        file1.write_text("# Topic A\n\nContent A.", encoding="utf-8")
        file2.write_text("# Topic B\n\nContent B.", encoding="utf-8")

        duplicates = [("topic-a.md", "topic-b.md", 0.92)]

        # LLM tries to write outside knowledge_dir
        llm_response = (
            "## 統合ファイル名\n"
            "../../etc/evil.md\n\n"
            "## 統合内容\n"
            "# Merged\n\nMerged content."
        )

        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_llm:
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = llm_response
            mock_llm.return_value = mock_response

            merged = await consolidation_engine._merge_knowledge_files(
                duplicates, model="test-model",
            )

        assert len(merged) == 1
        # The merged file should be inside knowledge_dir
        for entry in merged:
            # entry format: "topic-a.md + topic-b.md → sanitized_name.md"
            merged_name = entry.split("→ ")[1].strip()
            merged_path = consolidation_engine.knowledge_dir / merged_name
            assert merged_path.exists()
            assert merged_path.parent == consolidation_engine.knowledge_dir


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
