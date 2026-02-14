from __future__ import annotations
# AnimaWorks - Digital Person Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# This file is part of AnimaWorks core/server, licensed under AGPL-3.0.
# See LICENSES/AGPL-3.0.txt for the full license text.


"""Tests for memory consolidation engine."""

import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.fixture
def temp_person_dir(tmp_path: Path) -> Path:
    """Create a temporary person directory structure."""
    person_dir = tmp_path / "test_person"
    episodes_dir = person_dir / "episodes"
    knowledge_dir = person_dir / "knowledge"
    episodes_dir.mkdir(parents=True)
    knowledge_dir.mkdir(parents=True)
    return person_dir


@pytest.fixture
def consolidation_engine(temp_person_dir: Path):
    """Create a ConsolidationEngine instance."""
    from core.memory.consolidation import ConsolidationEngine

    return ConsolidationEngine(
        person_dir=temp_person_dir,
        person_name="test_person",
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
        assert "AUTO-CONSOLIDATED" in content

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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
