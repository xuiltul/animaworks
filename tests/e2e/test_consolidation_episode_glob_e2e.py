from __future__ import annotations
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.


"""E2E tests for daily_consolidate() with glob-based episode file discovery.

Verifies the full consolidation pipeline when episodes are stored in
suffixed files (YYYY-MM-DD_xxx.md) in addition to the standard
YYYY-MM-DD.md format.
"""

import os
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ── Fixtures ────────────────────────────────────────────────────────


@pytest.fixture
def temp_anima_dir(tmp_path: Path) -> Path:
    """Create a temporary anima directory structure for E2E tests."""
    anima_dir = tmp_path / "test_anima"
    episodes_dir = anima_dir / "episodes"
    knowledge_dir = anima_dir / "knowledge"
    episodes_dir.mkdir(parents=True)
    knowledge_dir.mkdir(parents=True)
    return anima_dir


@pytest.fixture
def consolidation_engine(temp_anima_dir: Path):
    """Create a ConsolidationEngine instance for E2E tests."""
    from core.memory.consolidation import ConsolidationEngine

    return ConsolidationEngine(
        anima_dir=temp_anima_dir,
        anima_name="test_anima",
    )


# ── Helper ──────────────────────────────────────────────────────────


def _make_mock_llm_response(knowledge_filename: str, knowledge_body: str):
    """Build a mock litellm.acompletion return value.

    The response follows the structured format expected by
    ``ConsolidationEngine._merge_to_knowledge()``.
    """
    llm_text = (
        "## 既存ファイル更新\n(なし)\n\n"
        f"## 新規ファイル作成\n"
        f"- ファイル名: knowledge/{knowledge_filename}\n"
        f"  内容: {knowledge_body}\n"
    )
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = llm_text
    return mock_response


# ── Tests ───────────────────────────────────────────────────────────


class TestDailyConsolidateWithSuffixedEpisodes:
    """E2E: daily_consolidate() discovers and processes suffixed episode files."""

    @pytest.mark.asyncio
    async def test_daily_consolidate_with_suffixed_episodes(
        self, consolidation_engine
    ):
        """Suffixed episode files are processed and knowledge is generated."""
        today = datetime.now().date()

        # Create only suffixed episode files — no standard file
        heartbeat = (
            consolidation_engine.episodes_dir / f"{today}_heartbeat_check.md"
        )
        heartbeat.write_text(
            "## 10:00 — サーバー監視\n\n"
            "**相手**: システム\n"
            "**要点**: CPU使用率は正常。メモリ使用量50%以下。\n\n"
            "## 10:30 — ディスクチェック\n\n"
            "**相手**: システム\n"
            "**要点**: ディスク空き容量は十分。\n",
            encoding="utf-8",
        )

        cron = consolidation_engine.episodes_dir / f"{today}_cron_batch.md"
        cron.write_text(
            "## 11:00 — バッチ処理完了\n\n"
            "**相手**: Cronスケジューラ\n"
            "**要点**: 日次レポート生成が完了。エラーなし。\n",
            encoding="utf-8",
        )

        mock_resp = _make_mock_llm_response(
            "server-health-knowledge.md",
            "# サーバー健全性\n\nCPU・メモリ・ディスクの日次監視で異常なし。",
        )

        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = mock_resp

            result = await consolidation_engine.daily_consolidate(
                min_episodes=1,
            )

        # 3 entries: 10:00, 10:30, 11:00
        assert result["skipped"] is False
        assert result["episodes_processed"] == 3
        assert len(result["knowledge_files_created"]) == 1
        assert "server-health-knowledge.md" in result["knowledge_files_created"]

        # Verify knowledge file exists on disk
        kf = consolidation_engine.knowledge_dir / "server-health-knowledge.md"
        assert kf.exists()
        content = kf.read_text(encoding="utf-8")
        assert "サーバー健全性" in content
        assert "auto_consolidated" in content

        # Verify LLM was called at least once (consolidation may invoke
        # multiple LLM calls for validation, duplicate-merge, etc.)
        assert mock_llm.await_count >= 1

    @pytest.mark.asyncio
    async def test_daily_consolidate_mixed_files(self, consolidation_engine):
        """Both standard and suffixed files are processed in daily consolidation."""
        today = datetime.now().date()

        # Standard file
        standard = consolidation_engine.episodes_dir / f"{today}.md"
        standard.write_text(
            "# エピソード記憶\n\n"
            "## 09:00 — 朝会\n\n"
            "**相手**: チームメンバー\n"
            "**要点**: 今日のタスク確認。Phase 3の計画を議論。\n",
            encoding="utf-8",
        )

        # Suffixed file
        suffixed = consolidation_engine.episodes_dir / f"{today}_heartbeat.md"
        suffixed.write_text(
            "## 12:00 — 定期巡回\n\n"
            "**相手**: システム\n"
            "**要点**: 全サービス正常稼働。レスポンスタイム良好。\n",
            encoding="utf-8",
        )

        # Another suffixed file without headers (fallback path)
        raw_file = consolidation_engine.episodes_dir / f"{today}_manual.md"
        raw_content = "手動メモ: クライアントから機能追加リクエストあり。要件整理が必要。"
        raw_file.write_text(raw_content, encoding="utf-8")
        recent_ts = (datetime.now() - timedelta(minutes=30)).timestamp()
        os.utime(raw_file, (recent_ts, recent_ts))

        mock_resp = _make_mock_llm_response(
            "daily-summary.md",
            "# 日次まとめ\n\n朝会でPhase 3議論。巡回正常。クライアントから機能追加リクエスト。",
        )

        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = mock_resp

            result = await consolidation_engine.daily_consolidate(
                min_episodes=1,
            )

        # 3 entries: 09:00 (standard), 12:00 (suffixed), mtime (raw)
        assert result["skipped"] is False
        assert result["episodes_processed"] == 3
        assert "daily-summary.md" in result["knowledge_files_created"]

        # Verify the LLM prompt received all 3 entries
        call_args = mock_llm.call_args
        prompt_content = call_args.kwargs["messages"][0]["content"]
        assert "朝会" in prompt_content
        assert "定期巡回" in prompt_content
        assert "手動メモ" in prompt_content
